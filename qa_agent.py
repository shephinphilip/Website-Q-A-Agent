import argparse
import requests
import time
import json
import os
import re
import hashlib
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import trafilatura
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from fastapi import FastAPI, HTTPException

# ------------- Helper Function for Section Extraction ------------------

def extract_sections(soup):
    """
    Extracts sections from the BeautifulSoup object based on headings (h1 to h6).
    
    This function iterates through all elements in the soup and groups content
    under heading tags into sections. Each section contains a title (from the heading)
    and a list of content elements (paragraphs, divs, spans, list items).
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        list: A list of dictionaries, each containing 'title' and 'content' for a section.
    """
    sections = []
    current_section = None
    for element in soup.find_all(True):  # Find all tags
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if current_section:
                sections.append(current_section)
            current_section = {'title': element.get_text().strip(), 'content': []}
        elif current_section and element.name in ['p', 'div', 'span', 'li']:
            current_section['content'].append(element.get_text().strip())
    if current_section:
        sections.append(current_section)
    # Filter out sections with empty content
    sections = [s for s in sections if s['content']]
    return sections

# ------------- Classes for Crawling, Indexing, and QA ------------------

class Crawler:
    """
    Handles recursive crawling of a website using BeautifulSoup.
    
    This class initializes with a base URL and a maximum depth for crawling.
    It provides a method to crawl the website recursively, extracting content
    from each page based on headings or falling back to full page content.
    """
    def __init__(self, base_url, max_depth=2):
        """
        Initializes the Crawler with a base URL and maximum crawling depth.
        
        Args:
            base_url (str): The base URL of the website to crawl.
            max_depth (int, optional): Maximum depth for recursive crawling. Defaults to 2.
        """
        self.base_url = base_url.rstrip('/')
        self.max_depth = max_depth

    def crawl(self, url, visited=None, depth=0):
        """
        Recursively scrapes the website starting from the given URL.
        
        This method crawls the website up to the specified maximum depth,
        extracting content from each page. It uses retry logic for network requests
        and extracts sections based on headings. If no sections are found, it falls
        back to extracting the full page content using trafilatura.
        
        Args:
            url (str): The URL to start crawling from.
            visited (set, optional): A set of visited URLs to avoid revisiting. Defaults to None.
            depth (int, optional): The current depth of crawling. Defaults to 0.
        
        Returns:
            list: A list of dictionaries containing 'url', 'page_title', 'section_title', and 'content' for each section.
        """
        if visited is None:
            visited = set()
        if url in visited or not url.startswith(self.base_url) or depth > self.max_depth:
            return []

        visited.add(url)
        print(f"Crawling: {url} (Depth: {depth})")

        # Retry logic for network requests
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    print(f"Failed to crawl {url} after 3 attempts: {e}")
                    return []
                time.sleep(2)

        soup = BeautifulSoup(response.text, 'html.parser')
        page_title = soup.title.string if soup.title else 'Untitled'

        # Extract sections based on headings
        sections = extract_sections(soup)
        documents = []
        for section in sections:
            section_content = ' '.join(section['content'])
            if section_content:
                documents.append({
                    'url': url,
                    'page_title': page_title,
                    'section_title': section['title'],
                    'content': section_content
                })

        # Fallback: Extract full content if no sections are found
        if not sections:
            content = trafilatura.extract(response.text)
            if content:
                documents.append({
                    'url': url,
                    'page_title': page_title,
                    'section_title': 'Full Page',
                    'content': content
                })

        # Collect links for recursion
        links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
        doc_links = [link for link in links if link.startswith(self.base_url)]

        for link in doc_links:
            documents.extend(self.crawl(link, visited, depth + 1))

        return documents

class Indexer:
    """
    Indexes documentation content for efficient querying.
    
    This class takes the crawled documents, creates chunks from the sections,
    generates embeddings using SentenceTransformer, and builds a FAISS index
    for fast similarity search.
    """
    def __init__(self, documents):
        """
        Initializes the Indexer with the crawled documents and builds the index.
        
        Args:
            documents (list): A list of dictionaries containing document sections.
        """
        self.documents = documents
        self.chunks = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.build_index()

    def build_index(self):
        """
        Builds a FAISS index from the document sections.
        
        This method creates chunks from the document sections, generates embeddings,
        and adds them to a FAISS index for efficient similarity search.
        """
        for doc in self.documents:
            self.chunks.append({
                'text': doc['content'],
                'url': doc['url'],
                'page_title': doc['page_title'],
                'section_title': doc['section_title']
            })

        if not self.chunks:
            raise ValueError("No valid content found to index.")

        texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)

    def search(self, query, k=5):
        """
        Searches for the top-k chunks most similar to the query.
        
        Args:
            query (str): The user's query.
            k (int, optional): The number of top chunks to retrieve. Defaults to 5.
        
        Returns:
            list: A list of the top-k chunks most similar to the query.
        """
        query_emb = self.model.encode([query])
        distances, indices = self.faiss_index.search(query_emb, k)
        return [self.chunks[i] for i in indices[0]]

class QASystem:
    """
    Answers user queries based on indexed documentation.
    """
    def __init__(self, indexer):
        """
        Initializes the QASystem with the indexer and sets up the QA pipeline.

        Args:
            indexer (Indexer): The indexer containing the FAISS index and chunks.
        """
        self.indexer = indexer
        self.qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

    def answer(self, question, k=5, threshold=0.5):
        """
        Generates an answer to the user's question from the most relevant chunks.

        This method retrieves the top-k relevant chunks using the indexer,
        applies the QA model to each chunk, and returns multiple relevant responses.

        Args:
            question (str): The user's question.
            k (int, optional): The number of top chunks to consider. Defaults to 5.
            threshold (float, optional): The minimum confidence score for an answer. Defaults to 0.5.

        Returns:
            list: A list of relevant answers with sources.
        """
        chunks = self.indexer.search(question, k)
        answers = []
        
        for chunk in chunks:
            result = self.qa_pipeline(question=question, context=chunk['text'])
            if result['score'] >= threshold:
                answers.append({
                    "answer": result['answer'],
                    "source": chunk['url'],
                    "confidence": result['score']
                })

        # If no relevant answers are found
        if not answers:
            return [{"answer": "Sorry, I couldn't find any information about that.", "source": None}]
        
        # If a step-by-step guide is detected (checks for "Step", "1.", "2." in content)
        for chunk in chunks:
            if any(keyword in chunk['text'].lower() for keyword in ["step", "1.", "2.", "instructions"]):
                return [{"answer": chunk['text'], "source": chunk['url']}]

        return answers


# ------------- Global Variables and Setup for FastAPI ------------------

qa_system = None  # Placeholder, initialized later

def is_valid_help_url(url):
    """
    Checks if the given URL belongs to a help documentation site.
    
    This function parses the URL and checks if 'help' is in the netloc or '/docs' is in the path.
    
    Args:
        url (str): The URL to check.
    
    Returns:
        bool: True if the URL is a valid help documentation URL, else False.
    """
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        return False
    return "help" in parsed_url.netloc or "/docs" in parsed_url.path

def setup_qa_system(url):
    """
    Sets up the Q&A system by crawling and indexing the website.
    
    This function validates the URL, crawls the website (or loads from cache),
    indexes the content, and initializes the QASystem.
    
    Args:
        url (str): The URL of the help website to crawl and index.
    
    Raises:
        ValueError: If the URL is invalid or crawling/indexing fails.
    """
    global qa_system

    if not is_valid_help_url(url):
        raise ValueError("Invalid URL: Only help documentation URLs are allowed.")

    # Generate cache filename based on URL hash
    cache_filename = hashlib.md5(url.encode()).hexdigest() + '.json'

    # Load from cache if available
    if os.path.exists(cache_filename):
        print(f"Loading cached documents from {cache_filename}")
        with open(cache_filename, 'r', encoding='utf-8') as f:
            documents = json.load(f)
    else:
        print(f"Crawling {url}...")
        start_time = time.time()
        crawler = Crawler(url, max_depth=2)
        documents = crawler.crawl(url)
        crawling_time = time.time() - start_time
        print(f"Crawling took {crawling_time:.2f} seconds")
        if not documents:
            raise ValueError("Failed to crawl the provided help documentation.")
        with open(cache_filename, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False)
        print(f"Saved crawled documents to {cache_filename}")

    print(f"Processing {len(documents)} sections.")

    print("Building index...")
    start_time = time.time()
    try:
        indexer = Indexer(documents)
    except ValueError as e:
        raise ValueError("Indexing failed due to insufficient valid content.")
    indexing_time = time.time() - start_time
    print(f"Indexing took {indexing_time:.2f} seconds")
    print(f"Indexed {len(indexer.chunks)} content chunks.")
    qa_system = QASystem(indexer)
    print("Q&A system is ready!")

# ------------- FastAPI Server ------------------

app = FastAPI()

@app.get("/setup")
def setup(url: str):
    """
    API Endpoint to setup the Q&A system.
    
    This endpoint initializes the Q&A system with the provided help website URL.
    
    Args:
        url (str): The URL of the help website.
    
    Returns:
        dict: A message indicating successful initialization.
    
    Raises:
        HTTPException: If the URL is invalid or setup fails.
    """
    try:
        setup_qa_system(url)
        return {"message": "Q&A system initialized with " + url}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/ask")
def ask(question: str):
    """
    API Endpoint to answer a user question.
    
    This endpoint takes a user's question and returns an answer based on the indexed documentation.
    
    Args:
        question (str): The user's question.
    
    Returns:
        dict: The answer and source URL if found, else a message indicating no relevant information.
    
    Raises:
        HTTPException: If the Q&A system is not initialized.
    """
    if qa_system is None:
        raise HTTPException(status_code=400, detail="Q&A system is not initialized. Please run /setup first.")
    
    answers = qa_system.answer(question)
    if answers:
        return {"answer": answers}
    else:
        return {"answer": "Sorry, I couldn't find any relevant information."}

# ------------- Main Function for CLI ------------------

def main():
    """
    Runs the Q&A agent in CLI mode.
    
    This function parses command-line arguments, sets up the Q&A system,
    and allows the user to ask questions interactively.
    """
    parser = argparse.ArgumentParser(description='Help Website Q&A Agent')
    parser.add_argument('--url', required=True, help='URL of the help website')
    args = parser.parse_args()

    try:
        setup_qa_system(args.url)  # Initialize the system
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("Ready! Ask me a question (type 'exit' to quit):")
    while True:
        question = input("> ")
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        if not question.strip():
            print("Please enter a valid question.")
            continue
        answer, source = qa_system.answer(question)
        if answer:
            print(f"**Answer:** {answer}\n**Source:** {source}")
        else:
            print("Sorry, I couldn't find any relevant information.")

if __name__ == '__main__':
    main()