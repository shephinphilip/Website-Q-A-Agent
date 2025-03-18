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
from fastapi import FastAPI, HTTPException, Query

# ------------- Helper Function for Section Extraction ------------------

def extract_sections(soup):
    """
    Extracts sections from the BeautifulSoup object based on headings (h1 to h6).
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        list: A list of dictionaries, each containing 'title' and 'content' for a section.
    """
    sections = []
    current_section = None
    for element in soup.find_all(True):
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if current_section:
                sections.append(current_section)
            current_section = {'title': element.get_text().strip(), 'content': []}
        elif current_section and element.name in ['p', 'div', 'span', 'li']:
            current_section['content'].append(element.get_text().strip())
    if current_section:
        sections.append(current_section)
    sections = [s for s in sections if s['content']]
    return sections

# ------------- Classes for Crawling, Indexing, and QA ------------------

class Crawler:
    """
    Handles recursive crawling of a website using BeautifulSoup.
    """
    def __init__(self, base_url, max_depth=2):
        self.base_url = base_url.rstrip('/')
        self.max_depth = max_depth

    def check_url_exists(self, url):
        """
        Checks if the URL exists and is reachable.
        
        Args:
            url (str): The URL to check.
        
        Returns:
            bool: True if URL exists and is reachable, False otherwise.
        """
        try:
            # Use HEAD request for efficiency, fall back to GET if HEAD fails
            response = requests.head(url, timeout=5, allow_redirects=True)
            if response.status_code >= 400:
                response = requests.get(url, timeout=5, allow_redirects=True)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"URL check failed for {url}: {e}")
            return False

    def crawl(self, url, visited=None, depth=0):
        """
        Recursively scrapes the website starting from the given URL.
        
        Args:
            url (str): The URL to start crawling from.
            visited (set, optional): A set of visited URLs to avoid revisiting.
            depth (int, optional): The current depth of crawling.
        
        Returns:
            list: A list of dictionaries with 'url', 'page_title', 'section_title', and 'content'.
        
        Raises:
            ValueError: If initial URL is not reachable or returns no content.
        """
        if not self.check_url_exists(url):
            raise ValueError(f"URL {url} is not reachable or does not exist.")

        if visited is None:
            visited = set()
        if url in visited or not url.startswith(self.base_url) or depth > self.max_depth:
            return []

        visited.add(url)
        print(f"Crawling: {url} (Depth: {depth})")

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

        if not sections:
            content = trafilatura.extract(response.text)
            if content:
                documents.append({
                    'url': url,
                    'page_title': page_title,
                    'section_title': 'Full Page',
                    'content': content
                })

        links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
        doc_links = [link for link in links if link.startswith(self.base_url)]

        for link in doc_links:
            documents.extend(self.crawl(link, visited, depth + 1))

        return documents

class Indexer:
    """
    Indexes documentation content from multiple sources for efficient querying.
    """
    def __init__(self, documents):
        """
        Initializes the Indexer with documents from multiple sources.
        
        Args:
            documents (list): A list of document dictionaries.
        """
        self.documents = documents
        self.chunks = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.build_index()

    def build_index(self):
        """
        Builds a FAISS index from document sections.
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
            k (int, optional): Number of top chunks to retrieve.
        
        Returns:
            list: Top-k chunks.
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

    def is_general_question(self, question):
        """
        Checks if the question is general (e.g., "What is...?").
        
        Args:
            question (str): The user's question.
        
        Returns:
            bool: True if general, False otherwise.
        """
        general_patterns = [r"what is .*?", r"tell me about .*", r"explain .*"]
        return any(re.match(pattern, question.lower()) for pattern in general_patterns)

    def answer(self, question, k=5, threshold=0.5):
        """
        Generates an answer to the user's question from the most relevant chunks.
        
        Args:
            question (str): The user's question.
            k (int, optional): Number of top chunks to consider.
            threshold (float, optional): Minimum confidence score.
        
        Returns:
            tuple: (answer, source) or (None, None) if no answer found.
        """
        chunks = self.indexer.search(question, k)
        best_answer = None
        best_score = -1
        best_source = None
        best_chunk = None

        if self.is_general_question(question):
            overview_chunks = [chunk for chunk in chunks 
                              if "overview" in chunk['section_title'].lower() 
                              or "introduction" in chunk['section_title'].lower()]
            chunks = overview_chunks if overview_chunks else chunks

        for chunk in chunks:
            result = self.qa_pipeline(question=question, context=chunk['text'])
            if result['score'] > best_score:
                best_score = result['score']
                best_answer = result['answer']
                best_source = chunk['url']
                best_chunk = chunk

        if best_score >= threshold:
            if len(best_answer.split()) < 5:
                best_answer = best_chunk['text']
            return best_answer, best_source
        return None, None

# ------------- Global Variables and Setup ------------------

qa_system = None

def is_valid_help_url(url):
    """
    Checks if the URL is a valid help documentation site.
    """
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        return False
    return "help" in parsed_url.netloc or "/docs" in parsed_url.path

def setup_qa_system(urls):
    """
    Sets up the Q&A system by crawling and indexing multiple websites.
    
    Args:
        urls (list): List of URLs to crawl and index.
    
    Raises:
        ValueError: If any URL is invalid, unreachable, or yields no content.
    """
    global qa_system
    all_documents = []

    for url in urls:
        if not is_valid_help_url(url):
            raise ValueError(f"Invalid URL format: {url}. Must contain 'help' or '/docs'.")

        cache_filename = hashlib.md5(url.encode()).hexdigest() + '.json'

        if os.path.exists(cache_filename):
            print(f"Loading cached documents from {cache_filename}")
            with open(cache_filename, 'r', encoding='utf-8') as f:
                documents = json.load(f)
        else:
            print(f"Crawling {url}...")
            start_time = time.time()
            crawler = Crawler(url, max_depth=2)
            documents = crawler.crawl(url)  # This will raise ValueError if URL doesn't exist
            if not documents:
                raise ValueError(f"No content retrieved from {url}. URL may not exist or contain crawlable data.")
            crawling_time = time.time() - start_time
            print(f"Crawling {url} took {crawling_time:.2f} seconds")
            with open(cache_filename, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False)
            print(f"Saved crawled documents to {cache_filename}")

        all_documents.extend(documents)

    if not all_documents:
        raise ValueError("No content retrieved from any provided URLs.")

    print(f"Processing {len(all_documents)} sections from all sites.")
    print("Building index...")
    start_time = time.time()
    try:
        indexer = Indexer(all_documents)
    except ValueError as e:
        raise ValueError("Indexing failed: " + str(e))
    indexing_time = time.time() - start_time
    print(f"Indexing took {indexing_time:.2f} seconds")
    print(f"Indexed {len(indexer.chunks)} content chunks.")
    qa_system = QASystem(indexer)
    print("Q&A system is ready!")

# ------------- FastAPI Server ------------------

app = FastAPI()

@app.get("/setup")
def setup(url: str = Query(None, description="A single help website URL"), 
          urls: str = Query(None, description="Comma-separated list of help website URLs")):
    """
    API Endpoint to setup the Q&A system with a single URL or multiple URLs.
    
    Args:
        url (str, optional): A single help website URL.
        urls (str, optional): Comma-separated list of help website URLs.
    
    Returns:
        dict: Success message indicating initialized URLs.
    
    Raises:
        HTTPException: If input is invalid or setup fails.
    """
    if url is None and urls is None:
        raise HTTPException(status_code=400, detail="Either 'url' or 'urls' parameter must be provided.")
    
    if url and urls:
        raise HTTPException(status_code=400, detail="Provide either 'url' or 'urls', not both.")
    
    if url:
        url_list = [url.strip()]
    else:
        url_list = [u.strip() for u in urls.split(',') if u.strip()]

    try:
        setup_qa_system(url_list)
        return {"message": f"Q&A system initialized with {', '.join(url_list)}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/ask")
def ask(question: str):
    """
    API Endpoint to answer a user question.
    
    Args:
        question (str): The user's question.
    
    Returns:
        dict: Answer and source.
    
    Raises:
        HTTPException: If system not initialized.
    """
    if qa_system is None:
        raise HTTPException(status_code=400, detail="Q&A system not initialized. Please run /setup first.")
    
    answer, source = qa_system.answer(question)
    if answer:
        return {"answer": answer, "source": source}
    else:
        return {"answer": "Sorry, I couldn't find any relevant information.", "source": None}

# ------------- Main Function for CLI ------------------

def main():
    """
    Runs the Q&A agent in CLI mode with multi-URL support.
    """
    parser = argparse.ArgumentParser(description='Help Website Q&A Agent')
    parser.add_argument('--urls', required=True, help='Comma-separated list of help website URLs')
    args = parser.parse_args()

    url_list = [url.strip() for url in args.urls.split(',')]
    try:
        setup_qa_system(url_list)
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