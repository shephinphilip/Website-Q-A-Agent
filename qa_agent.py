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
import numpy as np
from transformers import pipeline
from fastapi import FastAPI, HTTPException, Query
import markdown
import PyPDF2
from cachetools import TTLCache
from typing import List, Dict, Optional, Tuple

# ------------- Helper Functions ------------------

def extract_sections(soup):
    """Extracts sections from BeautifulSoup object based on headings (h1 to h6)."""
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
    return [s for s in sections if s['content']]

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_markdown(content: str) -> str:
    """Converts Markdown to plain text."""
    html = markdown.markdown(content)
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

# ------------- Classes for Crawling, Indexing, and QA ------------------

class Crawler:
    """Handles recursive crawling and content extraction from multiple formats."""
    def __init__(self, base_url: str, max_depth: int = 2):
        self.base_url = base_url.rstrip('/')
        self.max_depth = max_depth

    def check_url_exists(self, url: str) -> bool:
        """Checks if the URL is reachable."""
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            if response.status_code >= 400:
                response = requests.get(url, timeout=5, allow_redirects=True)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"URL check failed for {url}: {e}")
            return False

    def crawl(self, url: str, visited: set = None, depth: int = 0) -> List[Dict]:
        """Recursively scrapes the website."""
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

        documents = []
        content_type = response.headers.get('content-type', '').lower()
        
        if 'text/html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            page_title = soup.title.string if soup.title else 'Untitled'
            sections = extract_sections(soup)
            for section in sections:
                section_content = ' '.join(section['content'])
                if section_content:
                    documents.append({
                        'url': url,
                        'page_title': page_title,
                        'section_title': section['title'],
                        'content': section_content,
                        'format': 'html'
                    })
            if not sections:
                content = trafilatura.extract(response.text)
                if content:
                    documents.append({
                        'url': url,
                        'page_title': page_title,
                        'section_title': 'Full Page',
                        'content': content,
                        'format': 'html'
                    })
        elif 'application/pdf' in content_type:
            with open('temp.pdf', 'wb') as f:
                f.write(response.content)
            content = extract_text_from_pdf('temp.pdf')
            os.remove('temp.pdf')
            if content:
                documents.append({
                    'url': url,
                    'page_title': 'PDF Document',
                    'section_title': 'Full Document',
                    'content': content,
                    'format': 'pdf'
                })
        elif 'text/markdown' in content_type or url.endswith('.md'):
            content = extract_text_from_markdown(response.text)
            if content:
                documents.append({
                    'url': url,
                    'page_title': 'Markdown Document',
                    'section_title': 'Full Document',
                    'content': content,
                    'format': 'markdown'
                })

        links = [urljoin(url, a['href']) for a in BeautifulSoup(response.text, 'html.parser').find_all('a', href=True)] if 'text/html' in content_type else []
        doc_links = [link for link in links if link.startswith(self.base_url)]

        for link in doc_links:
            documents.extend(self.crawl(link, visited, depth + 1))

        return documents

class Indexer:
    """Indexes documentation content with performance optimizations."""
    def __init__(self, documents: List[Dict]):
        self.documents = documents
        self.chunks = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # Optimize for CPU
        self.faiss_index = None
        self.build_index()

    def build_index(self):
        """Builds a FAISS index with batch processing."""
        for doc in self.documents:
            self.chunks.append({
                'text': doc['content'],
                'url': doc['url'],
                'page_title': doc['page_title'],
                'section_title': doc['section_title'],
                'format': doc.get('format', 'html')
            })

        if not self.chunks:
            raise ValueError("No valid content found to index.")

        texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)  # Batch processing
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        self.faiss_index.add(embeddings)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Searches for top-k chunks using cosine similarity."""
        query_emb = self.model.encode([query])
        faiss.normalize_L2(query_emb)
        distances, indices = self.faiss_index.search(query_emb, k)
        return [(self.chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]

class QASystem:
    """Advanced QA system with caching and confidence scores."""
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # Cache up to 1000 answers for 1 hour

    def is_general_question(self, question: str) -> bool:
        """Checks if the question is general."""
        general_patterns = [r"what is .*?", r"tell me about .*", r"explain .*"]
        return any(re.match(pattern, question.lower()) for pattern in general_patterns)

    def answer(self, question: str, k: int = 5, threshold: float = 0.5) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Generates an answer with caching and confidence score."""
        cache_key = hashlib.md5(question.encode()).hexdigest()
        if cache_key in self.cache:
            print(f"Cache hit for question: {question}")
            answer, source, score = self.cache[cache_key]
            return answer, source, score

        chunks_with_scores = self.indexer.search(question, k)
        best_answer = None
        best_score = -1
        best_source = None
        best_chunk = None

        if self.is_general_question(question):
            chunks_with_scores = [(chunk, score) for chunk, score in chunks_with_scores 
                                  if "overview" in chunk['section_title'].lower() or "introduction" in chunk['section_title'].lower()] or chunks_with_scores

        for chunk, _ in chunks_with_scores:
            result = self.qa_pipeline(question=question, context=chunk['text'])
            score = result['score']
            if score > best_score:
                best_score = score
                best_answer = result['answer']
                best_source = chunk['url']
                best_chunk = chunk

        if best_score >= threshold:
            if len(best_answer.split()) < 5:
                best_answer = best_chunk['text']
            self.cache[cache_key] = (best_answer, best_source, best_score)
            return best_answer, best_source, best_score
        return None, None, 0.0

# ------------- Global Variables and Setup ------------------

qa_system = None

def is_valid_help_url(url: str) -> bool:
    """Checks if the URL is a valid help documentation site."""
    parsed_url = urlparse(url)
    return parsed_url.scheme and parsed_url.netloc and ("help" in parsed_url.netloc or "/docs" in parsed_url.path)

def setup_qa_system(urls: List[str]):
    """Sets up the Q&A system with multiple documentation sources."""
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
            documents = crawler.crawl(url)
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
    indexer = Indexer(all_documents)
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
    """API Endpoint to setup the Q&A system."""
    if url is None and urls is None:
        raise HTTPException(status_code=400, detail="Either 'url' or 'urls' parameter must be provided.")
    if url and urls:
        raise HTTPException(status_code=400, detail="Provide either 'url' or 'urls', not both.")
    
    url_list = [url.strip()] if url else [u.strip() for u in urls.split(',') if u.strip()]
    try:
        setup_qa_system(url_list)
        return {"message": f"Q&A system initialized with {', '.join(url_list)}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/ask")
def ask(question: str):
    """API Endpoint to answer a question with confidence score."""
    if qa_system is None:
        raise HTTPException(status_code=400, detail="Q&A system not initialized. Please run /setup first.")
    
    answer, source, confidence = qa_system.answer(question)
    if answer:
        return {"answer": answer, "source": source, "confidence": float(confidence)}
    return {"answer": "Sorry, I couldn't find any relevant information.", "source": None, "confidence": 0.0}

@app.get("/clear_cache")
def clear_cache():
    """API Endpoint to clear the answer cache."""
    if qa_system is None:
        raise HTTPException(status_code=400, detail="Q&A system not initialized.")
    qa_system.cache.clear()
    return {"message": "Cache cleared successfully."}

# ------------- Main Function for CLI ------------------

def main():
    """Runs the Q&A agent in CLI mode."""
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
        answer, source, confidence = qa_system.answer(question)
        if answer:
            print(f"**Answer:** {answer}\n**Source:** {source}\n**Confidence:** {confidence:.2f}")
        else:
            print("Sorry, I couldn't find any relevant information.")

if __name__ == '__main__':
    main()