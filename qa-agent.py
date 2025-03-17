import argparse
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import trafilatura
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import faiss
from transformers import pipeline

class Crawler:
    """Handles recursive crawling of a website using BeautifulSoup."""
    
    def __init__(self, base_url, max_depth=2):
        self.base_url = base_url.rstrip('/')
        self.max_depth = max_depth

    def crawl(self, url, visited=None, depth=0):
        """Recursively scrapes a website using BeautifulSoup."""
        if visited is None:
            visited = set()
        if url in visited or not url.startswith(self.base_url) or depth > self.max_depth:
            return []

        visited.add(url)
        print(f"Crawling: {url} (Depth: {depth})")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors

            soup = BeautifulSoup(response.text, 'html.parser')
            content = trafilatura.extract(response.text)  # Extract main content
            
            if not content:
                print(f"No content extracted from {url}")
                return []

            title = soup.title.string if soup.title else 'Untitled'

            # Extract all internal links
            links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
            doc_links = [link for link in links if link.startswith(self.base_url)]

            # Store document content
            documents = [{'url': url, 'title': title, 'content': content}]

            for link in doc_links:
                documents.extend(self.crawl(link, visited, depth + 1))
            
            return documents

        except requests.exceptions.RequestException as e:
            print(f"Error crawling {url}: {e}")
            return []

class Indexer:
    """Indexes documentation content for efficient querying."""
    def __init__(self, documents):
        self.documents = documents
        self.chunks = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.build_index()

    def build_index(self):
        """Splits content into chunks and builds a FAISS index."""
        for doc in self.documents:
            paragraphs = doc['content'].split('\n\n')
            for para in paragraphs:
                if para.strip():
                    self.chunks.append({'text': para, 'url': doc['url'], 'title': doc['title']})
        if not self.chunks:
            raise ValueError("No valid content found to index.")
        texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)

    def search(self, query, k=5):
        """Searches for the top-k chunks most similar to the query."""
        query_emb = self.model.encode([query])
        distances, indices = self.faiss_index.search(query_emb, k)
        return [self.chunks[i] for i in indices[0]]

class QASystem:
    """Answers user queries based on indexed documentation."""
    def __init__(self, indexer):
        self.indexer = indexer
        self.qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

    def answer(self, question, k=5, threshold=0.5):
        """Generates an answer from the most relevant chunks."""
        chunks = self.indexer.search(question, k)
        best_answer = None
        best_score = -1
        best_source = None
        for chunk in chunks:
            result = self.qa_pipeline(question=question, context=chunk['text'])
            if result['score'] > best_score:
                best_score = result['score']
                best_answer = result['answer']
                best_source = chunk['url']
        if best_score >= threshold:
            return best_answer, best_source
        return None, None

def main():
    """Main function to run the Q&A agent."""
    parser = argparse.ArgumentParser(description='Help Website Q&A Agent')
    parser.add_argument('--url', required=True, help='URL of the help website')
    args = parser.parse_args()

    print(f"Crawling {args.url}...")
    crawler = Crawler(args.url, max_depth=2)  # Limit depth for faster crawling
    documents = crawler.crawl(args.url)
    if not documents:
        print("Failed to crawl any pages. Check the URL and your setup.")
        return
    print(f"Crawled {len(documents)} pages.")

    print("Building index...")
    try:
        indexer = Indexer(documents)
    except ValueError as e:
        print(f"Error: {e}")
        return
    print(f"Indexed {len(indexer.chunks)} content chunks.")

    qa_system = QASystem(indexer)
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
            print("Sorry, I couldn't find any information about that.")

if __name__ == '__main__':
    main()