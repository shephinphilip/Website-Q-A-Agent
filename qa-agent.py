import argparse
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import trafilatura
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from fastapi import FastAPI

# ------------- CLASSES FOR CRAWLING, INDEXING, AND QA ------------------

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
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            content = trafilatura.extract(response.text)
            
            if not content:
                print(f"No content extracted from {url}")
                return []

            title = soup.title.string if soup.title else 'Untitled'
            links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
            doc_links = [link for link in links if link.startswith(self.base_url)]

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

    def search(self, query, k=5, min_score=0.2):
        """Search for top-k chunks with a minimum relevance score."""
        query_emb = self.model.encode([query])
        distances, indices = self.faiss_index.search(query_emb, k)
        results = []
        for i, index in enumerate(indices[0]):
            if distances[0][i] < min_score:  # Ensure relevance
                continue
            results.append(self.chunks[index])
        return results

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


# ------------- GLOBAL VARIABLES FOR FASTAPI ------------------

qa_system = None  # Placeholder, will be initialized later

def setup_qa_system(url):
    """Sets up the Q&A system by crawling and indexing the website."""
    global qa_system
    print(f"Crawling {url}...")
    crawler = Crawler(url, max_depth=2)
    documents = crawler.crawl(url)
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
    print("Q&A system is ready!")


# ------------- FASTAPI SERVER ------------------

app = FastAPI()

@app.get("/setup")
def setup(url: str):
    """API Endpoint to setup the Q&A system"""
    setup_qa_system(url)
    return {"message": "Q&A system initialized with " + url}

@app.get("/ask")
def ask(question: str):
    """API Endpoint to answer a user question"""
    if qa_system is None:
        return {"error": "Q&A system is not initialized. Please run /setup first."}
    
    answer, source = qa_system.answer(question)
    if answer:
        return {"answer": answer, "source": source}
    else:
        return {"answer": "Sorry, I couldn't find any information on that.", "source": None}


# ------------- MAIN FUNCTION FOR CLI ------------------

def main():
    """Runs the Q&A agent in CLI mode."""
    parser = argparse.ArgumentParser(description='Help Website Q&A Agent')
    parser.add_argument('--url', required=True, help='URL of the help website')
    args = parser.parse_args()

    setup_qa_system(args.url)  # Initialize the system

    if qa_system is None:
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
            print("Sorry, I couldn't find any information about that.")

if __name__ == '__main__':
    main()
