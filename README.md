# Help Website Q&A Agent

## Overview
This project is an AI-powered Question Answering (QA) Agent that crawls, indexes, and retrieves answers from help documentation websites. It uses Natural Language Processing (NLP) and semantic search to extract and return the most relevant information in response to user queries.

The system supports multiple formats, including HTML, PDFs, and Markdown, and utilizes FAISS for fast similarity search.

---

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

If `faiss-cpu` fails to install, try:

```bash
pip install faiss-cpu --no-cache-dir
```

### 2. Start the FastAPI Server
To run the API server:

```bash
uvicorn qa_agent:app --host 0.0.0.0 --port 8000
```

This will start the FastAPI server at `http://localhost:8000/`.

### 3. Setup the Q&A System (Crawl & Index)
Before querying the system, initialize it by crawling a help documentation site:

```bash
curl "http://localhost:8000/setup?url=https://help.com"
```

This command will crawl and index the documentation.

### 4. Ask a Question
Once the setup is complete, ask a question:

```bash
curl "http://localhost:8000/ask?question=What integrations are available?"
```

Example Response:
```json
{
    "answer": "You can integrate with Slack, Google Workspace, and Azure AD.",
    "source": "https://help.com/docs/integrations-overview",
    "confidence": 0.92
}
```

### 5. Clear Cache (Optional)
To remove cached responses and refresh the index:

```bash
curl "http://localhost:8000/clear_cache"
```

---

## Dependencies
This project relies on several libraries for crawling, indexing, and answering questions:

- Requests – Fetches web pages and API data  
- BeautifulSoup4 – Parses and extracts meaningful content from HTML  
- Trafilatura – Extracts text from web pages while preserving formatting  
- FAISS – Enables fast vector-based similarity search  
- SentenceTransformers – Converts text into vector embeddings for semantic search  
- Transformers – Provides NLP models for question answering  
- FastAPI – Serves the API for querying the agent  
- Uvicorn – ASGI server to run the FastAPI application  
- PyPDF2 – Extracts text from PDF documentation  
- CacheTools – Implements caching for faster responses  

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage Examples

### Running in CLI Mode
You can also interact with the system through a command-line interface:

```bash
python main.py --urls https://help.com
```

Example Interaction:
```plaintext
> What integrations are available?
Answer: You can integrate with Slack, Google Workspace, and Azure AD.
Source: https://help.com/docs/integrations-overview
Confidence: 0.92
```

Type "exit" to quit.

---

## Design Decisions

### Why FAISS for Search?
FAISS is a highly efficient vector search library that provides fast similarity search compared to traditional full-text search databases.

### Why Transformer Models for QA?
Instead of relying on keyword-based search, DistilBERT and SentenceTransformers are used to understand the context of the question and extract precise answers.

### Why FastAPI?
FastAPI was chosen due to its performance, scalability, and built-in support for asynchronous processing, making it an ideal choice for handling multiple user queries efficiently.

---

## Known Limitations

### 1. Limited JavaScript Handling
The system does not support JavaScript-heavy websites that require client-side rendering to load content. It only works with static HTML pages.

### 2. Requires Well-Structured Documentation
For accurate answers, documentation should be well-formatted and structured. Poorly formatted content may lead to less relevant responses.

### 3. No Support for Private Documentation
The system does not handle authentication-protected documentation. It only works with publicly accessible help sites.

---

## Future Enhancements
- Support API documentation crawling (e.g., OpenAPI specifications)  
- Improve search ranking using better NLP models  
- Use a larger QA model (e.g., GPT-4) for more complex answers  
- Integrate Pinecone for scalable cloud-based vector search  
- Handle private documentation with authentication  

---

## Testing and Benchmarking

- Unit Tests - Validate crawling, indexing, and QA functionality  
- Integration Tests - Ensure the system works as a whole  
- Performance Benchmarks - Measure response times and optimize FAISS indexing  

To run tests:

```bash
pytest tests/
```

### Demo Video
Watch the demo video to see the project in action:  
https://github.com/shephinphilip/Website-Q-A-Agent/blob/main/demo/Website-QA.mp4

If the video does not play, **[click here to download](https://github.com/shephinphilip/Website-Q-A-Agent/blob/main/demo/Website-QA.mp4)**
