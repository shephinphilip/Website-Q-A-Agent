# Help Website Q&A Agent

## Overview
This project is an AI-powered Question Answering (QA) Agent that crawls help documentation websites, indexes the content, and provides accurate answers to user queries.

It uses natural language processing (NLP) and semantic search to retrieve relevant answers from documentation.



## Features
- Accepts a help website URL as input (e.g., `help.zluri.com`).  
- Crawls and indexes documentation content for efficient search.  
- Accepts natural language questions via a command-line interface (CLI) and an API.  
- Provides accurate answers based on the indexed documentation.  
- Returns source references (URLs) for each answer.  
- Includes error handling for invalid URLs and unsupported websites.  



## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.8 or later installed.  
Run the following command to install all required dependencies:

```bash
pip install -r requirements.txt
```

If `faiss-cpu` fails to install, use:
```bash
pip install faiss-cpu --no-cache-dir
```



### 2. Run the FastAPI Server
To start the API server, run:
```bash
uvicorn qa_agent:app --host 0.0.0.0 --port 8000
```
This will launch the FastAPI server on `http://localhost:8000/`.



### 3. Setup the QA System (Crawl and Index)
Before asking questions, initialize the Q&A system by running:

```bash
curl "http://localhost:8000/setup?url=https://help.com"
```

This command will crawl and index the documentation.



### 4. Ask Questions
Once setup is complete, ask a question:

```bash
curl "http://localhost:8000/ask?question=What integrations are available?"
```

Example Response:
```json
{
    "answer": "You can integrate with Slack, Google Workspace, and Azure AD.",
    "source": "https://help.zluri.com/docs/integrations-overview"
}
```



## CLI Usage
The agent can also be used in the terminal:

```bash
python main.py --url https://help.com
```

Example Interaction:
```plaintext
> What integrations are available?
Answer: You can integrate with Slack, Google Workspace, and Azure AD.
Source: https://help.zluri.com/docs/integrations-overview
```



## Dependencies
- Python 3.8+
- Requests - Used for HTTP requests to fetch web pages  
- BeautifulSoup4 - Extracts meaningful content from HTML  
- Trafilatura - Extracts text from HTML pages  
- FAISS - Provides efficient vector-based similarity search  
- SentenceTransformers - Converts text into embeddings for semantic search  
- Transformers - NLP model for question answering  
- FastAPI - Provides API endpoints for querying the agent  
- Uvicorn - ASGI server to run the API  

Install all dependencies using:
```bash
pip install -r requirements.txt
```



## Technical Architecture
1. Crawler - Scrapes the help website and extracts relevant documentation.  
2. Indexer - Converts the extracted content into semantic vector embeddings using FAISS.  
3. QASystem - Uses semantic search to find relevant content and extracts answers using an NLP model (`distilbert-base-uncased-distilled-squad`).  
4. API (FastAPI) - Provides endpoints to set up and query the agent.



## Design Decisions
- FAISS is used for fast retrieval instead of a full-text search database.  
- A transformer-based NLP model is used instead of keyword-based search.  
- FastAPI is implemented for scalability and ease of use.  
- Caching and error handling are added to improve efficiency.  



## Known Limitations
- The system may not handle heavily JavaScript-rendered websites (e.g., sites requiring authentication).  
- The accuracy of answers depends on how well-structured the documentation is.  
- If the documentation is too small or lacks details, answers may not be meaningful.  



## Future Improvements
- Support for PDF and Markdown documentation using `PyMuPDF` and `markdown-it-py`.  
- Use Pinecone instead of FAISS for cloud-based scalable vector search.  
- Improve answer ranking using a larger NLP model such as GPT-4.  
- Add authentication support for private help documentation.  
- Support API documentation crawling (e.g., OpenAPI specifications).  



## Testing Approach
- Unit Tests - Validate crawling, indexing, and search functionality.  
- Integration Tests - Ensure the system works end-to-end.  
- Performance Benchmarks - Measure response times and optimize FAISS search.  

To run unit tests:
```bash
pytest tests/
```



## Submission Process
- Push the code to a private GitHub repository.  
- Share repository access with `vatsal@pulsegen.io`.  
- Include a short demo video (maximum 5 minutes) explaining:
  - How the agent works  
  - How to use it via API and CLI  
  - Any limitations and areas for improvement  



## Contact and Support
For any issues, please contact `your_email@example.com` or create a GitHub issue.