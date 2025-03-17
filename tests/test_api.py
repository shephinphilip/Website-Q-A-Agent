# Import required testing and HTTP client libraries
import pytest
from fastapi.testclient import TestClient
import os
import sys

# Add parent directory to Python path to allow importing from parent folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qa_agent import app  

# Create a test client instance for the FastAPI application
client = TestClient(app)

def test_api_setup():
    """
    Test the /setup endpoint
    - Sends a GET request to /setup with a test URL
    - Verifies that:
        1. Response status code is 200 (OK)
        2. Response JSON contains a 'message' field
    """
    response = client.get("/setup?url=https://help.zluri.com")
    assert response.status_code == 200
    assert "message" in response.json()

def test_api_ask():
    """
    Test the /ask endpoint
    - Sends a GET request to /ask with a test question
    - Verifies that:
        1. Response status code is 200 (OK)
        2. Response JSON contains an 'answer' field
    """
    response = client.get("/ask?question=What integrations are available?")
    assert response.status_code == 200
    assert "answer" in response.json()
