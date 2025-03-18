import pytest
from fastapi.testclient import TestClient
import os
import sys

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qa_agent import app

# Create test client instance for FastAPI
client = TestClient(app)

def test_api_setup():
    """
    Test the /setup endpoint
    - Sends a GET request to /setup with a test URL
    - Verifies response status and message
    """
    response = client.get("/setup?url=https://help.zluri.com")
    if response.status_code == 400:
        pytest.skip("Skipping test: Unable to crawl the given URL.")
    assert response.status_code == 200
    assert "message" in response.json()

def test_api_ask():
    """
    Test the /ask endpoint
    - Ensures system is initialized before querying
    - Verifies response contains an 'answer'
    """
    setup_response = client.get("/setup?url=https://help.zluri.com")
    if setup_response.status_code == 400:
        pytest.skip("Skipping test: Setup failed, so question answering cannot be tested.")

    response = client.get("/ask?question=What integrations are available?")
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "source" in response.json()
    assert "confidence" in response.json()
