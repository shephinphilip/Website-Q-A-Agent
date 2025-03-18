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
    - Sends a GET request to /setup with test URLs
    - Verifies response status and message
    """
    response = client.get("/setup?urls=https://help.zluri.com,https://help.com")
    
    # If partial success, it should return 206
    if response.status_code == 400:
        pytest.skip("Skipping test: Unable to crawl all test URLs.")
    
    assert response.status_code in [200, 206]  # Allow partial success
    json_response = response.json()
    assert "message" in json_response
    if response.status_code == 206:
        assert "warnings" in json_response  # Ensure warnings are present for partial failures

def test_api_ask():
    """
    Test the /ask endpoint
    - Ensures system is initialized before querying
    - Verifies response contains an 'answer'
    """
    setup_response = client.get("/setup?urls=https://help.zluri.com,https://help.com")
    if setup_response.status_code == 400:
        pytest.skip("Skipping test: Setup failed, so question answering cannot be tested.")

    response = client.get("/ask?question=What integrations are available?")
    assert response.status_code == 200
    json_response = response.json()
    assert "answer" in json_response
    assert "source" in json_response
    assert "confidence" in json_response
