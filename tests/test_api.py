import pytest
from fastapi.testclient import TestClient
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Adding parent directory to path
from qa_agent import app  # Import your FastAPI app

client = TestClient(app)

def test_api_setup():
    response = client.get("/setup?url=https://help.zluri.com")
    assert response.status_code == 200
    assert "message" in response.json()

def test_api_ask():
    response = client.get("/ask?question=What integrations are available?")
    assert response.status_code == 200
    assert "answer" in response.json()
