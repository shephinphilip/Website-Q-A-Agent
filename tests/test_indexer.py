import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Adding parent directory to path
from qa_agent import Indexer  # Import your actual Indexer class

def test_faiss_indexing():
    documents = [
        {"url": "https://help.example.com", "title": "Test", "content": "This is a test document."}
    ]
    indexer = Indexer(documents)
    
    results = indexer.search("test", k=1)
    assert len(results) > 0  # Should return at least one result
    assert "text" in results[0]  # Result should contain text
