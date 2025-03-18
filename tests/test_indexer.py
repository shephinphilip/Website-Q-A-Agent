import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qa_agent import Indexer

def test_faiss_indexing():
    """
    Tests FAISS indexing functionality.
    Ensures correct indexing, retrieval, and field presence.
    """
    documents = [
        {
            "url": "https://help.example.com",
            "page_title": "Test Page",
            "section_title": "Introduction",
            "content": "This is a test document."
        }
    ]

    indexer = Indexer(documents)
    results = indexer.search("test", k=1)

    assert len(results) > 0, "Search should return at least one result"

    # The search returns a tuple (chunk_dict, distance)
    result_data, _ = results[0]
    
    assert isinstance(result_data, dict), "Returned result should be a dictionary"
    assert "text" in result_data, "Result should contain 'text' field"
    assert "url" in result_data, "Result should contain 'url' field"
    assert "page_title" in result_data, "Result should contain 'page_title' field"
    assert "section_title" in result_data, "Result should contain 'section_title' field"
