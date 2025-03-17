import pytest
import os
import sys
# Add parent directory to Python path for importing local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qa_agent import Indexer

def test_faiss_indexing():
    """
    Test the FAISS indexing functionality.
    
    This test verifies that:
    1. The indexer can process and index a simple document
    2. The search function returns expected results
    3. The results contain the required fields
    """
    # Sample test document with required fields
    documents = [
        {
            "url": "https://help.example.com",
            "page_title": "Test Page",  
            "section_title": "Introduction",  
            "content": "This is a test document."
        }
    ]
    
    # Initialize indexer with test documents
    indexer = Indexer(documents)
    
    # Perform search with test query
    results = indexer.search("test", k=1)
    
    # Verify search results
    assert len(results) > 0, "Search should return at least one result"
    assert "text" in results[0], "Result should contain 'text' field"
    assert "url" in results[0], "Result should contain 'url' field"
    assert "page_title" in results[0], "Result should contain 'page_title' field"
    assert "section_title" in results[0], "Result should contain 'section_title' field"
