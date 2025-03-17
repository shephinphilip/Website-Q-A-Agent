import pytest
import os
import sys
# Add parent directory to Python path to allow imports from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qa_agent import Crawler  

def test_crawler_valid_url():
    """
    Test the crawler's behavior with a valid website URL.
    
    This test verifies that:
    1. The crawler successfully retrieves documents from a valid URL
    2. The returned data structure is correct
    3. The documents contain required fields (url and content)
    """
    url = "https://help.zluri.com"
    crawler = Crawler(url, max_depth=1)  # Initialize crawler with depth limit of 1
    documents = crawler.crawl(url)

    # Verify the crawler returns a list
    assert isinstance(documents, list)  
    
    # Verify at least one page was crawled
    assert len(documents) > 0  
    
    # Verify the document structure contains required fields
    assert "url" in documents[0]  
    assert "content" in documents[0]  

def test_crawler_invalid_url():
    """
    Test the crawler's behavior with an invalid/non-existent website URL.
    
    This test verifies that:
    1. The crawler handles invalid URLs gracefully
    2. Returns an empty list when the URL is invalid
    3. Doesn't crash on invalid input
    """
    url = "https://invalid.help.site.com"
    crawler = Crawler(url, max_depth=1)  # Initialize crawler with depth limit of 1
    documents = crawler.crawl(url)

    # Verify the crawler returns a list even for invalid URLs
    assert isinstance(documents, list)  
    
    # Verify no documents were crawled from invalid URL
    assert len(documents) == 0  
