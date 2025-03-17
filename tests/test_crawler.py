import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Adding parent directory to path
from qa_agent import Crawler  

# Test a valid help website URL
def test_crawler_valid_url():
    url = "https://help.zluri.com"
    crawler = Crawler(url, max_depth=1)
    documents = crawler.crawl(url)

    assert isinstance(documents, list)  # Should return a list of documents
    assert len(documents) > 0  # Should have at least one page crawled
    assert "url" in documents[0]  # Document must have a URL
    assert "content" in documents[0]  # Document must have extracted content

# Test an invalid URL
def test_crawler_invalid_url():
    url = "https://invalid.help.site.com"
    crawler = Crawler(url, max_depth=1)
    documents = crawler.crawl(url)

    assert isinstance(documents, list)  # Should return an empty list
    assert len(documents) == 0  # No pages should be crawled
