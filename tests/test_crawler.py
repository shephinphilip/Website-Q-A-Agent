import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qa_agent import Crawler

def test_crawler_valid_url():
    """
    Tests the crawler with a valid URL.
    Ensures pages are retrieved correctly and contain required fields.
    """
    url = "https://help.zluri.com"
    crawler = Crawler(url, max_depth=1)

    try:
        documents = crawler.crawl(url)
    except ValueError:
        pytest.skip("Skipping test: URL is unreachable.")

    assert isinstance(documents, list)
    assert len(documents) > 0
    assert all("url" in doc and "content" in doc for doc in documents)

def test_crawler_invalid_url():
    """
    Tests the crawler with an invalid URL.
    Ensures it raises an appropriate exception.
    """
    url = "https://invalid.help.site.com"
    crawler = Crawler(url, max_depth=1)

    with pytest.raises(ValueError, match="URL .* is not reachable or does not exist."):
        crawler.crawl(url)
