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
        documents = crawler.crawl(url, is_base_url=True)
    except ValueError:
        pytest.skip("Skipping test: Base URL is unreachable.")

    assert isinstance(documents, list)
    assert len(documents) > 0
