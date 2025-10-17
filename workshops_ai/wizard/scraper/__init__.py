# workshops_ai/wizard/scraper/__init__.py
"""
Traditional web scraping implementation using BeautifulSoup.

Main exports:
- get_website(): Scrape website content (homepage + services + about pages)
- flatten_website_content(): Convert detailed output to simple (url, content) tuple
- ScraperError: Exception for scraping failures
- RateLimitError: Exception for rate limit errors
- reach_url(): Check URL reachability
- fetch(): Simple content fetching
- fetch_with_final_url(): Fetch with redirect tracking
"""
from __future__ import annotations

from workshops_ai.wizard.scraper.core import get_website, flatten_website_content
from workshops_ai.wizard.scraper.exceptions import ScraperError
from workshops_ai.wizard.scraper.http_client import (
    RateLimitError,
    reach_url,
    fetch,
    fetch_with_final_url,
    get_http_client,
)

__all__ = [
    "get_website",
    "flatten_website_content",
    "ScraperError",
    "RateLimitError",
    "reach_url",
    "fetch",
    "fetch_with_final_url",
    "get_http_client",
]
