# workshops_ai/wizard/scraper/exceptions.py
"""
Exceptions for web scraping.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional


class ScraperErrorType(str, Enum):
    """Types of scraping errors."""
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    FORBIDDEN = "forbidden"
    NETWORK = "network_error"
    PARSE = "parse_error"
    EMPTY_CONTENT = "empty_content"
    SECURITY = "security_warning"
    INVALID_URL = "invalid_url"


class ScraperError(Exception):
    """Base exception for scraping errors."""

    def __init__(
        self,
        url: str,
        reason: str,
        error_type: Optional[ScraperErrorType] = None,
        details: Optional[dict] = None,
    ):
        self.url = url
        self.reason = reason
        self.error_type = error_type or ScraperErrorType.NETWORK
        self.details = details or {}
        super().__init__(f"Scraping failed for {url}: {reason}")
