# workshops_ai/wizard/scraper/http_client.py
"""
HTTP client utilities for web scraping with retry logic and encoding detection.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, AsyncIterator
from urllib.parse import urlparse, urlunparse
from contextlib import asynccontextmanager

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception,
)

from workshops_ai.settings import get_settings
from workshops_ai.wizard.scraper.exceptions import ScraperError, ScraperErrorType

logger = logging.getLogger(__name__)

# Try to import charset-normalizer (optional dependency for encoding detection)
try:
    from charset_normalizer import from_bytes
    HAS_CHARSET_NORMALIZER = True
except ImportError:
    HAS_CHARSET_NORMALIZER = False
    logger.debug(
        "charset-normalizer not available, using fallback encoding detection. "
        "Install with: pip install charset-normalizer"
    )

# Shared HTTP client for connection reuse
_shared_client: Optional[httpx.AsyncClient] = None


class RateLimitError(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, retry_after: int, message: str = "Rate limit exceeded"):
        self.retry_after = retry_after
        self.message = message
        super().__init__(message)


def is_rate_limit_exception(exception) -> bool:
    """Check if exception is a rate limit related error."""
    if isinstance(exception, RateLimitError):
        return True

    # Check for HTTP 429 status code
    if hasattr(exception, 'response') and exception.response:
        status_code = getattr(exception.response, 'status_code', None)
        if status_code == 429:
            return True

    # Check for generic rate limit indicators in error message
    error_message = str(exception).lower()
    if any(keyword in error_message for keyword in ['rate limit', 'quota exceeded', 'too many requests']):
        return True

    return False


def rate_limit_retry_decorator(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator for rate limit aware retry logic."""

    def decorator(func):
        @retry(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=base_delay, min=base_delay, max=60),
            retry=retry_if_exception(is_rate_limit_exception)
        )
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if is_rate_limit_exception(e):
                    # Extract retry_after from exception or response headers
                    retry_after = getattr(e, 'retry_after', None)
                    if not retry_after and hasattr(e, 'response'):
                        response = getattr(e, 'response', None)
                        if response and hasattr(response, 'headers'):
                            retry_after = response.headers.get('retry-after')
                            if retry_after:
                                try:
                                    retry_after = int(retry_after)
                                except ValueError:
                                    retry_after = 60  # Default fallback

                    if not retry_after:
                        retry_after = int(base_delay * (2 ** (max_attempts - 1)))

                    logger.warning(
                        "rate_limit_hit",
                        extra={"retry_after": retry_after}
                    )

                    raise RateLimitError(retry_after, f"Rate limit exceeded. Retry after {retry_after} seconds")
                raise e

        return wrapper
    return decorator


def normalize_url(url: str) -> str:
    """
    Normalize URL for fetching.

    - Add http:// if scheme missing
    - Drop query and fragment
    - Ensure path at least '/'

    Args:
        url: Raw URL string

    Returns:
        Normalized URL
    """
    # Strip whitespace
    url = url.strip()

    # Add scheme if missing (urlparse needs it to correctly identify netloc)
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    parsed = urlparse(url)

    # Remove query and fragment
    normalized = parsed._replace(query="", fragment="")

    # Ensure path
    if not normalized.path:
        normalized = normalized._replace(path="/")

    return urlunparse(normalized)


def _build_http_client() -> httpx.AsyncClient:
    """
    Build configured HTTP client for scraping.

    Returns:
        Configured async HTTP client
    """
    settings = get_settings()

    limits = httpx.Limits(
        max_keepalive_connections=20,
        max_connections=40
    )

    timeout = httpx.Timeout(
        connect=settings.wizard.scrape_timeout_sec,
        read=settings.wizard.scrape_timeout_sec,
        write=settings.wizard.scrape_timeout_sec,
        pool=settings.wizard.scrape_timeout_sec * 2,
    )

    headers = {"User-Agent": settings.wizard.scrape_user_agent}

    return httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        headers=headers,
        follow_redirects=True
    )


@asynccontextmanager
async def get_http_client() -> AsyncIterator[httpx.AsyncClient]:
    """
    Get shared HTTP client with connection reuse.

    Uses a global shared client to reuse connections across requests
    within the same process, improving performance.

    Yields:
        Configured async HTTP client
    """
    global _shared_client

    if _shared_client is None or _shared_client.is_closed:
        _shared_client = _build_http_client()
        logger.debug("http_client_created")

    try:
        yield _shared_client
    finally:
        # Do not close; reuse across requests within process
        pass


@retry(
    reraise=True,
    stop=stop_after_attempt(get_settings().wizard.scrape_max_retries + 1),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type((
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.NetworkError
    )),
)
async def fetch_with_final_url(url: str) -> Optional[Tuple[str, str]]:
    """
    Fetch page and return (text, final_url) or None.

    Features:
    - Shared HTTP client with connection reuse
    - Automatic redirect following
    - Enhanced encoding detection (charset-normalizer)
    - Retry on transient errors

    Args:
        url: URL to fetch

    Returns:
        Tuple of (page_text, final_url_after_redirects) or None if failed

    Raises:
        ScraperError: On 4xx/5xx or fatal errors
    """
    settings = get_settings()

    async with get_http_client() as client:
        try:
            resp = await client.get(url)

            # Handle error status codes
            if resp.status_code >= 400:
                error_type = (
                    ScraperErrorType.NOT_FOUND if resp.status_code == 404
                    else ScraperErrorType.FORBIDDEN if resp.status_code == 403
                    else ScraperErrorType.NETWORK
                )

                logger.warning(
                    "http_error",
                    extra={
                        "url": url,
                        "status_code": resp.status_code,
                        "error_type": error_type.value
                    }
                )

                raise ScraperError(
                    url=url,
                    reason=f"HTTP {resp.status_code}",
                    error_type=error_type,
                    details={"status_code": resp.status_code}
                )

            # Enhanced encoding detection
            if HAS_CHARSET_NORMALIZER:
                detected = from_bytes(resp.content).best()
                text = str(detected) if detected else resp.text

                logger.debug(
                    "encoding_detected",
                    extra={
                        "url": url,
                        "detected_encoding": detected.encoding if detected else "fallback",
                        "confidence": getattr(detected, 'confidence', 0) if detected else 0
                    }
                )
            else:
                # Fallback: use httpx default with ISO-8859-1 → UTF-8 heuristic
                encoding = resp.encoding or "ISO-8859-1"
                if encoding.upper() == "ISO-8859-1":
                    resp.encoding = "utf-8"  # Heuristic: try UTF-8 first
                text = resp.text

                logger.debug(
                    "encoding_fallback",
                    extra={"url": url, "encoding": encoding}
                )

            final_url = str(resp.url)

            logger.info(
                "fetch_success",
                extra={
                    "url": url,
                    "final_url": final_url,
                    "status_code": resp.status_code,
                    "content_length": len(text)
                }
            )

            return text, final_url

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.NetworkError) as e:
            # Retry-able errors - tenacity will handle
            logger.debug(
                "fetch_retry",
                extra={"url": url, "error": str(e) or type(e).__name__, "error_type": type(e).__name__}
            )
            raise

        except httpx.HTTPError as e:
            # Non-retryable HTTP errors
            error_msg = str(e) or type(e).__name__
            logger.error(
                "fetch_failed",
                extra={"url": url, "error": error_msg, "error_type": type(e).__name__}
            )

            raise ScraperError(
                url=url,
                reason=f"HTTP error: {error_msg}",
                error_type=ScraperErrorType.NETWORK,
                details={"error_type": type(e).__name__}
            ) from e

        except Exception as e:
            # Unexpected errors
            logger.error(
                "fetch_unexpected_error",
                extra={"url": url, "error": str(e), "error_type": type(e).__name__}
            )

            raise ScraperError(
                url=url,
                reason=f"Unexpected error: {str(e)}",
                error_type=ScraperErrorType.NETWORK,
                details={"error_type": type(e).__name__}
            ) from e


@retry(
    reraise=True,
    stop=stop_after_attempt(get_settings().wizard.scrape_max_retries + 1),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type((
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.NetworkError
    )),
)
async def reach_url(url: str, timeout: Optional[float] = None) -> bool:
    """
    Check whether a URL is reachable using HEAD falling back to GET.

    Uses HEAD request first for efficiency, falls back to GET if HEAD fails
    or returns error status.

    Args:
        url: URL to check
        timeout: Optional timeout override in seconds

    Returns:
        True if 2xx/3xx status code is received, False otherwise

    Raises:
        ScraperError: On connection errors or timeouts
    """
    async with get_http_client() as client:
        try:
            # Try HEAD first (lightweight)
            response = await client.head(url, timeout=timeout)

            # If HEAD returns error, try GET
            if response.status_code >= 400:
                response = await client.get(url, timeout=timeout)

            is_reachable = 200 <= response.status_code < 400

            logger.debug(
                "url_reachability_check",
                extra={
                    "url": url,
                    "status_code": response.status_code,
                    "reachable": is_reachable
                }
            )

            return is_reachable

        except httpx.HTTPError as exc:
            logger.warning(
                "url_unreachable",
                extra={"url": url, "error": str(exc)}
            )
            raise ScraperError(
                url=url,
                reason=f"Reachability check failed: {str(exc)}",
                error_type=ScraperErrorType.NETWORK,
                details={"error_type": type(exc).__name__}
            ) from exc


async def fetch(url: str) -> Optional[str]:
    """
    Fetch text content for a URL. Returns None on non-OK responses.

    Simple helper for fetching page content without detailed error handling.

    Args:
        url: URL to fetch

    Returns:
        Page text content or None if request failed
    """
    async with get_http_client() as client:
        try:
            resp = await client.get(url)

            if resp.status_code >= 400:
                logger.debug(
                    "fetch_error_status",
                    extra={"url": url, "status_code": resp.status_code}
                )
                return None

            # Improve decoding similar to fetch_with_final_url
            if HAS_CHARSET_NORMALIZER:
                detected = from_bytes(resp.content).best()
                text = str(detected) if detected else resp.text
            else:
                encoding = resp.encoding or "ISO-8859-1"
                if encoding.upper() == "ISO-8859-1":
                    resp.encoding = "utf-8"
                text = resp.text

            logger.debug(
                "fetch_success_simple",
                extra={"url": url, "content_length": len(text)}
            )

            return text

        except httpx.HTTPError as e:
            logger.debug(
                "fetch_failed_simple",
                extra={"url": url, "error": str(e)}
            )
            return None
