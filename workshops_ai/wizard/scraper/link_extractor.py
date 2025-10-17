# workshops_ai/wizard/scraper/link_extractor.py
"""
Link extraction and domain validation using Public Suffix List.
"""
from __future__ import annotations

import logging
from typing import List, Iterable
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from workshops_ai.wizard.scraper.constants import SERVICES_SUFFIXES, ABOUT_US_SUFFIXES, HTML_EXTENSIONS

logger = logging.getLogger(__name__)

# Try to import publicsuffix2 (optional dependency)
try:
    from publicsuffix2 import get_sld
    HAS_PSL = True
except ImportError:
    HAS_PSL = False
    logger.warning(
        "publicsuffix2 not available, using simple domain comparison. "
        "Install with: pip install publicsuffix2"
    )


def _is_same_domain(url_host: str, base_host: str) -> bool:
    """
    Check if two hosts belong to the same domain using Public Suffix List.

    Uses publicsuffix2 library for accurate domain comparison.
    Handles multi-level TLDs correctly (e.g., .co.uk, .github.io).

    Args:
        url_host: Hostname from URL
        base_host: Base hostname to compare against

    Returns:
        True if same registered domain
    """
    if not url_host or not base_host:
        return False

    if HAS_PSL:
        # Use Public Suffix List for accurate domain matching
        url_domain = get_sld(url_host)
        base_domain = get_sld(base_host)
        return url_domain == base_domain
    else:
        # Fallback: simple comparison (less accurate for multi-level TLDs)
        # Extract last 2 parts (naive approach)
        url_parts = url_host.split('.')
        base_parts = base_host.split('.')

        if len(url_parts) >= 2 and len(base_parts) >= 2:
            return '.'.join(url_parts[-2:]) == '.'.join(base_parts[-2:])

        return url_host == base_host


def extract_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    """
    Extract all same-domain links from HTML.

    Filtering rules:
    - Must have href attribute
    - HTTP or HTTPS only (excludes mailto:, tel:, javascript:)
    - Same domain only (uses PSL for accurate matching)
    - Query and fragment stripped
    - Trailing slash removed
    - Deduplicated

    Args:
        soup: BeautifulSoup parsed HTML
        base_url: Base URL for relative link resolution

    Returns:
        List of unique same-domain URLs
    """
    base_host = urlparse(base_url).hostname or ""
    results: List[str] = []

    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue

        # Convert relative to absolute URL
        href = urljoin(base_url, href)
        parsed = urlparse(href)

        # Filter out non-HTTP(S) links
        if parsed.scheme not in {"http", "https"}:
            continue

        # Skip if no hostname
        if not parsed.hostname:
            continue

        # Filter out external domains
        if not _is_same_domain(parsed.hostname, base_host):
            continue

        # Clean URL: remove query and fragment
        cleaned = urlunparse(parsed._replace(query="", fragment=""))
        cleaned = cleaned.rstrip("/")
        results.append(cleaned)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in results:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    logger.debug(
        "links_extracted",
        extra={
            "base_url": base_url,
            "total_links": len(results),
            "unique_links": len(unique)
        }
    )

    return unique


def find_page_candidates(links: Iterable[str], suffixes: List[str]) -> List[str]:
    """
    Find and rank candidate URLs by suffix matching.

    Matching strategy:
    - Path ends with suffix: /services, /about
    - With extensions: /services.html, /services.php
    - Substring match: /services/consulting, /about/team
    - Dash variants: /about-us, /about_us, /aboutus
    - HTTPS upgrade: Prefers HTTPS if any link uses it

    Args:
        links: List of URLs to search
        suffixes: List of URL suffixes to match (e.g., SERVICES_SUFFIXES)

    Returns:
        List of matching URLs
    """
    # Expand suffixes: handle dash variations
    expanded: set[str] = set(suffixes)
    for s in list(suffixes):
        if "-" in s:
            expanded.add(s.replace("-", "_"))  # about-us → about_us
            expanded.add(s.replace("-", ""))   # about-us → aboutus

    # Prefer HTTPS if available
    https_available = any(str(l).startswith("https://") for l in links)

    candidates: set[str] = set()

    for link in links:
        candidate = link.replace("http://", "https://") if https_available else link
        path = urlparse(candidate).path.lower()

        for suffix in expanded:
            # Path ends with suffix: /services, /company/about
            if path.endswith(suffix):
                candidates.add(candidate)
                continue

            # With extension: /services.html, /services.php
            if any(path.endswith(f"{suffix}.{ext}") for ext in HTML_EXTENSIONS):
                candidates.add(candidate)
                continue

            # Substring match: /services/ or /about/team
            if f"{suffix}/" in path:
                candidates.add(candidate)

                # Also add truncated version: /about/team → /about
                truncated = candidate.split(f"{suffix}/")[0] + suffix
                candidates.add(truncated)

    result = list(candidates)

    logger.debug(
        "candidates_found",
        extra={
            "suffix_count": len(suffixes),
            "candidates_found": len(result)
        }
    )

    return result
