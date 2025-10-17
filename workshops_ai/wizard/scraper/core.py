# workshops_ai/wizard/scraper/core.py
"""
Core web scraping implementation - main get_website() function.

Scrapes homepage + services + about pages with security and quality checks.
"""
from __future__ import annotations

import logging
import asyncio
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from bs4 import BeautifulSoup

from workshops_ai.settings import get_settings
from workshops_ai.wizard.scraper.http_client import fetch_with_final_url, normalize_url
from workshops_ai.wizard.scraper.text_processing import preprocess_text, limit_text
from workshops_ai.wizard.scraper.validation import is_valid_content
from workshops_ai.wizard.scraper.sanitizer import ContentSanitizer
from workshops_ai.wizard.scraper.link_extractor import extract_links, find_page_candidates
from workshops_ai.wizard.scraper.constants import SERVICES_SUFFIXES, ABOUT_US_SUFFIXES
from workshops_ai.wizard.scraper.exceptions import ScraperError, ScraperErrorType
from workshops_ai.wizard.exceptions import AIResponseError

logger = logging.getLogger(__name__)


class PageType(str, Enum):
    """Types of pages scraped."""
    HOME = "home"
    SERVICES = "services"
    ABOUT = "about"


@dataclass
class WebpageContent:
    """Content from a single scraped page."""
    url: str
    content: str
    page_type: PageType


@dataclass
class WebsiteContent:
    """Complete scraped website content."""
    base_url: str
    pages: List[WebpageContent]
    errors: Optional[List[str]] = None
    security_warnings: Optional[List[str]] = None


async def _fetch_and_process_page(
    url: str,
    page_type: PageType,
    sanitizer: ContentSanitizer,
) -> Optional[WebpageContent]:
    """
    Fetch single page, extract text, sanitize, and validate.

    Args:
        url: URL to fetch
        page_type: Type of page (home/services/about)
        sanitizer: Content sanitizer instance

    Returns:
        WebpageContent if successful and valid, None otherwise
    """
    settings = get_settings()

    try:
        # Fetch HTML
        fetched = await fetch_with_final_url(url)
        if not fetched:
            return None

        html, final_url = fetched

        # Extract text from HTML
        soup = BeautifulSoup(html, "lxml")
        raw_text = soup.get_text()

        # Preprocess
        clean_text = preprocess_text(raw_text)

        # Sanitize (security)
        sanitized_text, warnings = sanitizer.sanitize(
            clean_text,
            strict=settings.wizard.scrape_strict_sanitization
        )

        if warnings:
            logger.warning(
                "page_security_warnings",
                extra={"url": url, "warnings": warnings}
            )

        # Validate content quality
        if not is_valid_content(sanitized_text, settings.wizard.scrape_min_page_chars):
            logger.warning(
                "page_content_invalid",
                extra={"url": url, "page_type": page_type.value}
            )
            return None

        # Limit content length
        limited_text = limit_text(sanitized_text, settings.wizard.scrape_max_page_chars)

        logger.info(
            "page_scraped",
            extra={
                "url": url,
                "page_type": page_type.value,
                "content_length": len(limited_text),
                "warnings_count": len(warnings)
            }
        )

        return WebpageContent(
            url=final_url,
            content=limited_text,
            page_type=page_type
        )

    except ScraperError as e:
        logger.warning(
            "page_fetch_failed",
            extra={
                "url": url,
                "page_type": page_type.value,
                "error": str(e),
                "error_type": e.error_type.value
            }
        )
        return None

    except Exception as e:
        logger.error(
            "page_processing_failed",
            extra={
                "url": url,
                "page_type": page_type.value,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        return None


async def _fetch_additional_pages_combined(
    candidates: List[str],
    page_type: PageType,
    sanitizer: ContentSanitizer,
) -> Optional[WebpageContent]:
    """
    Fetch multiple candidate pages in parallel and combine content.

    Args:
        candidates: List of candidate URLs
        page_type: Type of pages (services/about)
        sanitizer: Content sanitizer instance

    Returns:
        Combined WebpageContent or None if all failed
    """
    settings = get_settings()
    sem = asyncio.Semaphore(settings.wizard.scrape_concurrent_limit)

    # Fetch all candidates in parallel with concurrency limit
    async def _task(u: str) -> Optional[WebpageContent]:
        async with sem:
            return await _fetch_and_process_page(u, page_type, sanitizer)

    results = await asyncio.gather(*[_task(u) for u in candidates], return_exceptions=True)

    # Filter successful results
    successful = [r for r in results if isinstance(r, WebpageContent)]

    if not successful:
        return None

    # Combine all fetched content up to limit
    total = 0
    pieces: List[str] = []
    first_url = successful[0].url
    max_total = settings.wizard.scrape_max_combined_chars

    for page in successful:
        if total >= max_total:
            break

        piece = page.content
        if total + len(piece) > max_total:
            piece = piece[: max_total - total]

        pieces.append(piece)
        total += len(piece)

    combined_content = " ".join(pieces)

    logger.info(
        "pages_combined",
        extra={
            "page_type": page_type.value,
            "pages_fetched": len(successful),
            "combined_length": len(combined_content)
        }
    )

    return WebpageContent(
        url=first_url,
        content=combined_content,
        page_type=page_type
    )


async def get_website(url: str) -> WebsiteContent:
    """
    Scrape website content including homepage, services, and about pages.

    Flow:
    1. Normalize and fetch homepage
    2. Extract all same-domain links
    3. Find services and about page candidates
    4. Fetch additional pages in parallel
    5. Sanitize and validate all content
    6. Return combined result

    Args:
        url: Website URL to scrape

    Returns:
        WebsiteContent with scraped pages

    Raises:
        ScraperError: If homepage cannot be fetched
        AIResponseError: If content fails security checks or is insufficient
    """
    settings = get_settings()

    # Check if scraping is enabled
    if not settings.wizard.scrape_enabled:
        raise AIResponseError(
            message="Traditional scraping is disabled in configuration",
            model="scraper"
        )

    # Normalize URL
    base_url = normalize_url(url)

    logger.info("scraping_started", extra={"url": base_url})

    # Initialize sanitizer
    sanitizer = ContentSanitizer()

    # Container for results
    pages: List[WebpageContent] = []
    errors: List[str] = []
    all_warnings: List[str] = []

    # 1. Fetch homepage
    try:
        fetched = await fetch_with_final_url(base_url)
        if not fetched:
            raise ScraperError(
                url=base_url,
                reason="Homepage unreachable",
                error_type=ScraperErrorType.NOT_FOUND
            )

        html, final_url = fetched

    except ScraperError as e:
        logger.error(
            "homepage_fetch_failed",
            extra={"url": base_url, "error": str(e)}
        )
        raise

    # 2. Extract links
    soup = BeautifulSoup(html, "lxml")
    links = extract_links(soup, base_url)

    logger.debug("links_extracted", extra={"count": len(links)})

    # 3. Find services and about candidates
    services_candidates = find_page_candidates(links, SERVICES_SUFFIXES)
    about_candidates = find_page_candidates(links, ABOUT_US_SUFFIXES)

    logger.debug(
        "candidates_found",
        extra={
            "services": len(services_candidates),
            "about": len(about_candidates)
        }
    )

    # 4. Process homepage
    homepage_page = await _fetch_and_process_page(final_url, PageType.HOME, sanitizer)

    if homepage_page:
        pages.append(homepage_page)
    else:
        errors.append("Homepage content invalid or too short")

    # 5. Fetch services pages (parallel)
    if services_candidates:
        services_page = await _fetch_additional_pages_combined(
            services_candidates,
            PageType.SERVICES,
            sanitizer
        )
        if services_page:
            pages.append(services_page)
        else:
            errors.append("No valid services pages found")

    # 6. Fetch about pages (parallel)
    if about_candidates:
        about_page = await _fetch_additional_pages_combined(
            about_candidates,
            PageType.ABOUT,
            sanitizer
        )
        if about_page:
            pages.append(about_page)
        else:
            errors.append("No valid about pages found")

    # 7. Final validation - check total content
    if not pages:
        raise AIResponseError(
            message="No valid content scraped from website",
            model="scraper",
            details={"url": base_url, "errors": errors}
        )

    total_content = " ".join(p.content for p in pages)

    # Simple check: just verify we have reasonable total length
    # (individual pages already passed full validation including uniqueness)
    if len(total_content) < settings.wizard.scrape_min_page_chars:
        raise AIResponseError(
            message=f"Scraped content insufficient: {len(total_content)} chars across {len(pages)} pages",
            model="scraper",
            details={
                "url": base_url,
                "total_chars": len(total_content),
                "pages_count": len(pages),
                "min_required": settings.wizard.scrape_min_page_chars
            }
        )

    # 8. Apply combined character limit
    combined_pages = []
    total_chars = 0
    max_total = settings.wizard.scrape_max_combined_chars

    for page in pages:
        if total_chars >= max_total:
            break

        content = page.content
        if total_chars + len(content) > max_total:
            content = content[: max_total - total_chars]

        combined_pages.append(
            WebpageContent(
                url=page.url,
                content=content,
                page_type=page.page_type
            )
        )
        total_chars += len(content)

    logger.info(
        "scraping_completed",
        extra={
            "url": base_url,
            "pages_count": len(combined_pages),
            "total_chars": total_chars,
            "errors_count": len(errors),
            "warnings_count": len(all_warnings)
        }
    )

    result = WebsiteContent(
        base_url=base_url,
        pages=combined_pages,
        errors=errors if errors else None,
        security_warnings=all_warnings if all_warnings else None
    )

    return result


def flatten_website_content(detailed: WebsiteContent) -> Tuple[str, str]:
    """
    Convert detailed scraper output to simple format compatible with web_search.

    This adapter maintains compatibility with the simple schema (url + content)
    while internally using the detailed multi-page structure.

    Args:
        detailed: Detailed WebsiteContent with multiple pages

    Returns:
        Tuple of (url, combined_content)
    """
    # Combine all page content with section markers
    combined_parts = []

    for page in detailed.pages:
        # Add section header for clarity
        section_header = f"[{page.page_type.upper()} PAGE]"
        combined_parts.append(f"{section_header}\n{page.content}")

    combined_content = "\n\n".join(combined_parts)

    logger.debug(
        "content_flattened",
        extra={
            "url": detailed.base_url,
            "pages_count": len(detailed.pages),
            "combined_length": len(combined_content),
            "had_errors": bool(detailed.errors),
            "had_warnings": bool(detailed.security_warnings)
        }
    )

    return detailed.base_url, combined_content
