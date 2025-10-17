# workshops_ai/wizard/content_resolver.py
"""
Content resolution for wizard endpoints.

Resolves website content from either:
1. Cached content in Redis session
2. Traditional scraping (BeautifulSoup) or AI web search (OpenAI)
3. Direct text provided in request

Uses Redis distributed locks to prevent duplicate scraping.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Literal
from redis.asyncio import Redis

from workshops_ai.wizard.redis_lock import redis_lock
from workshops_ai.wizard.exceptions import ContentNotFoundError, ScrapingError

logger = logging.getLogger(__name__)

# Type for search method
SearchMethod = Literal["scrape", "web_search"]


# Mock website content for testing
MOCK_WEBSITE_CONTENT = {
    "example.com": """
        Example Corporation - Leading Provider of Digital Solutions

        About Us:
        Example Corporation is a technology company specializing in cloud infrastructure,
        cybersecurity solutions, and digital transformation services.

        Our Services:
        - Cloud Infrastructure Management
        - Managed Security Services
        - Data Analytics and Business Intelligence
        - Custom Software Development
        - IT Consulting and Strategy

        Industries Served:
        We serve clients across finance, healthcare, retail, and manufacturing sectors,
        providing tailored solutions to meet their specific needs.

        Compliance:
        Our services comply with ISO 27001, GDPR, HIPAA, and SOC 2 standards.

        Contact: info@example.com | +1-555-0100
    """,
    "acme.org": """
        ACME Energy Solutions - Powering Tomorrow's World

        Company Overview:
        ACME Energy Solutions is a leading provider of renewable energy systems
        and smart grid technologies.

        Core Offerings:
        - Solar Power Installation and Maintenance
        - Wind Energy Systems
        - Smart Grid Management
        - Energy Storage Solutions
        - Energy Efficiency Consulting

        Market Presence:
        Operating across North America and Europe, serving utility companies,
        commercial enterprises, and residential customers.

        Certifications:
        ISO 50001 (Energy Management), ISO 9001 (Quality Management)

        Sustainability:
        Committed to reducing carbon emissions and promoting clean energy adoption.
    """,
    "medtech.io": """
        MedTech Innovations - Advanced Healthcare Technology

        Mission:
        Delivering cutting-edge medical devices and digital health solutions
        to improve patient outcomes.

        Product Lines:
        - Diagnostic Imaging Systems
        - Patient Monitoring Solutions
        - Electronic Health Records (EHR) Platform
        - Telemedicine Services
        - Medical Data Analytics

        Regulatory Compliance:
        FDA approved devices, HIPAA compliant systems, CE marked products

        Target Customers:
        Hospitals, clinics, diagnostic centers, and healthcare providers

        Research & Development:
        Continuous innovation in AI-driven diagnostics and remote patient care
    """,
}


class ContentResolver:
    """
    Resolves website content from various sources with caching.

    Resolution order:
    1. Direct text from request (if provided)
    2. Cached content in Redis session
    3. Mock scraper (production scraper to be implemented later)
    """

    def __init__(self, redis_client: Redis):
        """
        Initialize ContentResolver.

        Args:
            redis_client: Redis client for caching
        """
        self.redis = redis_client

    async def _get_cached_content(
        self,
        session_key: str,
        url: Optional[str]
    ) -> Optional[str]:
        """
        Get cached website content from Redis session.

        Args:
            session_key: Redis session key (e.g., "wizard:org123:user456")
            url: Website URL (optional, for logging)

        Returns:
            Cached content if found, None otherwise
        """
        try:
            content = await self.redis.hget(session_key, "website_content")

            if content:
                logger.info(
                    "content_cache_hit",
                    extra={
                        "session_key": session_key,
                        "url": url,
                        "content_length": len(content),
                    }
                )
                return content.decode("utf-8") if isinstance(content, bytes) else content

            logger.debug(
                "content_cache_miss",
                extra={"session_key": session_key, "url": url}
            )
            return None

        except Exception as e:
            logger.error(
                "content_cache_read_failed",
                extra={
                    "session_key": session_key,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            return None

    async def _set_cached_content(
        self,
        session_key: str,
        content: str,
        url: Optional[str] = None
    ) -> None:
        """
        Cache website content in Redis session.

        Args:
            session_key: Redis session key
            content: Website content to cache
            url: Website URL (optional, for logging)
        """
        try:
            await self.redis.hset(
                session_key,
                "website_content",
                content
            )

            # Also store URL if provided
            if url:
                await self.redis.hset(session_key, "website_url", url)

            logger.info(
                "content_cached",
                extra={
                    "session_key": session_key,
                    "url": url,
                    "content_length": len(content),
                }
            )

        except Exception as e:
            logger.error(
                "content_cache_write_failed",
                extra={
                    "session_key": session_key,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            # Don't raise - caching is optional

    async def _scrape_website(self, url: str) -> str:
        """
        Scrape website using BeautifulSoup-based scraper.

        Scrapes homepage + services + about pages with security checks.

        Args:
            url: Website URL to scrape

        Returns:
            Combined website content from all scraped pages

        Raises:
            ScrapingError: If scraping fails
            AIResponseError: If content fails security checks
        """
        from workshops_ai.wizard.scraper import get_website, flatten_website_content
        from workshops_ai.wizard.scraper.exceptions import ScraperError
        from workshops_ai.wizard.exceptions import AIResponseError

        try:
            # Scrape website (returns detailed structure internally)
            website_content = await get_website(url)

            # Flatten to simple format (url, content)
            final_url, combined_content = flatten_website_content(website_content)

            logger.info(
                "scrape_success",
                extra={
                    "url": url,
                    "final_url": final_url,
                    "pages_scraped": len(website_content.pages),
                    "content_length": len(combined_content),
                    "had_errors": bool(website_content.errors),
                    "had_warnings": bool(website_content.security_warnings),
                }
            )

            return combined_content

        except ScraperError as e:
            # Convert ScraperError to ScrapingError (our internal exception)
            logger.error(
                "scrape_failed",
                extra={
                    "url": url,
                    "error": str(e),
                    "error_type": e.error_type.value,
                }
            )
            raise ScrapingError(
                url=url,
                reason=str(e),
                details={"scraper_error_type": e.error_type.value}
            ) from e

        except AIResponseError:
            # Already the correct exception type, just log and re-raise
            logger.error(
                "scrape_security_failed",
                extra={"url": url}
            )
            raise

        except Exception as e:
            logger.error(
                "scrape_unexpected_error",
                extra={
                    "url": url,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            raise ScrapingError(
                url=url,
                reason=f"Unexpected scraping error: {str(e)}",
                details={"error_type": type(e).__name__}
            ) from e

    async def resolve_content(
        self,
        session_key: str,
        website_text: Optional[str] = None,
        website_url: Optional[str] = None,
        search_method: SearchMethod = "scrape",
        ai_model: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """
        Resolve website content from various sources.

        Resolution priority:
        1. Direct text from request (website_text)
        2. Cached content in Redis session
        3. Fetch from URL using specified search_method:
           - "scrape": Traditional mock scraper (default)
           - "web_search": AI-powered web search using OpenAI

        Args:
            session_key: Redis session key
            website_text: Optional direct website content
            website_url: Optional website URL to scrape/search
            search_method: Content retrieval method ("scrape" or "web_search")
            ai_model: Optional AI model override for web_search

        Returns:
            Tuple of (content, metadata dict with source info)

        Raises:
            ContentNotFoundError: If no content source available
            ScrapingError: If scraping fails
            AIResponseError: If web search fails
            AITimeoutError: If web search times out
        """
        metadata = {}

        # Priority 1: Direct text from request
        if website_text:
            logger.info(
                "content_resolved_from_text",
                extra={
                    "session_key": session_key,
                    "content_length": len(website_text),
                }
            )

            # Cache for future requests
            await self._set_cached_content(
                session_key=session_key,
                content=website_text,
                url=website_url
            )

            metadata["source"] = "direct_text"
            metadata["content_length"] = len(website_text)
            return website_text, metadata

        # Priority 2: Cached content
        cached_content = await self._get_cached_content(session_key, website_url)
        if cached_content:
            metadata["source"] = "cache"
            metadata["content_length"] = len(cached_content)
            return cached_content, metadata

        # Priority 3: Fetch from URL using specified method
        if website_url:
            # Use distributed lock to prevent multiple requests fetching same URL
            lock_key = f"wizard:lock:{search_method}:{website_url}"

            try:
                async with redis_lock(
                    redis_client=self.redis,
                    lock_key=lock_key,
                    timeout=30,  # Lock expires after 30s
                    blocking_timeout=10,  # Wait up to 10s for lock
                ):
                    # Double-check cache (another request might have fetched while we waited)
                    cached_content = await self._get_cached_content(session_key, website_url)
                    if cached_content:
                        logger.info(
                            "content_resolved_from_cache_after_lock",
                            extra={"session_key": session_key, "url": website_url}
                        )
                        metadata["source"] = "cache_after_lock"
                        metadata["content_length"] = len(cached_content)
                        return cached_content, metadata

                    # Fetch content using specified method
                    if search_method == "web_search":
                        # Use AI web search
                        from workshops_ai.wizard.web_search import get_website_via_search

                        content, usage_metrics, debug_info = await get_website_via_search(
                            url=website_url,
                            ai_model=ai_model,
                        )

                        metadata["source"] = "web_search"
                        metadata["url"] = website_url
                        metadata["content_length"] = len(content)
                        metadata["model"] = usage_metrics.model
                        metadata["tokens"] = usage_metrics.total_tokens
                        metadata["cost_usd"] = usage_metrics.cost_usd
                    else:
                        # Use traditional scraping
                        content = await self._scrape_website(website_url)

                        metadata["source"] = "scrape"
                        metadata["url"] = website_url
                        metadata["content_length"] = len(content)

                    # Cache the result
                    await self._set_cached_content(
                        session_key=session_key,
                        content=content,
                        url=website_url
                    )

                    return content, metadata

            except Exception as e:
                # Re-raise known exceptions
                from workshops_ai.wizard.exceptions import AIResponseError, AITimeoutError
                if isinstance(e, (ScrapingError, AIResponseError, AITimeoutError)):
                    raise

                logger.error(
                    f"{search_method}_failed",
                    extra={
                        "url": website_url,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )

                # Wrap generic errors based on method
                if search_method == "web_search":
                    from workshops_ai.wizard.exceptions import AIResponseError
                    raise AIResponseError(
                        f"Web search error: {str(e)}",
                        model=ai_model or "unknown",
                        details={"error_type": type(e).__name__}
                    ) from e
                else:
                    raise ScrapingError(
                        url=website_url,
                        reason=str(e),
                        details={"error_type": type(e).__name__}
                    ) from e

        # No content source available
        raise ContentNotFoundError(
            message="No website content available. Provide either 'website_text' or 'website_url'.",
            details={
                "website_text_provided": website_text is not None,
                "website_url_provided": website_url is not None,
                "cache_checked": True,
            }
        )
