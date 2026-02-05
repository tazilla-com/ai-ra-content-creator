# workshops_ai/routers/scrape_router.py
"""
FastAPI router for web scraping endpoints.

Provides endpoints for extracting information from websites.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends
from typing import Dict, Any

from workshops_ai.app.security import require_auth
from workshops_ai.wizard.schemas import (
    ScrapeWebsiteInput,
    ScrapeWebsiteOutput,
)

logger = logging.getLogger(__name__)

# Router (auth applied in main.py when mounting)
router = APIRouter(
    prefix="/scrape",
    tags=["scrape"],
)


@router.post(
    "/website",
    response_model=ScrapeWebsiteOutput,
    summary="Scrape company website",
    description="Extract business information from website using traditional scraping or AI search"
)
async def scrape_website(
    request: ScrapeWebsiteInput,
    identity: Dict[str, Any] = Depends(require_auth),
) -> ScrapeWebsiteOutput:
    """
    Scrape website content using specified method.

    Methods:
    - "scrape": Traditional BeautifulSoup-based scraping (homepage + services + about)
    - "web_search": AI-powered web search using OpenAI

    Note: This endpoint does NOT use session caching. Each request fetches fresh content.
    """
    from workshops_ai.llm.langfuse_helper import get_langfuse_client
    from workshops_ai.settings import get_settings

    # Apply default search_method from settings if not provided
    settings = get_settings()
    if request.search_method is None:
        request.search_method = settings.wizard.default_search_method

    logger.info(
        "scrape_website_request",
        extra={
            "url": request.url,
            "search_method": request.search_method,
            "user_id": identity.get("sub"),
        }
    )

    langfuse_client = get_langfuse_client()

    # Wrap entire endpoint in Langfuse trace for e2e visibility
    if langfuse_client:
        with langfuse_client.start_as_current_span(
            name="scrape_website_endpoint",
            input={
                "url": request.url,
                "search_method": request.search_method,
            },
            metadata={
                "endpoint": "/scrape/website",
                "method": request.search_method,
            }
        ) as root_span:
            try:
                result = await _scrape_website_impl(request, identity, root_span)

                # Update root span with success output
                root_span.update(
                    output={
                        "url": result.url,
                        "content_length": len(result.content),
                    }
                )

                return result

            except Exception as e:
                # Update root span with error
                root_span.update(level="ERROR", status_message=str(e))
                raise
    else:
        # Fallback without Langfuse
        return await _scrape_website_impl(request, identity, None)


async def _scrape_website_impl(
    request: ScrapeWebsiteInput,
    identity: Dict[str, Any],
    root_span = None,
) -> ScrapeWebsiteOutput:
    """Implementation of scrape_website logic (extracted for Langfuse wrapping)."""
    try:
        if request.search_method == "web_search":
            # Use AI web search (uses default model from settings)
            from workshops_ai.wizard.web_search import get_website_via_search

            content, usage_metrics, debug_info = await get_website_via_search(
                url=request.url,
                ai_model=None,  # Use default from settings.wizard.web_search_model
            )

            # Extract sources from debug_info
            sources = debug_info.get("sources", [])

            logger.info(
                "scrape_website_success_web_search",
                extra={
                    "url": request.url,
                    "content_length": len(content),
                    "model": usage_metrics.model,
                    "tokens": usage_metrics.total_tokens,
                    "cost_usd": usage_metrics.cost_usd,
                    "num_sources": len(sources),
                }
            )

            return ScrapeWebsiteOutput(
                url=request.url,
                content=content,
                sources=sources,
            )

        else:
            # Use traditional scraping
            from workshops_ai.wizard.scraper import get_website, flatten_website_content
            from workshops_ai.wizard.scraper.exceptions import ScraperError
            from workshops_ai.wizard.exceptions import AIResponseError

            # Scrape website
            website_content = await get_website(request.url)

            # Flatten to simple format
            final_url, combined_content = flatten_website_content(website_content)

            logger.info(
                "scrape_website_success_scrape",
                extra={
                    "url": request.url,
                    "final_url": final_url,
                    "pages_scraped": len(website_content.pages),
                    "content_length": len(combined_content),
                    "had_errors": bool(website_content.errors),
                    "had_warnings": bool(website_content.security_warnings),
                }
            )

            return ScrapeWebsiteOutput(
                url=final_url,
                content=combined_content,
            )

    except Exception as e:
        logger.error(
            "scrape_website_failed",
            extra={
                "url": request.url,
                "search_method": request.search_method,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise
