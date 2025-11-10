# workshops_ai/agents/tools/scraping.py
"""
Website scraping tool for LangGraph agents.

Design:
- Wrapper around existing get_website() function (reuse, no duplication!)
- Caches scraped content in session (avoid re-scraping)
- Returns flattened content suitable for LLM consumption
- Same scraping engine as wizard (consistency!)
"""
from __future__ import annotations

import logging
from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field, HttpUrl

from workshops_ai.wizard.scraper import get_website, flatten_website_content
from workshops_ai.agents.session import get_agent_session, save_agent_session

logger = logging.getLogger(__name__)

__all__ = ["scrape_company_website"]


# -----------------------------------------------------------------------------
# Tool Input Schemas
# -----------------------------------------------------------------------------

class ScrapeWebsiteInput(BaseModel):
    """Input for scraping a company website."""
    session_id: str = Field(..., description="Session identifier")
    url: HttpUrl = Field(..., description="Website URL to scrape (e.g., https://example.com)")
    purpose: str | None = Field(
        None,
        description="What information to extract (e.g., 'find IT systems', 'identify services')"
    )


# -----------------------------------------------------------------------------
# Tool
# -----------------------------------------------------------------------------

@tool
async def scrape_company_website(input: ScrapeWebsiteInput) -> dict:
    """
    Scrape company website to extract business information for asset creation.

    This tool:
    - Uses the same scraping engine as the wizard (consistent results!)
    - Scrapes homepage + services + about pages
    - Caches content in session (no re-scraping on same URL)
    - Returns combined text suitable for LLM analysis

    Args:
        input: ScrapeWebsiteInput with session_id, url, purpose

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - content: Combined text from all scraped pages
        - final_url: Final URL after redirects
        - pages_scraped: Number of pages successfully scraped
        - errors: List of any errors encountered
        - cached: Whether content was loaded from session cache

    Example:
        Scrape company website to find IT assets:
        url = "https://example.com"
        purpose = "identify IT systems and infrastructure"

    Note:
        - Content is automatically saved to session.scraped_content
        - Subsequent calls with same URL will use cached content
        - Use cached content to avoid hammering the website
    """
    try:
        # Load session
        session = await get_agent_session(input.session_id)
        if not session:
            return {
                "status": "error",
                "message": f"Session not found: {input.session_id}",
            }

        url_str = str(input.url)

        # Check cache first
        if session.scraped_url == url_str and session.scraped_content:
            logger.info(
                "Using cached scraped content",
                extra={
                    "session_id": input.session_id,
                    "url": url_str,
                    "content_length": len(session.scraped_content),
                }
            )
            return {
                "status": "success",
                "content": session.scraped_content,
                "final_url": session.scraped_url,
                "cached": True,
                "message": "Using cached content from previous scrape",
            }

        # Scrape website (reuse existing function!)
        logger.info(
            "Scraping website",
            extra={"session_id": input.session_id, "url": url_str}
        )

        website_content = await get_website(url_str)

        # Flatten for LLM consumption
        final_url, combined_text = flatten_website_content(website_content)

        # Cache in session
        session.scraped_url = str(final_url)
        session.scraped_content = combined_text
        await save_agent_session(session)

        logger.info(
            "Website scraped successfully",
            extra={
                "session_id": input.session_id,
                "url": url_str,
                "final_url": final_url,
                "pages_scraped": len(website_content.pages),
                "content_length": len(combined_text),
                "has_errors": bool(website_content.errors),
            }
        )

        return {
            "status": "success",
            "content": combined_text,
            "final_url": str(final_url),
            "pages_scraped": len(website_content.pages),
            "errors": website_content.errors or [],
            "has_warnings": bool(website_content.security_warnings),
            "cached": False,
            "message": f"Successfully scraped {len(website_content.pages)} page(s) from {final_url}",
        }

    except Exception as e:
        logger.error(
            "Failed to scrape website",
            extra={
                "session_id": input.session_id,
                "url": str(input.url),
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        return {
            "status": "error",
            "message": f"Failed to scrape website: {str(e)}",
            "error_type": type(e).__name__,
        }
