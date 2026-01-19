"""
Company information scraping tool.
"""

from __future__ import annotations

import logging
from typing import Annotated, Optional

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from workshops_ai.wizard.scraper import get_website, flatten_website_content
from workshops_ai.session import get_session, save_session

logger = logging.getLogger(__name__)


@tool
async def scrape_company_info(
    website_url: Optional[str] = None,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]
) -> Command:
    """
    Get company-specific information for asset context by scraping company website.

    This tool retrieves information about the SPECIFIC COMPANY - their tech stack,
    systems, size, industry details, products, and services.

    Args:
        website_url: (Optional) Company website URL. Only provide if user gave you a URL.
            Examples: "https://example.com", "www.company.sk"
            If not provided, tool checks cache first.

    DATA SOURCES (checked in order, stops when found):
    1. Session cache - already scraped this session (fastest)
    2. Web scraping - scrapes company website if URL provided

    USAGE FLOW:
    1. Call without website_url first: scrape_company_info()
        → checks cache, requests URL if needed
    2. If tool returns "I need your company's website URL..." → ask user for URL
    3. User provides URL → call again: scrape_company_info(website_url="www.example.sk")
    4. Tool scrapes website and caches result for session

    USE THIS TOOL WHEN:
    - You need company-specific technology information
    - User mentions "our systems", "we use", "our company"
    - You need to know what specific tech stack they have
    - Before proposing CIA classifications for assets

    DO NOT USE THIS TOOL WHEN:
    - You need general industry best practices (use lookup_knowledge instead)
    - User is asking about generic/theoretical assets

    Returns plain text with company information, or message requesting URL.
    """
    session_id = state.get("session_id")
    if not session_id:
        error_msg = "Error: Session ID not found in state"
        return Command(
            update={
                "tools_used": state.get("tools_used", []) + ["scrape_company_info"],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]
            }
        )

    session = await get_session(session_id)

    if not session:
        error_msg = f"Error: Session not found: {session_id}"
        return Command(
            update={
                "tools_used": state.get("tools_used", []) + ["scrape_company_info"],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]
            }
        )

    # 1. Check session cache (wizard stores as website_content)
    cached = session.wizard_metadata.get("website_content")
    if cached:
        logger.info(
            "Using cached company info from wizard",
            extra={"session_id": session_id, "content_length": len(cached)}
        )
        return Command(
            update={
                "tools_used": state.get("tools_used", []) + ["scrape_company_info"],
                "messages": [ToolMessage(content=cached, tool_call_id=tool_call_id)]
            }
        )

    # 2. Web search fallback
    # Try: 1) provided URL parameter, 2) organization.website
    company_website = website_url or state.get("organization", {}).get("website")

    if not company_website:
        error_msg = (
            "I need your company's website URL to fetch information about your organization. "
            "Please provide your company website address (e.g., https://example.com)."
        )
        return Command(
            update={
                "tools_used": state.get("tools_used", []) + ["scrape_company_info"],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]
            }
        )

    # Use existing scraping infrastructure
    try:
        logger.info(
            "Scraping company website",
            extra={"session_id": session_id, "url": company_website}
        )

        website_content = await get_website(company_website)
        final_url, combined_text = flatten_website_content(website_content)

        # Cache for this session (consistent with wizard)
        session.wizard_metadata["website_url"] = str(final_url)
        session.wizard_metadata["website_content"] = combined_text
        await save_session(session)

        logger.info(
            "Company website scraped successfully",
            extra={
                "session_id": session_id,
                "final_url": str(final_url),
                "pages_scraped": len(website_content.pages),
                "content_length": len(combined_text)
            }
        )

        return Command(
            update={
                "tools_used": state.get("tools_used", []) + ["scrape_company_info"],
                "messages": [ToolMessage(content=combined_text, tool_call_id=tool_call_id)]
            }
        )

    except Exception as e:
        logger.error(
            "Failed to scrape company website",
            extra={"session_id": session_id, "url": company_website, "error": str(e)}
        )
        error_msg = f"Error scraping company website: {str(e)}"
        return Command(
            update={
                "tools_used": state.get("tools_used", []) + ["scrape_company_info"],
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]
            }
        )
