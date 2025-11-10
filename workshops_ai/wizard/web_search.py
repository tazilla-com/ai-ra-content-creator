# workshops_ai/wizard/web_search.py
"""
AI-powered web search for retrieving website content.

Uses OpenAI's native web_search tool to intelligently gather and synthesize
information about an organization directly from the web. This provides an
alternative to traditional BeautifulSoup scraping.

Key Features:
- AI decides what to search for and synthesizes results
- Domain-filtered to restrict search to target domain only
- Returns structured content compatible with ContentResolver
- Single comprehensive call (no pagination)
"""
from __future__ import annotations

import logging
import asyncio
from urllib.parse import urlparse
from typing import Optional, Tuple

from openai import AsyncOpenAI, RateLimitError, APIConnectionError

from workshops_ai.settings import get_settings
from workshops_ai.wizard.exceptions import AIResponseError, AITimeoutError
from workshops_ai.wizard.cost_calculator import create_usage_metrics, UsageMetrics

logger = logging.getLogger(__name__)


# System prompt for web search
WEB_SEARCH_PROMPT_TEMPLATE = """Search for comprehensive information about the company at {url}.

Focus on gathering information that helps understand their business operations and assets:

WHAT THEY DO:
- Products and services offered
- Customer types (consumers/businesses)
- Industry sector

TECHNICAL OPERATIONS:
- Digital platforms, apps, or online services
- Infrastructure type (if relevant)
- Data they collect or process

BUSINESS MODEL:
- Revenue sources
- Key partners or suppliers
- Critical dependencies

ABSOLUTE RULES:
• NO citations, URLs, links, or source references in any format
• NO conversational phrases ("Let me know...", "Feel free to...", "Here's...", etc.)
• NO questions, offers, suggestions, or call-to-action phrases
• NO introductory or closing remarks
• NO markdown links like [text](url)
• Format: Direct bullet points only
• Deliver: Pure facts, no commentary
• Terminate: Immediately after last fact"""


async def get_website_via_search(
    url: str,
    ai_model: Optional[str] = None,
) -> Tuple[str, UsageMetrics, dict]:
    """
    Retrieve website content using OpenAI's web search tool.

    Uses OpenAI's native web search to gather comprehensive information
    about the company website in one call. AI decides what information to search for.

    Args:
        url: The website URL to search for
        ai_model: Optional AI model to use for search (defaults to settings.wizard.web_search_model)

    Returns:
        Tuple of (synthesized content, usage metrics, debug info dict)

    Raises:
        AITimeoutError: If web search times out
        AIResponseError: If response is invalid or insufficient
    """
    settings = get_settings()

    # Check if web search is enabled
    if not settings.wizard.web_search_enabled:
        raise AIResponseError(
            "Web search is disabled in configuration",
            model=ai_model or settings.wizard.web_search_model
        )

    # Parse URL and extract domain for filtering
    try:
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else url
        domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]

        if not domain:
            raise AIResponseError(
                f"Could not extract domain from URL: {url}",
                model=ai_model or settings.wizard.web_search_model
            )

        logger.info(
            "web_search_started",
            extra={
                "url": url,
                "base_url": base_url,
                "domain": domain,
            }
        )

    except Exception as e:
        logger.error(
            "web_search_url_parse_failed",
            extra={"url": url, "error": str(e)}
        )
        raise AIResponseError(
            f"Invalid URL format: {url}",
            model=ai_model or settings.wizard.web_search_model,
            details={"error": str(e)}
        ) from e

    # Initialize OpenAI client
    if not settings.llm.api_key:
        raise AIResponseError(
            "OpenAI API key not configured",
            model=settings.wizard.web_search_model
        )

    client = AsyncOpenAI(
        api_key=settings.llm.api_key,
        timeout=settings.wizard.web_search_timeout_sec,
    )
    model = settings.wizard.web_search_model  # Always use configured model, ignore ai_model parameter

    # Construct search prompt
    task_prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(url=url)

    logger.debug(
        "web_search_calling_openai",
        extra={
            "model": model,
            "domain_filter": domain,
            "timeout_sec": settings.wizard.web_search_timeout_sec,
        }
    )

    # Call OpenAI Responses API with web search tool
    try:
        response = await client.responses.create(
            model=model,
            tools=[
                {
                    "type": "web_search",
                    "filters": {
                        "allowed_domains": [domain]  # Restrict to target domain only
                    }
                }
            ],
            input=task_prompt,
        )

        logger.debug(
            "web_search_response_received",
            extra={
                "model": model,
                "usage": response.usage.model_dump() if response.usage else None,
            }
        )

    except RateLimitError as e:
        logger.error(
            "web_search_rate_limit",
            extra={"url": url, "model": model, "error": str(e)}
        )
        raise AIResponseError(
            f"OpenAI rate limit exceeded: {str(e)}",
            model=model,
            details={"error_type": "RateLimitError"}
        ) from e

    except APIConnectionError as e:
        logger.error(
            "web_search_connection_error",
            extra={"url": url, "model": model, "error": str(e)}
        )
        raise AIResponseError(
            f"OpenAI API connection failed: {str(e)}",
            model=model,
            details={"error_type": "APIConnectionError"}
        ) from e

    except Exception as e:
        # Check if it's a timeout error from OpenAI/httpx
        error_msg = str(e).lower()
        if "timeout" in error_msg or "timed out" in error_msg:
            logger.error(
                "web_search_timeout",
                extra={
                    "url": url,
                    "model": model,
                    "timeout_sec": settings.wizard.web_search_timeout_sec,
                    "error": str(e),
                }
            )
            raise AITimeoutError(
                timeout_sec=settings.wizard.web_search_timeout_sec,
                model=model
            ) from e

        # Other errors
        logger.error(
            "web_search_failed",
            extra={
                "url": url,
                "model": model,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise AIResponseError(
            f"Web search error: {str(e)}",
            model=model,
            details={"error_type": type(e).__name__}
        ) from e

    # Extract text content without citations using output_text
    message = response.output_text

    if not message:
        logger.warning(
            "web_search_no_output",
            extra={"url": url, "model": model}
        )
        raise AIResponseError(
            "Web search returned no output",
            model=model,
            details={"url": url}
        )

    if not message or len(message.strip()) < settings.wizard.web_search_min_content_chars:
        logger.warning(
            "web_search_insufficient_content",
            extra={
                "url": url,
                "model": model,
                "content_length": len(message) if message else 0,
                "min_required": settings.wizard.web_search_min_content_chars,
            }
        )
        raise AIResponseError(
            f"Web search returned insufficient content (< {settings.wizard.web_search_min_content_chars} chars)",
            model=model,
            details={
                "url": url,
                "content_length": len(message) if message else 0,
            }
        )

    # Limit content length
    clean_content = message.strip()
    limited_content = clean_content[:settings.wizard.max_content_length]

    if len(clean_content) > settings.wizard.max_content_length:
        logger.info(
            "web_search_content_truncated",
            extra={
                "url": url,
                "original_length": len(clean_content),
                "truncated_length": len(limited_content),
            }
        )

    # Extract usage from response metadata (Responses API has different usage format)
    usage_metrics = UsageMetrics(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        cost_usd=0.0,
        model=model,
    )

    if response.usage:
        # Responses API usage attributes: input_tokens, output_tokens, total_tokens
        prompt_tokens = getattr(response.usage, 'input_tokens', 0) or getattr(response.usage, 'prompt_tokens', 0)
        completion_tokens = getattr(response.usage, 'output_tokens', 0) or getattr(response.usage, 'completion_tokens', 0)

        usage_metrics = create_usage_metrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
            pricing=settings.wizard.openai_pricing
        )

    # Build debug info
    debug_info = {
        "url": url,
        "domain": domain,
        "model": model,
        "content_length": len(limited_content),
        "original_content_length": len(clean_content),
        "response_preview": limited_content[:500],
    }

    logger.info(
        "web_search_completed",
        extra={
            "url": url,
            "model": model,
            "content_length": len(limited_content),
            "tokens": usage_metrics.total_tokens,
            "cost_usd": usage_metrics.cost_usd,
        }
    )

    return limited_content, usage_metrics, debug_info
