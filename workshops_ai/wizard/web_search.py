# workshops_ai/wizard/web_search.py
"""
AI-powered web search for retrieving website content.

Uses OpenAI's native web_search tool to intelligently gather and synthesize
information about an organization directly from the web. This provides an
alternative to traditional BeautifulSoup scraping.

Key Features:
- AI decides what to search for and synthesizes results
- No domain restrictions - searches anywhere on the web
- Supports agentic search with reasoning models (gpt-5, o1, etc)
- Returns structured content compatible with ContentResolver
- Single comprehensive call (no pagination)
- Extracts and returns all source URLs consulted
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
from workshops_ai.wizard.prompts import WEB_SEARCH_PROMPT_TEMPLATE
from workshops_ai.llm.langfuse_helper import get_langfuse_client, get_text_prompt_from_langfuse

logger = logging.getLogger(__name__)


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
        Tuple of (synthesized content, usage metrics, debug info dict with sources)

    Raises:
        AITimeoutError: If web search times out
        AIResponseError: If response is invalid or insufficient
    """
    settings = get_settings()

    # Wrap entire function in Langfuse span for tracing
    langfuse_client = get_langfuse_client()

    if langfuse_client:
        with langfuse_client.start_as_current_span(
            name="web_search",
            input={"url": url, "ai_model": ai_model},
            metadata={"method": "web_search"}
        ) as search_span:
            try:
                result = await _get_website_via_search_impl(url, ai_model, settings, search_span)

                # Update span with success output
                content, usage_metrics, debug_info = result
                search_span.update(
                    output={
                        "content_length": len(content),
                        "tokens": usage_metrics.total_tokens,
                        "cost_usd": usage_metrics.cost_usd,
                        "model": usage_metrics.model,
                        "prompt_source": debug_info.get("prompt_source"),
                    }
                )

                return result
            except Exception as e:
                search_span.update(level="ERROR", status_message=str(e))
                raise
    else:
        # Fallback without Langfuse
        return await _get_website_via_search_impl(url, ai_model, settings, None)


async def _get_website_via_search_impl(
    url: str,
    ai_model: Optional[str],
    settings,
    search_span=None,
) -> Tuple[str, UsageMetrics, dict]:
    """Implementation of get_website_via_search (extracted for Langfuse wrapping)."""
    # Check if web search is enabled
    if not settings.wizard.web_search_enabled:
        raise AIResponseError(
            "Web search is disabled in configuration",
            model=ai_model or settings.wizard.web_search_model
        )

    # Fetch prompt from Langfuse with fallback to local (using helper)
    prompt_data = get_text_prompt_from_langfuse(
        prompt_name="helper/web_search",
        fallback_text=WEB_SEARCH_PROMPT_TEMPLATE,
    )

    task_prompt_template = prompt_data["prompt_text"]
    prompt_source = prompt_data["prompt_source"]
    prompt_version = prompt_data["version"]
    prompt_tags = prompt_data["tags"]
    prompt_labels = prompt_data["labels"]

    # Update span with prompt metadata (if span exists)
    if search_span:
        metadata = {
            "method": "web_search",
            "tool": "openai_web_search",
            "prompt_name": "helper/web_search",
            "prompt_version": prompt_version,
            "prompt_source": prompt_source,
        }

        # Add Langfuse prompt tags/labels to metadata
        if prompt_tags:
            metadata["prompt_tags"] = ",".join(prompt_tags)
        if prompt_labels:
            metadata["prompt_labels"] = ",".join(prompt_labels)

        search_span.update(metadata=metadata)

    # Parse URL and extract domain for filtering
    try:
        # Add https:// if scheme is missing
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"

        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else url
        domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]

        # Normalize domain: remove www prefix for consistent filtering
        if domain.startswith('www.'):
            domain = domain[4:]

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

    # Construct search prompt from template (either Langfuse or local)
    # Use base_url (with scheme) for clarity to AI model
    task_prompt = task_prompt_template.format(url=base_url)

    logger.debug(
        "web_search_calling_openai",
        extra={
            "model": model,
            "url": url,
            "timeout_sec": settings.wizard.web_search_timeout_sec,
            "prompt_source": prompt_source,
        }
    )

    # Call OpenAI Responses API with web search tool
    try:
        # Build request parameters
        request_params = {
            "model": model,
            "tools": [
                {
                    "type": "web_search"
                    # No domain restrictions - search anywhere on the web
                }
            ],
            "input": task_prompt,
            "include": ["web_search_call.action.sources"],  # Request sources field
        }

        # Add reasoning parameter only for reasoning models (gpt-5, o1, etc)
        # Non-reasoning models (gpt-4o, gpt-4o-mini) don't support this parameter
        reasoning_models = ("gpt-5", "o1", "o3", "o4")
        is_reasoning_model = any(model.startswith(prefix) for prefix in reasoning_models)
        if settings.wizard.web_search_reasoning_effort and is_reasoning_model:
            request_params["reasoning"] = {"effort": settings.wizard.web_search_reasoning_effort}
            logger.info(
                "web_search_using_reasoning",
                extra={
                    "model": model,
                    "reasoning_effort": settings.wizard.web_search_reasoning_effort,
                }
            )

        response = await client.responses.create(**request_params)

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

    # Extract sources from web_search_call output item
    sources = []
    if hasattr(response, 'output') and response.output:
        for output_item in response.output:
            if hasattr(output_item, 'type') and output_item.type == 'web_search_call':
                if hasattr(output_item, 'action') and hasattr(output_item.action, 'sources'):
                    raw_sources = output_item.action.sources or []
                    # Extract URL from each ActionSearchSource object
                    for source in raw_sources:
                        if hasattr(source, 'url'):
                            sources.append(source.url)
                    break

    if sources:
        logger.info(
            "web_search_sources_extracted",
            extra={
                "url": url,
                "model": model,
                "num_sources": len(sources),
                "sources": sources[:10],  # Log first 10 sources
            }
        )

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
        "prompt_source": prompt_source,
        "content_length": len(limited_content),
        "original_content_length": len(clean_content),
        "response_preview": limited_content[:500],
        "sources": sources,  # List of URLs consulted during web search
        "num_sources": len(sources),
    }

    logger.info(
        "web_search_completed",
        extra={
            "url": url,
            "model": model,
            "content_length": len(limited_content),
            "tokens": usage_metrics.total_tokens,
            "cost_usd": usage_metrics.cost_usd,
            "num_sources": len(sources),
        }
    )

    return limited_content, usage_metrics, debug_info
