# workshops_ai/wizard/llm_helper.py
"""
LLM helper for wizard endpoints.

Provides a unified interface for calling LLM with structured output:
- Reuses cached ChatOpenAI client from factory
- Validates model against allowed list
- Supports model + temperature override per request
- Automatic retry on validation errors (3x with exponential backoff)
- Timeout handling (30s default from settings)
- Returns structured Pydantic output
"""
from __future__ import annotations

import logging
from typing import TypeVar, Type, Optional, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pydantic import BaseModel, ValidationError

from workshops_ai.settings import get_settings
from workshops_ai.graph.factory import _get_llm_client_cached

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Model validation
# =============================================================================

def validate_model(model: Optional[str], allowed_models: list[str]) -> str:
    """
    Validate LLM model name against allowed list.

    Args:
        model: Model name (e.g., "gpt-4o", "gpt-5-mini")
        allowed_models: List of allowed model names

    Returns:
        Validated model name

    Raises:
        ValueError: If model not in allowed list
    """
    if not model:
        raise ValueError("Model name is required")

    if model not in allowed_models:
        raise ValueError(
            f"Model '{model}' not allowed. Must be one of: {', '.join(allowed_models)}"
        )

    return model


# =============================================================================
# LLM client builder
# =============================================================================

def get_llm_client(
    model: Optional[str] = None,
) -> Any:
    """
    Get cached ChatOpenAI client with optional model override.

    Args:
        model: Model override (if None, uses settings default)

    Returns:
        Cached ChatOpenAI client

    Raises:
        ValueError: If model not in allowed list
        RuntimeError: If settings missing required fields
    """
    settings = get_settings()

    # Get wizard settings
    wizard_settings = settings.wizard
    allowed_models = wizard_settings.allowed_models

    # Get LLM settings for defaults
    llm_settings = settings.llm
    default_model = llm_settings.model
    default_temperature = llm_settings.temperature
    api_key = llm_settings.api_key or settings.openai_api_key
    base_url = llm_settings.base_url or settings.openai_base_url
    http_timeout = wizard_settings.llm_timeout_sec
    max_retries = wizard_settings.llm_max_retries

    # Resolve model (override or default)
    final_model = model or default_model
    if not final_model:
        raise RuntimeError("No model specified and no default model in settings")

    # Validate model
    final_model = validate_model(final_model, allowed_models)

    # Validate API key
    if not api_key:
        raise RuntimeError("Missing OpenAI API key in settings")

    # Get cached client (reuses factory cache)
    client = _get_llm_client_cached(
        model=final_model,
        temperature=default_temperature,
        http_timeout_sec=http_timeout,
        max_retries=max_retries,
        api_key=api_key,
        base_url=base_url,
    )

    logger.info(
        "Wizard LLM client ready",
        extra={
            "model": final_model,
            "temperature": default_temperature,
            "timeout_sec": http_timeout,
            "max_retries": max_retries,
        }
    )

    return client


# =============================================================================
# Structured LLM call with retry
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(ValidationError),
    reraise=True,
)
async def call_llm_with_structured_output(
    prompt: str,
    output_schema: Type[T],
    model: Optional[str] = None,
) -> T:
    """
    Call LLM with structured output and automatic retry on validation errors.

    Uses LangChain's with_structured_output() to enforce schema.
    Retries up to 3 times with exponential backoff on ValidationError.
    Timeout errors fail immediately (no retry).

    Args:
        prompt: System/user prompt for LLM
        output_schema: Pydantic model class for structured output
        model: Model override (optional)

    Returns:
        Structured output matching output_schema

    Raises:
        ValidationError: After 3 retries if LLM output still invalid
        TimeoutError: If LLM call exceeds timeout
        RuntimeError: If settings misconfigured
        ValueError: If model not allowed
    """
    # Get LLM client (cached, validated)
    client = get_llm_client(model=model)

    # Attach structured output schema
    structured_client = client.with_structured_output(output_schema)

    logger.info(
        "Calling LLM with structured output",
        extra={
            "schema": output_schema.__name__,
            "model": model or "default",
        }
    )

    try:
        # Call LLM (async)
        result = await structured_client.ainvoke(prompt)

        logger.info(
            "LLM call successful",
            extra={
                "schema": output_schema.__name__,
                "result_type": type(result).__name__,
            }
        )

        return result

    except ValidationError as e:
        logger.warning(
            "LLM output validation failed, retrying",
            extra={
                "schema": output_schema.__name__,
                "error": str(e),
            }
        )
        raise  # Let tenacity retry

    except TimeoutError as e:
        logger.error(
            "LLM call timeout (no retry)",
            extra={
                "schema": output_schema.__name__,
                "timeout_sec": get_settings().wizard.llm_timeout_sec,
                "error": str(e),
            }
        )
        raise  # Don't retry timeouts

    except Exception as e:
        logger.error(
            "LLM call failed",
            extra={
                "schema": output_schema.__name__,
                "error_type": type(e).__name__,
                "error": str(e),
            }
        )
        raise
