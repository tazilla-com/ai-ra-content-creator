from __future__ import annotations

from typing import Optional, Any

from workshops_ai.llm.throttled_client import ThrottledChatOpenAI
from workshops_ai.settings import get_settings
from workshops_ai.wizard.exceptions import ModelNotAllowedError


def create_wizard_llm_client(
    model: Optional[str] = None,
    response_format: Optional[Any] = None
) -> ThrottledChatOpenAI:
    """
    Factory pre wizard LLM klient s throttlingom.

    Args:
        model: Model name (ak None, použije sa default z settings)
        response_format: OpenAI structured outputs schema (optional)

    Returns:
        ThrottledChatOpenAI instance

    Raises:
        ModelNotAllowedError: Ak model nie je v allowed_models
    """
    settings = get_settings()

    resolved_model = model or settings.llm.model

    if resolved_model not in settings.wizard.allowed_models:
        raise ModelNotAllowedError(resolved_model, settings.wizard.allowed_models)

    client_kwargs = {
        "model": resolved_model,
        "temperature": settings.llm.temperature,
        "openai_api_key": settings.llm.api_key,
        "timeout": settings.wizard.llm_timeout_sec,
        "max_retries": 0,
        "include_response_headers": True  # Enable access to OpenAI rate limit headers
    }

    # Add response_format if provided (OpenAI structured outputs)
    if response_format is not None:
        client_kwargs["model_kwargs"] = {"response_format": response_format}

    return ThrottledChatOpenAI(**client_kwargs)
