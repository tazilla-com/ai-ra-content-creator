from __future__ import annotations

from typing import Optional, Any

from workshops_ai.llm.throttled_client import ThrottledChatOpenAI, ThrottledAzureChatOpenAI
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

    resolved_model = model or settings.wizard.default_model

    if resolved_model not in settings.wizard.allowed_models:
        raise ModelNotAllowedError(resolved_model, settings.wizard.allowed_models)

    # Reasoning models (gpt-5*, o1*, o3*, o4*) don't support temperature parameter
    _REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
    is_reasoning = any(resolved_model.startswith(p) for p in _REASONING_PREFIXES)

    # Azure OpenAI provider
    if settings.llm.provider == "azure":
        if ThrottledAzureChatOpenAI is None:
            raise RuntimeError("langchain_openai AzureChatOpenAI is not available. Install langchain-openai.")

        client_kwargs = {
            "azure_deployment": resolved_model,
            "azure_endpoint": settings.llm.azure_endpoint,
            "api_key": settings.llm.api_key,
            "api_version": settings.llm.azure_api_version,
            "timeout": settings.wizard.llm_timeout_sec,
            "max_retries": 0,
            "include_response_headers": True,
        }

        if not is_reasoning:
            client_kwargs["temperature"] = settings.wizard.temperature

        if response_format is not None:
            client_kwargs["model_kwargs"] = {"response_format": response_format}

        return ThrottledAzureChatOpenAI(**client_kwargs)

    # Default: OpenAI provider
    client_kwargs = {
        "model": resolved_model,
        "openai_api_key": settings.llm.api_key,
        "timeout": settings.wizard.llm_timeout_sec,
        "max_retries": 0,
        "include_response_headers": True,
    }

    if not is_reasoning:
        client_kwargs["temperature"] = settings.wizard.temperature

    if response_format is not None:
        client_kwargs["model_kwargs"] = {"response_format": response_format}

    return ThrottledChatOpenAI(**client_kwargs)
