# File: workshops_ai/llm/client.py

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable, Literal
from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, StrictStr, ConfigDict, field_validator

# Lazy import; fail-fast with a clear error message if missing.
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore


DEFAULT_MODEL = "gpt-5-mini"


@runtime_checkable
class _LLMSettingsP(Protocol):
    """Minimal contract that our Settings.llm must satisfy."""
    model: str
    temperature: float
    call_timeout_sec: int
    openai_max_retries: int
    api_key: Optional[str]  # optional; may be None


@runtime_checkable
class _LangChainSettingsP(Protocol):
    """Optional tracing settings (LangSmith)."""
    tracing_v2: bool
    api_key: Optional[str]
    endpoint: Optional[str]
    project: Optional[str]


@runtime_checkable
class _AppSettingsP(Protocol):
    """Duck-typed app settings used by this factory.

    Expected shape:
        settings.llm: _LLMSettingsP
        settings.langchain: _LangChainSettingsP (optional)
    """
    llm: _LLMSettingsP
    langchain: Optional[_LangChainSettingsP]  # may be absent or None


class LLMConfig(BaseModel):
    """Strict config built from Settings (no os.getenv)."""
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "azure"] = Field("openai", description="LLM provider identifier.")
    model: StrictStr = Field(DEFAULT_MODEL, description="Model name (e.g., gpt-4o-mini).")
    temperature: StrictFloat = Field(0.2, ge=0.0, le=2.0, description="Sampling temperature.")
    call_timeout_sec: StrictInt = Field(30, ge=1, le=300, description="Per-call LLM/agent timeout in seconds.")
    max_retries: StrictInt = Field(3, ge=0, le=10, description="HTTP retry attempts.")
    api_key: Optional[StrictStr] = Field(default=None, description="Provider API key.")

    # Azure OpenAI settings
    azure_endpoint: Optional[StrictStr] = Field(default=None, description="Azure OpenAI endpoint URL.")
    azure_api_version: StrictStr = Field("2025-01-01-preview", description="Azure OpenAI API version.")
    azure_deployment: Optional[StrictStr] = Field(default=None, description="Azure deployment name (defaults to model).")

    # Tracing (LangSmith / LangChain v2)
    tracing_v2_enabled: StrictBool = Field(False, description="Enable LangSmith tracing v2.")
    tracing_api_key: Optional[StrictStr] = Field(default=None, description="LangSmith API key.")
    tracing_endpoint: Optional[StrictStr] = Field(default=None, description="LangSmith endpoint URL.")
    tracing_project: Optional[StrictStr] = Field(default=None, description="LangSmith project name.")

    @field_validator("model")
    @classmethod
    def _strip_model(cls, v: str) -> str:
        v = v.strip()
        return v or DEFAULT_MODEL

    @classmethod
    def from_settings(
        cls,
        settings: _AppSettingsP,
        *,
        override_model: Optional[str] = None,
        override_temperature: Optional[float] = None,
    ) -> "LLMConfig":
        """Build config from strongly-typed Settings (no env lookups)."""
        if not isinstance(settings, _AppSettingsP):
            raise TypeError(
                "Settings object does not satisfy required interface. "
                "Required: settings.llm.(model, temperature, call_timeout_sec, openai_max_retries, api_key) "
                "and optional settings.langchain.(tracing_v2, api_key, endpoint, project)."
            )

        llm = settings.llm
        lc = getattr(settings, "langchain", None)

        model = override_model if override_model is not None else getattr(llm, "model", DEFAULT_MODEL)
        temperature = override_temperature if override_temperature is not None else float(getattr(llm, "temperature", 0.2))

        cfg = cls(
            provider=getattr(llm, "provider", "openai") or "openai",
            model=model,
            temperature=temperature,
            call_timeout_sec=int(getattr(llm, "call_timeout_sec", 30)),
            max_retries=int(getattr(llm, "openai_max_retries", 3)),
            api_key=getattr(llm, "api_key", None),
            azure_endpoint=getattr(llm, "azure_endpoint", None),
            azure_api_version=getattr(llm, "azure_api_version", "2025-01-01-preview") or "2025-01-01-preview",
            azure_deployment=getattr(llm, "azure_deployment", None),
            tracing_v2_enabled=bool(getattr(lc, "tracing_v2", False)) if lc else False,
            tracing_api_key=(getattr(lc, "api_key", None) if lc else None),
            tracing_endpoint=(getattr(lc, "endpoint", None) if lc else None),
            tracing_project=(getattr(lc, "project", None) if lc else None),
        )
        return cfg


def _assert_openai_available() -> None:
    """Ensure `langchain_openai` is present."""
    if ChatOpenAI is None:
        raise RuntimeError(
            "langchain_openai is not available. Install it via: `pip install langchain-openai`"
        )


def _apply_tracing(cfg: LLMConfig) -> None:
    """Configure LangSmith tracing via LangChain envs (set locally, no global .env reads)."""
    import os
    if not cfg.tracing_v2_enabled:
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if cfg.tracing_api_key:
        os.environ["LANGCHAIN_API_KEY"] = cfg.tracing_api_key
    if cfg.tracing_project:
        os.environ["LANGCHAIN_PROJECT"] = cfg.tracing_project
    if cfg.tracing_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = cfg.tracing_endpoint


def _is_reasoning_model(model: str) -> bool:
    """Reasoning models (gpt-5*, o1*, o3*, o4*) don't support temperature parameter."""
    return any(model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))


def make_llm_client(cfg: LLMConfig) -> Any:
    """
    Create and return a LangChain chat model instance according to config.
    This function never reads from process env except for setting tracing envs when enabled.
    """
    _assert_openai_available()
    _apply_tracing(cfg)

    if cfg.provider == "azure":
        if not cfg.api_key:
            raise RuntimeError("Azure OpenAI API key is missing in Settings.llm.api_key.")
        if not cfg.azure_endpoint:
            raise RuntimeError("Azure OpenAI endpoint is missing in Settings.llm.azure_endpoint.")
        try:
            from langchain_openai import AzureChatOpenAI  # type: ignore
        except ImportError:
            raise RuntimeError("langchain_openai AzureChatOpenAI not available. Install langchain-openai.")
        kwargs = {
            "azure_deployment": cfg.azure_deployment or cfg.model,
            "azure_endpoint": cfg.azure_endpoint,
            "api_key": cfg.api_key,
            "api_version": cfg.azure_api_version,
            "timeout": cfg.call_timeout_sec,
            "max_retries": cfg.max_retries,
        }
        if not _is_reasoning_model(cfg.model):
            kwargs["temperature"] = cfg.temperature
        return AzureChatOpenAI(**kwargs)  # type: ignore[call-arg]

    if cfg.provider == "openai":
        if not cfg.api_key:
            raise RuntimeError(
                "OpenAI API key is missing in Settings.llm.api_key. "
                "Provide it via your Settings loader (e.g., Pydantic BaseSettings)."
            )
        kwargs = {
            "model": cfg.model,
            "timeout": cfg.call_timeout_sec,
            "max_retries": cfg.max_retries,
            "api_key": cfg.api_key,
        }
        if not _is_reasoning_model(cfg.model):
            kwargs["temperature"] = cfg.temperature
        return ChatOpenAI(**kwargs)  # type: ignore[call-arg]

    raise ValueError(f"Unsupported LLM provider: {cfg.provider!r}")


def make_llm_client_from_settings(
    settings: _AppSettingsP,
    *,
    override_model: Optional[str] = None,
    override_temperature: Optional[float] = None,
) -> Any:
    """
    Preferred entry point: build LLMConfig from Settings and return an LLM client.
    """
    cfg = LLMConfig.from_settings(
        settings,
        override_model=override_model,
        override_temperature=override_temperature,
    )
    return make_llm_client(cfg)
