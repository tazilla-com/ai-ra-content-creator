# File: workshops_ai/tools/tracing.py
# Purpose: normalize env for LangSmith/LangChain tracing (V1/V2), then enable.
from __future__ import annotations
import os

def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}

def setup_langsmith_env() -> None:
    """
    Accept both:
      - V2 (preferované): LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
      - legacy:           LANGSMITH_TRACING,   LANGSMITH_ENDPOINT, LANGSMITH_API_KEY, LANGSMITH_PROJECT
    Fill missing LANGCHAIN_* from LANGSMITH_* pokud jsou k dispozici.
    """
    # If legacy is set, map to new envs if new are absent.
    if os.getenv("LANGSMITH_TRACING") and not os.getenv("LANGCHAIN_TRACING_V2"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true" if _truthy(os.getenv("LANGSMITH_TRACING")) else "false"
    if os.getenv("LANGSMITH_ENDPOINT") and not os.getenv("LANGCHAIN_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "")
    if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")

    # Optional QoL
    if not os.getenv("LANGCHAIN_TRACING_V2") and _truthy(os.getenv("LANGCHAIN_TRACING_V2", "")):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

def enable_tracing_if_configured() -> None:
    """Call once on startup, before LangChain is used."""
    setup_langsmith_env()
    # No explicit SDK init needed; LangChain respects env on import/use.
