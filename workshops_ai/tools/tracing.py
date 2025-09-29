# File: workshops_ai/tools/tracing.py
# -----------------------------------------------------------------------------
# Purpose
# Programmatic LangSmith / LangChain tracing configured strictly via app settings
# (no environment variable reads/writes). This module exposes:
#
#   - tracing_enabled()                 -> bool
#   - get_langsmith_client()            -> langsmith.Client    (cached)
#   - make_tracer()                     -> Optional[LangChainTracer]
#   - get_callbacks()                   -> list[LangChainTracer] for LC calls
#   - tracing_session(enabled=None)     -> context manager
#   - enable_tracing_if_configured(...) -> back-compat shim for app/main.py
#
# Design notes
# - We rely on workshops_ai.settings.get_settings() as the single source of truth.
# - We do NOT mutate or read os.environ here (no side effects).
# - All factories are idempotent and safe to import at app startup.
# -----------------------------------------------------------------------------

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator, Optional, List

from langsmith import Client, tracing_context
from langchain_core.tracers.langchain import LangChainTracer

from workshops_ai.settings import get_settings  # single source of truth

__all__ = [
    "tracing_enabled",
    "get_langsmith_client",
    "make_tracer",
    "get_callbacks",
    "tracing_session",
    "enable_tracing_if_configured",  # back-compat shim
]


# ---------- Feature flag ------------------------------------------------------


def tracing_enabled() -> bool:
    """
    Returns True if LangChain/LangSmith tracing is enabled via settings.
    """
    return bool(get_settings().langchain.tracing_v2)


# ---------- Client / tracer factories (cached) --------------------------------


@lru_cache(maxsize=1)
def get_langsmith_client() -> Client:
    """
    Build a LangSmith Client from settings.langchain (no env variables).
    The endpoint and API key are taken directly from settings.
    """
    s = get_settings()
    lc = s.langchain
    # Explicit values only; Client will not fall back to environment.
    return Client(
        api_url=lc.endpoint or "https://api.smith.langchain.com",
        api_key=lc.api_key,
    )


def make_tracer() -> Optional[LangChainTracer]:
    """
    Create a LangChain tracer bound to our LangSmith Client and project.
    Returns None when tracing is disabled.
    """
    if not tracing_enabled():
        return None
    s = get_settings()
    return LangChainTracer(
        project_name=s.langchain.project,
        client=get_langsmith_client(),
    )


def get_callbacks() -> List[LangChainTracer]:
    """
    Convenience helper for passing callbacks=[...] into LangChain chains/LLMs.
    Returns [] when tracing is disabled.
    """
    t = make_tracer()
    return [t] if t else []


# ---------- Context manager for per-request/session control -------------------


@contextmanager
def tracing_session(enabled: Optional[bool] = None) -> Iterator[None]:
    """
    Context manager that enables LangSmith tracing for the enclosed code block.
    Uses LangSmith's `tracing_context` with our programmatic Client.
    If disabled (globally or via argument), it yields a no-op.
    """
    if enabled is None:
        enabled = tracing_enabled()
    if not enabled:
        # No tracing -> just run the block without side effects.
        yield None
        return

    client = get_langsmith_client()
    # Inside this block, LC/traceable runs stream to LangSmith using our client.
    with tracing_context(enabled=True, client=client):
        yield None


# ---------- Backward-compatibility shim ---------------------------------------


def enable_tracing_if_configured(settings=None) -> None:
    """
    Back-compat shim expected by workshops_ai.app.main.

    Historically this function set environment variables for LangChain v2 tracing.
    We now configure tracing strictly via code (no env). The shim is:
      - idempotent
      - side-effect free (no os.environ writes)
      - safe to call during application startup

    It "warms up" the internal caches so that downstream code can immediately
    grab callbacks or start a tracing_session without additional setup.
    """
    try:
        s = settings or get_settings()
        if getattr(s.langchain, "tracing_v2", False):
            # Warm up factories; any exceptions must never stop app startup.
            _ = get_langsmith_client()
            _ = make_tracer()
        # If tracing is disabled, this is a no-op.
    except Exception:
        # Tracing is optional; never fail application startup because of it.
        pass
