# File: workshops_ai/tools/tracing.py
# Purpose: Export LANGCHAIN_* env vars purely from centralized settings (no getenv reads).
#          LangChain/LangSmith pick these up at import/use time.

from __future__ import annotations

import os
from typing import Any

from workshops_ai.settings import get_settings  # single source of truth


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return (str(v or "")).strip().lower() in {"1", "true", "yes", "on", "y"}


def enable_tracing_if_configured() -> None:
    """
    Read values from settings.langchain.* and set LANGCHAIN_* env vars accordingly.
    We never read from process env — settings is the only source of truth.
    Call once on startup, before LangChain is imported/used.
    """
    s = get_settings()
    lc = getattr(s, "langchain", None)

    tracing_v2 = _truthy(getattr(lc, "tracing_v2", False)) if lc is not None else False
    endpoint = getattr(lc, "endpoint", None)
    api_key = getattr(lc, "api_key", None)
    project = getattr(lc, "project", None)

    # Always write explicit boolean so downstream libs have a deterministic value
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if tracing_v2 else "false"

    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = str(endpoint)
    else:
        os.environ.pop("LANGCHAIN_ENDPOINT", None)

    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = str(api_key)
    else:
        os.environ.pop("LANGCHAIN_API_KEY", None)

    if project:
        os.environ["LANGCHAIN_PROJECT"] = str(project)
    else:
        os.environ.pop("LANGCHAIN_PROJECT", None)
