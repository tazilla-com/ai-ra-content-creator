"""Runtime bootstrap for driving the generators/agent outside the FastAPI server.

The harness runs inside a container built from the application image, so it reuses
the application's own settings, throttle, catalog provider and provider clients.
This module replicates the small amount of server-lifespan setup the generators
need (throttle init) and provides preflight checks + a redaction-safe config
summary for the report header.
"""
from __future__ import annotations

import urllib.request
import uuid
from typing import Any


def init_runtime() -> Any:
    """Initialise settings + the OpenAI throttle exactly as the server lifespan does."""
    from workshops_ai.settings import get_settings

    settings = get_settings()
    if settings.openai_throttle.enabled:
        try:
            from workshops_ai.llm.openai_throttle import init_throttle

            init_throttle(
                rpm=settings.openai_throttle.rpm,
                tpm=settings.openai_throttle.tpm,
                max_concurrent=settings.openai_throttle.max_concurrent,
                per_request_limits=settings.wizard.openai_per_request_limits,
            )
        except Exception:
            # Throttle is best-effort; generators still run without it.
            pass
    return settings


def new_session_id(prefix: str = "harness") -> str:
    """A fresh session id. Required when the catalog provider runs in ``http`` mode."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def config_summary(settings) -> dict:
    """Redaction-safe snapshot of the active configuration for the report header.

    Never includes API keys, endpoints containing credentials, or tokens.
    """
    return {
        "provider": settings.llm.provider,
        "wizard_default_model": settings.wizard.default_model,
        "agent_model": settings.llm.model,
        "reasoning_effort": settings.wizard.reasoning_effort,
        "catalog_mode": settings.catalog.mode,
        "catalog_endpoint": settings.catalog.endpoint if settings.catalog.mode == "http" else None,
        "environment": settings.env,
        "throttle_enabled": settings.openai_throttle.enabled,
    }


def preflight_catalog(settings) -> dict:
    """Verify the *configured* catalog provider is reachable; raise a clear error otherwise.

    Honours whatever ``WORKSHOPS_AI_CATALOG__MODE`` is set to — ``file`` needs no
    network; ``http`` is probed with a lightweight request to the sectors catalog.
    """
    if settings.catalog.mode != "http":
        return {"mode": settings.catalog.mode, "ok": True, "detail": "file mode (bundled CSVs)"}

    endpoint = (settings.catalog.endpoint or "").rstrip("/")
    if not endpoint:
        raise RuntimeError(
            "Catalog mode is 'http' but WORKSHOPS_AI_CATALOG__ENDPOINT is empty."
        )
    url = f"{endpoint}/catalogs/sectors"
    try:
        with urllib.request.urlopen(url, timeout=settings.catalog.timeout_sec or 10) as resp:  # noqa: S310
            return {
                "mode": "http",
                "endpoint": endpoint,
                "ok": resp.status == 200,
                "http_status": resp.status,
            }
    except Exception as exc:
        raise RuntimeError(
            f"Configured catalog provider is unreachable at {url}: {exc}. "
            f"Run the harness inside the stack "
            f"(docker compose ... run --rm harness) so the catalog service name resolves, "
            f"or set WORKSHOPS_AI_CATALOG__MODE=file."
        ) from exc


def catalog_control_names(settings, session_id: str) -> set[str] | None:
    """Best-effort set of control names from the configured catalog, for verifying
    that generated controls were selected from (and enriched by) the catalog.

    Returns ``None`` if the catalog cannot be read the same way the app reads it,
    so the caller can skip the membership check rather than fail spuriously.
    """
    # 1) Preferred: the application's own catalog provider.
    try:
        from workshops_ai.wizard.catalog_provider import get_catalog_provider

        provider = get_catalog_provider()
        for call in (
            lambda: provider.get_catalog_names("controls", session_id=session_id),
            lambda: provider.get_catalog_names("controls"),
        ):
            try:
                names = call()
                if names:
                    return {str(n) for n in names}
            except TypeError:
                continue
            except Exception:
                break
    except Exception:
        pass

    # 2) Fallback: read the configured catalog endpoint directly (http mode).
    if settings.catalog.mode == "http" and settings.catalog.endpoint:
        try:
            import json

            url = settings.catalog.endpoint.rstrip("/") + "/catalogs/controls"
            with urllib.request.urlopen(url, timeout=settings.catalog.timeout_sec or 10) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
            items = data.get("controls", data) if isinstance(data, dict) else data
            names = {
                str(it.get("name"))
                for it in items
                if isinstance(it, dict) and it.get("name")
            }
            return names or None
        except Exception:
            return None
    return None
