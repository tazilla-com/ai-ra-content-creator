# File: workshops_ai/app/services/tazila_client.py
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import httpx
from pydantic_settings import BaseSettings

from workshops_ai.models.changeset import ChangeSet


class Settings(BaseSettings):
    # Mode: "file" | "http" | "disabled"
    TAZILA_MODE: str = "file"
    # Local outbox path for JSON patches (file mode)
    TAZILA_OUTBOX_PATH: str = "data/outbox"
    # HTTP endpoint (http mode)
    TAZILA_ENDPOINT: str | None = None
    # Optional Bearer token (http mode)
    TAZILA_AUTH_TOKEN: str | None = None
    # Request timeout seconds (http mode)
    TAZILA_TIMEOUT: int = 30


settings = Settings()  # reads from env if present

__all__ = ["submit_changeset", "asubmit_changeset"]


def _payload(cs: ChangeSet) -> Dict[str, Any]:
    return cs.model_dump(mode="json")


def _build_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.TAZILA_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {settings.TAZILA_AUTH_TOKEN}"
    return headers


def _safe_component(s: str) -> str:
    # Keep alnum, dash, underscore; replace others by dash; strip/shorten to stay friendly cross-platform
    import re
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    return s[:64] or "x"


def _write_file_sync(payload: Dict[str, Any], cs: ChangeSet) -> str:
    outdir = Path(settings.TAZILA_OUTBOX_PATH)
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Be defensive in case idempotency_key/org_id are missing
    ikey = _safe_component(str(getattr(cs, "idempotency_key", "nokey"))[:12])
    org = _safe_component(str(getattr(cs, "org_id", "noorg")))
    fname = f"{ts}_{org}_{ikey}.json"

    fpath = outdir / fname
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(fpath)


async def _write_file_async(payload: Dict[str, Any], cs: ChangeSet) -> str:
    # Offload to a thread to avoid blocking the event loop
    return await asyncio.to_thread(_write_file_sync, payload, cs)


# --------------------------------------------------------------------------------------
# Synchronous API (backward compatible)
# --------------------------------------------------------------------------------------
def submit_changeset(cs: ChangeSet) -> Dict[str, Any]:
    """
    Synchronous submit suitable for CLI/scripts.
    NOTE: In async code paths (e.g., WS/agents), prefer `asubmit_changeset` to avoid blocking.
    """
    payload = _payload(cs)

    if settings.TAZILA_MODE == "file":
        fpath = _write_file_sync(payload, cs)
        return {"mode": "file", "path": fpath, "idempotency_key": cs.idempotency_key}

    if settings.TAZILA_MODE == "http" and settings.TAZILA_ENDPOINT:
        with httpx.Client(timeout=settings.TAZILA_TIMEOUT) as client:
            r = client.post(settings.TAZILA_ENDPOINT, json=payload, headers=_build_headers())
            r.raise_for_status()
            try:
                body = r.json()
            except Exception:
                body = r.text
        return {"mode": "http", "status": r.status_code, "response": body, "idempotency_key": cs.idempotency_key}

    return {"mode": "disabled", "reason": "TAZILA_MODE is disabled or misconfigured"}


# --------------------------------------------------------------------------------------
# Asynchronous API (non-blocking; recommended for WS / async handlers)
# --------------------------------------------------------------------------------------
async def asubmit_changeset(cs: ChangeSet) -> Dict[str, Any]:
    """
    Asynchronous submit:
      - file mode: writes JSON via asyncio.to_thread (non-blocking)
      - http mode: posts via httpx.AsyncClient (non-blocking)
      - disabled: returns {"mode":"disabled"}
    """
    payload = _payload(cs)

    if settings.TAZILA_MODE == "file":
        fpath = await _write_file_async(payload, cs)
        return {"mode": "file", "path": fpath, "idempotency_key": cs.idempotency_key}

    if settings.TAZILA_MODE == "http" and settings.TAZILA_ENDPOINT:
        async with httpx.AsyncClient(timeout=settings.TAZILA_TIMEOUT) as client:
            r = await client.post(settings.TAZILA_ENDPOINT, json=payload, headers=_build_headers())
            r.raise_for_status()
            try:
                body = r.json()
            except Exception:
                body = r.text
        return {"mode": "http", "status": r.status_code, "response": body, "idempotency_key": cs.idempotency_key}

    return {"mode": "disabled", "reason": "TAZILA_MODE is disabled or misconfigured"}
