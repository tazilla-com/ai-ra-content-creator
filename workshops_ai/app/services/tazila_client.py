# File: workshops_ai/app/services/tazila_client.py
# Purpose: Submit ChangeSet payloads to Tazila in three modes:
#          - "file": write JSON to outbox (non-blocking via to_thread in async path)
#          - "http": POST JSON to HTTP endpoint (httpx / AsyncClient)
#          - "disabled": no-op (returns a structured reason)
#
# Notes:
# - Configuration is centralized via workshops_ai.settings.
# - HTTP auth token supports both raw token and already-prefixed "Bearer ...".
# - File names are defensive and cross-platform friendly.
# - Asynchronous variant is recommended from WS/agent code paths.

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import httpx

from workshops_ai.settings import get_settings
from workshops_ai.models.changeset import ChangeSet

__all__ = ["submit_changeset", "asubmit_changeset"]


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------
def _payload(cs: ChangeSet) -> Dict[str, Any]:
    return cs.model_dump(mode="json")


def _safe_component(s: str | None) -> str:
    # Keep alnum, dot, dash, underscore; replace others by dash; trim length
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    return s[:64] or "x"


def _headers() -> Dict[str, str]:
    cfg = get_settings().tazila
    headers = {"Content-Type": "application/json"}
    if cfg.auth_token:
        tok = cfg.auth_token.strip()
        if not tok.lower().startswith("bearer "):
            tok = f"Bearer {tok}"
        headers["Authorization"] = tok
    return headers


def _write_file_sync(payload: Dict[str, Any], cs: ChangeSet) -> str:
    cfg = get_settings().tazila
    outdir = Path(cfg.outbox_path)
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ikey = _safe_component(str(getattr(cs, "idempotency_key", "nokey"))[:12])
    org = _safe_component(str(getattr(cs, "org_id", "noorg")))
    fname = f"{ts}_{org}_{ikey}.json"

    fpath = outdir / fname
    with fpath.open("w", encoding="utf-8") as f:
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
    cfg = get_settings().tazila
    payload = _payload(cs)

    if cfg.mode == "file":
        fpath = _write_file_sync(payload, cs)
        return {"mode": "file", "path": fpath, "idempotency_key": cs.idempotency_key}

    if cfg.mode == "http" and cfg.endpoint:
        with httpx.Client(timeout=cfg.timeout_sec) as client:
            r = client.post(cfg.endpoint, json=payload, headers=_headers())
            r.raise_for_status()
            try:
                body = r.json()
            except Exception:
                body = r.text
        return {"mode": "http", "status": r.status_code, "response": body, "idempotency_key": cs.idempotency_key}

    return {
        "mode": "disabled",
        "reason": "TAZILA mode is disabled or misconfigured",
        "idempotency_key": cs.idempotency_key,
    }


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
    cfg = get_settings().tazila
    payload = _payload(cs)

    if cfg.mode == "file":
        fpath = await _write_file_async(payload, cs)
        return {"mode": "file", "path": fpath, "idempotency_key": cs.idempotency_key}

    if cfg.mode == "http" and cfg.endpoint:
        async with httpx.AsyncClient(timeout=cfg.timeout_sec) as client:
            r = await client.post(cfg.endpoint, json=payload, headers=_headers())
            r.raise_for_status()
            try:
                body = r.json()
            except Exception:
                body = r.text
        return {"mode": "http", "status": r.status_code, "response": body, "idempotency_key": cs.idempotency_key}

    return {
        "mode": "disabled",
        "reason": "TAZILA mode is disabled or misconfigured",
        "idempotency_key": cs.idempotency_key,
    }
