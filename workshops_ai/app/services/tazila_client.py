# File: workshops_ai/app/services/tazila_client.py
# -----------------------------------------------------------------------------
# Purpose
# Submit ChangeSet payloads to Tazila in three modes:
#   - "file": write JSON to outbox (non-blocking via to_thread in async path)
#   - "http": POST JSON to HTTP endpoint (httpx / AsyncClient, with retry/backoff)
#   - "disabled": no-op (returns a structured reason)
#
# Notes
# - Configuration is centralized via workshops_ai.settings (see Tazila-related fields).
# - Accepts either a Pydantic model (with .model_dump()) OR a plain dict payload.
# - HTTP auth token supports both raw token and already-prefixed "Bearer ...".
# - File names are defensive and cross-platform friendly.
# - Asynchronous variant is recommended from WS/agent code paths.
# - Designed for send-now Impact/Asset agents; supports idempotency keys end-to-end.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import httpx
from pydantic import BaseModel

from workshops_ai.settings import get_settings

__all__ = ["submit_changeset", "asubmit_changeset"]


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------

def _payload(cs: BaseModel | Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalize an incoming ChangeSet to a JSON-serializable dict.
    Supports:
      - Pydantic v2 BaseModel (uses .model_dump(mode="json"))
      - dict-like objects (Mapping)
    """
    if isinstance(cs, BaseModel):
        return cs.model_dump(mode="json")
    if isinstance(cs, Mapping):
        return dict(cs)
    raise TypeError("ChangeSet must be a Pydantic BaseModel or a Mapping[str, Any].")


def _safe_component(s: str | None) -> str:
    # Keep alnum, dot, dash, underscore; replace others by dash; trim length
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    return s[:64] or "x"


def _headers(idempotency_key: Optional[str] = None) -> Dict[str, str]:
    cfg = get_settings().tazila
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    tok = (getattr(cfg, "auth_token", None) or "").strip()
    if tok:
        if not tok.lower().startswith("bearer "):
            tok = f"Bearer {tok}"
        headers["Authorization"] = tok
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def _file_name(payload: Dict[str, Any]) -> str:
    """
    Build a defensive filename: <ts>_<kind>_<org>_<ikey>.json
    Falls back to placeholders if fields are missing.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kind = _safe_component(str(payload.get("kind") or "changeset"))
    ikey = _safe_component(str(payload.get("idempotency_key") or "nokey")[:24])
    org = _safe_component(str(payload.get("org_id") or "noorg"))
    return f"{ts}_{kind}_{org}_{ikey}.json"


def _write_file_sync(payload: Dict[str, Any]) -> str:
    cfg = get_settings().tazila
    outdir = Path(getattr(cfg, "outbox_path", "data/outbox"))
    outdir.mkdir(parents=True, exist_ok=True)
    fpath = outdir / _file_name(payload)
    with fpath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(fpath)


async def _write_file_async(payload: Dict[str, Any]) -> str:
    # Offload to a thread to avoid blocking the event loop
    return await asyncio.to_thread(_write_file_sync, payload)


def _should_retry(exc_or_status: Any) -> bool:
    """
    Retry pouze pro:
      - síťové chyby/timeouty,
      - HTTP 5xx,
      - HTTP 429.
    Na 4xx (krom 429) NERETRYUJEME.
    """
    # přímo status kód (int)
    if isinstance(exc_or_status, int):
        return exc_or_status >= 500 or exc_or_status == 429

    # httpx HTTPStatusError (má response se status kódem)
    if isinstance(exc_or_status, httpx.HTTPStatusError):
        sc = getattr(exc_or_status, "response", None)
        sc = getattr(sc, "status_code", None)
        return (sc is not None) and (sc >= 500 or sc == 429)

    # ostatní request/network chyby (timeout, connect, atd.) → retry
    if isinstance(exc_or_status, (httpx.TimeoutException, httpx.RequestError)):
        return True

    return False


async def _http_post_async(payload: Dict[str, Any], *, idempotency_key: Optional[str]) -> Dict[str, Any]:
    cfg = get_settings().tazila
    endpoint = getattr(cfg, "endpoint", None)
    timeout = getattr(cfg, "timeout_sec", 30)
    max_retries = int(getattr(cfg, "max_retries", 3))
    if not endpoint:
        return {"mode": "http", "status": None, "response": "No endpoint configured", "idempotency_key": idempotency_key}

    async with httpx.AsyncClient(timeout=timeout) as client:
        attempt = 0
        last_error: str | None = None
        while True:
            attempt += 1
            try:
                r = await client.post(endpoint, json=payload, headers=_headers(idempotency_key))
                if _should_retry(r.status_code) and attempt <= max_retries:
                    # Exponential backoff with jitter
                    await asyncio.sleep((2 ** (attempt - 1)) + random.random())
                    continue
                r.raise_for_status()
                try:
                    body = r.json()
                except Exception:
                    body = r.text
                return {
                    "mode": "http",
                    "status": r.status_code,
                    "response": body,
                    "idempotency_key": idempotency_key,
                }
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_error = str(e)
                if attempt <= max_retries and _should_retry(e):
                    await asyncio.sleep((2 ** (attempt - 1)) + random.random())
                    continue
                return {
                    "mode": "http",
                    "status": None,
                    "error": last_error,
                    "idempotency_key": idempotency_key,
                }


def _http_post_sync(payload: Dict[str, Any], *, idempotency_key: Optional[str]) -> Dict[str, Any]:
    cfg = get_settings().tazila
    endpoint = getattr(cfg, "endpoint", None)
    timeout = getattr(cfg, "timeout_sec", 30)
    max_retries = int(getattr(cfg, "max_retries", 3))
    if not endpoint:
        return {"mode": "http", "status": None, "response": "No endpoint configured", "idempotency_key": idempotency_key}

    with httpx.Client(timeout=timeout) as client:
        attempt = 0
        last_error: str | None = None
        while True:
            attempt += 1
            try:
                r = client.post(endpoint, json=payload, headers=_headers(idempotency_key))
                if _should_retry(r.status_code) and attempt <= max_retries:
                    # Simple sync backoff
                    import time
                    time.sleep((2 ** (attempt - 1)) + random.random())
                    continue
                r.raise_for_status()
                try:
                    body = r.json()
                except Exception:
                    body = r.text
                return {
                    "mode": "http",
                    "status": r.status_code,
                    "response": body,
                    "idempotency_key": idempotency_key,
                }
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_error = str(e)
                if attempt <= max_retries and _should_retry(e):
                    import time
                    time.sleep((2 ** (attempt - 1)) + random.random())
                    continue
                return {
                    "mode": "http",
                    "status": None,
                    "error": last_error,
                    "idempotency_key": idempotency_key,
                }


# -----------------------------------------------------------------------------
# Synchronous API (backward compatible)
# -----------------------------------------------------------------------------

def submit_changeset(
    cs: BaseModel | Mapping[str, Any],
    *,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronous submit suitable for CLI/scripts.
    NOTE: In async code paths (e.g., WS/agents), prefer `asubmit_changeset` to avoid blocking.
    """
    cfg = get_settings().tazila
    payload = _payload(cs)
    # propagate idempotency (header + body + filename)
    if idempotency_key and "idempotency_key" not in payload:
        payload["idempotency_key"] = idempotency_key

    mode = str(getattr(cfg, "mode", "file")).lower()
    if mode == "file":
        fpath = _write_file_sync(payload)
        return {"mode": "file", "path": fpath, "idempotency_key": payload.get("idempotency_key")}

    if mode == "http":
        result = _http_post_sync(payload, idempotency_key=idempotency_key or payload.get("idempotency_key"))
        return result

    return {
        "mode": "disabled",
        "reason": "TAZILA mode is disabled or misconfigured",
        "idempotency_key": payload.get("idempotency_key") or idempotency_key,
    }


# -----------------------------------------------------------------------------
# Asynchronous API (non-blocking; recommended for WS / async handlers)
# -----------------------------------------------------------------------------

async def asubmit_changeset(
    cs: BaseModel | Mapping[str, Any],
    *,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Asynchronous submit:
      - file mode: writes JSON via asyncio.to_thread (non-blocking)
      - http mode: posts via httpx.AsyncClient with retry/backoff (non-blocking)
      - disabled: returns {"mode":"disabled"}
    """
    cfg = get_settings().tazila
    payload = _payload(cs)
    # propagate idempotency (header + body + filename)
    if idempotency_key and "idempotency_key" not in payload:
        payload["idempotency_key"] = idempotency_key

    mode = str(getattr(cfg, "mode", "file")).lower()
    if mode == "file":
        fpath = await _write_file_async(payload)
        return {"mode": "file", "path": fpath, "idempotency_key": payload.get("idempotency_key")}

    if mode == "http":
        result = await _http_post_async(payload, idempotency_key=idempotency_key or payload.get("idempotency_key"))
        return result

    return {
        "mode": "disabled",
        "reason": "TAZILA mode is disabled or misconfigured",
        "idempotency_key": payload.get("idempotency_key") or idempotency_key,
    }
