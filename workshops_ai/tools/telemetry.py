# File: workshops_ai/tools/telemetry.py
# Purpose:
#   Lightweight, dependency-free telemetry/audit logger for Workshop Agents.
#   - Structured JSON logs to STDOUT (optionally also to file).
#   - Non-blocking by default (QueueHandler + background listener).
#   - Redacts sensitive fields and caps payload size.
#   - Adds audit context (user_id, org_id, batch_tags, request_id, source, env, service).
#
# Usage:
#   from workshops_ai.tools.telemetry import telemetry_log
#   telemetry_log("chat.request", {"message_preview": "list assets"}, user_id="u-1", org_id="acme")
#
# Notes:
#   - Never raise exceptions; failures are swallowed (best-effort telemetry).
#   - request_id is auto-fetched if available via main app; can also be passed in payload.
#   - This module aims to be import-safe (no heavy side effects).

from __future__ import annotations

import json
import logging
import os
import queue
import re
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from typing import Any, Iterable, Mapping, Optional

try:
    # Centralized settings (preferred source of truth)
    from workshops_ai.settings import get_settings  # type: ignore
    _SETTINGS = get_settings()
except Exception:  # pragma: no cover
    _SETTINGS = None  # graceful fallback below

# --------------------------------------------------------------------------------------
# Redaction utilities (dependency-free)
# --------------------------------------------------------------------------------------

# Common sensitive key names to redact (case-insensitive exact field name match)
_SENSITIVE_KEYS = {
    "password", "pass", "pwd",
    "token", "access_token", "refresh_token",
    "secret", "client_secret", "secret_key",
    "authorization", "authorization_header", "auth",
    "api_key", "apikey", "x-api-key",
    "bearer", "session", "cookie",
}

# PII patterns (emails, phone numbers, credit card-like, UUIDs if considered sensitive)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?[0-9][0-9\s\-()]{6,}")
_CC_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")  # very loose, just to avoid logging numbers
# Keep UUIDs visible by default (often harmless in logs), set to True to mask them:
_MASK_UUIDS = False
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)

def redact_pii(s: str | None) -> str | None:
    """Redact common PII patterns in arbitrary strings (best-effort)."""
    try:
        if not s:
            return s
        out = s
        out = _EMAIL_RE.sub("[redacted-email]", out)
        out = _PHONE_RE.sub("[redacted-phone]", out)
        out = _CC_RE.sub("[redacted-cc]", out)
        if _MASK_UUIDS:
            out = _UUID_RE.sub("[redacted-uuid]", out)
        return out
    except Exception:  # pragma: no cover
        return s

def _redact_value(val: Any) -> Any:
    """Apply PII redaction to leaf values while preserving types."""
    try:
        if isinstance(val, str):
            return redact_pii(val)
        # numbers/bools/None unchanged
        if isinstance(val, (int, float, bool)) or val is None:
            return val
        # For other simple leafs, attempt string redaction but keep original if conversion fails
        if isinstance(val, (bytes, bytearray)):
            try:
                return redact_pii(val.decode("utf-8", errors="ignore"))
            except Exception:
                return "[redacted-bytes]"
        return val
    except Exception:  # pragma: no cover
        return val

def redact_for_log(obj: Any, *, sensitive_keys: Iterable[str] = _SENSITIVE_KEYS) -> Any:
    """
    Recursively redact values for keys considered sensitive (case-insensitive),
    and scrub PII patterns from strings. Keeps structure (dict/list/tuple).
    """
    try:
        sens = {str(k).lower() for k in sensitive_keys}

        def _walk(x: Any) -> Any:
            if isinstance(x, Mapping):
                red: dict[str, Any] = {}
                for k, v in x.items():
                    k_str = str(k)
                    if k_str.lower() in sens:
                        red[k_str] = "[redacted]"
                    else:
                        red[k_str] = _walk(v)
                return red
            if isinstance(x, list):
                return [_walk(v) for v in x]
            if isinstance(x, tuple):
                return tuple(_walk(v) for v in x)
            # leaf
            return _redact_value(x)

        return _walk(obj)
    except Exception:  # pragma: no cover
        return obj

def safe_model_dump(obj: Any, *, max_len: int = 2048) -> Any:
    """
    Best-effort serialization guard: returns JSON-serializable shape or
    a pruned textual repr not exceeding max_len.
    """
    try:
        # If already JSON-ish (dict/list/tuple/scalar), return as-is
        if isinstance(obj, (Mapping, list, tuple, str, int, float, bool)) or obj is None:
            # If it's a huge string, trim it
            if isinstance(obj, str) and len(obj) > max_len:
                return obj[:max_len] + "…"
            return obj
        # pydantic v2 model?
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump(mode="json")
            except Exception:
                pass
        # Try generic 'dict()'
        try:
            return dict(obj)  # type: ignore[arg-type]
        except Exception:
            pass
        # Fallback to repr (trimmed)
        s = repr(obj)
        return (s[:max_len] + "…") if len(s) > max_len else s
    except Exception:  # pragma: no cover
        return "<unserializable>"

# --------------------------------------------------------------------------------------
# Config (from settings; small env-free fallback for import-time safety)
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Config:
    enabled: bool
    service: str
    version: str
    environment: str
    level: int
    log_to_file: bool
    log_file_path: str
    max_payload_len: int

def _level_from_str(val: str | None) -> int:
    mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARN,
        "WARNING": logging.WARN,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return mapping.get((val or "").strip().upper(), logging.INFO)

def _build_cfg_from_settings() -> _Config:
    if _SETTINGS is None:
        # Minimal sane fallback if settings import failed early
        return _Config(
            enabled=True,
            service="workshops_ai",
            version="0.1.0",
            environment="dev",
            level=logging.INFO,
            log_to_file=False,
            log_file_path="./telemetry.log",
            max_payload_len=16_000,
        )
    tel = getattr(_SETTINGS, "telemetry", None)
    return _Config(
        enabled=bool(getattr(tel, "enabled", True)),
        service=str(getattr(_SETTINGS, "service_name", "workshops_ai")),
        version=str(getattr(_SETTINGS, "version", "0.1.0")),
        environment=str(getattr(_SETTINGS, "env", "dev")),
        level=_level_from_str(getattr(tel, "level", "INFO")),
        log_to_file=bool(getattr(tel, "log_to_file", False)),
        log_file_path=str(getattr(tel, "log_file_path", "./telemetry.log")),
        max_payload_len=int(getattr(tel, "max_payload_len", 16_000)),
    )

_CFG = _build_cfg_from_settings()

# --------------------------------------------------------------------------------------
# Logger (non-blocking)
# --------------------------------------------------------------------------------------

_logger_name = "workshops_ai.telemetry"
_logger = logging.getLogger(_logger_name)
_logger.setLevel(_CFG.level)

# Create a queue-based logging pipeline once
if not _logger.handlers:
    _log_queue: "queue.Queue[logging.LogRecord]" = queue.Queue(-1)
    qh = QueueHandler(_log_queue)
    qh.setLevel(_CFG.level)
    _logger.addHandler(qh)

    # Sink handlers (stdout + optional rotating file)
    sinks: list[logging.Handler] = []

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(_CFG.level)
    # Pass-through formatter (we log pre-rendered JSON strings as message)
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    sinks.append(stdout_handler)

    if _CFG.log_to_file:
        file_handler = RotatingFileHandler(_CFG.log_file_path, maxBytes=10_000_000, backupCount=3, encoding="utf-8")
        file_handler.setLevel(_CFG.level)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        sinks.append(file_handler)

    _listener = QueueListener(_log_queue, *sinks, respect_handler_level=True)
    _listener.daemon = True  # do not block process exit
    try:
        _listener.start()
    except Exception:
        # If QueueListener fails (rare), fall back to direct stdout handler
        for h in list(_logger.handlers):
            _logger.removeHandler(h)
        _logger.addHandler(stdout_handler)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

_hostname = socket.gethostname()
_pid = os.getpid()

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _try_get_request_id() -> Optional[str]:
    """
    Try import of request-id accessor from the main app. Returns None if unavailable.
    """
    try:
        from workshops_ai.app.main import get_request_id  # type: ignore
        rid = get_request_id()  # may return "n/a"
        return str(rid) if rid and rid != "n/a" else None
    except Exception:
        return None

def _level_to_int(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    return _level_from_str(str(level or "INFO"))

def _coerce_mapping(obj: Any) -> Mapping[str, Any] | None:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj  # type: ignore[return-value]
    # Best-effort: convert simple objects
    try:
        return dict(obj)  # type: ignore[arg-type]
    except Exception:
        return {"value": safe_model_dump(obj, max_len=_CFG.max_payload_len)}

# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def telemetry_log(
    event: str,
    payload: Mapping[str, Any] | None = None,
    *,
    level: str | int = "INFO",
    user_id: str | None = None,
    org_id: str | None = None,
    batch_tags: Iterable[str] | None = None,
    source: str = "AI",
) -> None:
    """
    Emit a structured JSON telemetry record. Best-effort; never raises.

    Args:
        event: Short event code (e.g., "chat.request", "manager.route_in").
        payload: Arbitrary mapping (will be redacted & size-capped).
        level: "DEBUG"|"INFO"|"WARN"|"ERROR" or logging level int.
        user_id: Optional user identifier.
        org_id: Optional tenant/org identifier.
        batch_tags: Optional iterable of tag strings (e.g., ["AI","batch-123"]).
        source: Logical source (default "AI").

    Behavior:
        - Adds ts, service, version, env, host, pid, request_id.
        - Redacts sensitive fields in payload, caps snapshot length.
        - Respects global enable flag; becomes a no-op if disabled.
    """
    if not _CFG.enabled:
        return
    try:
        lvl = _level_to_int(level)

        # Base envelope
        record: dict[str, Any] = {
            "ts": _iso_utc_now(),
            "level": logging.getLevelName(lvl),
            "event": str(event),
            "source": str(source),
            "service": _CFG.service,
            "version": _CFG.version,
            "environment": _CFG.environment,
            "host": _hostname,
            "pid": _pid,
        }

        rid = _try_get_request_id()
        if rid:
            record["request_id"] = rid
        if user_id:
            record["user_id"] = str(user_id)
        if org_id:
            record["org_id"] = str(org_id)
        if batch_tags:
            # Normalize and deduplicate while preserving order
            seen: set[str] = set()
            tags_norm: list[str] = []
            for t in batch_tags:
                s = str(t).strip()
                if s and s not in seen:
                    seen.add(s)
                    tags_norm.append(s)
            record["batch_tags"] = tags_norm

        # Payload (redacted, size-capped)
        if payload is not None:
            shaped = _coerce_mapping(payload) or {}
            shaped = redact_for_log(shaped)
            snapshot = safe_model_dump(shaped, max_len=_CFG.max_payload_len)
            record["payload"] = snapshot

        # Serialize once; keep formatter simple
        msg = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        _logger.log(lvl, msg)
    except Exception:
        # Best-effort: never raise from telemetry
        pass

__all__ = ["telemetry_log", "redact_pii", "redact_for_log", "safe_model_dump"]
