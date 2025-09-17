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
# Environment:
#   WORKSHOPS_AI_TELEMETRY_ENABLED   (default: "true")
#   WORKSHOPS_AI_SERVICE_NAME        (default: "workshops_ai")
#   WORKSHOPS_AI_SERVICE_VERSION     (default: "0.1.0")
#   WORKSHOPS_AI_ENVIRONMENT         (default: "dev")
#   WORKSHOPS_AI_LOG_LEVEL           (default: "INFO")  # DEBUG|INFO|WARN|ERROR
#   WORKSHOPS_AI_LOG_TO_FILE         (default: "false")
#   WORKSHOPS_AI_LOG_FILE_PATH       (default: "./telemetry.log")
#   WORKSHOPS_AI_MAX_PAYLOAD_LEN     (default: "4096")  # characters for payload snapshot
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
import socket
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from typing import Any, Iterable, Mapping, MutableMapping, Optional

try:
    # Optional helpers for safe dumps/redaction
    from workshops_ai.tools.guard_validate_payload import safe_model_dump, redact_for_log  # type: ignore
except Exception:  # pragma: no cover
    def safe_model_dump(obj: Any, *, max_len: int = 2048) -> Any:  # type: ignore
        try:
            s = repr(obj)
            return (s[:max_len] + "…") if len(s) > max_len else obj
        except Exception:
            return "<unserializable>"

    def redact_for_log(obj: Any, *, sensitive_keys: Iterable[str] = ("password", "token", "secret", "authorization", "api_key", "apikey", "bearer")) -> Any:  # type: ignore
        return obj  # minimal fallback


# --------------------------------------------------------------------------------------
# Config
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

def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _level_from_env(val: str | None) -> int:
    mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARN,
        "WARNING": logging.WARN,
        "ERROR": logging.ERROR,
    }
    return mapping.get((val or "").strip().upper(), logging.INFO)

_CFG = _Config(
    enabled=_env_bool("WORKSHOPS_AI_TELEMETRY_ENABLED", True),
    service=os.getenv("WORKSHOPS_AI_SERVICE_NAME", "workshops_ai"),
    version=os.getenv("WORKSHOPS_AI_SERVICE_VERSION", "0.1.0"),
    environment=os.getenv("WORKSHOPS_AI_ENVIRONMENT", "dev"),
    level=_level_from_env(os.getenv("WORKSHOPS_AI_LOG_LEVEL")),
    log_to_file=_env_bool("WORKSHOPS_AI_LOG_TO_FILE", False),
    log_file_path=os.getenv("WORKSHOPS_AI_LOG_FILE_PATH", "./telemetry.log"),
    max_payload_len=_env_int("WORKSHOPS_AI_MAX_PAYLOAD_LEN", 4096),
)


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
    return _level_from_env(str(level or "INFO"))

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


__all__ = ["telemetry_log"]
