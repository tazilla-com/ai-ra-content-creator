# File: workshops_ai/graph/checkpointer.py
# -----------------------------------------------------------------------------
# Purpose
# LangGraph checkpointer factory with graceful fallbacks.
# Order of preference:
#   1) Redis (when WORKSHOPS_AI_REDIS_URL is provided) — optional TTL support
#   2) SQLite (file path configurable)
#   3) In-memory (no persistence; last resort)
#
# Configuration (.env / settings)
#   WORKSHOPS_AI_CHECKPOINTER=redis|sqlite|memory   # optional explicit preference
#   WORKSHOPS_AI_REDIS_URL=redis://localhost:6379/0
#   WORKSHOPS_AI_CHECKPOINT_NS=lg:ckpt              # Redis key namespace
#   WORKSHOPS_AI_CHECKPOINT_TTL_SEC=86400           # optional TTL (if supported)
#   WORKSHOPS_AI_SQLITE_CKPT_PATH=/app/workshops_ai/data/state/checkpoints.db
#
# Notes
# - Robust across langgraph versions (from_url/from_conn_string, TTL optional).
# - Ensures the returned object is a *saver instance*, not a context manager.
# - If a context manager form is encountered, it is unwrapped and closed on exit.
# -----------------------------------------------------------------------------

from __future__ import annotations

import atexit
import inspect
import logging
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["make_checkpointer", "get_checkpointer"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _parse_int(val: Optional[str]) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        logger.warning("Invalid int for WORKSHOPS_AI_CHECKPOINT_TTL_SEC: %s", val)
        return None


def _collect_cfg(settings: Any | None) -> dict[str, Any]:
    """
    Build a config dict from (preferred) settings object with environment fallbacks.
    The settings object is expected to expose attributes with intuitive names; we also
    accept direct env-variables for portability.
    """
    env = os.getenv
    cfg: dict[str, Any] = {}

    # Pull from settings if available
    if settings is not None:
        cfg["checkpointer"] = getattr(settings, "checkpointer", None) or getattr(
            settings, "WORKSHOPS_AI_CHECKPOINTER", None
        )
        cfg["redis_url"] = (
            getattr(settings, "redis_url", None)
            or getattr(getattr(settings, "redis", None), "url", None)
            or getattr(settings, "WORKSHOPS_AI_REDIS_URL", None)
            or getattr(settings, "WORKSHOPS_AI_REDIS__URL", None)
        )
        cfg["checkpoint_ns"] = getattr(settings, "checkpoint_ns", None) or getattr(
            settings, "WORKSHOPS_AI_CHECKPOINT_NS", None
        )
        cfg["checkpoint_ttl_sec"] = (
            getattr(settings, "checkpoint_ttl_sec", None)
            or getattr(settings, "WORKSHOPS_AI_CHECKPOINT_TTL_SEC", None)
        )
        cfg["sqlite_ckpt_path"] = getattr(settings, "sqlite_ckpt_path", None) or getattr(
            settings, "WORKSHOPS_AI_SQLITE_CKPT_PATH", None
        )

    # Env fallbacks
    cfg["checkpointer"] = (cfg.get("checkpointer") or env("WORKSHOPS_AI_CHECKPOINTER", "")).strip() or None
    cfg["redis_url"] = (
        (cfg.get("redis_url") or env("WORKSHOPS_AI_REDIS_URL", "") or env("WORKSHOPS_AI_REDIS__URL", ""))
        .strip()
        or None
    )
    cfg["checkpoint_ns"] = (cfg.get("checkpoint_ns") or env("WORKSHOPS_AI_CHECKPOINT_NS", "lg:ckpt")).strip()
    cfg["checkpoint_ttl_sec"] = (
        cfg.get("checkpoint_ttl_sec") if isinstance(cfg.get("checkpoint_ttl_sec"), int)
        else _parse_int(env("WORKSHOPS_AI_CHECKPOINT_TTL_SEC", "").strip())
    )
    cfg["sqlite_ckpt_path"] = (
        cfg.get("sqlite_ckpt_path") or env("WORKSHOPS_AI_SQLITE_CKPT_PATH", "/app/workshops_ai/data/state/checkpoints.db")
    )

    return cfg


def _is_valid_saver(saver: Any) -> bool:
    """
    Minimal capability check across langgraph versions.
    Accept either the new API (get_next_version) or the async/sync tuple APIs.
    """
    if saver is None:
        return False
    has_next_ver = hasattr(saver, "get_next_version")
    has_async = any(hasattr(saver, name) for name in ("aget_tuple", "aput", "apatch"))
    has_sync = any(hasattr(saver, name) for name in ("get_tuple", "put", "patch"))
    return bool(has_next_ver or has_async or has_sync)


def _unwrap_context_manager(obj: Any) -> Any:
    """
    If the provided object is a context manager, enter it immediately and
    register its exit handler for interpreter shutdown. Return the inner saver.
    """
    if hasattr(obj, "__enter__") and hasattr(obj, "__exit__"):
        cm = obj
        saver = cm.__enter__()  # type: ignore[misc]
        def _finalize():
            try:
                cm.__exit__(None, None, None)  # type: ignore[misc]
            except Exception:
                pass
        atexit.register(_finalize)
        return saver

    if isinstance(obj, AbstractContextManager):
        saver = obj.__enter__()
        def _finalize():
            try:
                obj.__exit__(None, None, None)
            except Exception:
                pass
        atexit.register(_finalize)
        return saver

    return obj


# -----------------------------------------------------------------------------
# Concrete saver builders
# -----------------------------------------------------------------------------

def _make_redis_saver(cfg: dict[str, Any]):
    url = cfg.get("redis_url")
    if not url:
        raise RuntimeError("Redis URL not provided")

    try:
        from langgraph.checkpoint.redis import RedisSaver  # type: ignore
    except Exception as e:
        raise RuntimeError(f"RedisSaver import failed: {e}") from e

    ns = cfg.get("checkpoint_ns") or "lg:ckpt"
    ttl = cfg.get("checkpoint_ttl_sec")

    def _supported_kwargs(factory: Any) -> set[str]:
        try:
            sig = inspect.signature(factory)
            return set(sig.parameters.keys())
        except Exception:
            return set()

    def _build_kwargs(supported: set[str]) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        if "namespace" in supported:
            kw["namespace"] = ns
        if "ttl" in supported and ttl:
            kw["ttl"] = int(ttl)
        return kw

    # 1) Try from_url
    if hasattr(RedisSaver, "from_url"):
        try:
            kwargs = _build_kwargs(_supported_kwargs(getattr(RedisSaver, "from_url")))
            saver = RedisSaver.from_url(url, **kwargs)  # type: ignore[call-arg]
            saver = _unwrap_context_manager(saver)
            if not _is_valid_saver(saver):
                raise RuntimeError("RedisSaver(from_url) returned an invalid saver")
            logger.info(
                "LangGraph checkpointer: Redis via from_url (ns=%s, ttl=%s)",
                kwargs.get("namespace"), kwargs.get("ttl")
            )
            return saver
        except Exception as e:
            logger.warning("RedisSaver.from_url failed: %s", e)

    # 2) Try from_conn_string
    if hasattr(RedisSaver, "from_conn_string"):
        try:
            kwargs = _build_kwargs(_supported_kwargs(getattr(RedisSaver, "from_conn_string")))
            saver = RedisSaver.from_conn_string(url, **kwargs)  # type: ignore[call-arg]
            saver = _unwrap_context_manager(saver)
            if not _is_valid_saver(saver):
                raise RuntimeError("RedisSaver(from_conn_string) returned an invalid saver")
            logger.info(
                "LangGraph checkpointer: Redis via from_conn_string (ns=%s, ttl=%s)",
                kwargs.get("namespace"), kwargs.get("ttl")
            )
            return saver
        except Exception as e:
            logger.warning("RedisSaver.from_conn_string failed: %s", e)

    # 3) Fallback: client=...
    try:
        from redis.asyncio import Redis as AsyncRedis  # type: ignore
    except Exception as e:
        raise RuntimeError(f"redis-py (async) not installed: {e}") from e

    try:
        client = AsyncRedis.from_url(url, decode_responses=False)
        try:
            saver = RedisSaver(client=client, namespace=ns)  # type: ignore[call-arg]
        except TypeError:
            saver = RedisSaver(client=client)  # type: ignore[call-arg]
        saver = _unwrap_context_manager(saver)
        if not _is_valid_saver(saver):
            raise RuntimeError("RedisSaver(client=...) returned an invalid saver")
        logger.info("LangGraph checkpointer: Redis via client (ns=%s)", ns)
        return saver
    except Exception as e:
        raise RuntimeError(f"RedisSaver(client=...) failed: {e}") from e


def _make_sqlite_saver(cfg: dict[str, Any]):
    """
    Construct a SQLite-backed saver. Ensures directory exists and is writable.
    Always returns a saver instance (never a context manager).
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
    except Exception as e:
        raise RuntimeError(f"SqliteSaver import failed: {e}") from e

    path = Path(cfg.get("sqlite_ckpt_path") or "/app/workshops_ai/data/state/checkpoints.db")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("mkdir for sqlite parent failed (%s): %s", path.parent, e)

    # Try modern helpers first, then basic ctor
    try_methods = []
    if hasattr(SqliteSaver, "from_conn_string"):
        try_methods.append(("from_conn_string", lambda: SqliteSaver.from_conn_string(f"sqlite:///{path}")))
    if hasattr(SqliteSaver, "from_url"):
        try_methods.append(("from_url", lambda: SqliteSaver.from_url(f"sqlite:///{path}")))  # type: ignore[attr-defined]
    try_methods.append(("ctor", lambda: SqliteSaver(str(path))))  # type: ignore[call-arg]

    last_err: Optional[Exception] = None
    for name, builder in try_methods:
        try:
            saver = builder()
            saver = _unwrap_context_manager(saver)
            if not _is_valid_saver(saver):
                raise RuntimeError(f"SqliteSaver ({name}) returned an invalid saver")
            logger.info("LangGraph checkpointer: SQLite (%s via %s)", str(path), name)
            return saver
        except Exception as e:
            last_err = e

    raise RuntimeError(f"SqliteSaver construction failed: {last_err}")  # pragma: no cover


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def make_checkpointer(prefer: Optional[str] = None, settings: Any | None = None) -> Any:
    """
    Build a LangGraph checkpointer saver instance.

    Resolution:
      1) Respect 'prefer' if provided.
      2) Else use WORKSHOPS_AI_CHECKPOINTER from settings/env.
      3) Else pick: Redis if url present → SQLite → Memory.
    """
    cfg = _collect_cfg(settings)

    target = (prefer or cfg.get("checkpointer") or "").strip().lower()
    if not target:
        target = "redis" if cfg.get("redis_url") else "sqlite"

    if target == "redis":
        try:
            return _make_redis_saver(cfg)
        except Exception as e:
            logger.warning("Falling back from Redis checkpointer: %s", e)
            try:
                return _make_sqlite_saver(cfg)
            except Exception as e2:
                logger.warning("Falling back from SQLite checkpointer: %s", e2)

    elif target == "sqlite":
        try:
            return _make_sqlite_saver(cfg)
        except Exception as e:
            logger.warning("Falling back from SQLite checkpointer: %s", e)
            if cfg.get("redis_url"):
                try:
                    return _make_redis_saver(cfg)
                except Exception as e2:
                    logger.warning("Falling back from Redis checkpointer: %s", e2)

    # Final fallback: in-memory (no persistence)
    try:
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore
        saver = MemorySaver()
        logger.info("LangGraph checkpointer: Memory (no persistence)")
        return saver
    except Exception as e:
        # Extremely unlikely — make the failure explicit
        raise RuntimeError(f"MemorySaver import failed: {e}") from e


def get_checkpointer(settings: Any | None = None) -> Any:
    """
    Compatibility wrapper used by callers that pass a settings object.
    Equivalent to make_checkpointer(prefer=None, settings=settings).
    """
    return make_checkpointer(settings=settings)
