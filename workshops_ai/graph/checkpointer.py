# File: workshops_ai/graph/checkpointer.py
# -----------------------------------------------------------------------------
# Purpose
# LangGraph checkpointer factory with robust async normalization.
# Order of preference:
#   1) Redis (WORKSHOPS_AI_REDIS_URL)
#   2) SQLite (file)
#   3) Memory (last resort)
#
# Guarantees:
# - Always returns an object that supports aget_tuple / aput / apatch.
# - If the underlying saver only has sync methods (get / get_tuple / put / patch / update),
#   we adapt them with a threadpool so LangGraph’s async runtime can use them.
# - If async methods exist but raise NotImplementedError (stubs), we fall back to sync.
# - Compatible with LangGraph builds that call aput_writes / apatch_writes.
# -----------------------------------------------------------------------------

from __future__ import annotations

import atexit
import asyncio
import inspect
import logging
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Optional, Callable

logger = logging.getLogger(__name__)

__all__ = ["make_checkpointer", "get_checkpointer"]


# -----------------------------------------------------------------------------
# Config helpers
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
    env = os.getenv
    cfg: dict[str, Any] = {}

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


# -----------------------------------------------------------------------------
# Context manager unwrap
# -----------------------------------------------------------------------------

def _unwrap_context_manager(obj: Any) -> Any:
    if hasattr(obj, "__enter__") and hasattr(obj, "__exit__"):
        cm = obj  # type: ignore[assignment]
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
# Universal async adapter (with NotImplementedError fallback)
# -----------------------------------------------------------------------------

class _AsyncAdapter:
    """
    Wraps any saver and guarantees async surface (aget_tuple/aput/apatch).
    Prefers native async if present; otherwise runs sync variants in a thread.
    If a native async method exists but raises NotImplementedError (stub),
    we gracefully fall back to the sync method.

    Also provides compatibility shims for aput_writes / apatch_writes used by
    newer LangGraph versions; when not present, we fall back to aput/apatch.
    """

    def __init__(self, inner: Any):
        self._inner = inner

    # --------- helpers
    def _get_method(self, *names: str) -> Optional[Callable]:
        for n in names:
            m = getattr(self._inner, n, None)
            if callable(m):
                return m
        return None

    async def _run_in_thread(self, fn: Callable, *a, **k):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*a, **k))

    async def _run_async_or_fallback_sync(
        self,
        pref_async: tuple[str, ...],
        fallback_sync: tuple[str, ...],
        *a, **k
    ):
        # 1) Try async method (if present)
        am = self._get_method(*pref_async)
        if am:
            try:
                res = am(*a, **k)
                if inspect.isawaitable(res):
                    return await res
                return res
            except NotImplementedError:
                # Async surface exists but is a stub — fall back to sync.
                pass

        # 2) Fallback to sync in a thread
        sm = self._get_method(*fallback_sync)
        if sm:
            return await self._run_in_thread(sm, *a, **k)

        # 3) Nothing usable
        raise NotImplementedError(
            f"Neither async {pref_async} nor sync {fallback_sync} are implemented on {type(self._inner)}"
        )

    # --------- async API expected by LangGraph
    async def aget_tuple(self, *a, **k):
        return await self._run_async_or_fallback_sync(
            ("aget_tuple",),
            ("get_tuple", "get"),
            *a, **k
        )

    async def aput(self, *a, **k):
        return await self._run_async_or_fallback_sync(
            ("aput",),
            ("put",),
            *a, **k
        )

    async def apatch(self, *a, **k):
        return await self._run_async_or_fallback_sync(
            ("apatch",),
            ("patch", "update"),
            *a, **k
        )

    # --------- compatibility shims for newer LangGraph
    async def aput_writes(self, *a, **k):
        # Prefer real aput_writes / put_writes if available; otherwise fallback to aput/put
        try:
            return await self._run_async_or_fallback_sync(
                ("aput_writes",),
                ("put_writes", "put"),
                *a, **k
            )
        except NotImplementedError:
            return await self.aput(*a, **k)

    async def apatch_writes(self, *a, **k):
        # Prefer real apatch_writes / patch_writes if available; otherwise fallback to apatch/patch
        try:
            return await self._run_async_or_fallback_sync(
                ("apatch_writes",),
                ("patch_writes", "patch", "update"),
                *a, **k
            )
        except NotImplementedError:
            return await self.apatch(*a, **k)

    # --------- optional surfaces
    def get_next_version(self, *a, **k):
        if hasattr(self._inner, "get_next_version"):
            return getattr(self._inner, "get_next_version")(*a, **k)
        raise AttributeError("get_next_version not available on inner saver")

    def __getattr__(self, name: str):
        # Delegate everything else to the inner saver
        return getattr(self._inner, name)


def _ensure_async(saver: Any) -> Any:
    # Always wrap — the adapter prefers native async but will handle fallbacks.
    return _AsyncAdapter(saver)


# -----------------------------------------------------------------------------
# Redis / SQLite builders
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
    ttl_seconds = int(ttl) if ttl else 0
    # IMPORTANT: Always pass a dict for TTL to avoid "int is not iterable" in some builds.
    ttl_map = {"default_ttl": ttl_seconds} if ttl_seconds > 0 else None

    def _supported_kwargs(factory: Any) -> set[str]:
        try:
            return set(inspect.signature(factory).parameters.keys())
        except Exception:
            return set()

    def _build_kwargs(supported: set[str]) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        # namespace compatibility
        if "namespace" in supported:
            kw["namespace"] = ns
        elif "ns" in supported:
            kw["ns"] = ns
        # ttl mapping (prefer ttl_config, fallback to legacy "ttl" but still a dict)
        if ttl_map:
            if "ttl_config" in supported:
                kw["ttl_config"] = ttl_map
            elif "ttl" in supported:
                kw["ttl"] = ttl_map
        return kw

    # Prefer helper constructors first
    if hasattr(RedisSaver, "from_url"):
        try:
            kwargs = _build_kwargs(_supported_kwargs(RedisSaver.from_url))
            saver = RedisSaver.from_url(url, **kwargs)  # type: ignore[call-arg]
            saver = _unwrap_context_manager(saver)
            saver = _ensure_async(saver)
            logger.info(
                "LangGraph checkpointer: Redis via from_url (ns=%s, ttl_config=%s)",
                kwargs.get("namespace") or kwargs.get("ns"),
                kwargs.get("ttl_config") or kwargs.get("ttl"),
            )
            return saver
        except Exception as e:
            logger.warning("RedisSaver.from_url failed: %s", e)

    if hasattr(RedisSaver, "from_conn_string"):
        try:
            kwargs = _build_kwargs(_supported_kwargs(RedisSaver.from_conn_string))
            saver = RedisSaver.from_conn_string(url, **kwargs)  # type: ignore[call-arg]
            saver = _unwrap_context_manager(saver)
            saver = _ensure_async(saver)
            logger.info(
                "LangGraph checkpointer: Redis via from_conn_string (ns=%s, ttl_config=%s)",
                kwargs.get("namespace") or kwargs.get("ns"),
                kwargs.get("ttl_config") or kwargs.get("ttl"),
            )
            return saver
        except Exception as e:
            logger.warning("RedisSaver.from_conn_string failed: %s", e)

    # Fallback: construct with client (broad compatibility)
    try:
        from redis.asyncio import Redis as AsyncRedis  # type: ignore
    except Exception as e:
        raise RuntimeError(f"redis-py (async) not installed: {e}") from e

    try:
        client = AsyncRedis.from_url(url, decode_responses=False)

        ctor_supported = _supported_kwargs(RedisSaver)
        saver_kwargs: dict[str, Any] = {"client": client}

        if "namespace" in ctor_supported:
            saver_kwargs["namespace"] = ns
        elif "ns" in ctor_supported:
            saver_kwargs["ns"] = ns

        if ttl_map:
            if "ttl_config" in ctor_supported:
                saver_kwargs["ttl_config"] = ttl_map
            elif "ttl" in ctor_supported:
                saver_kwargs["ttl"] = ttl_map

        try:
            saver = RedisSaver(**saver_kwargs)  # type: ignore[call-arg]
        except TypeError:
            # Very old signature — no kwargs except client
            saver = RedisSaver(client=client)  # type: ignore[call-arg]

        saver = _unwrap_context_manager(saver)
        saver = _ensure_async(saver)
        logger.info("LangGraph checkpointer: Redis via client (ns=%s, ttl_config=%s)", ns, ttl_map)
        return saver
    except Exception as e:
        raise RuntimeError(f"RedisSaver(client=...) failed: {e}") from e


def _make_sqlite_saver(cfg: dict[str, Any]):
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
    except Exception as e:
        raise RuntimeError(f"SqliteSaver import failed: {e}") from e

    path = Path(cfg.get("sqlite_ckpt_path") or "/app/workshops_ai/data/state/checkpoints.db")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("mkdir for sqlite parent failed (%s): %s", path.parent, e)

    attempts: list[tuple[str, Callable[[], Any]]] = []
    if hasattr(SqliteSaver, "from_conn_string"):
        attempts.append(("from_conn_string", lambda: SqliteSaver.from_conn_string(f"sqlite:///{path}")))
    if hasattr(SqliteSaver, "from_url"):
        attempts.append(("from_url", lambda: SqliteSaver.from_url(f"sqlite:///{path}")))  # type: ignore[attr-defined]
    attempts.append(("ctor", lambda: SqliteSaver(str(path))))  # type: ignore[call-arg]

    last_err: Optional[Exception] = None
    for name, build in attempts:
        try:
            saver = build()
            saver = _unwrap_context_manager(saver)
            saver = _ensure_async(saver)
            logger.info("LangGraph checkpointer: SQLite (%s via %s)", str(path), name)
            return saver
        except Exception as e:
            last_err = e

    raise RuntimeError(f"SqliteSaver construction failed: {last_err}")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def make_checkpointer(prefer: Optional[str] = None, settings: Any | None = None) -> Any:
    """
    Build and return an async-capable LangGraph checkpointer saver.
    Resolution:
      1) Explicit 'prefer' if provided.
      2) Settings/env preference.
      3) Otherwise Redis if URL present; else SQLite; else Memory.
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

    # Final fallback: memory
    try:
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore
        saver = MemorySaver()
        saver = _ensure_async(saver)
        logger.info("LangGraph checkpointer: Memory (no persistence)")
        return saver
    except Exception as e:
        raise RuntimeError(f"MemorySaver import failed: {e}") from e


def get_checkpointer(settings: Any | None = None) -> Any:
    return make_checkpointer(settings=settings)
