# File: workshops_ai/graph/checkpointer.py
"""
Redis-only LangGraph Checkpointer (robust, fail-fast)
====================================================

Purpose
-------
Build a Redis-backed LangGraph checkpointer and **fail fast** if Redis is not
available or misconfigured. No SQLite or in-memory fallbacks (by request).

Behavior
--------
- Reads configuration from provided `settings` and/or environment variables:
    WORKSHOPS_AI_CHECKPOINTER        → must be "redis"
    WORKSHOPS_AI_REDIS_URL           → e.g., "redis://redis:6379/0"
    WORKSHOPS_AI_CHECKPOINT_NS       → key namespace for LangGraph (default: "lg:ckpt")
                                      (NOTE: this namespace is *not* passed to the saver;
                                       it is intended for RunnableConfig.configurable.checkpoint_ns)
    WORKSHOPS_AI_CHECKPOINT_TTL_SEC  → optional default TTL in seconds (int)

- Supports common RedisSaver factories across langgraph-checkpoint-redis versions:
    RedisSaver.from_url(url, ttl=?)
    RedisSaver.from_conn_string(url, ttl=?)
  We try TTL as:
    1) {"default_ttl": <int>}
    2) no TTL (persistent keys)
  We intentionally do not pass "ttl_config" variants that some versions reject.

- Guarantees an **async** surface even if the underlying saver is sync-only:
    aget_tuple / aput / apatch
    aput_writes / apatch_writes (compat shims if missing)

Important detail (prevents recursion)
-------------------------------------
Some code (e.g., a factory) probes for context-manager methods using hasattr().
If our adapter *exposes* inner __enter__/__exit__ via __getattr__, that probe may
try to enter the adapter itself (and/or loop through delegated lookups). To avoid
this:
  - The adapter’s __getattr__ explicitly *does not* expose "__enter__" or "__exit__".
  - We prefer to unwrap context managers ourselves (class-level check) before
    wrapping into the async adapter.

Notes
-----
- No explicit network PING at construction (keeps startup synchronous). If you need
  connectivity checks, do it in your app startup.
- Construction failures raise immediately (fail-fast by design).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import atexit
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = ["make_checkpointer", "get_checkpointer"]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _parse_int(val: Optional[str]) -> Optional[int]:
    """Parse string to int or return None; warn on invalid or negative values."""
    if val is None or val == "":
        return None
    try:
        parsed = int(val)
        if parsed < 0:
            logger.warning("Negative TTL ignored: %s", val)
            return None
        return parsed
    except ValueError:
        logger.warning("Invalid int for WORKSHOPS_AI_CHECKPOINT_TTL_SEC: %s", val)
        return None


def _maybe_enter_context(obj: Any) -> Any:
    """
    Enter a *real* context manager without accidentally unwrapping our async adapter.

    Why:
      - Our adapter uses __getattr__ for delegation. Instance-level checks like
        hasattr(obj, "__enter__") could delegate into the wrapped saver and cause
        the caller to try entering the adapter instead of the inner object.

    Strategy:
      1) If `obj` is our adapter, return as-is (never unwrap).
      2) Detect context-manager capability on the **class**, not the instance.
      3) If it is a CM, call __enter__() and register best-effort atexit cleanup.
    """
    # 1) Never unwrap our async adapter
    if obj.__class__.__name__ == "_AsyncAdapter":
        return obj

    # 2) Check on the class to avoid triggering __getattr__ on the instance
    cls = type(obj)
    has_enter = callable(getattr(cls, "__enter__", None))
    has_exit = callable(getattr(cls, "__exit__", None))
    if has_enter and has_exit:
        # 3) Enter and arrange best-effort cleanup
        entered = obj.__enter__()  # bound to the object; no delegation
        try:
            atexit.register(lambda: obj.__exit__(None, None, None))
        except Exception:
            pass
        return entered

    return obj


# -----------------------------------------------------------------------------
# Universal async adapter
# Ensures we always expose async methods even if the saver is sync-only or uses
# different method names across LangGraph versions.
# -----------------------------------------------------------------------------
class _AsyncAdapter:
    """
    Wrap any saver and guarantee an async surface:

      - aget_tuple / aput / apatch
      - aput_writes / apatch_writes (compat shims)

    Selection strategy:
      1) Prefer a native async method if present.
      2) If missing or raising NotImplementedError → call a sync variant in a thread.

    Recursion prevention:
      We **do not** expose "__enter__" / "__exit__" via __getattr__. This prevents
      external hasattr(obj, "__enter__") checks from mistaking the adapter for a
      context manager and trying to enter it.
    """

    def __init__(self, inner: Any):
        self._inner = inner

    def _get(self, *names: str) -> Optional[Callable]:
        for n in names:
            m = getattr(self._inner, n, None)
            if callable(m):
                return m
        return None

    async def _run_in_thread(self, fn: Callable, *a, **k):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*a, **k))

    async def _prefer_async_else_sync(
        self,
        async_names: tuple[str, ...],
        sync_names: tuple[str, ...],
        *a,
        **k,
    ):
        # 1) Try async candidates
        for name in async_names:
            fn = getattr(self._inner, name, None)
            if not callable(fn):
                continue
            try:
                try:
                    res = fn(*a, **k)
                except TypeError:
                    # Older signatures may not accept kwargs
                    res = fn(*a)
                if inspect.isawaitable(res):
                    try:
                        return await res
                    except NotImplementedError:
                        # Async surface exists but not implemented → try next/sync
                        continue
                    except TypeError:
                        # Argument mismatch surfaced late → retry without kwargs
                        res2 = fn(*a)
                        return await res2 if inspect.isawaitable(res2) else res2
                else:
                    # Some "async" APIs may actually be sync; return value directly
                    return res
            except NotImplementedError:
                continue

        # 2) Try sync candidates (off the event loop)
        for name in sync_names:
            fn = getattr(self._inner, name, None)
            if not callable(fn):
                continue
            try:
                try:
                    return await self._run_in_thread(fn, *a, **k)
                except TypeError:
                    return await self._run_in_thread(fn, *a)
            except NotImplementedError:
                continue

        # 3) Nothing usable
        raise NotImplementedError(
            f"No suitable method on {type(self._inner).__name__}: "
            f"tried async={async_names} then sync={sync_names}"
        )

    # --- LangGraph async API
    async def aget_tuple(self, *a, **k):
        try:
            return await self._prefer_async_else_sync(("aget_tuple", "aget"), ("get_tuple", "get"), *a, **k)
        except (TypeError, NotImplementedError):
            # Some versions expose only aget/get (no tuple). Normalize to (value, writes).
            res = await self._prefer_async_else_sync(("aget",), ("get",), *a, **k)
            return res if isinstance(res, tuple) else (res, None)

    async def aput(self, *a, **k):
        return await self._prefer_async_else_sync(("aput",), ("put",), *a, **k)

    async def apatch(self, *a, **k):
        return await self._prefer_async_else_sync(("apatch",), ("patch", "update"), *a, **k)

    async def aput_writes(self, *a, **k):
        try:
            return await self._prefer_async_else_sync(("aput_writes",), ("put_writes", "put"), *a, **k)
        except NotImplementedError:
            # Older savers may not support writes-specific methods
            return await self.aput(*a, **k)

    async def apatch_writes(self, *a, **k):
        try:
            return await self._prefer_async_else_sync(("apatch_writes",), ("patch_writes", "patch", "update"), *a, **k)
        except NotImplementedError:
            return await self.apatch(*a, **k)

    # --- Transparent passthrough for anything else, but do NOT expose CM hooks
    def __getattr__(self, name: str):
        if name in ("__enter__", "__exit__"):
            # Hide CM methods so hasattr(obj, "__enter__") returns False on adapter
            raise AttributeError(name)
        return getattr(self._inner, name)

    # Friendlier introspection (optional): hide noisy sync names in dir()
    def __dir__(self):
        names = set(dir(self._inner))
        for n in ("get", "get_tuple", "put", "patch", "update", "put_writes", "patch_writes",
                  "__enter__", "__exit__"):
            names.discard(n)
        names.update({
            "aget_tuple", "aput", "apatch", "aput_writes", "apatch_writes",
            "_get", "_run_in_thread", "_prefer_async_else_sync", "_inner",
        })
        return sorted(names)


def _ensure_async(saver: Any) -> Any:
    """Wrap a saver with the async adapter to provide a consistent async API."""
    return _AsyncAdapter(saver)


# -----------------------------------------------------------------------------
# Configuration collection (settings → env)
# -----------------------------------------------------------------------------
def _collect_cfg(settings: Any | None) -> dict[str, Any]:
    """
    Collect and normalize configuration from `settings` and environment variables.

    Supports both single-underscore and double-underscore env variants:
      WORKSHOPS_AI_REDIS_URL  and  WORKSHOPS_AI_REDIS__URL
    """
    env = os.getenv
    cfg: dict[str, Any] = {}

    # Prefer settings object, if provided
    if settings is not None:
        cfg["checkpointer"] = getattr(settings, "checkpointer", None) or getattr(
            settings, "WORKSHOPS_AI_CHECKPOINTER", None
        )
        cfg["redis_url"] = (
            getattr(settings, "redis_url", None)
            or getattr(getattr(settings, "redis", None), "url", None)  # nested settings.redis.url
            or getattr(settings, "WORKSHOPS_AI_REDIS_URL", None)
            or getattr(settings, "WORKSHOPS_AI_REDIS__URL", None)
        )
        cfg["checkpoint_ns"] = getattr(settings, "checkpoint_ns", None) or getattr(
            settings, "WORKSHOPS_AI_CHECKPOINT_NS", None
        )
        ttl_raw = getattr(settings, "checkpoint_ttl_sec", None) or getattr(
            settings, "WORKSHOPS_AI_CHECKPOINT_TTL_SEC", None
        )
        cfg["checkpoint_ttl_sec"] = ttl_raw

    # Env fallbacks
    cfg["checkpointer"] = (cfg.get("checkpointer") or env("WORKSHOPS_AI_CHECKPOINTER", "redis")).strip()
    cfg["redis_url"] = (
        (cfg.get("redis_url") or env("WORKSHOPS_AI_REDIS_URL", "") or env("WORKSHOPS_AI_REDIS__URL", ""))
        .strip()
        or None
    )
    cfg["checkpoint_ns"] = (cfg.get("checkpoint_ns") or env("WORKSHOPS_AI_CHECKPOINT_NS", "lg:ckpt")).strip()

    # TTL normalization
    ttl = cfg.get("checkpoint_ttl_sec")
    if isinstance(ttl, str):
        ttl = _parse_int(ttl)
    if ttl is None:
        ttl = _parse_int(env("WORKSHOPS_AI_CHECKPOINT_TTL_SEC", "").strip())
    cfg["checkpoint_ttl_sec"] = ttl  # int | None

    return cfg


# -----------------------------------------------------------------------------
# Redis builders (handle multiple LangGraph signatures; no non-Redis fallbacks)
# -----------------------------------------------------------------------------
def _attempt_builder(builder: Callable[..., Any], url: str, ttl_seconds: int) -> Any:
    """
    Build a Redis-backed LangGraph saver using the given builder
    (e.g., RedisSaver.from_url / from_conn_string) with a TTL configuration
    compatible across langgraph-checkpoint-redis releases.
    """
    attempts: list[dict[str, Any]] = []

    # Prefer dict-based TTL config (works on current releases)
    if ttl_seconds and ttl_seconds > 0:
        attempts.append({"ttl": {"default_ttl": ttl_seconds}})

    # No-TTL fallback (persistent keys)
    attempts.append({})

    last_err: Optional[Exception] = None
    for kw in attempts:
        try:
            # Builder may return a raw saver or a context manager
            saver_or_cm = builder(url, **kw)

            # If a context manager was returned, enter it immediately
            saver = _maybe_enter_context(saver_or_cm)

            # Optional setup() to create indices etc. (idempotent)
            setup = getattr(saver, "setup", None)
            if callable(setup):
                res = setup()
                if inspect.isawaitable(res):
                    logger.warning("RedisSaver.setup() returned awaitable; ignoring in sync context.")

            # Wrap to guarantee async surface
            wrapped = _ensure_async(saver)

            # Sanity check: catch incompatible int-typed TTL attributes early
            ttl_attr = getattr(saver, "ttl", None)
            if isinstance(ttl_attr, int):
                raise RuntimeError(
                    "Incompatible TTL type (int) detected on RedisSaver; expected a dict-like TTL config."
                )

            logger.info(
                "LangGraph checkpointer ready: %s ttl=%s",
                getattr(builder, "__name__", "builder"),
                kw or None,
            )
            return wrapped
        except Exception as e:
            last_err = e
            logger.debug(
                "RedisSaver attempt failed via %s with kwargs=%r, err=%r",
                getattr(builder, "__name__", "builder"),
                kw,
                e,
            )

    raise RuntimeError(
        f"RedisSaver build failed for {getattr(builder, '__name__', 'builder')}: {last_err}"
    )


def _build_redis_saver(cfg: dict[str, Any]) -> Any:
    """
    Construct a Redis-backed LangGraph checkpointer.

    Order of attempts:
      1) Full RedisSaver via from_url → from_conn_string
      2) ShallowRedisSaver (still Redis, no JSON/FT modules) via from_url → from_conn_string
      3) If all fail → raise (fail-fast, no SQLite/memory)
    """
    url = cfg.get("redis_url")
    if not url:
        raise RuntimeError("Redis URL not provided (WORKSHOPS_AI_REDIS_URL is empty).")

    ttl_seconds = int(cfg.get("checkpoint_ttl_sec") or 0)

    # Import primary and shallow Redis savers
    try:
        from langgraph.checkpoint.redis import RedisSaver  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "langgraph-checkpoint-redis is required but not importable. "
            "Install it and ensure compatible LangGraph versions."
        ) from e

    try:
        from langgraph.checkpoint.redis.shallow import ShallowRedisSaver  # type: ignore
    except Exception:
        ShallowRedisSaver = None  # type: ignore

    # 1) Full RedisSaver — prefer from_url, then from_conn_string
    if hasattr(RedisSaver, "from_url"):
        try:
            return _attempt_builder(RedisSaver.from_url, url, ttl_seconds)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning("RedisSaver.from_url failed: %s", e)
    if hasattr(RedisSaver, "from_conn_string"):
        try:
            return _attempt_builder(RedisSaver.from_conn_string, url, ttl_seconds)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning("RedisSaver.from_conn_string failed: %s", e)

    # 2) ShallowRedisSaver (still Redis) — if available
    if ShallowRedisSaver is not None:
        if hasattr(ShallowRedisSaver, "from_url"):
            try:
                return _attempt_builder(ShallowRedisSaver.from_url, url, ttl_seconds)  # type: ignore[arg-type]
            except Exception as e:
                logger.warning("ShallowRedisSaver.from_url failed: %s", e)
        if hasattr(ShallowRedisSaver, "from_conn_string"):
            try:
                return _attempt_builder(ShallowRedisSaver.from_conn_string, url, ttl_seconds)  # type: ignore[arg-type]
            except Exception as e:
                logger.warning("ShallowRedisSaver.from_conn_string failed: %s", e)

    # 3) Give up — Redis-only policy, so we do not degrade to memory/SQLite
    raise RuntimeError(
        "Unable to construct a Redis-backed LangGraph checkpointer. "
        "Ensure Redis is reachable and the installed langgraph / langgraph-checkpoint-redis "
        "versions support your environment (consider redis/redis-stack if modules are required)."
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def make_checkpointer(settings: Any | None = None) -> Any:
    """
    Build and return a Redis-backed LangGraph checkpointer.

    Rules:
      - WORKSHOPS_AI_CHECKPOINTER must be "redis" (or settings.checkpointer == "redis").
      - WORKSHOPS_AI_REDIS_URL must be set (e.g., "redis://redis:6379/0").
      - No fallback to SQLite or Memory. If we cannot construct RedisSaver, raise.

    Additionally:
      - If the saver (unexpectedly) is a context manager, we **enter and unwrap**
        it here so LangGraph receives the actual saver object (with required methods),
        and then we wrap it with our async adapter.
    """
    cfg = _collect_cfg(settings)

    target = (cfg.get("checkpointer") or "redis").strip().lower()
    if target != "redis":
        raise RuntimeError(
            f"checkpointer must be 'redis' (no fallbacks). Got: {target!r}. "
            "Set WORKSHOPS_AI_CHECKPOINTER=redis."
        )

    saver = _build_redis_saver(cfg)

    # Defensive: if the saver itself is (unexpectedly) a context manager, unwrap it.
    saver = _maybe_enter_context(saver)

    return saver


def get_checkpointer(settings: Any | None = None) -> Any:
    """Alias used by factories/manager to build the Redis-backed checkpointer."""
    return make_checkpointer(settings=settings)
