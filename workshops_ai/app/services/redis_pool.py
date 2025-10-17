# File: workshops_ai/app/services/redis_pool.py
from __future__ import annotations

from typing import Optional
import inspect
from redis.asyncio import Redis as AsyncRedis
from redis import Redis as SyncRedis
from workshops_ai.settings import get_settings

_r_async: Optional[AsyncRedis] = None  # cached async client
_r_sync: Optional[SyncRedis] = None    # cached sync client


async def _safe_close_async(r: AsyncRedis) -> None:
    """
    Close async Redis client across redis-py 4.x/5.x differences.
    Tries aclose()/close(); awaits if needed.
    """
    try:
        close = getattr(r, "aclose", None)
        if callable(close):
            await close()
            return
    except Exception:
        pass
    try:
        close2 = getattr(r, "close", None)
        if callable(close2):
            res = close2()
            if inspect.isawaitable(res):
                await res  # type: ignore[func-returns-value]
    except Exception:
        pass


def _safe_close_sync(r: SyncRedis) -> None:
    """Close sync Redis client (no-op safe)."""
    try:
        close = getattr(r, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


async def get_redis() -> Optional[AsyncRedis]:
    """
    Return a healthy ASYNC Redis client instance.
    - Reuses cached client if PING succeeds
    - Recreates on failure
    - Returns None if no URL configured
    """
    global _r_async

    url = (get_settings().redis.url or "").strip()
    if not url:
        return None

    if _r_async is not None:
        try:
            await _r_async.ping()
            return _r_async
        except Exception:
            try:
                await _safe_close_async(_r_async)
            finally:
                _r_async = None

    _r_async = AsyncRedis.from_url(url, decode_responses=True, encoding="utf-8")

    try:
        await _r_async.ping()
    except Exception:
        pass

    return _r_async


def get_redis_sync() -> Optional[SyncRedis]:
    """
    Return a healthy SYNC Redis client instance.
    - Reuses cached client if PING succeeds
    - Recreates on failure
    - Returns None if no URL configured
    """
    global _r_sync

    url = (get_settings().redis.url or "").strip()
    if not url:
        return None

    if _r_sync is not None:
        try:
            _r_sync.ping()
            return _r_sync
        except Exception:
            try:
                _safe_close_sync(_r_sync)
            finally:
                _r_sync = None

    _r_sync = SyncRedis.from_url(url, decode_responses=True, encoding="utf-8")

    try:
        _r_sync.ping()
    except Exception:
        pass

    return _r_sync


async def close_redis() -> None:
    """Explicit shutdown hook for app shutdown (async client)."""
    global _r_async
    if _r_async is not None:
        try:
            await _safe_close_async(_r_async)
        finally:
            _r_async = None


def close_redis_sync() -> None:
    """Explicit shutdown hook for app shutdown (sync client)."""
    global _r_sync
    if _r_sync is not None:
        try:
            _safe_close_sync(_r_sync)
        finally:
            _r_sync = None
