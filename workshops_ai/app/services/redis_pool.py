# File: workshops_ai/app/services/redis_pool.py
from __future__ import annotations

from typing import Optional
import inspect
from redis.asyncio import Redis
from workshops_ai.settings import get_settings

_r: Optional[Redis] = None  # cached client


async def _safe_close(r: Redis) -> None:
    """
    Close Redis client across redis-py 4.x/5.x differences.
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


async def get_redis() -> Optional[Redis]:
    """
    Return a healthy Redis client instance.
    - Reuses cached client if PING succeeds
    - Recreates on failure
    - Returns None if no URL configured
    """
    global _r

    url = (get_settings().redis.url or "").strip()
    if not url:
        return None

    if _r is not None:
        try:
            await _r.ping()
            return _r
        except Exception:
            # stale/broken client → close & recreate
            try:
                await _safe_close(_r)
            finally:
                _r = None

    # (re)create client
    _r = Redis.from_url(url, decode_responses=True, encoding="utf-8")

    # best-effort warmup; even if it fails, return client (caller will handle)
    try:
        await _r.ping()
    except Exception:
        pass

    return _r


async def close_redis() -> None:
    """Explicit shutdown hook for app shutdown."""
    global _r
    if _r is not None:
        try:
            await _safe_close(_r)
        finally:
            _r = None
