# workshops_ai/app/services/redis_pool.py
from typing import Optional
from redis.asyncio import Redis
from workshops_ai.settings import get_settings

_r: Optional[Redis] = None

async def get_redis() -> Optional[Redis]:
    global _r
    if _r and not _r.closed:
        return _r
    url = (get_settings().redis.url or "").strip()
    if not url:
        return None
    _r = Redis.from_url(url, decode_responses=True)
    try:
        await _r.ping()
    except Exception:
        pass
    return _r

async def close_redis():
    global _r
    if _r:
        await _r.aclose()
        _r = None
