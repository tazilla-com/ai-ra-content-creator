# File: workshops_ai/app/middlewares/rate_limit.py
from __future__ import annotations

import time
from typing import Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, status, HTTPException

from workshops_ai.settings import get_settings
from workshops_ai.app.services.redis_pool import get_redis

HEALTH_PATHS = {"/livez", "/readyz"}  # never rate-limit health probes


def _get_request_id(request: Request) -> str:
    """
    Best-effort correlation id used in error envelopes.
    Prefers state (if a RequestID middleware populated it), then header, then fallback.
    """
    rid = getattr(request.state, "request_id", None) or request.headers.get("x-request-id")
    return rid or "n/a"


async def _allow(key: str, per_minute: int) -> Tuple[bool, int]:
    """
    Fixed 60s window counter using Redis INCR + EXPIRE.
    Returns (allowed, seconds_until_reset).
    - If Redis is unavailable or errors, allow the request (fail-open).
    """
    try:
        r = await get_redis()
    except Exception:
        return True, 60

    if not r:
        return True, 60  # fail-open (no Redis configured)

    try:
        now = int(time.time())
        window = now // 60
        rk = f"rl:{key}:{window}"

        n = await r.incr(rk)
        if n == 1:
            # keep key a bit longer than a minute to be safe
            await r.expire(rk, 120)

        # seconds left until the current minute window resets
        reset_in = 60 - (now % 60)
        return n <= per_minute, reset_in
    except Exception:
        # Any Redis error → fail-open
        return True, 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    HTTP rate limiting middleware.
    - Reads config from settings().ratelimit (enabled, per_minute).
    - Keys by path + session_id (if present) else client IP.
    - On limit → raises HTTPException(429, ...) so the global handler
      builds the unified error envelope and attaches the request_id.
    - Adds Retry-After header with seconds until reset.
    """

    def __init__(self, app):
        super().__init__(app)
        s = getattr(get_settings(), "ratelimit", None)
        self.enabled = bool(getattr(s, "enabled", True)) if s is not None else True
        self.per_min = int(getattr(s, "per_minute", 60)) if s is not None else 60

    async def dispatch(self, request: Request, call_next):
        # Bypass health endpoints completely
        if request.url.path in HEALTH_PATHS:
            return await call_next(request)

        if not self.enabled:
            return await call_next(request)

        # Build a stable key: path + session or IP
        sid: Optional[str] = getattr(request.state, "session_id", None)
        ip = request.client.host if request.client else "ip"
        key = f"http:{request.url.path}:{sid or ip}"

        allowed, retry_after = await _allow(key, self.per_min)
        if not allowed:
            rid = _get_request_id(request)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "rate_limited",
                    "message": "Too many requests",
                    "request_id": rid,
                    "details": {
                        "path": str(request.url.path),
                        "limit_per_minute": self.per_min,
                    },
                },
                headers={"Retry-After": str(max(1, retry_after))},
            )

        return await call_next(request)


# Helper for WebSocket gating (mirrors HTTP keying)
async def allow_ws(event: str, sid: str | None, per_minute: int) -> bool:
    key = f"ws:{event}:{sid or 'anon'}"
    allowed, _ = await _allow(key, per_minute)
    return allowed
