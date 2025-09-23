# File: workshops_ai/app/middlewares/rate_limit.py
from __future__ import annotations

import time
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, status, HTTPException

from workshops_ai.settings import get_settings
from workshops_ai.app.services.redis_pool import get_redis


def _get_request_id(request: Request) -> str:
    """
    Best-effort correlation id used in error envelopes.
    Prefers state (if a RequestID middleware populated it), then header, then fallback.
    """
    rid = getattr(request.state, "request_id", None) or request.headers.get("x-request-id")
    return rid or "n/a"


async def _allow(key: str, per_minute: int) -> tuple[bool, int]:
    """
    Fixed 60s window counter using Redis INCR + EXPIRE.
    Returns (allowed, seconds_until_reset).
    - If Redis is unavailable, allow the request (fail-open).
    """
    r = await get_redis()
    if not r:
        return True, 60  # fail-open, behave generously

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


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    HTTP rate limiting middleware.
    - Reads config from settings().ratelimit (enabled, per_minute).
    - Keys by path + session_id (if present) else client IP.
    - On limit → raises HTTPException(429, ...) so global error handler
      builds the unified error envelope and attaches the request_id.
    - Adds Retry-After header with seconds until reset.
    """

    def __init__(self, app):
        super().__init__(app)
        s = getattr(get_settings(), "ratelimit", None)
        self.enabled = bool(getattr(s, "enabled", True)) if s is not None else True
        self.per_min = int(getattr(s, "per_minute", 60)) if s is not None else 60

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        # Build a stable key: path + session or IP
        sid: Optional[str] = getattr(request.state, "session_id", None)
        ip = request.client.host if request.client else "ip"
        key = f"http:{request.url.path}:{sid or ip}"

        allowed, retry_after = await _allow(key, self.per_min)
        if not allowed:
            rid = _get_request_id(request)
            # Raise HTTPException → picked by global handler for a unified JSON envelope.
            # Attach Retry-After header for good API hygiene.
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


# Helper pro WebSocket
async def allow_ws(event: str, sid: str | None, per_minute: int) -> bool:
    """
    WS guard used v routeru; sjednoceno s HTTP klíčováním.
    """
    key = f"ws:{event}:{sid or 'anon'}"
    allowed, _ = await _allow(key, per_minute)
    return allowed
