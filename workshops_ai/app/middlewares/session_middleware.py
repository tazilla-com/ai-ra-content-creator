# workshops_ai/app/middlewares/session_middleware.py
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
import json, secrets
from workshops_ai.app.services.redis_pool import get_redis

COOKIE = "wsai.sid"
TTL = 8 * 60 * 60  # 8 h
PROBE_PATHS = {"/livez", "/readyz", "/health"}  # no session for health probes: each one wrote a new key

class SessionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, secure_cookie: bool):
        super().__init__(app)
        self.secure_cookie = secure_cookie

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PROBE_PATHS:
            return await call_next(request)
        sid = request.cookies.get(COOKIE) or secrets.token_urlsafe(18)
        r = await get_redis()
        data = {}
        if r:
            raw = await r.get(f"sess:{sid}")
            data = json.loads(raw) if raw else {}
        request.state.session_id = sid
        request.state.session = data

        resp: Response = await call_next(request)

        if r:
            await r.set(f"sess:{sid}", json.dumps(request.state.session or {}), ex=TTL)
        # nastav cookie jen když nebyla
        if COOKIE not in request.cookies:
            resp.set_cookie(COOKIE, sid, httponly=True, samesite="lax", secure=self.secure_cookie)
        return resp
