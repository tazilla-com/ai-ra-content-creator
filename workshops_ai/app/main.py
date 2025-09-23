# File: workshops_ai/app/main.py
# Purpose:
#   FastAPI bootstrap: health endpoints + mounts /chat router + /ws router + /assets router + ChatUI.
#   - Strict Pydantic v2 validation (fail-closed at edges)
#   - Request ID + timing middleware
#   - Sessions (Redis) + simple rate-limit middleware
#   - CORS (configurable via settings.cors_origins)
#   - Consistent JSON error model (prefers orjson if available)
#   - LangChain/LangSmith tracing env export via centralized settings (no getenv reads)
#   - Safe fallbacks if /chat, /ws, or /assets router are not implemented
#   - Chat UI is provided by workshops_ai/chatui/router.py

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterable

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

try:
    from fastapi.responses import ORJSONResponse  # uses orjson if installed
    _DEFAULT_RESPONSE_CLASS = ORJSONResponse
except Exception:  # pragma: no cover
    _DEFAULT_RESPONSE_CLASS = JSONResponse

from pydantic import BaseModel, Field, ConfigDict

# Load .env early if present (non-fatal) — BaseSettings will also load it.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

# Centralized settings (single source of truth)
from workshops_ai.settings import get_settings
settings = get_settings()

# Optional auth dependency (API-key/JWT); if module not present we no-op auth.
try:
    from workshops_ai.app.deps.auth import require_auth as _require_auth  # noqa: F401
    _AUTH_DEP = [Depends(_require_auth)]
except Exception:  # pragma: no cover
    _AUTH_DEP = []

# Chat UI (demo)
from workshops_ai.chatui.router import router as chatui_router

# Sessions + Rate-limit middlewares & Redis lifecycle
from workshops_ai.app.middlewares.session_middleware import SessionMiddleware
from workshops_ai.app.middlewares.rate_limit import RateLimitMiddleware
from workshops_ai.app.services.redis_pool import get_redis, close_redis

# Tracing (export LANGCHAIN_* from settings.langchain, no getenv reads)
from workshops_ai.tools.tracing import enable_tracing_if_configured


# ------------------------------------------------------------------------------
# Logging (structured-ish)
# ------------------------------------------------------------------------------
logger = logging.getLogger("workshops_ai")
_handler = logging.StreamHandler(sys.stdout)
_formatter = logging.Formatter(
    fmt='{"ts":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
)
_handler.setFormatter(_formatter)

_level_map = {
    "CRITICAL": logging.CRITICAL, "ERROR": logging.ERROR, "WARNING": logging.WARNING,
    "INFO": logging.INFO, "DEBUG": logging.DEBUG
}
logger.setLevel(_level_map.get((settings.log_level or "INFO").upper(), logging.INFO))
if not logger.handlers:
    logger.addHandler(_handler)


# ------------------------------------------------------------------------------
# Request context (request-id)
# ------------------------------------------------------------------------------
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")

def get_request_id() -> str:
    rid = _request_id_ctx.get("")
    return rid or "n/a"


# ------------------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------------------
class RequestIDMiddleware:
    """Propagate X-Request-Id/X-Correlation-Id through the stack, generate if missing."""
    def __init__(self, app, header_names: Iterable[str] = ("X-Request-Id", "X-Correlation-Id")) -> None:
        self.app = app
        self.header_names = tuple(header_names)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        rid_in = headers.get("x-request-id") or headers.get("x-correlation-id")
        _request_id_ctx.set(rid_in.strip() if rid_in else str(uuid.uuid4()))

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                rid = _request_id_ctx.get("") or "n/a"
                hdrs = message.get("headers", [])
                new_headers = [(k, v) for (k, v) in hdrs if k.lower() not in (b"x-request-id", b"x-correlation-id")]
                new_headers.extend([
                    (b"x-request-id", rid.encode()),
                    (b"x-correlation-id", rid.encode()),
                ])
                message["headers"] = new_headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


class TimingMiddleware:
    """Measure processing time and expose it in a response header."""
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                duration = time.perf_counter() - start
                headers = message.get("headers", [])
                headers = [(k, v) for (k, v) in headers if k.lower() != b"x-process-time"]
                headers.append((b"x-process-time", f"{duration:.6f}s".encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


# ------------------------------------------------------------------------------
# Error model & handlers (unified envelope)
# ------------------------------------------------------------------------------
class ErrorPayload(BaseModel):
    type: str
    message: str
    details: Any | None = None
    request_id: str = Field(default_factory=get_request_id)
    model_config = ConfigDict(extra="forbid")

def error_response(status_code: int, *, type_: str, message: str, details: Any | None = None) -> JSONResponse:
    payload = ErrorPayload(type=type_, message=message, details=details, request_id=get_request_id())
    return _DEFAULT_RESPONSE_CLASS(status_code=status_code, content=json.loads(payload.model_dump_json()))

async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """
    Map HTTPException.detail into our envelope.
    If detail is a dict with {"code","message",...}, use code as `type` and message as `message`.
    Otherwise use "http_error".
    """
    if isinstance(exc.detail, dict):
        type_ = str(exc.detail.get("code") or "http_error")
        message = str(exc.detail.get("message") or "HTTP error")
        details = exc.detail
    else:
        type_ = "http_error"
        message = str(exc.detail) if isinstance(exc.detail, str) else "HTTP error"
        details = {"detail": exc.detail}
    return error_response(exc.status_code, type_=type_, message=message, details=details)

async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        type_="validation_error",
        message="Request validation failed",
        details=exc.errors(),
    )

async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception: %s", exc)
    return error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        type_="internal_error",
        message="Unexpected server error",
    )


# ------------------------------------------------------------------------------
# Lifespan (startup/shutdown hooks)
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # Export LANGCHAIN_* envs purely from settings (no getenv reads)
    try:
        enable_tracing_if_configured()
    except Exception as e:
        logger.warning("tracing.setup_failed: %s", e)

    # Log redacted settings (no secrets)
    try:
        logger.info("startup.begin %s", settings.redacted_dict())
    except Exception:
        logger.info("startup.begin")

    # Initialize Redis connection (optional; non-fatal)
    try:
        await get_redis()
    except Exception as e:
        logger.warning("redis.init_failed: %s", e)

    yield

    # Close Redis on shutdown
    try:
        await close_redis()
    except Exception as e:
        logger.warning("redis.close_failed: %s", e)

    logger.info("shutdown.end")


# ------------------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.service_name,
        version=settings.version,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
        default_response_class=_DEFAULT_RESPONSE_CLASS,
    )

    # Core middleware (order matters)
    app.add_middleware(RequestIDMiddleware)  # adds headers on the way out
    app.add_middleware(TimingMiddleware)
    # Sessions (cookie + Redis). Secure cookie if env != "dev".
    app.add_middleware(SessionMiddleware, secure_cookie=(settings.env != "dev"))
    # HTTP rate-limit (per session/IP per route, fixed 1-minute window)
    app.add_middleware(RateLimitMiddleware)

    # CORS (supports "*" or CSV/JSON list via settings.cors_origins)
    allow_origins = settings.cors_origins
    allow_credentials = False if allow_origins == ["*"] else True  # Starlette rule
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-Id", "X-Correlation-Id", "X-API-Key"],
        expose_headers=["X-Request-Id", "X-Correlation-Id", "X-Process-Time"],
    )

    # Error handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Health & meta
    @app.get("/health", tags=["_meta"], response_class=_DEFAULT_RESPONSE_CLASS)
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": settings.service_name,
            "version": settings.version,
            "environment": settings.env,
        }

    @app.get("/livez", tags=["_meta"], response_class=PlainTextResponse)
    async def livez() -> str:
        return "OK"

    @app.get("/readyz", tags=["_meta"], response_class=PlainTextResponse)
    async def readyz() -> str:
        # In future, check dependencies (catalogs, storage, etc.)
        return "READY"

    # --------------------------------------------------------------------------
    # Chat UI ("/" → /demo)
    # --------------------------------------------------------------------------
    app.include_router(chatui_router, tags=["_demo"])

    # --------------------------------------------------------------------------
    # Mount /chat (HTTP router) + safe fallback
    # Prefer the API router we implemented: workshops_ai.api.app.router
    # Protect with auth dependency if available.
    # --------------------------------------------------------------------------
    try:
        from workshops_ai.api.app import router as chat_router  # PREFERRED (our new /chat)
        app.include_router(chat_router, tags=["chat"], dependencies=_AUTH_DEP)
        logger.info("chat_router mounted (api.app)")
    except Exception:
        try:
            from workshops_ai.routers.chat_router import router as chat_router  # NEW canonical (if exists)
            app.include_router(chat_router, tags=["chat"], dependencies=_AUTH_DEP)
            logger.info("chat_router mounted (routers.chat_router)")
        except Exception:
            try:
                from workshops_ai.app.routers.chat_router import router as chat_router  # LEGACY path
                app.include_router(chat_router, tags=["chat"], dependencies=_AUTH_DEP)
                logger.info("chat_router mounted (app.routers.chat_router)")
            except Exception as e:  # pragma: no cover
                logger.warning('chat_router not mounted: %s', e)
                fallback = APIRouter()

                @fallback.post(
                    "/chat",
                    tags=["chat"],
                    response_class=_DEFAULT_RESPONSE_CLASS,
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
                async def chat_not_ready(_: Request) -> dict[str, Any]:
                    return {
                        "error": {
                            "type": "not_ready",
                            "message": "Chat router not mounted yet. Implement workshops_ai.api.app.router",
                        },
                        "request_id": get_request_id(),
                    }

                app.include_router(fallback)

    # --------------------------------------------------------------------------
    # Mount /ws (WebSocket router) + safe fallback (prefer new path, then legacy)
    # --------------------------------------------------------------------------
    try:
        from workshops_ai.routers.ws_router import router as ws_router  # NEW canonical
        app.include_router(ws_router, tags=["ws"], dependencies=_AUTH_DEP)
        logger.info("ws_router mounted")
    except Exception:
        try:
            from workshops_ai.app.routers.ws_router import router as ws_router  # LEGACY path
            app.include_router(ws_router, tags=["ws"], dependencies=_AUTH_DEP)
            logger.info("ws_router mounted (legacy path)")
        except Exception as e:  # pragma: no cover
            logger.warning("ws_router not mounted: %s", e)
            ws_fallback = APIRouter()

            @ws_fallback.websocket("/ws")
            async def ws_not_ready(websocket: WebSocket) -> None:
                await websocket.accept()
                await websocket.send_json({
                    "error": {
                        "type": "not_ready",
                        "message": "WS router not mounted yet. Implement workshops_ai.app.routers.ws_router.router"
                    },
                    "request_id": get_request_id(),
                })
                await websocket.close()

            app.include_router(ws_fallback)

    # --------------------------------------------------------------------------
    # Mount /assets (HTTP router) + safe fallback (preferred path only; MVP exists)
    # --------------------------------------------------------------------------
    try:
        from workshops_ai.routers.assets import router as assets_router  # NEW canonical
        app.include_router(assets_router, tags=["assets"], dependencies=_AUTH_DEP)
        logger.info("assets_router mounted")
    except Exception as e:  # pragma: no cover
        logger.warning("assets_router not mounted: %s", e)
        assets_fallback = APIRouter()

        @assets_fallback.get(
            "/api/assets",
            tags=["assets"],
            response_class=_DEFAULT_RESPONSE_CLASS,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        async def assets_not_ready() -> dict[str, Any]:
            return {
                "error": {
                    "type": "not_ready",
                    "message": "Assets router not mounted yet. Implement workshops_ai.routers.assets.router",
                },
                "request_id": get_request_id(),
            }

        app.include_router(assets_fallback)

    return app


# FastAPI instance for ASGI servers
app = create_app()

# Optional entrypoint for local dev: `python -m workshops_ai.app.main`
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(
        "workshops_ai.app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.reload,
        log_level=(settings.log_level or "INFO").lower(),
    )
