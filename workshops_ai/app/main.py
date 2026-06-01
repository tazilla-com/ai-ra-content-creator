# File: workshops_ai/app/main.py
# Purpose:
#   FastAPI bootstrap: health endpoints + mounts /chat router + /ws router + /assets router + ChatUI.
#   - Strict Pydantic v2 validation (fail-closed at edges)
#   - Request ID + timing middleware
#   - Sessions (Redis) + simple rate-limit middleware
#   - CORS (configurable via settings.cors_origins) with prod hardening
#   - Consistent JSON error model (prefers orjson if available)
#   - LangChain/LangSmith tracing env export via centralized settings (no getenv reads)
#   - Canonical router imports (fail-fast in prod; no legacy fallbacks)
#   - Chat UI is provided by workshops_ai/chatui/router.py
#
# Change:
#   - Use central `mask_secrets()` to redact config in the startup log (replaces settings.redacted_dict()).
#   - Added session propagation helpers for HTTP: a stable cookie/header name and a
#     `get_session_id_from_request()` fallback that chat_router can use when body lacks `session_id`.
#   - ✅ NEW: Enable faulthandler at startup and register SIGUSR2 to dump stacks (for recursion/debug).
#   - Kept the project's patch style: old code commented above new code.

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterable, Optional  # <- Optional for helper return

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.openapi.models import SecuritySchemeType
from fastapi.security import APIKeyHeader, HTTPBearer

try:
    from fastapi.responses import ORJSONResponse  # uses orjson if installed
    _DEFAULT_RESPONSE_CLASS = ORJSONResponse
except Exception:
    _DEFAULT_RESPONSE_CLASS = JSONResponse

from pydantic import BaseModel, Field, ConfigDict

# Load .env early if present (non-fatal)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

# Centralized settings
from workshops_ai.settings import get_settings
settings = get_settings()

# Central secret masking (for startup log redaction)
from workshops_ai.tools.telemetry import redact_for_log as mask_secrets

# Optional HTTP auth dependency (used for HTTP routers, NOT for WebSocket)
try:
    from workshops_ai.app.security import require_auth as _require_auth
    _AUTH_DEP = [Depends(_require_auth)]
except Exception as e:  # pragma: no cover
    logging.getLogger("workshops_ai").warning("auth.disabled: %s", e)
    _AUTH_DEP = []

# Chat UI (demo)
from workshops_ai.chatui.router import router as chatui_router

# Sessions + Rate-limit middlewares & Redis lifecycle
from workshops_ai.app.middlewares.session_middleware import SessionMiddleware
from workshops_ai.app.middlewares.rate_limit import RateLimitMiddleware
from workshops_ai.app.services.redis_pool import get_redis, close_redis

# Tracing (export LANGCHAIN_* from settings.langchain, no getenv reads)
from workshops_ai.tools.tracing import enable_tracing_if_configured

# enable on-demand/full-crash Python stack dumps 
import faulthandler  
import signal        


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


# Suppress noisy health-probe access logs (/livez, /readyz)
class _HealthProbeFilter(logging.Filter):
    _PROBE_PATHS = {"/livez", "/readyz"}

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(f"GET {p} " in msg for p in self._PROBE_PATHS)

logging.getLogger("uvicorn.access").addFilter(_HealthProbeFilter())


# ------------------------------------------------------------------------------
# Request context (request-id)
# ------------------------------------------------------------------------------
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")

def get_request_id() -> str:
    rid = _request_id_ctx.get("")
    return rid or "n/a"


# ------------------------------------------------------------------------------
# Session identifiers (HTTP)
# ------------------------------------------------------------------------------
# Keep cookie name aligned with ws_router so both HTTP and WS share the same session.
SESSION_COOKIE_NAME = "wsai.sid"
# Custom HTTP header that FE may send alongside or instead of cookie.
SESSION_HEADER_NAME = "X-Session-Id"

def get_session_id_from_request(req: Request) -> Optional[str]:
    """
    Extract session id for HTTP requests. This is used by chat_router as a fallback
    when the client doesn't send `session_id` in the JSON body.

    Precedence:
      1) Cookie `wsai.sid` (stable name shared with WS)
      2) Header `X-Session-Id` (handy for non-cookie environments)
      3) Legacy/loose header `x-sessionid` (best-effort)
    """
    try:
        sid = req.cookies.get(SESSION_COOKIE_NAME)
    except Exception:
        sid = None
    if sid:
        return sid
    # Header fallback
    return req.headers.get(SESSION_HEADER_NAME) or req.headers.get("x-sessionid")


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
                headers.append((b"x-process-time", f"{duration:.6f}".encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


class SecurityHeadersMiddleware:
    """
    Add a few basic security headers. Keep CSP minimal to avoid breaking the demo UI.
    For production, consider serving UI with a stricter CSP policy.
    """
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        # >>> Only handle HTTP responses; skip websockets <<<
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))

                def _set(name: bytes, val: str):
                    headers[name.lower()] = val.encode()

                _set(b"x-content-type-options", "nosniff")
                _set(b"x-frame-options", "DENY")
                _set(b"referrer-policy", "no-referrer")
                # Only set HSTS outside dev; assumes HTTPS termination upstream
                if settings.env != "dev":
                    _set(b"strict-transport-security", "max-age=63072000; includeSubDomains; preload")

                message["headers"] = list(headers.items())
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

async def wizard_exception_handler(_: Request, exc: "WizardException") -> JSONResponse:
    """Handle wizard-specific exceptions with structured error responses."""
    from workshops_ai.wizard.exceptions import WizardException
    if not isinstance(exc, WizardException):
        return await unhandled_exception_handler(_, exc)

    logger.error(
        f"wizard_exception: code={exc.code}, message={exc.message}, details={exc.details}, request_id={get_request_id()}"
    )
    return error_response(
        exc.status_code,
        type_=exc.code,
        message=exc.message,
        details=exc.details
    )


# ------------------------------------------------------------------------------
# Lifespan (startup/shutdown hooks)
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    try:
        # Dumps go to stderr → visible in `docker logs`
        faulthandler.enable()
        # Manual dump anytime: `docker kill -s USR2 workshops-ai-api`
        faulthandler.register(signal.SIGUSR2, all_threads=True)
    except Exception:
        # Be tolerant to environments lacking signals (e.g., Windows dev shells)
        pass

    # Export LANGCHAIN_* envs purely from settings (no getenv reads)
    try:
        enable_tracing_if_configured()
    except Exception as e:
        logger.warning("tracing.setup_failed: %s", e)

    # Log redacted settings (no secrets)
    try:
        logger.info("startup.begin %s", mask_secrets(settings))
    except Exception:
        logger.info("startup.begin")

    # Initialize Redis connection (optional; non-fatal)
    try:
        await get_redis()
    except Exception as e:
        logger.warning("redis.init_failed: %s", e)

    # Initialize OpenAI throttle
    if settings.openai_throttle.enabled:
        try:
            from workshops_ai.llm.openai_throttle import init_throttle
            init_throttle(
                rpm=settings.openai_throttle.rpm,
                tpm=settings.openai_throttle.tpm,
                max_concurrent=settings.openai_throttle.max_concurrent,
                per_request_limits=settings.wizard.openai_per_request_limits,
                capacity_limits=settings.wizard.openai_capacity_limits,
            )
        except Exception as e:
            logger.warning("openai_throttle.init_failed: %s", e)

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
    # CORS hardening: if "*" in prod, fail fast to avoid accidental wide-open CORS
    if settings.env != "dev" and settings.cors_origins == ["*"]:
        raise RuntimeError("CORS '*' is not allowed outside dev. Configure settings.cors_origins with explicit domains.")

    app = FastAPI(
        title=settings.service_name,
        version=settings.version,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
        default_response_class=_DEFAULT_RESPONSE_CLASS,
    )

    # Add OpenAPI security schemes for Swagger UI authentication
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        from fastapi.openapi.utils import get_openapi
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "APIKeyHeader": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API Key authentication via X-API-Key header"
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT Bearer token authentication"
            }
        }

        # Add security requirement to all paths that need authentication
        # Paths that DON'T require auth
        public_paths = {"/health", "/livez", "/readyz", "/", "/demo", "/openapi.json", "/docs", "/redoc"}

        for path, path_item in openapi_schema.get("paths", {}).items():
            # Skip public endpoints
            if path in public_paths or path.startswith("/demo"):
                continue

            # Add security to all methods in this path
            for method in path_item:
                if method in ["get", "post", "put", "patch", "delete", "options", "head"]:
                    # Allow either API key OR Bearer token
                    path_item[method]["security"] = [
                        {"APIKeyHeader": []},
                        {"BearerAuth": []}
                    ]

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    # Core middleware (order matters)
    app.add_middleware(RequestIDMiddleware)  # adds headers on the way out
    app.add_middleware(TimingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    # Sessions (cookie + Redis). Secure cookie if env != "dev".
    # NOTE: we don't force the cookie name here to avoid coupling to middleware internals.
    #       The helper `get_session_id_from_request()` reads the standard cookie/header names.
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
        allow_headers=["Authorization", "Content-Type", "X-Request-Id", "X-Correlation-Id", "X-API-Key", "Origin"],
        expose_headers=["X-Request-Id", "X-Correlation-Id", "X-Process-Time"],
    )

    # Error handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # Wizard exception handler
    try:
        from workshops_ai.wizard.exceptions import WizardException
        app.add_exception_handler(WizardException, wizard_exception_handler)
    except ImportError:
        pass

    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Health & meta (hidden from API docs - wizard-only API)
    @app.get(
        "/health",
        tags=["Health"],
        response_class=_DEFAULT_RESPONSE_CLASS,
        summary="Health check endpoint",
        description="Returns application health status, version, and environment information"
    )
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": settings.service_name,
            "version": settings.version,
            "environment": settings.env,
        }

    @app.get("/livez", tags=["_meta"], response_class=PlainTextResponse, include_in_schema=False)
    async def livez() -> str:
        return "OK"

    @app.get("/readyz", tags=["_meta"], response_class=PlainTextResponse, include_in_schema=False)
    async def readyz() -> str:
        # In future, check dependencies (catalogs, storage, etc.)
        return "READY"

    @app.get("/health/openai", tags=["Health"], response_class=_DEFAULT_RESPONSE_CLASS, include_in_schema=False)
    async def openai_throttle_metrics() -> dict[str, Any]:
        if not settings.openai_throttle.enabled:
            return {"enabled": False}
        try:
            from workshops_ai.llm.openai_throttle import get_throttle
            throttle = get_throttle()
            return throttle.get_metrics()
        except Exception as e:
            return {"error": str(e)}

    # --------------------------------------------------------------------------
    # Chat UI ("/" → /demo) - hidden from API docs (wizard-only API)
    # --------------------------------------------------------------------------
    app.include_router(chatui_router, tags=["_demo"], include_in_schema=False)

    # --------------------------------------------------------------------------
    # Canonical routers (fail-fast outside dev)
    # --------------------------------------------------------------------------
    def _mount_or_fallback(name: str, mount: Callable[[], None], fallback: Callable[[], None]) -> None:
        try:
            mount()
            logger.info("%s mounted", name)
        except Exception as e:
            if settings.env != "dev":
                raise
            logger.warning("%s not mounted, using dev fallback: %s", name, e)
            fallback()

    # /api/chat (note: chat_router ALREADY has prefix="/api" + its own auth deps)
    # Hidden from API docs - wizard-only API
    def _mount_chat():
        from workshops_ai.routers.chat_router import router as chat_router
        app.include_router(chat_router, include_in_schema=False)
        logger.info("chat_router mounted at /api/chat")

    def _fallback_chat():
        router = APIRouter()
        @router.post(
            "/api/chat",
            tags=["chat"],
            response_class=_DEFAULT_RESPONSE_CLASS,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            include_in_schema=False,
        )
        async def chat_not_ready(_: Request) -> dict[str, Any]:
            return {
                "error": {"type": "not_ready", "message": "Chat router not mounted."},
                "request_id": get_request_id(),
            }
        app.include_router(router)

    _mount_or_fallback("chat_router", _mount_chat, _fallback_chat)

    # /ws (NO HTTP auth dependencies here; ws_router does its own verify_ws_auth)
    # Hidden from API docs - wizard-only API
    def _mount_ws():
        from workshops_ai.routers.ws_router import router as ws_router
        app.include_router(ws_router, tags=["ws"], include_in_schema=False)

    def _fallback_ws():
        router = APIRouter()
        @router.websocket("/ws")
        async def ws_not_ready(websocket: WebSocket) -> None:
            await websocket.accept()
            await websocket.send_json({
                "error": {"type": "not_ready", "message": "WS router not mounted."},
                "request_id": get_request_id(),
            })
            await websocket.close()
        app.include_router(router)

    _mount_or_fallback("ws_router", _mount_ws, _fallback_ws)

    # /api/assets (public read-only) + /graph/assets (tool-first)
    # Hidden from API docs - wizard-only API
    def _mount_assets():
        from workshops_ai.routers.assets import router as assets_router, graph_router as assets_graph_router
        # Public read-only API
        app.include_router(assets_router, tags=["assets"], dependencies=_AUTH_DEP, include_in_schema=False)
        # Tool-first endpoints
        app.include_router(assets_graph_router, tags=["graph:assets"], dependencies=_AUTH_DEP, include_in_schema=False)

    def _fallback_assets():
        router = APIRouter()
        # Public list fallback
        @router.get(
            "/api/assets",
            tags=["assets"],
            response_class=_DEFAULT_RESPONSE_CLASS,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            include_in_schema=False,
        )
        async def assets_not_ready() -> dict[str, Any]:
            return {
                "error": {"type": "not_ready", "message": "Assets router not mounted."},
                "request_id": get_request_id(),
            }
        # Tool-first fallbacks
        @router.get(
            "/graph/assets/list",
            tags=["graph:assets"],
            response_class=_DEFAULT_RESPONSE_CLASS,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            include_in_schema=False,
        )
        async def assets_list_not_ready() -> dict[str, Any]:
            return {"error": {"type": "not_ready", "message": "Graph assets list not mounted."}, "request_id": get_request_id()}

        @router.post(
            "/graph/assets/create",
            tags=["graph:assets"],
            response_class=_DEFAULT_RESPONSE_CLASS,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            include_in_schema=False,
        )
        async def assets_create_not_ready() -> dict[str, Any]:
            return {"error": {"type": "not_ready", "message": "Graph assets create not mounted."}, "request_id": get_request_id()}

        @router.put(
            "/graph/assets/update",
            tags=["graph:assets"],
            response_class=_DEFAULT_RESPONSE_CLASS,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            include_in_schema=False,
        )
        async def assets_update_not_ready() -> dict[str, Any]:
            return {"error": {"type": "not_ready", "message": "Graph assets update not mounted."}, "request_id": get_request_id()}

        app.include_router(router)

    _mount_or_fallback("assets_router(s)", _mount_assets, _fallback_assets)

    # /graph/threats (HTTP)
    # Hidden from API docs - wizard-only API
    def _mount_threats():
        from workshops_ai.routers.threats import router as threats_router
        app.include_router(threats_router, tags=["threats"], dependencies=_AUTH_DEP, include_in_schema=False)

    def _fallback_threats():
        router = APIRouter()
        @router.post(
            "/graph/threats",
            tags=["threats"],
            response_class=_DEFAULT_RESPONSE_CLASS,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            include_in_schema=False,
        )
        async def threats_not_ready(_: Request) -> dict[str, Any]:
            return {
                "error": {"type": "not_ready", "message": "Threats router not mounted."},
                "request_id": get_request_id(),
            }
        @router.get("/graph/threats/ping", tags=["threats"], include_in_schema=False)
        async def threats_ping() -> dict[str, Any]:
            return {"status": "not_ready", "router": "threats"}
        app.include_router(router)

    _mount_or_fallback("threats_router", _mount_threats, _fallback_threats)

    # /graph/impacts (HTTP)
    # Hidden from API docs - wizard-only API
    def _mount_impacts():
        from workshops_ai.routers.impacts import router as impacts_router
        app.include_router(impacts_router, tags=["impacts"], dependencies=_AUTH_DEP, include_in_schema=False)

    def _fallback_impacts():
        router = APIRouter()
        @router.post(
            "/graph/impacts",
            tags=["impacts"],
            response_class=_DEFAULT_RESPONSE_CLASS,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            include_in_schema=False,
        )
        async def impacts_not_ready(_: Request) -> dict[str, Any]:
            return {
                "error": {"type": "not_ready", "message": "Impacts router not mounted."},
                "request_id": get_request_id(),
            }
        @router.get("/graph/impacts/ping", tags=["impacts"], include_in_schema=False)
        async def impacts_ping() -> dict[str, Any]:
            return {"status": "not_ready", "router": "impacts"}
        app.include_router(router)

    _mount_or_fallback("impacts_router", _mount_impacts, _fallback_impacts)

    # /wizard (AI wizard for onboarding)
    def _mount_wizard():
        from workshops_ai.wizard.router import router as wizard_router
        app.include_router(wizard_router, tags=["wizard"], dependencies=_AUTH_DEP)

    def _fallback_wizard():
        router = APIRouter()
        @router.get(
            "/wizard/state",
            tags=["wizard"],
            response_class=_DEFAULT_RESPONSE_CLASS,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            include_in_schema=False,
        )
        async def wizard_not_ready() -> dict[str, Any]:
            return {
                "error": {"type": "not_ready", "message": "Wizard router not mounted."},
                "request_id": get_request_id(),
            }
        app.include_router(router)

    _mount_or_fallback("wizard_router", _mount_wizard, _fallback_wizard)

    # /session (Session management)
    def _mount_session():
        from workshops_ai.routers.session_router import router as session_router
        app.include_router(session_router, tags=["session"], dependencies=_AUTH_DEP)

    def _fallback_session():
        router = APIRouter()
        @router.get(
            "/session",
            tags=["session"],
            response_class=_DEFAULT_RESPONSE_CLASS,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            include_in_schema=False,
        )
        async def session_not_ready() -> dict[str, Any]:
            return {
                "error": {"type": "not_ready", "message": "Session router not mounted."},
                "request_id": get_request_id(),
            }
        app.include_router(router)

    _mount_or_fallback("session_router", _mount_session, _fallback_session)

    # /scrape (Web scraping endpoints)
    def _mount_scrape():
        from workshops_ai.routers.scrape_router import router as scrape_router
        app.include_router(scrape_router, tags=["scrape"], dependencies=_AUTH_DEP)

    def _fallback_scrape():
        router = APIRouter()
        @router.post(
            "/scrape/website",
            tags=["scrape"],
            response_class=_DEFAULT_RESPONSE_CLASS,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            include_in_schema=False,
        )
        async def scrape_not_ready() -> dict[str, Any]:
            return {
                "error": {"type": "not_ready", "message": "Scrape router not mounted."},
                "request_id": get_request_id(),
            }
        app.include_router(router)

    _mount_or_fallback("scrape_router", _mount_scrape, _fallback_scrape)

    # /api/catalogs (Read-only catalog endpoints - Impacts, Threats, Controls)
    def _mount_catalogs():
        from workshops_ai.routers.catalogs import router as catalogs_router
        app.include_router(catalogs_router, tags=["catalogs"], dependencies=_AUTH_DEP)

    def _fallback_catalogs():
        router = APIRouter()
        @router.get(
            "/api/catalogs/impacts",
            tags=["catalogs"],
            response_class=_DEFAULT_RESPONSE_CLASS,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            include_in_schema=False,
        )
        async def catalogs_not_ready() -> dict[str, Any]:
            return {
                "error": {"type": "not_ready", "message": "Catalogs router not mounted."},
                "request_id": get_request_id(),
            }
        app.include_router(router)

    _mount_or_fallback("catalogs_router", _mount_catalogs, _fallback_catalogs)

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
