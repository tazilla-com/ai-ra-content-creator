# File: workshops_ai/app/main.py
# FastAPI bootstrap: health endpoints + mounts /chat router + /ws router + /assets router.
# - Strict Pydantic v2 validation (fail-closed at edges)
# - Request ID + timing middleware
# - CORS (configurable; supports "*" or CSV list)
# - Consistent JSON error model (prefers orjson if available)
# - LangSmith (legacy) -> LangChain V2 tracing env normalization
# - Safe fallbacks if /chat, /ws, or /assets router are not implemented
# - Simple /demo page to talk to the AI agent over /chat + /ws

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterable

from fastapi import APIRouter, FastAPI, HTTPException, Request, WebSocket, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    HTMLResponse,
    RedirectResponse,
)

try:
    from fastapi.responses import ORJSONResponse  # uses orjson if installed
    _DEFAULT_RESPONSE_CLASS = ORJSONResponse
except Exception:  # pragma: no cover
    _DEFAULT_RESPONSE_CLASS = JSONResponse

from pydantic import BaseModel, Field, ConfigDict

try:
    # Optional but recommended; used for env parsing
    from pydantic_settings import BaseSettings
except Exception:  # pragma: no cover
    # Lightweight fallback if pydantic-settings isn’t available yet
    class BaseSettings(BaseModel):  # type: ignore[misc]
        model_config = ConfigDict(arbitrary_types_allowed=True)


# ------------------------------------------------------------------------------
# LangSmith <-> LangChain tracing env normalization (supports both legacy & V2)
# ------------------------------------------------------------------------------

def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}

def _enable_tracing_env_normalization() -> None:
    """
    Accept both:
      - V2 (prefer): LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
      - legacy:      LANGSMITH_TRACING,    LANGSMITH_ENDPOINT, LANGSMITH_API_KEY, LANGSMITH_PROJECT
    If legacy is provided, map into LANGCHAIN_* unless already set.
    """
    if os.getenv("LANGSMITH_TRACING") and not os.getenv("LANGCHAIN_TRACING_V2"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true" if _truthy(os.getenv("LANGSMITH_TRACING")) else "false"
    if os.getenv("LANGSMITH_ENDPOINT") and not os.getenv("LANGCHAIN_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "")
    if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")


# ------------------------------------------------------------------------------
# Settings (env-driven, safe defaults)
# ------------------------------------------------------------------------------

class Settings(BaseSettings):
    app_name: str = "workshops_ai"
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    environment: str = os.getenv("ENVIRONMENT", "dev")  # dev|test|stage|prod

    # Respect your .env naming
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "8080"))

    enable_docs: bool = (os.getenv("ENABLE_DOCS") or "true").lower() not in ("0", "false", "no")

    # CORS_ORIGINS supports "*" or comma-separated list
    cors_origins_raw: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    cors_credentials: bool = True  # will be forced to False if wildcard origins are used
    cors_methods: list[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    cors_headers: list[str] = Field(default_factory=lambda: [
        "Authorization", "Content-Type", "X-Request-Id", "X-Correlation-Id"
    ])

    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    model_config = ConfigDict(extra="ignore")

    # Helper
    def parsed_cors_origins(self) -> list[str]:
        raw = (self.cors_origins_raw or "").strip()
        if raw in {"*", "'*'", '"*"'}:
            return ["*"]
        return [x.strip() for x in raw.split(",") if x.strip()]


settings = Settings()


# ------------------------------------------------------------------------------
# Logging (structured-ish)
# ------------------------------------------------------------------------------

logger = logging.getLogger("workshops_ai")
_handler = logging.StreamHandler(sys.stdout)
_formatter = logging.Formatter(
    fmt='{"ts":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
)
_handler.setFormatter(_formatter)

# Map LOG_LEVEL safely
_level_map = {
    "CRITICAL": logging.CRITICAL, "ERROR": logging.ERROR, "WARNING": logging.WARNING,
    "INFO": logging.INFO, "DEBUG": logging.DEBUG
}
logger.setLevel(_level_map.get(settings.log_level.upper(), logging.INFO))
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

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # ensure request id headers are present
                rid = _request_id_ctx.get("") or "n/a"
                headers_out = [
                    (b"x-request-id", rid.encode()),
                    (b"x-correlation-id", rid.encode()),
                ]
                # rebuild headers list: keep originals + enforce our two headers (overwrite if exist)
                new_headers = [(k, v) for k, v in message.get("headers", []) if k.lower() not in (b"x-request-id", b"x-correlation-id")]
                new_headers.extend(headers_out)
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
# Error model & handlers
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
    detail = exc.detail if isinstance(exc.detail, str) else "HTTP error"
    return error_response(
        exc.status_code,
        type_="http_error",
        message=detail,
        details=getattr(exc, "detail", None),
    )

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
async def lifespan(app: FastAPI):  # noqa: ARG001 (app is part of FastAPI signature)
    _enable_tracing_env_normalization()
    logger.info("startup.begin env=%s version=%s", settings.environment, settings.app_version)
    yield
    logger.info("shutdown.end")


# ------------------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
        default_response_class=_DEFAULT_RESPONSE_CLASS,
    )

    # Core middleware
    app.add_middleware(RequestIDMiddleware)  # adds headers on the way out
    app.add_middleware(TimingMiddleware)

    # CORS (supports "*" or CSV list)
    allow_origins = settings.parsed_cors_origins()
    allow_credentials = settings.cors_credentials and allow_origins != ["*"]  # Starlette rule
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
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
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
        }

    @app.get("/livez", tags=["_meta"], response_class=PlainTextResponse)
    async def livez() -> str:
        return "OK"

    @app.get("/readyz", tags=["_meta"], response_class=PlainTextResponse)
    async def readyz() -> str:
        # In future, check dependencies (catalogs, storage, etc.)
        return "READY"

    # ------------------------------------------------------------------------------
    # Demo UI (simple HTML page that talks to /chat and listens on /ws)
    # ------------------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/demo", status_code=302)

    @app.get("/demo", include_in_schema=False, response_class=HTMLResponse)
    async def demo_page() -> str:
        return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>WorkshopsAI — Demo Chat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
    body { margin: 0; background: #0b1020; color: #e6edf3; }
    header { padding: 14px 16px; background: #101830; border-bottom: 1px solid #1d2750; display: flex; gap: 12px; align-items: center; }
    header input, header select, header button { background: #0f1a38; color: #e6edf3; border: 1px solid #2a3a72; border-radius: 8px; padding: 8px 10px; }
    header input::placeholder { color: #93a1c6; }
    main { display: grid; grid-template-columns: 1fr 360px; gap: 12px; padding: 12px; height: calc(100vh - 64px); box-sizing: border-box; }
    .chat, .events { background: #0f1730; border: 1px solid #1d2750; border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; }
    .chat-log { flex: 1; overflow: auto; padding: 12px; }
    .msg { margin: 8px 0; line-height: 1.4; }
    .msg.me { color: #c8e1ff; }
    .msg.ai { color: #a7f3d0; }
    .msg.sys { color: #facc15; }
    .input-row { display: flex; gap: 8px; padding: 10px; border-top: 1px solid #1d2750; }
    .input-row textarea { flex: 1; min-height: 44px; resize: vertical; background: #0f1a38; color: #e6edf3; border: 1px solid #2a3a72; border-radius: 8px; padding: 8px 10px; }
    .events-log { flex: 1; overflow: auto; padding: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
    .pill { font-size: 12px; padding: 2px 8px; border-radius: 999px; border: 1px solid #334155; }
    a { color: #8ab4ff; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <header>
    <strong>WorkshopsAI Demo</strong>
    <span class="pill">/chat + /ws</span>
    <span style="margin-left:auto"></span>
    <label>org_id <input id="org" value="demo-org" size="10"/></label>
    <label>user_id <input id="user" value="demo-user" size="10"/></label>
    <label>tab
      <select id="tab">
        <option value="assets" selected>assets</option>
        <option value="impacts">impacts</option>
        <option value="threats">threats</option>
        <option value="controls">controls</option>
      </select>
    </label>
    <button id="ws-btn">Connect WS</button>
    <a href="/docs" target="_blank" rel="noreferrer" style="margin-left:8px">/docs</a>
  </header>

  <main>
    <section class="chat">
      <div id="chat" class="chat-log"></div>
      <div class="input-row">
        <textarea id="msg" placeholder="Type to talk to the AI AssetAgent (e.g., 'List primary Application software', or 'Create asset CRM as primary type Application software')"></textarea>
        <button id="send">Send</button>
      </div>
    </section>

    <aside class="events">
      <div class="events-log" id="events"></div>
    </aside>
  </main>

  <script>
    const chat = document.getElementById('chat');
    const eventsEl = document.getElementById('events');
    const sendBtn = document.getElementById('send');
    const wsBtn = document.getElementById('ws-btn');
    const org = document.getElementById('org');
    const user = document.getElementById('user');
    const tab = document.getElementById('tab');
    const msg = document.getElementById('msg');

    let ws = null;
    let connId = null;

    function line(kind, text) {
      const div = document.createElement('div');
      div.className = `msg ${kind}`;
      div.textContent = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }

    function ev(obj) {
      const pre = document.createElement('pre');
      pre.textContent = JSON.stringify(obj, null, 2);
      eventsEl.appendChild(pre);
      eventsEl.scrollTop = eventsEl.scrollHeight;
    }

    async function sendMessage() {
      const payload = {
        org_id: org.value.trim(),
        active_tab: tab.value,
        selection: {},   // extend if you want to pass asset_id, etc.
        message: msg.value,
        staging: false,
        user_id: user.value.trim(),
      };
      line('me', payload.message || '(empty)');
      msg.value = '';

      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (res.ok) {
          (data.messages || []).forEach(t => line('ai', t));
          if (data.proposals) {
            line('sys', '[proposals attached]');
            ev({ type: 'proposals', proposals: data.proposals });
          }
        } else {
          line('sys', `ERROR ${res.status}: ${data?.message || 'Request failed'}`);
          ev({ type: 'error', data });
        }
      } catch (e) {
        line('sys', `Network error: ${e}`);
      }
    }

    function connectWs() {
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close(1000, 'reconnect');
      }
      const url = new URL(window.location.href);
      url.protocol = url.protocol.replace('http', 'ws');
      url.pathname = '/ws';
      url.searchParams.set('org_id', org.value.trim());
      url.searchParams.set('user_id', user.value.trim());
      url.searchParams.set('active_tab', tab.value);

      ws = new WebSocket(url.toString());
      ws.onopen = () => { line('sys', 'WS connected'); wsBtn.textContent = 'Reconnect WS'; };
      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          // AgentEvent frames are raw JSON; chat frames may be wrapped {type, payload}
          if (data && typeof data === 'object') {
            if (data.type && data.payload) {
              ev({ ws_type: data.type, payload: data.payload });
              if (data.type === 'info' && data.payload && data.payload.message) {
                line('sys', data.payload.message);
              }
            } else {
              // AgentEvent (asset.create / asset.set_meta etc.)
              ev(data);
              if (data.event) {
                line('sys', `event: ${data.event}`);
              }
            }
          }
        } catch (e) {
          ev({ parse_error: String(e), raw: evt.data });
        }
      };
      ws.onclose = (evt) => { line('sys', `WS closed (${evt.code})`); };
      ws.onerror = (evt) => { line('sys', 'WS error'); };
    }

    sendBtn.addEventListener('click', sendMessage);
    msg.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) sendMessage(); });
    wsBtn.addEventListener('click', connectWs);
    window.addEventListener('load', connectWs);
    window.addEventListener('beforeunload', () => { if (ws) try { ws.close(1000, 'leaving'); } catch {} });
  </script>
</body>
</html>
        """

    # ------------------------------------------------------------------------------
    # Mount /chat (HTTP router) + safe fallback (prefer new path, then legacy)
    # ------------------------------------------------------------------------------
    try:
        from workshops_ai.routers.chat_router import router as chat_router  # NEW
        app.include_router(chat_router, tags=["chat"])
        logger.info("chat_router mounted")
    except Exception:
        try:
            from workshops_ai.app.routers.chat_router import router as chat_router  # LEGACY
            app.include_router(chat_router, tags=["chat"])
            logger.info("chat_router mounted (legacy path)")
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
                        "message": "Chat router not mounted yet. Implement workshops_ai.routers.chat_router.router",
                    },
                    "request_id": get_request_id(),
                }

            app.include_router(fallback)

    # ------------------------------------------------------------------------------
    # Mount /ws (WebSocket router) + safe fallback (prefer new path, then legacy)
    # ------------------------------------------------------------------------------
    try:
        from workshops_ai.routers.ws_router import router as ws_router  # NEW
        app.include_router(ws_router, tags=["ws"])
        logger.info("ws_router mounted")
    except Exception:
        try:
            from workshops_ai.app.routers.ws_router import router as ws_router  # LEGACY
            app.include_router(ws_router, tags=["ws"])
            logger.info("ws_router mounted (legacy path)")
        except Exception as e:  # pragma: no cover
            logger.warning("ws_router not mounted: %s", e)
            ws_fallback = APIRouter()

            @ws_fallback.websocket("/ws")
            async def ws_not_ready(websocket: WebSocket) -> None:
                # Minimal non-breaking fallback: accept, inform, close.
                await websocket.accept()
                await websocket.send_json({
                    "error": {
                        "type": "not_ready",
                        "message": "WS router not mounted yet. Implement workshops_ai.routers.ws_router.router"
                    },
                    "request_id": get_request_id(),
                })
                await websocket.close()

            app.include_router(ws_fallback)

    # ------------------------------------------------------------------------------
    # Mount /assets (HTTP router) + safe fallback (preferred path only; MVP exists)
    # ------------------------------------------------------------------------------
    try:
        # Expected standard location for the assets router:
        # workshops_ai/routers/assets.py -> router = APIRouter(prefix="/api/assets")
        from workshops_ai.routers.assets import router as assets_router  # NEW
        app.include_router(assets_router, tags=["assets"])
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

    # Per-request request-id setter (ensures ContextVar is filled from headers or generated)
    @app.middleware("http")
    async def set_request_id_from_headers(request: Request, call_next: Callable[[Request], Any]):
        header_val = request.headers.get("X-Request-Id") or request.headers.get("X-Correlation-Id")
        _request_id_ctx.set(header_val.strip() if header_val else str(uuid.uuid4()))
        return await call_next(request)

    return app


# FastAPI instance for ASGI servers
app = create_app()

# Optional entrypoint for local dev: `python -m workshops_ai.app.main`
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(
        "workshops_ai.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=(os.getenv("RELOAD") or "true").lower() in ("1", "true", "yes"),
        log_level=settings.log_level.lower(),
    )
