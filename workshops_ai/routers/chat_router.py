# File: workshops_ai/routers/chat_router.py
from __future__ import annotations

"""
Chat Router (UIRequest → route_chat → UIResponse)
-------------------------------------------------
- Auth-protected POST /api/chat endpoint used by the UI.
- Strictly validates incoming payload as UIRequest; falls back to legacy ChatIn.
- Returns a strictly validated UIResponse.

Defensive design goals:
- Never blindly access optional UIRequest attributes (e.g., conn_id) — always use getattr().
- Do not crash if UIRequest model forbids extra attributes; use a safe setter helper.
- Hydrate missing conn_id/session_id and language hints best-effort (WS hub + headers).
- Enforce a single timeout (settings.chat.timeout_sec) and message size limit.
- Keep the contract stable for the manager/graphs (DI via make_default_deps()).
"""

import asyncio
import inspect
import uuid
from typing import Any, Optional, Callable

from fastapi import APIRouter, Depends, HTTPException, Request, status, Body
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from workshops_ai.settings import get_settings
from workshops_ai.schemas.ui_contract import UIRequest, UIResponse

# Security dependency (API key / JWT) — prefer new path, fallback to legacy if present
try:
    from workshops_ai.app.security import require_auth  # preferred path
    _AUTH_DEPS = [Depends(require_auth)]
except Exception:  # pragma: no cover
    try:
        from workshops_ai.app.deps.auth import require_auth  # type: ignore
        _AUTH_DEPS = [Depends(require_auth)]
    except Exception:  # pragma: no cover
        _AUTH_DEPS = []  # no-op if security is disabled in dev

# Telemetry (best-effort, entirely optional)
try:
    from workshops_ai.tools.telemetry import telemetry_log  # type: ignore
except Exception:  # pragma: no cover
    def telemetry_log(event: str, payload: dict) -> None:  # type: ignore
        return

# DI factory (ensures llm_client + checkpointer + memory + retriever)
from workshops_ai.graph.factory import make_default_deps

settings = get_settings()
router = APIRouter(prefix="/api", dependencies=_AUTH_DEPS)

# Router runtime limits (single source of truth from settings)
CHAT_TIMEOUT_SEC: float = float(getattr(getattr(settings, "chat", object()), "timeout_sec", 60))
MAX_MESSAGE_LEN: int = int(getattr(getattr(settings, "chat", object()), "max_message_len", 4000))


# ----------------------------------------------------------------------
# Utilities / Helpers
# ----------------------------------------------------------------------
def _truncate_for_log(text: Optional[str], limit: int = 512) -> Optional[str]:
    """Return a compact preview for telemetry to avoid PII/log bloat."""
    if text is None:
        return None
    return text if len(text) <= limit else (text[:limit] + "…")


def _normalize_language(payload: dict[str, Any]) -> None:
    """
    Accept `accept_language` (UI/browser hint) and map it to UIRequest.language.
    Drop the original key to satisfy UIRequest(extra='forbid') and avoid 422.
    NOTE: Header fallback is applied later in _lock_user_language().
    """
    al = payload.get("accept_language")
    if isinstance(al, str) and al.strip():
        payload.setdefault("language", al.strip()[:256])
    payload.pop("accept_language", None)


def _header_accept_language(request: Request) -> Optional[str]:
    """Extract HTTP 'Accept-Language' header (trimmed, capped) if present."""
    h = request.headers.get("accept-language")
    if isinstance(h, str):
        h = h.strip()
        if h:
            return h[:256]
    return None


def _maybe_conn_id_from_headers_or_cookie(request: Request) -> Optional[str]:
    """
    Try to recover a connection id if the client didn't send it in the payload.
    Looks into a custom header and a cookie name that the WS layer may set.
    This is best-effort and safe to fail.
    """
    hdr = request.headers.get("x-conn-id")
    if hdr and isinstance(hdr, str) and hdr.strip():
        return hdr.strip()[:128]
    # Optional cookie fallback
    try:
        cookie_val = request.cookies.get("ws_conn_id")
        if cookie_val and isinstance(cookie_val, str) and cookie_val.strip():
            return cookie_val.strip()[:128]
    except Exception:
        pass
    return None


async def _run_with_timeout(awaitable_or_result: Any, timeout_sec: float) -> Any:
    """
    Await a coroutine/future or return a sync result with a timeout.
    If a sync callable slips through, offload it to a thread pool.
    """
    if inspect.isawaitable(awaitable_or_result):
        return await asyncio.wait_for(awaitable_or_result, timeout=timeout_sec)
    if callable(awaitable_or_result):
        return await asyncio.wait_for(asyncio.to_thread(awaitable_or_result), timeout=timeout_sec)
    return awaitable_or_result


def _safe_set(model: Any, field: str, value: Any) -> None:
    """
    Set `model.field = value` only if the model seems to support the attribute.
    This prevents crashes with Pydantic v2 when extra fields are forbidden.
    """
    if value is None:
        return
    try:
        # hasattr() is safe here; Pydantic v2 raises AttributeError for unknown fields,
        # which hasattr() then converts to False without crashing.
        if hasattr(model, field):
            setattr(model, field, value)
    except Exception:
        # Best-effort: ignore if the model forbids setting this field.
        pass


def _to_ui_request(payload: dict[str, Any]) -> UIRequest:
    """
    Prefer the new UIRequest; if that fails, try legacy ChatIn and bridge to UIRequest.
    Normalize language hints before validation.
    """
    _normalize_language(payload)

    try:
        return UIRequest.model_validate(payload)
    except ValidationError:
        # Legacy bridge (schemas/chat.py) — optional, backward-compatible
        try:
            from workshops_ai.schemas.chat import ChatIn  # lazy import
            legacy = ChatIn.model_validate(payload)
            return legacy.to_ui()
        except Exception as e2:
            # Raise a unified 422 so clients get a consistent error shape
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"message": "Invalid UIRequest (and legacy ChatIn fallback failed)"},
            ) from e2


def _ensure_session_and_conn(ui_req: UIRequest, request: Request) -> None:
    """
    Make sure we have useful conn_id and session_id:
      - If conn_id is missing, try to recover it from headers/cookies.
      - If session_id is missing, derive a stable UUIDv5 from org_id + (conn_id or user_id).
    This runs in the router (not the manager) to keep the contract strong and predictable.

    IMPORTANT: Do NOT directly access ui_req.conn_id (it may not exist if the model was older).
               Always use getattr() and safe setters to avoid AttributeError.
    """
    # Recover/ensure conn_id
    conn_id = getattr(ui_req, "conn_id", None)
    if not conn_id:
        recovered = _maybe_conn_id_from_headers_or_cookie(request)
        conn_id = recovered or conn_id
        _safe_set(ui_req, "conn_id", conn_id)

    # Ensure session_id (deterministic UUIDv5) if missing
    session_id = getattr(ui_req, "session_id", None)
    if not session_id:
        org_id = getattr(ui_req, "org_id", None) or "anon-org"
        user_id = getattr(ui_req, "user_id", None)
        conn_or_user = conn_id or user_id or "anon"
        basis = f"workshops_ai/{org_id}/{conn_or_user}"
        session_id = uuid.uuid5(uuid.NAMESPACE_URL, basis).hex
        _safe_set(ui_req, "session_id", session_id)


async def _hydrate_from_ws_context(ui_req: UIRequest) -> None:
    """
    Best-effort: if the WS hub stores per-connection context (staging, accept_language, tab),
    load it and fill missing fields on UIRequest. Never overrides explicit client values.
    Safe to no-op when the hub is unavailable.
    """
    conn_id = getattr(ui_req, "conn_id", None)
    if not conn_id:
        return  # nothing to hydrate by

    try:
        # Lazy, optional import to avoid hard dependency
        from workshops_ai.app.services.ws_hub import get_client_context  # type: ignore
    except Exception:
        return

    try:
        ctx = await get_client_context(conn_id)  # expected to return a dict-like or None
    except Exception:
        ctx = None

    if not ctx:
        return

    # Do not overwrite explicit values coming from the caller
    if getattr(ui_req, "staging", None) is None and "staging" in ctx:
        try:
            _safe_set(ui_req, "staging", bool(ctx["staging"]))
        except Exception:
            pass

    if not getattr(ui_req, "active_tab", None):
        at = ctx.get("active_tab")
        if isinstance(at, str) and at:
            _safe_set(ui_req, "active_tab", at)

    if not getattr(ui_req, "language", None):
        al = ctx.get("accept_language")
        if isinstance(al, str) and al.strip():
            _safe_set(ui_req, "language", al.strip()[:256])


async def _lock_user_language(
    *,
    request: Request,
    ui_req: UIRequest,
    deps,  # GraphDeps
    ttl_sec: int = 1800,
) -> Optional[str]:
    """
    Compute a stable user language for this conversation and persist a short-lived lock in memory.

    Precedence:
      1) ui_req.language (explicit from client payload or WS context)
      2) HTTP header 'Accept-Language'
    If a lock already exists in memory, it wins over both (to prevent drift).

    Memory key:
      namespace='ux', key='lang', thread_id=deps.thread_id_for(ui_req)
    """
    # Candidate from payload (already normalized) or HTTP header
    lang_attr = getattr(ui_req, "language", None)
    candidate = lang_attr or _header_accept_language(request)
    candidate = (candidate.strip()[:256] if isinstance(candidate, str) else None)

    # Memory not guaranteed; guard each use
    mem_get: Optional[Callable[..., Any]] = getattr(deps, "memory_get", None)
    mem_set: Optional[Callable[..., Any]] = getattr(deps, "memory_set", None)

    locked: Optional[str] = None
    try:
        if callable(mem_get):
            locked = await mem_get(
                org_id=getattr(ui_req, "org_id", None),
                user_id=getattr(ui_req, "user_id", None),
                thread_id=deps.thread_id_for(ui_req),
                namespace="ux",
                key="lang",
            )
    except Exception:
        locked = None  # fail-open

    # Decide the effective language
    effective = locked or candidate  # may remain None if unknown

    # Persist the candidate only when no lock existed yet
    if (locked is None) and candidate and callable(mem_set):
        try:
            await mem_set(
                org_id=getattr(ui_req, "org_id", None),
                user_id=getattr(ui_req, "user_id", None),
                thread_id=deps.thread_id_for(ui_req),
                namespace="ux",
                key="lang",
                value=candidate,
                ttl_sec=int(ttl_sec),
            )
        except Exception:
            pass  # best-effort

    return effective


# ----------------------------------------------------------------------
# Route
# ----------------------------------------------------------------------
@router.post(
    "/chat",
    response_model=UIResponse,
    response_model_exclude_none=True,
    response_class=JSONResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat entrypoint: routes UIRequest to the LangGraph manager",
    tags=["chat"],
)
async def chat_endpoint(
    request: Request,
    payload: dict[str, Any] = Body(...),
) -> UIResponse:
    """
    Contract:
      UIRequest → (workshops_ai.graph.manager:route_chat) → UIResponse

    Safety guarantees:
    - Strict Pydantic v2 validation (fail-closed).
    - Message length guard (MAX_MESSAGE_LEN).
    - Unified timeout (CHAT_TIMEOUT_SEC).
    - Backward-compatible acceptance of legacy ChatIn (auto-bridged).
    - DI: manager receives GraphDeps via make_default_deps() with a centralized llm_client.
    - Language lock: Accept-Language (payload/header) persisted per user/thread (TTL ~30m).
    - Context hydration: try to source missing staging/active_tab/language from WS hub.
    - Fully defensive around optional UIRequest attributes (getattr + safe setters).
    """
    # 1) Validate inbound payload (UIRequest or legacy), with language normalization
    ui_req = _to_ui_request(payload)

    # 1.5) Extract auth identity from request.state (populated by security middleware)
    identity = getattr(request.state, "identity", None)
    # Note: identity is tracked in request.state, no need to copy to ui_req

    # 2) Message-length guard (only if message is provided)
    msg = getattr(ui_req, "message", None)
    if msg and len(msg) > MAX_MESSAGE_LEN:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"message": f"message too long (>{MAX_MESSAGE_LEN} chars)"},
        )

    # 3) Ensure conn_id/session_id are present (derive if missing)
    _ensure_session_and_conn(ui_req, request)

    # 4) Best-effort: hydrate missing fields from WS hub (staging, active_tab, language)
    try:
        await _hydrate_from_ws_context(ui_req)
    except Exception:
        # Hydration is best-effort; failure should not block the request
        pass

    # 5) Build DI (GraphDeps) with centralized llm_client and friends
    deps = make_default_deps(tab=getattr(ui_req, "active_tab", None))

    # 6) Determine + lock the user language (best-effort)
    try:
        locked_lang = await _lock_user_language(request=request, ui_req=ui_req, deps=deps, ttl_sec=1800)
        if locked_lang:
            _safe_set(ui_req, "language", locked_lang)
    except Exception:
        # If locking fails, we still proceed; manager may infer/respond
        pass

    # 7) Telemetry IN (best-effort)
    try:
        telemetry_log(
            "chat.request",
            {
                "user_id": getattr(ui_req, "user_id", None),
                "org_id": getattr(ui_req, "org_id", None),
                "active_tab": getattr(ui_req, "active_tab", None),
                "selection": getattr(ui_req, "selection", None),
                "staging": getattr(ui_req, "staging", None),
                "message_preview": _truncate_for_log(getattr(ui_req, "message", None), 256),
                "language": getattr(ui_req, "language", None),
                "locale": getattr(ui_req, "locale", None),
                "has_history_token": bool(getattr(ui_req, "history_token", None)),
                "has_org_profile": bool(getattr(ui_req, "organization_profile", None)),
                "has_services": bool(getattr(ui_req, "services", None)),
                "has_existing_assets": bool(getattr(ui_req, "existing_assets", None)),
                "has_website_content": bool(getattr(ui_req, "website_content", None)),
                "emit_changes_enabled": getattr(getattr(ui_req, "policy", None), "emit_changes_enabled", True),
                "source": "API.router",
            },
        )
    except Exception:
        pass

    # 8) Route strictly to the LangGraph manager (with unified timeout)
    try:
        from workshops_ai.graph.manager import route_chat  # lazy import to avoid import cycles

        # Prefer calling with DI if supported; else fallback for legacy manager signature
        if "deps" in inspect.signature(route_chat).parameters:
            result = await _run_with_timeout(route_chat(ui_req, deps=deps), timeout_sec=CHAT_TIMEOUT_SEC)
        else:  # pragma: no cover
            result = await _run_with_timeout(route_chat(ui_req), timeout_sec=CHAT_TIMEOUT_SEC)

        # 9) Validate outbound payload (strict → UIResponse)
        if isinstance(result, dict):
            ui_resp = UIResponse.model_validate(result)
        elif isinstance(result, UIResponse):
            ui_resp = result
        else:
            raise TypeError(f"route_chat returned unsupported type: {type(result)!r}")

    except asyncio.TimeoutError as te:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"message": "Chat routing timed out"},
        ) from te
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Response validation failed", "errors": ve.errors()},
        ) from ve
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Router error", "error": str(ex)},
        ) from ex

    # 10) Telemetry OUT (best-effort)
    try:
        telemetry_log(
            "chat.response",
            {
                "user_id": getattr(ui_req, "user_id", None),
                "org_id": getattr(ui_req, "org_id", None),
                "active_tab": getattr(ui_req, "active_tab", None),
                "messages_count": len(getattr(ui_resp, "messages", []) or []),
                "has_proposals": bool(getattr(ui_resp, "proposals", None)),
                "staging": getattr(ui_resp, "staging", None),
                "has_agent_events": bool(getattr(ui_resp, "agent_events", None)),
                "language_used": getattr(ui_req, "language", None),
                "source": "API.router",
            },
        )
    except Exception:
        pass

    return ui_resp
