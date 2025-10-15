# File: workshops_ai/routers/chat_router.py
from __future__ import annotations

"""
Chat Router (UIRequest → route_chat → UIResponse)
-------------------------------------------------
- Auth-protected POST /api/chat endpoint used by the UI.
- Strictly validates incoming payload as UIRequest; falls back to legacy ChatIn.
- Returns a strictly validated UIResponse.
- Calls the unified LangGraph manager with DI built via graph.factory.make_default_deps()
  so the manager/graphs receive a centralized `llm_client` and related deps.
- Single timeout from settings.chat.timeout_sec.
- Accept-Language handling:
    • Prefer explicit UIRequest.language
    • else payload.accept_language
    • else HTTP header "Accept-Language"
  The chosen value is *session-locked* (Redis) for this user/thread (TTL ~30 min) so the
  model does not drift between CS/EN/SK mid-conversation.
"""

import asyncio
import inspect
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status, Body
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from workshops_ai.settings import get_settings
from workshops_ai.schemas.ui_contract import UIRequest, UIResponse

# Security dependency (API key / JWT)
try:
    from workshops_ai.app.security import require_auth  # preferred path
    _AUTH_DEPS = [Depends(require_auth)]
except Exception:  # pragma: no cover
    try:
        # Optional legacy fallback
        from workshops_ai.app.deps.auth import require_auth  # type: ignore
        _AUTH_DEPS = [Depends(require_auth)]
    except Exception:  # pragma: no cover
        _AUTH_DEPS = []

# Telemetry (best-effort, optional)
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
# Helpers
# ----------------------------------------------------------------------
def _truncate_for_log(text: str | None, limit: int = 512) -> str | None:
    """Compact long strings for telemetry to avoid PII/log bloat."""
    if text is None:
        return None
    return text if len(text) <= limit else (text[:limit] + "…")


def _normalize_language(payload: dict[str, Any]) -> None:
    """
    Accept `accept_language` (from UI/browser) and map it to UIRequest.language.
    Drop unknown key to satisfy UIRequest(extra='forbid').

    NOTE: We only set language from payload here. HTTP header fallback is applied
    after we have a validated UIRequest (see _lock_user_language).
    """
    al = payload.get("accept_language")
    if isinstance(al, str) and al.strip():
        # Do not overwrite explicit `language` if client provided it already
        payload.setdefault("language", al.strip()[:256])
    # Remove unrecognized key to avoid 422
    if "accept_language" in payload:
        payload.pop("accept_language", None)


def _header_accept_language(request: Request) -> Optional[str]:
    """Extract the HTTP 'Accept-Language' header (trimmed, capped)."""
    h = request.headers.get("accept-language")
    if isinstance(h, str):
        h = h.strip()
        if h:
            return h[:256]
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


def _to_ui_request(payload: dict[str, Any]) -> UIRequest:
    """
    Prefer the new UIRequest; if that fails, try legacy ChatIn and bridge.
    Also normalize language hints before validation.
    """
    # Normalize language hint (accept_language → language)
    _normalize_language(payload)

    try:
        return UIRequest.model_validate(payload)
    except ValidationError:
        # Legacy bridge (schemas/chat.py) — optional and backward-compatible
        try:
            from workshops_ai.schemas.chat import ChatIn  # lazy import
            legacy = ChatIn.model_validate(payload)
            return legacy.to_ui()
        except Exception as e2:
            # Re-raise a unified 422 so clients get a consistent error shape
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"message": "Invalid UIRequest (and legacy ChatIn fallback failed)"},
            ) from e2


async def _lock_user_language(
    *,
    request: Request,
    ui_req: UIRequest,
    deps,  # GraphDeps
    ttl_sec: int = 1800,
) -> str | None:
    """
    Compute a stable user language for this conversation and persist a short-lived lock in memory.

    Precedence:
      1) ui_req.language (explicit from client payload)
      2) HTTP header 'Accept-Language'
    If a lock already exists in memory, it wins over both (to prevent drift).

    Memory key:
      namespace='ux', key='lang', thread_id=deps.thread_id_for(ui_req)
    """
    # Candidate from payload (already normalized) or HTTP header
    candidate = (ui_req.language or _header_accept_language(request))
    candidate = (candidate.strip()[:256] if isinstance(candidate, str) else None)

    # Memory not guaranteed; guard each use
    mem_get = getattr(deps, "memory_get", None)
    mem_set = getattr(deps, "memory_set", None)

    locked: Optional[str] = None
    try:
        if callable(mem_get):
            locked = await mem_get(
                org_id=ui_req.org_id,
                user_id=ui_req.user_id,
                thread_id=deps.thread_id_for(ui_req),
                namespace="ux",
                key="lang",
            )
    except Exception:
        locked = None  # fail-open

    # Decide the effective language
    effective = locked or candidate

    # If nothing known yet, you can choose a safe default (optional).
    # Keeping None here means the manager/graphs may decide based on policy.
    # If you prefer a default, uncomment one of the lines below:
    # effective = effective or "cs"
    # effective = effective or "en"

    # Persist the candidate only when no lock existed yet
    if (locked is None) and candidate and callable(mem_set):
        try:
            await mem_set(
                org_id=ui_req.org_id,
                user_id=ui_req.user_id,
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

    Safety:
    - Strict Pydantic v2 validation (fail-closed).
    - Message length guard (MAX_MESSAGE_LEN).
    - Unified timeout (CHAT_TIMEOUT_SEC).
    - Backward-compatible acceptance of legacy ChatIn (auto-bridged).
    - DI: manager receives GraphDeps via make_default_deps() with a centralized llm_client.
    - Language lock: Accept-Language (payload/header) is persisted per user/thread (TTL ~30m).
    """
    # 1) Validate inbound payload (UIRequest or legacy), with language normalization
    ui_req = _to_ui_request(payload)

    # 2) Message-length guard
    if ui_req.message and len(ui_req.message) > MAX_MESSAGE_LEN:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"message": f"message too long (>{MAX_MESSAGE_LEN} chars)"},
        )

    # 3) Build DI (GraphDeps) with centralized llm_client and friends
    deps = make_default_deps(tab=ui_req.active_tab)

    # 4) Determine + lock the user language (best-effort)
    try:
        locked_lang = await _lock_user_language(request=request, ui_req=ui_req, deps=deps, ttl_sec=1800)
        if locked_lang:
            # Mutate the request object so manager/graphs get a stable hint
            ui_req.language = locked_lang
    except Exception:
        # If locking fails, we still proceed; manager may infer/respond
        pass

    # 5) Telemetry IN (best-effort)
    try:
        telemetry_log(
            "chat.request",
            {
                "user_id": ui_req.user_id,
                "org_id": ui_req.org_id,
                "active_tab": ui_req.active_tab,
                "selection": getattr(ui_req, "selection", None),
                "staging": ui_req.staging,
                "message_preview": _truncate_for_log(ui_req.message, 256),
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

    # 6) Route strictly to the LangGraph manager
    try:
        from workshops_ai.graph.manager import route_chat  # lazy import to avoid import cycles

        # Prefer calling with DI if supported; else fallback (maintains backward compatibility)
        if "deps" in inspect.signature(route_chat).parameters:
            result = await _run_with_timeout(route_chat(ui_req, deps=deps), timeout_sec=CHAT_TIMEOUT_SEC)
        else:
            # Legacy manager signature without DI (not recommended)
            result = await _run_with_timeout(route_chat(ui_req), timeout_sec=CHAT_TIMEOUT_SEC)

        # 7) Validate outbound payload (strict → UIResponse)
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

    # 8) Telemetry OUT (best-effort)
    try:
        telemetry_log(
            "chat.response",
            {
                "user_id": ui_req.user_id,
                "org_id": ui_req.org_id,
                "active_tab": ui_req.active_tab,
                "messages_count": len(ui_resp.messages or []),
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
