# File: workshops_ai/api/app.py
# -----------------------------------------------------------------------------
# Purpose
# Auth-protected HTTP API router for chat (+ mounts /rag):
#   - POST /chat  → validates UIRequest and routes to LangGraph manager (route_chat)
#   - /rag/*      → mounts ingest/purge endpoints from rag_router (if available)
#
# Guarantees
# - Strict Pydantic validation (fail-closed).
# - Unified timeout from settings.chat.timeout_sec.
# - Consistent UIResponse shape (re-validated on output).
# - Best-effort telemetry; safe error handling.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from workshops_ai.settings import get_settings
from workshops_ai.schemas.ui_contract import UIRequest, UIResponse

# Centralized HTTP auth (API-key/JWT).
# Keep import location consistent with the rest of the app.
try:
    from workshops_ai.app.security import require_auth  # ✅ correct import
except Exception as _imp_err:  # pragma: no cover
    # If security middleware isn't available at import time, make it explicit.
    raise RuntimeError("Auth dependency 'require_auth' not available") from _imp_err

# Telemetry (best-effort, no-op if missing)
try:
    from workshops_ai.tools.telemetry import telemetry_log  # type: ignore
except Exception:  # pragma: no cover
    def telemetry_log(event: str, payload: dict) -> None:  # type: ignore
        return

# Optional centralized guard; if missing, fall back to native .model_validate()
try:
    from workshops_ai.tools.guard_validate_payload import guard_validate_payload  # type: ignore
except Exception:  # pragma: no cover
    guard_validate_payload = None  # type: ignore[misc,assignment]

# -----------------------------------------------------------------------------
# Settings (centralized)
# -----------------------------------------------------------------------------
settings = get_settings()

# Align with the rest of the codebase: use nested `settings.chat.*`
CHAT_TIMEOUT_SEC: float = float(getattr(getattr(settings, "chat", object()), "timeout_sec", 60))
MAX_MESSAGE_LEN: int = int(getattr(getattr(settings, "chat", object()), "max_message_len", 4000))

# Auth applies to all routes mounted under this router
router = APIRouter(dependencies=[Depends(require_auth)])


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
class ChatAcceptedResponse(UIResponse):
    """Alias kept for OpenAPI examples; the route returns UIResponse directly."""


def _read_request_id(request: Request) -> str:
    """
    Read request id injected by RequestIDMiddleware. Fallback to 'n/a' if missing.
    Supports common header names (lower/upper).
    """
    for h in ("x-request-id", "x-correlation-id"):
        rid = request.headers.get(h) or request.headers.get(h.upper())
        if rid:
            return rid.strip()
    return "n/a"


def _truncate_for_log(text: Optional[str], limit: int = 512) -> Optional[str]:
    """Truncate long strings for telemetry logs (keeps logs bounded)."""
    if text is None:
        return None
    return text if len(text) <= limit else (text[:limit] + "…")


async def _run_with_timeout(awaitable_or_result: Any, timeout_sec: float) -> Any:
    """
    Await either a coroutine/future or return a sync result with a timeout.
    If a sync callable leaked, run it in a thread pool.
    """
    if inspect.isawaitable(awaitable_or_result):
        return await asyncio.wait_for(awaitable_or_result, timeout=timeout_sec)
    if callable(awaitable_or_result):
        return await asyncio.wait_for(asyncio.to_thread(awaitable_or_result), timeout=timeout_sec)
    return awaitable_or_result


# -----------------------------------------------------------------------------
# Route
# -----------------------------------------------------------------------------
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
    payload: dict[str, Any],
) -> UIResponse:
    """
    Contract:
      UIRequest → Manager/Router (LangGraph) → UIResponse

    Security/validation:
    - Fail-closed: Pydantic v2 strict parsing (or optional centralized guard).
    - Message length capped (MAX_MESSAGE_LEN).
    - Telemetry audited with user_id, org_id, active_tab, and request_id.
    """
    request_id = _read_request_id(request)

    # 1) Validate inbound payload (strict)
    try:
        if guard_validate_payload:
            ui_req = guard_validate_payload(UIRequest, payload)  # type: ignore[operator]
        else:
            ui_req = UIRequest.model_validate(payload)
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Invalid UIRequest", "errors": ve.errors(), "request_id": request_id},
        ) from ve

    # 2) Soft guard: message size
    if ui_req.message and len(ui_req.message) > MAX_MESSAGE_LEN:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"message": f"message too long (>{MAX_MESSAGE_LEN} chars)", "request_id": request_id},
        )

    # 3) Telemetry (best-effort)
    try:
        sel = getattr(ui_req, "selection", None)
        sel_dump = sel.model_dump(mode="json") if hasattr(sel, "model_dump") else (sel or None)
        telemetry_log(
            "chat.request",
            {
                "request_id": request_id,
                "user_id": ui_req.user_id,
                "org_id": ui_req.org_id,
                "active_tab": ui_req.active_tab,
                "selection": sel_dump,
                "staging": ui_req.staging,
                "message_preview": _truncate_for_log(ui_req.message, 256),
                "locale": getattr(ui_req, "locale", None),
                "emit_changes_enabled": getattr(getattr(ui_req, "policy", None), "emit_changes_enabled", True),
                "source": "API",
            },
        )
    except Exception:
        # Telemetry must never break the request
        pass

    # 4) Route strictly via our component (Manager/Router)
    try:
        # Local import avoids early import cycles at startup
        from workshops_ai.graph.manager import route_chat

        # route_chat accepts UIRequest; call it and guard with timeout
        result = await _run_with_timeout(route_chat(ui_req), timeout_sec=CHAT_TIMEOUT_SEC)

        # 5) Validate outbound response (strict)
        if guard_validate_payload:
            ui_resp = guard_validate_payload(UIResponse, result)  # type: ignore[operator]
        else:
            if isinstance(result, dict):
                ui_resp = UIResponse.model_validate(result)
            elif isinstance(result, UIResponse):
                ui_resp = result
            else:
                raise TypeError(f"route_chat returned unsupported type: {type(result)!r}")

    except asyncio.TimeoutError as te:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"message": "Chat routing timed out", "request_id": request_id},
        ) from te
    except HTTPException:
        # Already shaped; re-raise
        raise
    except ValidationError as ve:
        # Outgoing strict validation failed → internal contract mismatch
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Response validation failed", "errors": ve.errors(), "request_id": request_id},
        ) from ve
    except Exception as ex:
        # Unknown internal error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Router error", "error": str(ex), "request_id": request_id},
        ) from ex

    # 6) Telemetry (best-effort)
    try:
        telemetry_log(
            "chat.response",
            {
                "request_id": request_id,
                "user_id": ui_req.user_id,
                "org_id": ui_req.org_id,
                "active_tab": ui_req.active_tab,
                "messages_count": len(ui_resp.messages or []),
                "has_proposals": bool(getattr(ui_resp, "proposals", None)),
                "staging": getattr(ui_resp, "staging", None),
                "has_agent_events": bool(getattr(ui_resp, "agent_events", None)),
                "source": "API",
            },
        )
    except Exception:
        pass

    return ui_resp


# -----------------------------------------------------------------------------
# Mount /rag (ingest/purge)
# -----------------------------------------------------------------------------
# Prefer mounting the dedicated RAG router if available. It should define endpoints
# like POST /ingest and POST /purge without a prefix; we mount it here under /rag.
try:
    from workshops_ai.app.routers.rag_router import router as rag_router  # type: ignore
    router.include_router(rag_router, prefix="/rag", tags=["rag"])
except Exception:
    # If the RAG router is not yet implemented, keep chat API operational.
    pass
