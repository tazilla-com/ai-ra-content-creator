# File: workshops_ai/routers/chat_router.py
from __future__ import annotations

"""
Chat Router (UIRequest → route_chat → UIResponse)
-------------------------------------------------
- Auth-protected POST /chat endpoint used by the UI.
- Strictly validates incoming payload as UIRequest and returns UIResponse.
- Calls the unified LangGraph manager (workshops_ai.graph.manager:route_chat).
- Single timeout from settings.chat.timeout_sec.
- Best-effort telemetry; safe error mapping.
"""

import asyncio
import inspect
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from workshops_ai.settings import get_settings
from workshops_ai.schemas.ui_contract import UIRequest, UIResponse

# --- Security dependency (API key / JWT) --------------------------------------
# Prefer the central path; fall back to legacy; dev may run without auth.
try:
    from workshops_ai.app.security import require_auth  # type: ignore[reportMissingImports]
    _AUTH_DEPS = [Depends(require_auth)]
except Exception:  # pragma: no cover
    try:
        from workshops_ai.app.deps.auth import require_auth  # type: ignore[reportMissingImports]
        _AUTH_DEPS = [Depends(require_auth)]
    except Exception:  # pragma: no cover
        _AUTH_DEPS = []

# --- Telemetry (best-effort, optional) ----------------------------------------
try:
    from workshops_ai.tools.telemetry import telemetry_log  # type: ignore
except Exception:  # pragma: no cover
    def telemetry_log(event: str, payload: dict) -> None:  # type: ignore
        return

settings = get_settings()
router = APIRouter(dependencies=_AUTH_DEPS)

CHAT_TIMEOUT_SEC: float = float(getattr(getattr(settings, "chat", object()), "timeout_sec", 60))
MAX_MESSAGE_LEN: int = int(getattr(getattr(settings, "chat", object()), "max_message_len", 4000))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _truncate_for_log(text: str | None, limit: int = 512) -> str | None:
    if text is None:
        return None
    return text if len(text) <= limit else (text[:limit] + "…")


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
    payload: dict[str, Any],
) -> UIResponse:
    """
    Contract:
      UIRequest → (workshops_ai.graph.manager:route_chat) → UIResponse

    Safety:
    - Strict Pydantic v2 validation (fail-closed).
    - Message length limit (MAX_MESSAGE_LEN).
    - Unified timeout (CHAT_TIMEOUT_SEC).
    """
    # 1) Validate inbound payload (strict → UIRequest)
    try:
        ui_req = UIRequest.model_validate(payload)
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Invalid UIRequest", "errors": ve.errors()},
        ) from ve

    # 2) Message-length guard
    if ui_req.message and len(ui_req.message) > MAX_MESSAGE_LEN:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"message": f"message too long (>{MAX_MESSAGE_LEN} chars)"},
        )

    # 3) Telemetry (best-effort)
    try:
        telemetry_log(
            "chat.request",
            {
                "user_id": ui_req.user_id,
                "org_id": ui_req.org_id,
                "active_tab": ui_req.active_tab,
                "selection": ui_req.selection.model_dump(mode="json") if hasattr(ui_req, "selection") else None,
                "staging": ui_req.staging,
                "message_preview": _truncate_for_log(ui_req.message, 256),
                "language": getattr(ui_req, "language", None),
                "number_of_assets": getattr(ui_req, "number_of_assets", None),
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

    # 4) Route strictly to the LangGraph manager
    try:
        from workshops_ai.graph.manager import route_chat  # lazy import to avoid early import cycles
        result = await _run_with_timeout(route_chat(ui_req), timeout_sec=CHAT_TIMEOUT_SEC)

        # 5) Validate outbound payload (strict → UIResponse)
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

    # 6) Telemetry (best-effort)
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
                "source": "API.router",
            },
        )
    except Exception:
        pass

    return ui_resp
