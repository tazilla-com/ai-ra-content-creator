# File: workshops_ai/api/app.py

from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.tools.telemetry import telemetry_log

try:
    # Optional centralized guard; if missing, we fall back to native .model_validate().
    from workshops_ai.tools.guard_validate_payload import guard_validate_payload  # type: ignore
except Exception:  # pragma: no cover
    guard_validate_payload = None  # type: ignore[misc,assignment]

CHAT_TIMEOUT_SEC: float = float(os.getenv("CHAT_TIMEOUT_SEC", "20"))
MAX_MESSAGE_LEN: int = int(os.getenv("MAX_MESSAGE_LEN", "4000"))

router = APIRouter()


class ChatAcceptedResponse(BaseModel):
    """Wrapper for OpenAPI examples; the real route returns UIResponse directly."""
    ok: bool = True
    data: UIResponse


def _read_request_id(request: Request) -> str:
    # Mirror headers set by RequestIDMiddleware; generate fallback correlation if missing
    for h in ("x-request-id", "x-correlation-id"):
        rid = request.headers.get(h) or request.headers.get(h.upper())
        if rid:
            return rid.strip()
    return "n/a"  # main middleware will inject a proper id on the way out


def _truncate_for_log(text: str | None, limit: int = 512) -> str | None:
    if text is None:
        return None
    return text if len(text) <= limit else (text[:limit] + "…")


async def _run_with_timeout(awaitable_or_result: Any, timeout_sec: float) -> Any:
    """
    Await either a coroutine/future or return a sync result with a timeout.
    If a sync value is passed (already computed result), it is just returned.
    If a sync callable leaked, wrap in a thread.
    """
    if inspect.isawaitable(awaitable_or_result):
        return await asyncio.wait_for(awaitable_or_result, timeout=timeout_sec)
    if callable(awaitable_or_result):
        return await asyncio.wait_for(asyncio.to_thread(awaitable_or_result), timeout=timeout_sec)
    return awaitable_or_result


@router.post(
    "/chat",
    response_model=UIResponse,
    response_model_exclude_none=True,
    response_class=JSONResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat entrypoint: routes UIRequest to the proper domain agent",
    tags=["chat"],
)
async def chat_endpoint(request: Request, payload: dict[str, Any]) -> UIResponse:
    """
    Contract:
    UIRequest → Manager/Router → UIResponse

    Security/validation:
    - Fail-closed: Pydantic v2 strict parsing (or optional guard).
    - Message length capped (MAX_MESSAGE_LEN).
    - Telemetry audited with userId, org_id, active_tab, and request_id.
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
            detail={
                "message": f"message too long (>{MAX_MESSAGE_LEN} chars)",
                "request_id": request_id,
            },
        )

    # 3) Telemetry (best-effort)
    try:
        telemetry_log(
            "chat.request",
            {
                "request_id": request_id,
                "user_id": ui_req.user_id,
                "org_id": ui_req.org_id,
                "active_tab": ui_req.active_tab,
                "selection": ui_req.selection,
                "staging": ui_req.staging,
                "message_preview": _truncate_for_log(ui_req.message, 256),
                "source": "API",
            },
        )
    except Exception:
        pass

    # 4) Route strictly via our component (Manager/Router)
    try:
        # Local import avoids early import cycles at startup
        from workshops_ai.graph.manager import route_chat

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
            detail={
                "message": "Response validation failed",
                "errors": ve.errors(),
                "request_id": request_id,
            },
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
                "source": "API",
            },
        )
    except Exception:
        pass

    return ui_resp
