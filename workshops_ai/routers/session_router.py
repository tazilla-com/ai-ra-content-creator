# File: workshops_ai/routers/session_router.py
# Purpose: Session management endpoints (state retrieval and deletion).

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from workshops_ai.wizard.session import (
    SESSION_ID_PATTERN,
    clear_wizard_state,
    get_wizard_state,
)

logger = logging.getLogger("workshops_ai.session_router")

router = APIRouter(
    prefix="/session",
    tags=["Session"],
)


# ----------------------------- Dependency -------------------------------


async def get_session_id(
    request: Request,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
) -> str:
    """
    Extract and validate session_id from X-Session-ID header or query param.
    """
    sid = x_session_id or request.query_params.get("session_id", "").strip()
    if not sid or not SESSION_ID_PATTERN.match(sid):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format. Expected: 1-40 alphanumeric characters (a-z, A-Z, 0-9, _, -)",
        )
    return sid


# ----------------------------- Models -----------------------------------


class SessionStateResponse(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    state: dict[str, Any] = Field(..., description="Current session state")


class SessionDeleteResponse(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    deleted: bool = Field(..., description="Whether the session was deleted")


# ----------------------------- Endpoints --------------------------------


@router.get(
    "/",
    response_model=SessionStateResponse,
    summary="Get session state",
    description="Retrieve the current wizard state for a session",
)
async def get_session_state(
    session_id: str = Header(..., alias="X-Session-ID"),
) -> SessionStateResponse:
    if not SESSION_ID_PATTERN.match(session_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format. Expected: 1-40 alphanumeric characters (a-z, A-Z, 0-9, _, -)",
        )

    state = await get_wizard_state(session_id)
    return SessionStateResponse(session_id=session_id, state=state)


@router.delete(
    "/",
    response_model=SessionDeleteResponse,
    summary="Delete session",
    description="Delete the session and all associated data from storage",
)
async def delete_session_state(
    session_id: str = Header(..., alias="X-Session-ID"),
) -> SessionDeleteResponse:
    if not SESSION_ID_PATTERN.match(session_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format. Expected: 1-40 alphanumeric characters (a-z, A-Z, 0-9, _, -)",
        )

    await clear_wizard_state(session_id)
    return SessionDeleteResponse(session_id=session_id, deleted=True)
