# File: workshops_ai/routers/ws_router.py
# -----------------------------------------------------------------------------
# Purpose
# WebSocket endpoint with strict event validation, centralized settings-based auth,
# timeout handling, and integration with WsHub for broadcasting AgentEvents.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, ValidationError

from workshops_ai.settings import get_settings
from workshops_ai.schemas.ui_contract import UIRequest
from workshops_ai.graph.manager import route_chat
from workshops_ai.app.security import verify_ws_auth  # unified WS auth (API-key/JWT)

# WS Hub (agent → client broadcasts)
try:
    from workshops_ai.app.services.ws_hub import WsHub
except Exception:  # pragma: no cover
    class WsHub:  # minimal no-op fallback
        @staticmethod
        async def register(ws: WebSocket) -> None: ...
        @staticmethod
        async def unregister(ws: WebSocket) -> None: ...

# WS rate-limit helper (e.g., Redis-based)
try:
    from workshops_ai.app.middlewares.rate_limit import allow_ws
except Exception:  # pragma: no cover
    async def allow_ws(event: str, key: str, per_minute: int) -> bool:
        return True

# Best-effort request_id getter (HTTP layer populates a ContextVar)
try:
    from workshops_ai.app.main import get_request_id  # returns "n/a" if missing
except Exception:  # pragma: no cover
    def get_request_id() -> str:  # type: ignore
        return "n/a"

router = APIRouter()
logger = logging.getLogger(__name__)

# Allowed tabs (fail-closed; aligned with available graphs)
ALLOWED_TABS = {"assets", "impacts", "threats"}

# Centralized settings
_settings = get_settings()
# Correct namespaced read (previous flat attr caused mismatches):
WS_EVENT_TIMEOUT_S: int = int(_settings.ws.timeout_sec)

# Cookie name used by SessionMiddleware for session id
COOKIE_NAME = "wsai.sid"


# ------------------------------ Inbound schema --------------------------------
class InboundEvent(BaseModel):
    """
    Unified inbound WS event schema.

    Types:
      - "ping"        : health check
      - "switch_tab"  : { active_tab: "assets|impacts|threats" }
      - "select"      : { asset_id: "..." }
      - "chat"        : { message: "..." }
      - "export"      : no payload (synthetic "export" message)

    Out-of-band:
      - Agent change events (create/set_meta) are broadcast via WsHub and delivered
        to the client as raw JSON frames (AgentEvent) with no envelope.
    """
    type: Literal["select", "chat", "export", "ping", "switch_tab"]
    message: str | None = None
    asset_id: str | None = None
    active_tab: str | None = None

    model_config = ConfigDict(extra="forbid")


# ------------------------------ Connection state ------------------------------
@dataclass
class Conn:
    org_id: str
    user_id: str
    active_tab: str
    selected_asset_id: str | None = None
    conn_id: str = uuid.uuid4().hex
    busy: bool = False  # simple anti-reentrancy guard
    # Session id (from cookie); may be None if cookie wasn't present
    sid: Optional[str] = None
    # Correlation id captured at connect time (echoed in error frames)
    request_id: str = "n/a"


# --------------------------------- Helpers ------------------------------------
def _build_ui_request(state: Conn, message: str, *, staging: bool = False) -> UIRequest:
    """Build UIRequest for the Manager/Router based on current connection state."""
    return UIRequest(
        org_id=state.org_id,
        active_tab=state.active_tab,
        selection={"asset_id": state.selected_asset_id} if state.selected_asset_id else {},
        message=message,
        staging=staging,
        user_id=state.user_id,
        session_id=state.sid,  # forward session id to the agent
        # language can be injected later if you decide to store it in session
    )


async def _ws_send(ws: WebSocket, event_type: str, payload: dict[str, Any] | list[Any] | None = None) -> None:
    try:
        await ws.send_json({"type": event_type, "payload": payload or {}})
    except WebSocketDisconnect:
        return
    except RuntimeError as e:
        if "Unexpected ASGI message" in str(e) or "once a close message has been sent" in str(e):
            return
        raise


def _cur_request_id(state: Optional[Conn]) -> str:
    """
    Resolve the current request id for correlation:
    1) prefer state.request_id captured at connect time
    2) fallback to ContextVar getter from the HTTP layer (may be "n/a")
    """
    if state and state.request_id and state.request_id != "n/a":
        return state.request_id
    return get_request_id() or "n/a"


async def _ws_error(
    ws: WebSocket,
    message: str,
    *,
    conn_id: Optional[str] = None,
    details: Any | None = None,
    state: Optional[Conn] = None,
) -> None:
    """
    Unified WS error frame:
    {
      "type":"error",
      "payload": {
        "message": "...",
        "details": ...,
        "conn_id": "...",
        "request_id": "..."
      }
    }
    """
    await _ws_send(
        ws,
        "error",
        {
            "message": message,
            "details": details,
            "conn_id": conn_id,
            "request_id": _cur_request_id(state),
        },
    )


# ----------------------------- WebSocket endpoint -----------------------------
@router.websocket("/ws")
async def ws_endpoint(
    ws: WebSocket,
    org_id: str = Query(..., description="Organization ID"),
    user_id: str = Query(..., description="User ID"),
    active_tab: str = Query("assets", description="assets|impacts|threats"),
) -> None:
    """
    Real-time chat endpoint:
    - Security: API key or JWT via verify_ws_auth (query ?api_key=... or header X-API-Key / Authorization: Bearer).
    - Strictly validates all inbound messages (Pydantic v2, fail-closed).
    - Prevents concurrent long-running actions per connection (busy-guard).
    - Wraps route_chat(...) with timeout and unified error frames (including conn_id & request_id).
    - Registers the socket in WsHub so agents can broadcast AgentEvent JSON frames.
    - Applies rate-limit per session (sid) for "chat" and "export" events.
    """
    # Normalize/validate the initial tab
    active_tab = active_tab if active_tab in ALLOWED_TABS else "assets"

    # Security validation BEFORE accept() to reject early if unauthorized
    auth_ok = await verify_ws_auth(ws)
    if not auth_ok:   
        return
    # Accept the connection after passing security
    await ws.accept()

    # Register this client in the hub to receive agent broadcasts
    await WsHub.register(ws)

    # Try to read session id (set by HTTP SessionMiddleware)
    sid = ws.cookies.get(COOKIE_NAME)

    # Capture correlation id right after we accept
    rid = get_request_id() or "n/a"

    state = Conn(org_id=org_id, user_id=user_id, active_tab=active_tab, sid=sid, request_id=rid)

    # Initial hello (+ info about broadcast mode)
    await _ws_send(
        ws,
        "hello",
        {
            "ok": True,
            "active_tab": state.active_tab,
            "conn_id": state.conn_id,
            "request_id": state.request_id,
            "org_id": state.org_id,
            "user_id": state.user_id,
            "sid": state.sid,
            "agent_events": {"subscribed": True, "mode": "raw_json"},
        },
    )

    try:
        while True:
            # Receive + validate
            try:
                raw = await ws.receive_json()
            except Exception as e:
                code = getattr(e, "code", None)
                if code in (1000, 1006):
                    logger.info("WS client disconnected (%s)", code)
                    return
                logger.warning("WS receive error: %s", e)
                continue

            try:
                evt = InboundEvent.model_validate(raw)
            except ValidationError as ve:
                await _ws_error(ws, "Invalid event payload", conn_id=state.conn_id, details=ve.errors(), state=state)
                continue
            except Exception as e:
                await _ws_error(ws, "Invalid event payload", conn_id=state.conn_id, details=str(e), state=state)
                continue

            # Health check
            if evt.type == "ping":
                await _ws_send(ws, "pong", {"conn_id": state.conn_id, "request_id": state.request_id})
                continue

            # Switch active tab (enforce allowed set)
            if evt.type == "switch_tab":
                new_tab = (evt.active_tab or "").strip().lower()
                if new_tab in ALLOWED_TABS:
                    state.active_tab = new_tab
                    await _ws_send(ws, "switch_tab_ack", {"active_tab": state.active_tab, "request_id": state.request_id})
                else:
                    await _ws_error(
                        ws,
                        f"Unsupported tab: {new_tab}",
                        conn_id=state.conn_id,
                        details={"allowed": sorted(ALLOWED_TABS)},
                        state=state,
                    )
                continue

            # Select asset
            if evt.type == "select":
                state.selected_asset_id = evt.asset_id or None
                await _ws_send(ws, "select_ack", {"asset_id": state.selected_asset_id, "request_id": state.request_id})
                continue

            # Long-running actions (chat/export) — guarded + with timeout
            if evt.type in {"chat", "export"}:
                # WS rate-limit per minute (uses session sid; fallback to anon:conn)
                if getattr(_settings, "ratelimit", None) and getattr(_settings.ratelimit, "enabled", False):
                    event_for_rl = "chat" if evt.type == "chat" else "export"
                    sid_for_rl = state.sid or f"anon:{state.conn_id[:8]}"
                    try:
                        ok = await allow_ws(event_for_rl, sid_for_rl, _settings.ratelimit.per_minute)
                    except Exception as rl_err:  # explicit; don't fail-open
                        await _ws_error(
                            ws,
                            "Rate-limit check failed",
                            conn_id=state.conn_id,
                            details=str(rl_err),
                            state=state,
                        )
                        continue
                    if not ok:
                        await _ws_error(ws, "Too many messages", conn_id=state.conn_id, state=state)
                        continue

                if state.busy:
                    await _ws_error(ws, "Busy: previous request is still running", conn_id=state.conn_id, state=state)
                    continue

                state.busy = True
                try:
                    text = ("export" if evt.type == "export" else (evt.message or "")).strip()
                    if evt.type == "chat" and not text:
                        await _ws_error(ws, "Empty message for 'chat' event", conn_id=state.conn_id, state=state)
                        continue

                    ui_req = _build_ui_request(state, text, staging=False)

                    await _ws_send(
                        ws,
                        "progress",
                        {
                            "status": "running",
                            "conn_id": state.conn_id,
                            "request_id": state.request_id,
                            "tab": state.active_tab,
                        },
                    )

                    ui_res = await asyncio.wait_for(route_chat(ui_req), timeout=WS_EVENT_TIMEOUT_S)

                    # Support both Pydantic model and plain dict
                    payload = ui_res.model_dump(mode="json") if hasattr(ui_res, "model_dump") else ui_res
                    await _ws_send(ws, "ui_response", payload)
                    await _ws_send(ws, "progress", {"status": "done", "conn_id": state.conn_id, "request_id": state.request_id})
                except asyncio.TimeoutError:
                    await _ws_error(ws, f"Timed out after {WS_EVENT_TIMEOUT_S}s", conn_id=state.conn_id, state=state)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    await _ws_error(
                        ws,
                        "Unhandled error while processing request",
                        conn_id=state.conn_id,
                        details=str(e),
                        state=state,
                    )
                finally:
                    state.busy = False
                continue

            # Fallback (should never happen due to strict validation)
            await _ws_error(ws, f"Unknown event type: {evt.type}", conn_id=state.conn_id, state=state)

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await WsHub.unregister(ws)
        except Exception:
            pass
