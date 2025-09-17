# File: workshops_ai/app/routers/ws_router.py
from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel, ConfigDict, ValidationError

from workshops_ai.schemas.ui_contract import UIRequest
from workshops_ai.graph.manager import route_chat
from workshops_ai.app.services.ws_hub import WsHub  # WS hub for agent broadcasts

router = APIRouter()

# Allowed tabs (fail-closed)
ALLOWED_TABS = {"assets", "impacts", "threats", "controls"}

# Timeout for long-running actions (seconds)
WS_EVENT_TIMEOUT_S = int(os.getenv("WS_EVENT_TIMEOUT_S", "75"))


# ------------------------------ Inbound schema --------------------------------
class InboundEvent(BaseModel):
    """
    Unified inbound WS event schema.

    Types:
      - "ping"        : health check
      - "switch_tab"  : { active_tab: "assets|impacts|threats|controls" }
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
    )


# ----------------------------- WebSocket endpoint -----------------------------
@router.websocket("/ws")
async def ws_endpoint(
    ws: WebSocket,
    org_id: str = Query(..., description="Organization ID"),
    user_id: str = Query(..., description="User ID"),
    active_tab: str = Query("assets", description="assets|impacts|threats|controls"),
) -> None:
    """
    Real-time chat endpoint:
    - Strictly validates all inbound messages (Pydantic v2, fail-closed).
    - Prevents concurrent long-running actions per connection (busy-guard).
    - Wraps route_chat(...) with timeout and unified error frames.
    - Registers the socket in WsHub so agents can broadcast AgentEvent JSON frames.
    """
    # Normalize/validate the initial tab
    active_tab = active_tab if active_tab in ALLOWED_TABS else "assets"

    await ws.accept()

    # Register this client in the hub to receive agent broadcasts
    await WsHub.register(ws)

    state = Conn(org_id=org_id, user_id=user_id, active_tab=active_tab)

    async def send(event_type: str, payload: dict[str, Any] | list[Any] | None = None) -> None:
        await ws.send_json({"type": event_type, "payload": payload or {}})

    async def send_error(message: str, details: Any | None = None) -> None:
        await send("error", {"message": message, "details": details, "conn_id": state.conn_id})

    # Initial hello (+ info about broadcast mode)
    await send(
        "hello",
        {
            "ok": True,
            "active_tab": state.active_tab,
            "conn_id": state.conn_id,
            "org_id": state.org_id,
            "user_id": state.user_id,
            # The client will additionally receive raw AgentEvent JSON via WsHub.broadcast_json(...)
            "agent_events": {"subscribed": True, "mode": "raw_json"},
        },
    )

    try:
        while True:
            # Receive + validate
            try:
                raw = await ws.receive_json()
            except Exception as e:
                await send_error("Invalid JSON frame", str(e))
                continue

            try:
                evt = InboundEvent.model_validate(raw)
            except ValidationError as ve:
                await send_error("Invalid event payload", ve.errors())
                continue
            except Exception as e:
                await send_error("Invalid event payload", str(e))
                continue

            # Health check
            if evt.type == "ping":
                await send("pong", {"conn_id": state.conn_id})
                continue

            # Switch active tab (enforce allowed set)
            if evt.type == "switch_tab":
                new_tab = (evt.active_tab or "").strip().lower()
                if new_tab in ALLOWED_TABS:
                    state.active_tab = new_tab
                    await send("switch_tab_ack", {"active_tab": state.active_tab})
                else:
                    await send_error(f"Unsupported tab: {new_tab}", {"allowed": sorted(ALLOWED_TABS)})
                continue

            # Select asset
            if evt.type == "select":
                state.selected_asset_id = evt.asset_id or None
                await send("select_ack", {"asset_id": state.selected_asset_id})
                continue

            # Long-running actions (chat/export) — guarded + with timeout
            if evt.type in {"chat", "export"}:
                if state.busy:
                    await send_error("Busy: previous request is still running")
                    continue

                state.busy = True
                try:
                    text = ("export" if evt.type == "export" else (evt.message or "")).strip()
                    if not text:
                        await send_error("Empty message for 'chat' event")
                        continue

                    ui_req = _build_ui_request(state, text, staging=False)
                    await send("progress", {"status": "running", "conn_id": state.conn_id, "tab": state.active_tab})

                    ui_res = await asyncio.wait_for(route_chat(ui_req), timeout=WS_EVENT_TIMEOUT_S)

                    await send("ui_response", ui_res.model_dump(mode="json"))
                    await send("progress", {"status": "done", "conn_id": state.conn_id})
                except asyncio.TimeoutError:
                    await send_error(f"Timed out after {WS_EVENT_TIMEOUT_S}s")
                except WebSocketDisconnect:
                    # Client disconnected during processing; break to cleanup.
                    break
                except Exception as e:
                    await send_error("Unhandled error while processing request", str(e))
                finally:
                    state.busy = False
                continue

            # Fallback (should never happen due to strict validation)
            await send_error(f"Unknown event type: {evt.type}")

    except WebSocketDisconnect:
        # Client disconnected — fall through to finally for unregister.
        pass
    finally:
        # Always unregister this socket from the hub (best-effort).
        try:
            await WsHub.unregister(ws)
        except Exception:
            pass
