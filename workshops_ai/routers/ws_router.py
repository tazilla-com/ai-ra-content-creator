# File: workshops_ai/routers/ws_router.py
# -----------------------------------------------------------------------------
# Purpose
# WebSocket endpoint with strict event validation, centralized settings-based auth,
# timeout handling, and integration with WsHub for broadcasting AgentEvents.
#
# Adds (language stability with minimal surface area):
#   - Deterministic Accept-Language resolution (query param > header > default).
#   - Normalization to ISO primary tag (e.g., "cs-CZ, cs" -> "cs") for downstream use.
#   - Per-connection "language lock": the chosen language (ISO) is stored in Conn
#     and sent on every UIRequest.language so agents keep replying in that language.
#   - "hello" payload now exposes both the canonical tag (accept_language) and the
#     normalized ISO ("lang") so the UI can display/debug what was chosen.
#
# NOTE:
# - No "force language" switch here on purpose. Language follows the user and then
#   stays stable during the connection (no drift), which is exactly the intended
#   behavior.
# - All other behavior (rate-limit, staging, deps wiring) remains unchanged.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, ValidationError

from workshops_ai.settings import get_settings
from workshops_ai.schemas.ui_contract import UIRequest
from workshops_ai.graph.manager import route_chat
from workshops_ai.graph.factory import make_default_deps
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
ALLOWED_TABS = {"assets", "impacts", "threats", "controls"}

# Centralized settings
_settings = get_settings()
WS_EVENT_TIMEOUT_S: int = int(_settings.ws.timeout_sec)

# Cookie name used by SessionMiddleware for session id
COOKIE_NAME = "wsai.sid"

# -----------------------------------------------------------------------------
# Env helpers
# -----------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# Default staging behavior (can be overridden via WS_DEFAULT_STAGING=0/false)
DEFAULT_STAGING: bool = _env_bool("WS_DEFAULT_STAGING", True)

# Default language tag if nothing is provided by client (kept simple & explicit)
DEFAULT_ACCEPT_LANGUAGE_TAG: str = os.getenv("WS_DEFAULT_LANGUAGE", "cs-CZ")


# -----------------------------------------------------------------------------
# Inbound schema
# -----------------------------------------------------------------------------


class InboundEvent(BaseModel):
    type: Literal["select", "chat", "export", "ping", "switch_tab"]
    message: str | None = None
    asset_id: str | None = None
    active_tab: str | None = None

    model_config = ConfigDict(extra="forbid")


# -----------------------------------------------------------------------------
# Connection state (per WebSocket)
# -----------------------------------------------------------------------------


@dataclass
class Conn:
    org_id: str
    user_id: str
    active_tab: str
    selected_asset_id: str | None = None
    conn_id: str = uuid.uuid4().hex
    busy: bool = False
    sid: Optional[str] = None
    request_id: str = "n/a"
    staging: bool = True
    # Language handling:
    # - accept_language: the canonical tag (as chosen from query/header/default), e.g., "cs-CZ"
    # - lang_iso: the normalized ISO primary tag used for agents, e.g., "cs"
    accept_language: Optional[str] = None
    lang_iso: Optional[str] = None


# -----------------------------------------------------------------------------
# Language helpers (deterministic selection + normalization)
# -----------------------------------------------------------------------------


def _parse_accept_language(raw: Optional[str]) -> list[tuple[str, float]]:
    """
    Parse an Accept-Language-like string into [(tag, q)], sorted by descending q.

    Examples:
        "cs-CZ,cs;q=0.9,en;q=0.7" -> [("cs-CZ", 1.0), ("cs", 0.9), ("en", 0.7)]
        None or "" -> []
    """
    if not raw:
        return []
    items: list[tuple[str, float]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        # Match: <tag>[;q=<float>]
        m = re.match(r"^([^;,\s]+)(?:\s*;\s*q=([0-9.]+))?$", part)
        if not m:
            continue
        tag = m.group(1).strip()
        try:
            q = float(m.group(2)) if m.group(2) else 1.0
        except ValueError:
            q = 1.0
        items.append((tag, q))
    # Highest q first, stable sort
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def _to_iso_primary(tag_or_list: Optional[str]) -> str:
    """
    Normalize a tag or a comma-separated list to its primary ISO language code.

    Examples:
        "cs-CZ, cs" -> "cs"
        "en-US"     -> "en"
        "sk"        -> "sk"
        None        -> "cs" (we default to Czech if absolutely nothing is known)
    """
    if not tag_or_list:
        return "cs"
    head = tag_or_list.split(",")[0].split(";")[0].strip()
    # "cs-CZ" -> "cs"
    primary = head.split("-")[0].strip().lower()
    return primary or "cs"


def _choose_lang(query_value: Optional[str], header_value: Optional[str], default_tag: str) -> tuple[str, str]:
    """
    Decide the canonical tag and normalized ISO for this connection.

    Precedence:
      1) Query param ?accept_language=...
      2) HTTP header "Accept-Language" from the WS upgrade
      3) Server default (DEFAULT_ACCEPT_LANGUAGE_TAG)

    Returns:
      (canonical_tag, iso_primary)
    """
    if query_value and query_value.strip():
        q_list = _parse_accept_language(query_value)
        tag = q_list[0][0] if q_list else query_value.split(",")[0].strip()
        return (tag, _to_iso_primary(tag))
    h_list = _parse_accept_language(header_value or "")
    if h_list:
        tag = h_list[0][0]
        return (tag, _to_iso_primary(tag))
    return (default_tag, _to_iso_primary(default_tag))


# -----------------------------------------------------------------------------
# UIRequest builder
# -----------------------------------------------------------------------------


def _build_ui_request(state: Conn, message: str, *, staging: bool = False) -> UIRequest:
    """
    Build UIRequest for the Manager/Router based on current connection state.

    IMPORTANT:
    - We pass `language=state.lang_iso`. This is the normalized ISO code (e.g., "cs")
      that agents will use to keep a stable language for the whole session.
    - This prevents language drift even if prompts/tools contain English fragments.
    """
    return UIRequest(
        org_id=state.org_id,
        active_tab=state.active_tab,
        selection={"asset_id": state.selected_asset_id} if state.selected_asset_id else {},
        message=message,
        staging=staging,
        user_id=state.user_id,
        session_id=state.sid,
        language=state.lang_iso or _to_iso_primary(state.accept_language),
    )


# -----------------------------------------------------------------------------
# WS send/error helpers
# -----------------------------------------------------------------------------


async def _ws_send(ws: WebSocket, event_type: str, payload: dict[str, Any] | list[Any] | None = None) -> None:
    try:
        await ws.send_json({"type": event_type, "payload": payload or {}})
    except WebSocketDisconnect:
        return
    except RuntimeError as e:
        # Safety: ASGI can raise if sending after close
        if "Unexpected ASGI message" in str(e) or "once a close message has been sent" in str(e):
            return
        raise


def _cur_request_id(state: Optional[Conn]) -> str:
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


# -----------------------------------------------------------------------------
# WebSocket endpoint
# -----------------------------------------------------------------------------


@router.websocket("/ws")
async def ws_endpoint(
    ws: WebSocket,
    org_id: str = Query(..., description="Organization ID"),
    user_id: str = Query(..., description="User ID"),
    active_tab: str = Query("assets", description="assets|impacts|threats|controls"),
    accept_language: Optional[str] = Query(None, description="Optional Accept-Language hint (e.g., 'cs, sk;q=0.9, en;q=0.7')"),
) -> None:
    """
    Real-time chat endpoint.

    Language behavior:
      - We select the language ONCE per connection (query > header > default).
      - We normalize to ISO primary (e.g., "cs") and send it on UIRequest.language.
      - This makes agents stick to the user's language for the whole session (no drift).
    """
    active_tab = active_tab if active_tab in ALLOWED_TABS else "assets"

    # Auth before accept
    auth_ok = await verify_ws_auth(ws)
    if not auth_ok:
        return
    await ws.accept()

    await WsHub.register(ws)

    sid = ws.cookies.get(COOKIE_NAME)
    rid = get_request_id() or "n/a"

    # Decide language once (per connection) and "lock" it into the Conn
    lang_tag, lang_iso = _choose_lang(
        accept_language,
        ws.headers.get("accept-language"),
        default_tag=DEFAULT_ACCEPT_LANGUAGE_TAG,
    )

    state = Conn(
        org_id=org_id,
        user_id=user_id,
        active_tab=active_tab,
        sid=sid,
        request_id=rid,
        staging=DEFAULT_STAGING,
        accept_language=lang_tag,  # e.g., "cs-CZ"
        lang_iso=lang_iso,         # e.g., "cs"
    )

    # Send hello with both values so the UI can clearly see what was chosen
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
            "staging": state.staging,
            "accept_language": state.accept_language,  # canonical tag (e.g., "cs-CZ")
            "lang": state.lang_iso,                    # normalized ISO (e.g., "cs")
        },
    )

    try:
        while True:
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

            if evt.type == "ping":
                await _ws_send(ws, "pong", {"conn_id": state.conn_id, "request_id": state.request_id})
                continue

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

            if evt.type == "select":
                state.selected_asset_id = evt.asset_id or None
                await _ws_send(ws, "select_ack", {"asset_id": state.selected_asset_id, "request_id": state.request_id})
                continue

            if evt.type in {"chat", "export"}:
                # Rate limit
                if getattr(_settings, "ratelimit", None) and getattr(_settings.ratelimit, "enabled", False):
                    event_for_rl = "chat" if evt.type == "chat" else "export"
                    sid_for_rl = state.sid or f"anon:{state.conn_id[:8]}"
                    try:
                        ok = await allow_ws(event_for_rl, sid_for_rl, _settings.ratelimit.per_minute)
                    except Exception as rl_err:
                        await _ws_error(ws, "Rate-limit check failed", conn_id=state.conn_id, details=str(rl_err), state=state)
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

                    deps = make_default_deps(tab=state.active_tab)
                    ui_req = _build_ui_request(state, text, staging=state.staging)

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

                    # Backward/forward compatible call: route_chat may or may not accept `deps`
                    if "deps" in route_chat.__code__.co_varnames:
                        ui_res = await asyncio.wait_for(route_chat(ui_req, deps=deps), timeout=WS_EVENT_TIMEOUT_S)
                    else:
                        ui_res = await asyncio.wait_for(route_chat(ui_req), timeout=WS_EVENT_TIMEOUT_S)

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

            await _ws_error(ws, f"Unknown event type: {evt.type}", conn_id=state.conn_id, state=state)

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await WsHub.unregister(ws)
        except Exception:
            pass
