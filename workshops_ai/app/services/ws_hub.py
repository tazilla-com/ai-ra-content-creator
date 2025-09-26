# File: workshops_ai/app/services/ws_hub.py
# -----------------------------------------------------------------------------
# WebSocket Hub (WsHub)
# - Minimal, FastAPI-friendly broadcast hub with async registration.
# - Per-connection metadata (org_id/user_id/tab) for targeted fan-out.
# - App-level keepalive to prevent idle disconnects.
# - Schema-aware emitters for impacts and risks.
#
# Public API
# - register(ws, *, org_id=None, user_id=None, tab=None)
# - unregister(ws)
# - update_client_meta(ws, *, org_id=None, user_id=None, tab=None)
# - connected_count() -> int
# - broadcast(event: str, payload: dict)         # will infer target tab from event
# - emit(event: str, payload: dict)              # alias of broadcast
# - broadcast_json(payload: dict)                # expects "event" in payload
# - aemit_impact_set_level(payload: dict)        # validates WSImpactSetEvent
# - aemit_risk_set_probability(payload: dict)    # validates WSRiskSetEvent or wraps plain item
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import WebSocket

from workshops_ai.schemas.impact_models import WSImpactSetEvent
from workshops_ai.schemas.risk_models import WSRiskSetEvent, RiskChangeItem


# ---- Internal structures ------------------------------------------------------


@dataclass
class _Client:
    """Tracked WebSocket connection with optional routing metadata."""
    ws: WebSocket
    org_id: Optional[str] = None
    user_id: Optional[str] = None
    tab: Optional[str] = None
    keepalive: Optional[asyncio.Task] = None


# ---- Hub ---------------------------------------------------------------------


class WsHub:
    """
    In-memory WebSocket hub for broadcasting JSON events.

    Notes
    - Best-effort: a failing client is unregistered silently.
    - Filtering: events are routed to clients with matching tab/org_id.
      Legacy clients without metadata still receive all events.
    - Keepalive: app-level JSON ping prevents idle disconnects on some stacks.
    """

    _clients: Dict[WebSocket, _Client] = {}
    _lock: asyncio.Lock = asyncio.Lock()

    # -- Registration -----------------------------------------------------------

    @classmethod
    async def register(
        cls,
        ws: WebSocket,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        tab: str | None = None,
    ) -> None:
        """
        Register an accepted WebSocket connection with optional routing metadata.
        IMPORTANT: call websocket.accept() in your router before registering.
        """
        client = _Client(ws=ws, org_id=org_id, user_id=user_id, tab=tab)
        # Start keepalive ticker (JSON-level ping; 20–30s is usually enough)
        client.keepalive = asyncio.create_task(cls._keepalive_task(client))

        async with cls._lock:
            cls._clients[ws] = client

    @classmethod
    async def unregister(cls, ws: WebSocket) -> None:
        """
        Unregister a WebSocket connection (idempotent) and stop keepalive ticker.
        """
        async with cls._lock:
            client = cls._clients.pop(ws, None)

        if client and client.keepalive:
            client.keepalive.cancel()
            # swallow cancellation exceptions
            try:
                await client.keepalive
            except Exception:
                pass

    @classmethod
    async def update_client_meta(
        cls,
        ws: WebSocket,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        tab: str | None = None,
    ) -> None:
        """
        Update routing metadata for a registered client (no-op if missing).
        """
        async with cls._lock:
            client = cls._clients.get(ws)
            if not client:
                return
            if org_id is not None:
                client.org_id = org_id
            if user_id is not None:
                client.user_id = user_id
            if tab is not None:
                client.tab = tab

    @classmethod
    def connected_count(cls) -> int:
        """Number of currently tracked WebSocket clients."""
        return len(cls._clients)

    # -- Broadcasting -----------------------------------------------------------

    @classmethod
    async def broadcast(cls, event: str, payload: dict[str, Any]) -> None:
        """
        Broadcast a JSON message adding the 'event' field automatically.
        The target tab is inferred from the event name.
        """
        message = {"event": event, **payload}
        await cls.broadcast_json(message)

    @classmethod
    async def emit(cls, event: str, payload: dict[str, Any]) -> None:
        """Alias of broadcast(...)."""
        await cls.broadcast(event, payload)

    @classmethod
    async def broadcast_json(cls, payload: dict[str, Any]) -> None:
        """
        Broadcast a JSON payload to matching clients.

        Routing rules:
        - Target tab is inferred from payload["event"] (impacts/threats/None).
        - If org_id is present (top-level or inside payload["payload"]), only
          clients with the same org_id (or no org_id set) receive the message.
        - Legacy clients without metadata receive all events.

        Failures while sending result in client removal.
        """
        event = str(payload.get("event") or "")
        target_tab = cls._infer_tab_from_event(event)
        target_org = cls._extract_org_id(payload)

        # Snapshot outside the lock to avoid holding it during I/O
        async with cls._lock:
            targets = list(cls._clients.values())

        def _match(c: _Client) -> bool:
            # Tab filter: deliver to clients matching the tab OR without tab (legacy)
            tab_ok = (c.tab is None) or (target_tab is None) or (c.tab == target_tab)
            # Org filter: deliver to clients matching org OR without org (legacy)
            org_ok = (c.org_id is None) or (target_org is None) or (c.org_id == target_org)
            return tab_ok and org_ok

        selected = [c for c in targets if _match(c)]
        if not selected:
            return

        async def _send(client: _Client) -> None:
            try:
                await client.ws.send_json(payload)
            except Exception:
                # Drop broken client quietly
                await cls.unregister(client.ws)

        await asyncio.gather(*(_send(c) for c in selected), return_exceptions=False)

    # -- Domain emitters (schema-validated) ------------------------------------

    @classmethod
    async def aemit_impact_set_level(cls, payload: dict[str, Any]) -> None:
        """
        Validate and emit an 'impact.set_level' event to all relevant clients.

        Expected fields (Pydantic-enforced):
          - event: "impact.set_level"
          - org_id, asset_id, impact_code, level_index (0..4), level_value (5/20/50/70/100)
          - note? (str), actor (str), ts (ISO8601)
        """
        evt = WSImpactSetEvent.model_validate(payload)
        await cls.broadcast_json(evt.model_dump())

    @classmethod
    async def aemit_risk_set_probability(cls, payload: dict[str, Any]) -> None:
        """
        Validate and emit a 'risk.set_probability' event to all relevant clients.

        Accepted payloads:
          1) Full event envelope:
             {"event": "risk.set_probability", "payload": {...RiskChangeItem...}}
          2) Plain item (auto-wrapped):
             {"org_id":"...", "asset_id":"...", "threat_code":"...", "probability": 3, "note": "...?"}

        Plain items are normalized into a RiskChangeItem, then wrapped with event.
        """
        # Full envelope path
        if "event" in payload and "payload" in payload:
            evt = WSRiskSetEvent.model_validate(payload)
            await cls.broadcast_json(evt.model_dump())
            return

        # Plain item path → normalize and wrap
        item = RiskChangeItem.model_validate(payload)
        evt = WSRiskSetEvent(event="risk.set_probability", payload=item)
        await cls.broadcast_json(evt.model_dump())

    # -- Internals --------------------------------------------------------------

    @staticmethod
    def _infer_tab_from_event(event: str) -> Optional[str]:
        """
        Map event names to UI tabs.
        """
        if not event:
            return None
        if event.startswith("impact."):
            return "impacts"
        # Both "risk.set_probability" and potential legacy "threat.set_probability"
        if event.startswith("risk.") or event.startswith("threat."):
            return "threats"
        return None

    @staticmethod
    def _extract_org_id(payload: dict[str, Any]) -> Optional[str]:
        """
        Extract org_id from either the top-level payload or the inner 'payload' field.
        """
        if "org_id" in payload and isinstance(payload["org_id"], str):
            return payload["org_id"]
        inner = payload.get("payload")
        if isinstance(inner, dict) and isinstance(inner.get("org_id"), str):
            return inner["org_id"]
        return None

    @classmethod
    async def _keepalive_task(cls, client: _Client) -> None:
        """
        Periodically send a lightweight JSON ping to keep the connection alive.
        """
        try:
            while True:
                await asyncio.sleep(25)  # 20–30s is a good compromise
                ts = datetime.now(timezone.utc).isoformat()
                try:
                    await client.ws.send_json({"type": "keepalive", "ts": ts})
                except Exception:
                    # Unregister on send failure to avoid leaks
                    await cls.unregister(client.ws)
                    return
        except asyncio.CancelledError:
            # Normal task shutdown
            return
