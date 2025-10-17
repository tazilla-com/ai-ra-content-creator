# File: workshops_ai/app/services/ws_hub.py
# -----------------------------------------------------------------------------
# WebSocket Hub (WsHub)
#
# Purpose
# - Minimal in-memory broadcast hub for FastAPI WebSocket connections.
# - Tracks per-connection metadata (org_id/user_id/tab) to route events.
# - Provides schema-validated emitters for impacts/risks domain events.
# - Adds lightweight emitters for asset create/update events.
# - Sends periodic JSON keepalives to prevent idle disconnects.
#
# Public API (class methods)
# - register(ws, *, org_id=None, user_id=None, tab=None) -> None
# - unregister(ws) -> None
# - update_client_meta(ws, *, org_id=None, user_id=None, tab=None) -> None
# - connected_count() -> int
# - broadcast(event: str, payload: dict) -> None         # sync wrapper
# - emit(event: str, payload: dict) -> None              # alias of broadcast
# - broadcast_json(payload: dict) -> Awaitable[None]     # expects "event"
# - aemit_impact_set_level(payload: dict) -> Awaitable[None]
# - aemit_impact_set_event(payload: dict) -> Awaitable[None]
# - aemit_risk_set_probability(payload: dict) -> Awaitable[None]
# - aemit_risk_set_event(payload: dict) -> Awaitable[None]
# - aemit_asset_event(payload: dict) -> Awaitable[None]            # NEW
# - aemit_asset_created(item: dict) -> Awaitable[None]             # NEW
# - aemit_asset_updated(item: dict) -> Awaitable[None]             # NEW
#
# Notes
# - This hub is intentionally simple: no persistence, no external broker.
# - Failures while sending to a client result in silent unregistration.
# - Routing is best-effort: if client metadata is missing, it receives events.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import WebSocket

# Single-item events:
#   * WSImpactSetEvent:  {"event":"impact.set_level", "payload":{...}}
#   * WSRiskSetEvent:    {"event":"risk.set_probability", "payload":{...}}
# Batch/list events:
#   * WSImpactSetListEvent: {"event":"impact.set_levels", "payload":{impacts:[...]} }
#   * WSRiskSetListEvent:   {"event":"risk.set_probabilities", "payload":{risks:[...]} }

from workshops_ai.schemas.impact_models import (
    WSImpactSetEvent,
    WSImpactSetListEvent,
)
from workshops_ai.schemas.risk_models import (
    WSRiskSetEvent,
    WSRiskSetListEvent,
    RiskChangeItem,
)

__all__ = ["WsHub"]


# ---- Internal structures -----------------------------------------------------


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

    - Filtering: events are routed to clients with matching tab/org_id.
      Legacy clients without metadata still receive all events.
    - Keepalive: a lightweight JSON ping prevents idle disconnects.
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
        # Start keepalive ticker (JSON-level ping; ~25s is a reasonable compromise)
        client.keepalive = asyncio.create_task(cls._keepalive_task(client))

        async with cls._lock:
            cls._clients[ws] = client

    @classmethod
    async def unregister(cls, ws: WebSocket) -> None:
        """Unregister a WebSocket connection (idempotent) and stop keepalive."""
        async with cls._lock:
            client = cls._clients.pop(ws, None)

        if client and client.keepalive:
            client.keepalive.cancel()
            try:
                await client.keepalive
            except Exception:
                # Swallow cancellation/transport exceptions during shutdown
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
        """Update routing metadata for a registered client (no-op if missing)."""
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

    # -- Broadcasting (sync wrappers for graphs' SEND-NOW calls) ----------------

    @classmethod
    def broadcast(cls, event: str, payload: dict[str, Any]) -> None:
        """
        Fire-and-forget broadcast of a JSON message, adding/overriding the 'event' field.
        Safe to call from non-async code (graphs use this in SEND-NOW paths).
        """
        # Ensure the provided event name wins over any accidental 'event' in payload
        message = {**payload, "event": event}
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(cls.broadcast_json(message))
        except RuntimeError:
            # No running loop (unlikely under FastAPI/Uvicorn) – run a short-lived loop.
            asyncio.run(cls.broadcast_json(message))

    @classmethod
    def emit(cls, event: str, payload: dict[str, Any]) -> None:
        """Alias of broadcast(...)."""
        cls.broadcast(event, payload)

    @classmethod
    async def broadcast_json(cls, payload: dict[str, Any]) -> None:
        """
        Broadcast a JSON payload to matching clients.

        Routing rules:
        - Target tab is inferred from payload["event"] (impacts/threats/assets/None).
        - If org_id is present (top-level or inside payload["payload"]), only
          clients with the same org_id (or no org_id set) receive the message.
        - Legacy clients without metadata receive all events.

        Failures while sending result in client removal.
        """
        event = str(payload.get("event") or "")
        target_tab = cls._infer_tab_from_event(event)
        target_org = cls._extract_org_id(payload)

        # Snapshot targets outside the lock to avoid holding it during I/O
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
        Validate and emit an 'impact.set_level' event (single-item) to all relevant clients.

        Expected envelope (Pydantic-enforced by WSImpactSetEvent):
          - event: "impact.set_level"
          - payload: {
              org_id, asset_id, asset_name?,
              impact_code, impact_name?,
              level, note?, ts?, idempotency_key?
            }
        """
        evt = WSImpactSetEvent.model_validate(payload)
        await cls.broadcast_json(evt.model_dump(mode="json"))

    @classmethod
    async def aemit_impact_set_event(cls, payload: dict[str, Any]) -> None:
        """
        Validate and emit an 'impact.set_levels' event (batch/list) to all relevant clients.

        Expected envelope (Pydantic-enforced by WSImpactSetListEvent):
          - event: "impact.set_levels"
          - payload: {
              org_id, asset_id, asset_name?,
              impacts: [{impact_code, level, note?, ts?, idempotency_key?}, ...]
            }
        """
        evt = WSImpactSetListEvent.model_validate(payload)
        await cls.broadcast_json(evt.model_dump(mode="json"))

    @classmethod
    async def aemit_risk_set_probability(cls, payload: dict[str, Any]) -> None:
        """
        Validate and emit a 'risk.set_probability' event (single-item).

        Accepted payloads:
          1) Full event envelope:
             {"event": "risk.set_probability", "payload": {...RiskChangeItem...}}
          2) Plain item (auto-wrapped):
             {"org_id":"...", "asset_id":"...", "threat_code":"...", "probability": 0..4, "note":"...?"}

        Plain items are normalized into a RiskChangeItem, then wrapped with 'event'.
        """
        # Full envelope path
        if "event" in payload and "payload" in payload:
            evt = WSRiskSetEvent.model_validate(payload)
            await cls.broadcast_json(evt.model_dump(mode="json"))
            return

        # Plain item path → normalize and wrap
        item = RiskChangeItem.model_validate(payload)
        evt = WSRiskSetEvent(event="risk.set_probability", payload=item)
        await cls.broadcast_json(evt.model_dump(mode="json"))

    @classmethod
    async def aemit_risk_set_event(cls, payload: dict[str, Any]) -> None:
        """
        Validate and emit a 'risk.set_probabilities' event (batch/list).

        Expected envelope (Pydantic-enforced by WSRiskSetListEvent):
          - event: "risk.set_probabilities"
          - payload: {
              org_id, asset_id, asset_name?,
              risks: [{threat_code, probability, note?, ts?, idempotency_key?}, ...]
            }
        """
        evt = WSRiskSetListEvent.model_validate(payload)
        await cls.broadcast_json(evt.model_dump(mode="json"))

    # -- Assets emitters (lightweight validation) -------------------------------

    _ALLOWED_ASSET_EVENTS = {"asset.created", "asset.updated"}

    @classmethod
    async def aemit_asset_event(cls, payload: dict[str, Any]) -> None:
        """
        Validate and emit an asset event (single-item) to the *assets* tab.

        Supported event names:
          - "asset.created"
          - "asset.updated"

        Accepted inputs:
          1) Full envelope:
             {
               "event": "asset.created" | "asset.updated",
               "payload": {
                  "org_id": "...",            # REQUIRED
                  "asset_id": "...",          # REQUIRED
                  "name": "CRM",              # OPTIONAL
                  "assetType": "APPLICATION", # OPTIONAL
                  "isPrimary": true,          # OPTIONAL
                  "description": "...",       # OPTIONAL
                  "ts": "...", "idempotency_key": "..."  # OPTIONAL
               }
             }
          2) Plain item with "op":
             {
               "op": "created" | "updated",
               "org_id": "...",
               "asset_id": "...",
               ...
             }
             (Will be wrapped into the envelope above.)

        Raises:
          ValueError if the shape is invalid or required keys are missing.
        """
        # Case 1: already an envelope with an explicit event
        if isinstance(payload, dict) and "event" in payload and "payload" in payload:
            event = str(payload["event"])
            if event not in cls._ALLOWED_ASSET_EVENTS:
                raise ValueError(f"Unsupported asset event: {event!r}")
            inner = payload["payload"]
            if not isinstance(inner, dict):
                raise ValueError("Asset event 'payload' must be an object.")
            if not isinstance(inner.get("org_id"), str) or not inner.get("org_id"):
                raise ValueError("Asset event requires non-empty payload.org_id.")
            if not isinstance(inner.get("asset_id"), str) or not inner.get("asset_id"):
                raise ValueError("Asset event requires non-empty payload.asset_id.")
            # Broadcast as-is (already normalized)
            await cls.broadcast_json({"event": event, "payload": inner})
            return

        # Case 2: plain item with 'op'
        if isinstance(payload, dict) and "op" in payload:
            op = str(payload.get("op") or "").strip().lower()
            if op not in ("created", "updated"):
                raise ValueError(f"Unsupported asset op: {op!r}")
            event = f"asset.{op}"
            if not isinstance(payload.get("org_id"), str) or not payload.get("org_id"):
                raise ValueError("Asset item requires non-empty org_id.")
            if not isinstance(payload.get("asset_id"), str) or not payload.get("asset_id"):
                raise ValueError("Asset item requires non-empty asset_id.")
            # Remove helper key 'op' and wrap
            item = dict(payload)
            item.pop("op", None)
            await cls.broadcast_json({"event": event, "payload": item})
            return

        raise ValueError("Unsupported asset event payload shape. Provide an envelope with 'event' and 'payload', or a plain item with 'op'.")

    @classmethod
    async def aemit_asset_created(cls, item: dict[str, Any]) -> None:
        """
        Convenience wrapper: emit 'asset.created' for a single item.

        Example 'item':
          {
            "org_id": "...", "asset_id": "...",
            "name": "CRM", "assetType": "APPLICATION", "isPrimary": true, "description": "...?",
            "ts": "...?", "idempotency_key": "...?"
          }
        """
        await cls.aemit_asset_event({"event": "asset.created", "payload": item})

    @classmethod
    async def aemit_asset_updated(cls, item: dict[str, Any]) -> None:
        """
        Convenience wrapper: emit 'asset.updated' for a single item.

        Example 'item':
          {
            "org_id": "...", "asset_id": "...",
            "name": "CRM", "assetType": "APPLICATION", "isPrimary": true, "description": "...?",
            "ts": "...?", "idempotency_key": "...?"
          }
        """
        await cls.aemit_asset_event({"event": "asset.updated", "payload": item})

    # -- Internals --------------------------------------------------------------

    @staticmethod
    def _infer_tab_from_event(event: str) -> Optional[str]:
        """Map event names to UI tabs."""
        if not event:
            return None
        if event.startswith("impact.") or event.startswith("impacts."):
            return "impacts"
        # Both "risk.set_probability" and potential legacy "threat.set_probability"
        if event.startswith("risk.") or event.startswith("threat."):
            return "threats"
        if event.startswith("asset."):
            return "assets"
        return None

    @staticmethod
    def _extract_org_id(payload: dict[str, Any]) -> Optional[str]:
        """Extract org_id from either the top-level payload or the inner 'payload' field."""
        org_top = payload.get("org_id")
        if isinstance(org_top, str) and org_top:
            return org_top
        inner = payload.get("payload")
        if isinstance(inner, dict):
            org_inner = inner.get("org_id")
            if isinstance(org_inner, str) and org_inner:
                return org_inner
        return None

    @classmethod
    async def _keepalive_task(cls, client: _Client) -> None:
        """Periodically send a lightweight JSON ping to keep the connection alive."""
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
