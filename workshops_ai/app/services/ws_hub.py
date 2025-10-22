# File: workshops_ai/app/services/ws_hub.py
# -----------------------------------------------------------------------------
# WebSocket Hub (WsHub)
#
# Purpose
# - Minimal in-memory broadcast hub for FastAPI WebSocket connections.
# - Tracks per-connection metadata so we can route/broadcast domain events.
# - Provides schema-validated emitters for impacts/risks, plus asset events.
# - Sends periodic JSON keepalives to prevent idle disconnects.
#
# New in this version
# - _Client expanded metadata: staging, accept_language, active_tab, conn_id.
# - register(...) captures initial metadata (best-effort from headers).
# - update_context(conn_id, **kwargs) to set/patch context after "hello".
# - Getters for the chat router to fetch per-connection context by conn_id.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
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
    """
    Tracked WebSocket connection with routing/context metadata.

    Notes:
    - accept_language is the canonical header tag (e.g. "cs-CZ"). The router
      can separately convert this to ISO primary ("cs") for model prompts.
    - active_tab follows the UI tab ("assets"|"impacts"|"threats"|...).
    - conn_id is the UI/router-generated connection id (stable for this socket).
    """
    ws: WebSocket
    org_id: Optional[str] = None
    user_id: Optional[str] = None
    active_tab: Optional[str] = None
    staging: bool = True
    accept_language: Optional[str] = None
    conn_id: Optional[str] = None
    keepalive: Optional[asyncio.Task] = None


# ---- Hub ---------------------------------------------------------------------


class WsHub:
    """
    In-memory WebSocket hub for broadcasting JSON events.

    - Filtering: events are routed to clients with matching tab/org_id.
      Legacy clients without metadata still receive all events.
    - Keepalive: a lightweight JSON ping prevents idle disconnects.
    """

    # Primary index by WebSocket object
    _clients: Dict[WebSocket, _Client] = {}
    # Secondary index by conn_id for quick lookups from HTTP chat router
    _by_conn_id: Dict[str, _Client] = {}

    _lock: asyncio.Lock = asyncio.Lock()

    # -- Registration / lifecycle ---------------------------------------------

    @classmethod
    async def register(
        cls,
        ws: WebSocket,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        active_tab: str | None = None,
        staging: bool | None = None,
        accept_language: str | None = None,
        conn_id: str | None = None,
    ) -> None:
        """
        Register an accepted WebSocket connection with optional metadata.

        IMPORTANT
        - Call websocket.accept() before registering.
        - The router can pass org/user/tab/staging/lang/conn_id if already known.
          Otherwise we best-effort fill some from headers and let the router
          patch later via update_context(conn_id, ...).

        Header fallbacks:
        - accept_language defaults to ws.headers.get("accept-language") if not given.
        """
        # Best-effort header fallback for language if the router didn't provide it yet
        if not accept_language:
            try:
                accept_language = ws.headers.get("accept-language")
            except Exception:
                accept_language = None

        client = _Client(
            ws=ws,
            org_id=org_id,
            user_id=user_id,
            active_tab=active_tab,
            staging=bool(True if staging is None else staging),
            accept_language=accept_language,
            conn_id=conn_id,
        )

        # Start keepalive ticker (JSON-level ping; ~25s is a reasonable compromise)
        client.keepalive = asyncio.create_task(cls._keepalive_task(client))

        async with cls._lock:
            cls._clients[ws] = client
            # Maintain secondary index if conn_id is already known
            if client.conn_id:
                cls._by_conn_id[client.conn_id] = client

    @classmethod
    async def unregister(cls, ws: WebSocket) -> None:
        """Unregister a WebSocket connection (idempotent) and stop keepalive."""
        async with cls._lock:
            client = cls._clients.pop(ws, None)
            # Clean secondary index
            if client and client.conn_id:
                cls._by_conn_id.pop(client.conn_id, None)

        if client and client.keepalive:
            client.keepalive.cancel()
            try:
                await client.keepalive
            except Exception:
                # Swallow cancellation/transport exceptions during shutdown
                pass

    # -- Context updates / getters --------------------------------------------

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
        Backward-compatible updater (legacy name/signature).
        Prefer update_context(conn_id, ...) in new code.
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
                client.active_tab = tab

    @classmethod
    async def update_context(
        cls,
        conn_id: str,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        active_tab: str | None = None,
        staging: bool | None = None,
        accept_language: str | None = None,
    ) -> None:
        """
        Update per-connection context by conn_id (preferred modern API).

        Typical usage in the WS router:
          - After sending "hello", call:
              await WsHub.update_context(
                  conn_id,
                  org_id=...,
                  user_id=...,
                  active_tab=...,
                  staging=...,
                  accept_language=...,
              )
        """
        async with cls._lock:
            client = cls._by_conn_id.get(conn_id)
            if not client:
                return
            if org_id is not None:
                client.org_id = org_id
            if user_id is not None:
                client.user_id = user_id
            if active_tab is not None:
                client.active_tab = active_tab
            if staging is not None:
                client.staging = bool(staging)
            if accept_language is not None:
                client.accept_language = accept_language

    @classmethod
    async def set_conn_id(cls, ws: WebSocket, conn_id: str) -> None:
        """
        Assign or change conn_id for a registered client and refresh the secondary index.
        Call this once you generate a conn_id in the WS router during handshake.
        """
        async with cls._lock:
            client = cls._clients.get(ws)
            if not client:
                return
            # Remove old index entry if any
            if client.conn_id:
                cls._by_conn_id.pop(client.conn_id, None)
            client.conn_id = conn_id
            cls._by_conn_id[conn_id] = client

    @classmethod
    def get_context(cls, conn_id: str) -> Optional[dict[str, Any]]:
        """
        Return a JSON-serializable view of the per-connection context for chat router.

        Includes:
          org_id, user_id, active_tab, staging, accept_language, conn_id

        Returns None if not found.
        """
        client = cls._by_conn_id.get(conn_id)
        if not client:
            return None
        data = asdict(client)
        # Strip non-serializable WebSocket and ephemeral keepalive task from the view
        data.pop("ws", None)
        data.pop("keepalive", None)
        return data  # {org_id, user_id, active_tab, staging, accept_language, conn_id}

    @classmethod
    def get_language(cls, conn_id: str) -> Optional[str]:
        """Convenience getter: canonical Accept-Language tag for this connection."""
        client = cls._by_conn_id.get(conn_id)
        return client.accept_language if client else None

    @classmethod
    def get_active_tab(cls, conn_id: str) -> Optional[str]:
        """Convenience getter: active UI tab for this connection."""
        client = cls._by_conn_id.get(conn_id)
        return client.active_tab if client else None

    @classmethod
    def get_staging(cls, conn_id: str) -> Optional[bool]:
        """Convenience getter: staging flag for this connection."""
        client = cls._by_conn_id.get(conn_id)
        return client.staging if client else None

    @classmethod
    def connected_count(cls) -> int:
        """Number of currently tracked WebSocket clients."""
        return len(cls._clients)

    # -- Broadcasting (sync wrappers for graphs' SEND-NOW calls) ---------------

    @classmethod
    def broadcast(cls, event: str, payload: dict[str, Any]) -> None:
        """
        Fire-and-forget broadcast of a JSON message, adding/overriding the 'event' field.
        Safe to call from non-async code (graphs use this in SEND-NOW paths).
        """
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
            tab_ok = (c.active_tab is None) or (target_tab is None) or (c.active_tab == target_tab)
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
        """
        evt = WSImpactSetEvent.model_validate(payload)
        await cls.broadcast_json(evt.model_dump(mode="json"))

    @classmethod
    async def aemit_impact_set_event(cls, payload: dict[str, Any]) -> None:
        """
        Validate and emit an 'impact.set_levels' event (batch/list) to all relevant clients.
        """
        evt = WSImpactSetListEvent.model_validate(payload)
        await cls.broadcast_json(evt.model_dump(mode="json"))

    @classmethod
    async def aemit_risk_set_probability(cls, payload: dict[str, Any]) -> None:
        """
        Validate and emit a 'risk.set_probability' event (single-item).
        Accepts either a full envelope or a plain item.
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
             {"event": "...", "payload": {...}}
          2) Plain item with "op": {"op":"created"|"updated", ...}
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
            item = dict(payload)
            item.pop("op", None)
            await cls.broadcast_json({"event": event, "payload": item})
            return

        raise ValueError("Unsupported asset event payload shape.")

    @classmethod
    async def aemit_asset_created(cls, item: dict[str, Any]) -> None:
        """Convenience wrapper: emit 'asset.created' for a single item."""
        await cls.aemit_asset_event({"event": "asset.created", "payload": item})

    @classmethod
    async def aemit_asset_updated(cls, item: dict[str, Any]) -> None:
        """Convenience wrapper: emit 'asset.updated' for a single item."""
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
