# File: workshops_ai/app/services/ws_hub.py
# -----------------------------------------------------------------------------
# WebSocket Hub (WsHub)
# - Minimal, FastAPI-friendly broadcast hub with async registration.
# - Used by agents/services to push UI events (optimistic updates).
# - Provides a dedicated emitter for ImpactAgent: aemit_impact_set_level(payload).
#
# Design goals
# - Thread-safe via a single asyncio.Lock for the clients set.
# - Best-effort broadcasting: a failing client is removed silently.
# - Schema-aware emitters validate payloads (Pydantic v2) before broadcasting.
#
# Public API
# - register(ws: WebSocket) / unregister(ws: WebSocket)
# - connected_count() -> int
# - broadcast(event: str, payload: dict)  # adds {"event": event, **payload}
# - broadcast_json(payload: dict)         # expects "event" already present
# - aemit_impact_set_level(payload: dict) # validates WSImpactSetEvent, then broadcast_json
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
from typing import Any, Set

from fastapi import WebSocket

from workshops_ai.schemas.impact_models import WSImpactSetEvent


class WsHub:
    """A simple in-memory WebSocket hub for broadcasting JSON events."""

    _clients: Set[WebSocket] = set()
    _lock: asyncio.Lock = asyncio.Lock()

    # -- Registration -----------------------------------------------------------

    @classmethod
    async def register(cls, ws: WebSocket) -> None:
        """
        Register an accepted WebSocket connection.
        Note: call ws.accept() in your router before registering.
        """
        async with cls._lock:
            cls._clients.add(ws)

    @classmethod
    async def unregister(cls, ws: WebSocket) -> None:
        """Unregister a WebSocket connection (idempotent)."""
        async with cls._lock:
            cls._clients.discard(ws)

    @classmethod
    def connected_count(cls) -> int:
        """Number of currently tracked WebSocket clients."""
        return len(cls._clients)

    # -- Broadcasting -----------------------------------------------------------

    @classmethod
    async def broadcast(cls, event: str, payload: dict[str, Any]) -> None:
        """
        Broadcast a JSON message adding the 'event' field automatically.
        """
        message = {"event": event, **payload}
        await cls.broadcast_json(message)

    @classmethod
    async def broadcast_json(cls, payload: dict[str, Any]) -> None:
        """
        Broadcast a JSON payload to all connected clients.
        If a client send fails, the client is unregistered silently.
        """
        # Snapshot to avoid holding the lock while sending (I/O)
        targets: list[WebSocket]
        async with cls._lock:
            targets = list(cls._clients)

        if not targets:
            return

        # Send concurrently; unregister failing sockets
        async def _send(ws: WebSocket) -> None:
            try:
                await ws.send_json(payload)
            except Exception:
                # Drop broken client quietly
                await cls.unregister(ws)

        await asyncio.gather(*(_send(ws) for ws in targets), return_exceptions=False)

    # -- Domain emitters (schema-validated) -------------------------------------

    @classmethod
    async def aemit_impact_set_level(cls, payload: dict[str, Any]) -> None:
        """
        Validate and emit an 'impact.set_level' event to all clients.
        Expected payload fields (Pydantic-enforced):
          - event: "impact.set_level"
          - org_id, asset_id, impact_code, level_index (0..4), level_value (5/20/50/70/100)
          - note?, actor, ts (ISO8601)
        """
        evt = WSImpactSetEvent.model_validate(payload)
        await cls.broadcast_json(evt.model_dump())
