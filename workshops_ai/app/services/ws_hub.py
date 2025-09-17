# File: workshops_ai/app/services/ws_hub.py
from __future__ import annotations
import asyncio
from typing import Set
from fastapi import WebSocket

class WsHub:
    _clients: Set[WebSocket] = set()
    _lock = asyncio.Lock()

    @classmethod
    async def register(cls, ws: WebSocket) -> None:
        async with cls._lock:
            cls._clients.add(ws)

    @classmethod
    async def unregister(cls, ws: WebSocket) -> None:
        async with cls._lock:
            cls._clients.discard(ws)

    @classmethod
    async def broadcast_json(cls, payload: dict) -> None:
        for ws in list(cls._clients):
            try:
                await ws.send_json(payload)
            except Exception:
                # drop broken connection silently
                await cls.unregister(ws)
