# File: workshops_ai/app/services/change_events.py
from __future__ import annotations
from typing import Optional
from workshops_ai.models.agent_event import AgentEvent
from workshops_ai.app.services.ws_hub import WsHub

async def aemit_change_event(
    event: str, *, org_id: str, user_id: str, data: dict, asset_id: Optional[str] = None
) -> dict:
    ev = AgentEvent(event=event, org_id=org_id, user_id=user_id, asset_id=asset_id, data=data)
    payload = ev.model_dump(mode="json")
    await WsHub.broadcast_json(payload)
    return payload
