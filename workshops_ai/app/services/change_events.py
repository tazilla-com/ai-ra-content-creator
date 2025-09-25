# File: workshops_ai/app/services/change_events.py
from __future__ import annotations

from typing import Optional

from workshops_ai.models.agent_event import AgentEvent
from workshops_ai.app.services.ws_hub import WsHub
from workshops_ai.schemas.risk_models import WSRiskSetEvent, RiskChangeItem


async def aemit_change_event(
    event: str,
    *,
    org_id: str,
    user_id: str,
    data: dict,
    asset_id: Optional[str] = None,
) -> dict:
    """
    Generic AgentEvent emitter (audit-friendly shape).
    Broadcasts a JSON object like:
      {
        "event": "<event>",
        "org_id": "...",
        "user_id": "...",
        "asset_id": "...?",    # optional
        "data": { ... }        # opaque domain-specific payload
      }
    """
    ev = AgentEvent(
        event=event,
        org_id=org_id,
        user_id=user_id,
        asset_id=asset_id,
        data=data,
    )
    payload = ev.model_dump(mode="json")
    await WsHub.broadcast_json(payload)
    return payload


# -----------------------------------------------------------------------------
# Domain-specific emitter (parallel to impact.set_level)
# -----------------------------------------------------------------------------

async def aemit_risk_set_probability(
    *,
    asset_id: str,
    threat_code: str,
    probability: int,
    note: Optional[str] = None,
) -> dict:
    """
    Emit a WS event for "risk.set_probability" (UI-optimized, schema-validated).

    Produces and broadcasts a WSRiskSetEvent with a normalized RiskChangeItem:
      {
        "event": "risk.set_probability",
        "payload": {
          "asset_id": "...",
          "threat_code": "...",
          "probability": 0..4,
          "note": "...?",
          "ts": "<ISO8601>",            # auto-added by RiskChangeItem
          "idempotency_key": "<uuid>"   # deterministic UUIDv5
        }
      }

    This is a UI-facing event; auditing/actor/org context should be handled
    separately (e.g., via aemit_change_event(...)) if needed.
    """
    item = RiskChangeItem.model_validate(
        {
            "asset_id": asset_id,
            "threat_code": threat_code,
            "probability": probability,
            **({"note": note} if note else {}),
        }
    )
    evt = WSRiskSetEvent(event="risk.set_probability", payload=item)
    wire = evt.model_dump(mode="json")
    # Broadcast to all connected WS clients under /ws
    await WsHub.broadcast_json(wire)
    return wire
