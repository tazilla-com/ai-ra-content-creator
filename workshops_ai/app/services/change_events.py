# File: workshops_ai/app/services/change_events.py
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union, List, Dict

from workshops_ai.models.agent_event import AgentEvent
from workshops_ai.app.services.ws_hub import WsHub

# Risk WS models
from workshops_ai.schemas.risk_models import (
    WSRiskSetEvent,
    RiskChangeItem,
)

# Impact WS models
from workshops_ai.schemas.impact_models import (
    WSImpactSetEvent,
    WSImpactSetListEvent,
    ImpactChangeItem,
)


__all__ = [
    "aemit_change_event",
    "aemit_risk_set_probability",
    "aemit_risk_set_probabilities",
    "aemit_impact_set_level",
    "aemit_impact_set_levels",
]


# -----------------------------------------------------------------------------
# Generic, audit-friendly event (not UI-optimized)
# -----------------------------------------------------------------------------
async def aemit_change_event(
    event: str,
    *,
    org_id: str,
    user_id: str,
    data: Dict,
    asset_id: Optional[str] = None,
) -> Dict:
    """
    Emit a generic AgentEvent (good for audit trails, logs, and analytics).

    Shape on the wire:
      {
        "event": "<event>",
        "org_id": "...",
        "user_id": "...",
        "asset_id": "...?",  # optional
        "data": { ... }      # domain-specific payload
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
# RISK: UI-optimized WS emissions (schema-validated)
# -----------------------------------------------------------------------------
async def aemit_risk_set_probability(
    *,
    asset_id: str,
    threat_code: str,
    probability: int,
    note: Optional[str] = None,
) -> Dict:
    """
    Emit a single 'risk.set_probability' event validated by WSRiskSetEvent.

    Wire format:
      {
        "event": "risk.set_probability",
        "payload": {
          "asset_id": "...",
          "threat_code": "...",
          "probability": 0..4,
          "note": "...?",             # optional
          "ts": "<ISO8601>",          # auto-populated by RiskChangeItem
          "idempotency_key": "<uuid>" # auto/deterministic in the model
        }
      }
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
    await WsHub.broadcast_json(wire)
    return wire


async def aemit_risk_set_probabilities(
    items: Sequence[Union[RiskChangeItem, Dict]],
) -> List[Dict]:
    """
    Convenience bulk helper for risks.
    NOTE: If your WS schema does not define a list event for risks, we emit
    individual 'risk.set_probability' events one-by-one (preserves UI behavior
    and keeps compatibility with existing consumers).

    Returns a list of serialized events that were broadcast.
    """
    out: List[Dict] = []
    for raw in items:
        item = raw if isinstance(raw, RiskChangeItem) else RiskChangeItem.model_validate(raw)
        evt = WSRiskSetEvent(event="risk.set_probability", payload=item)
        wire = evt.model_dump(mode="json")
        await WsHub.broadcast_json(wire)
        out.append(wire)
    return out


# -----------------------------------------------------------------------------
# IMPACT: UI-optimized WS emissions (schema-validated)
# -----------------------------------------------------------------------------
async def aemit_impact_set_level(
    *,
    asset_id: str,
    impact_code: str,
    level: int,
    note: Optional[str] = None,
) -> Dict:
    """
    Emit a single 'impact.set_level' event validated by WSImpactSetEvent.

    Wire format:
      {
        "event": "impact.set_level",
        "payload": {
          "asset_id": "...",
          "impact_code": "...",
          "level": 0..4,              # validated by ImpactChangeItem
          "note": "...?",             # optional
          "ts": "<ISO8601>",
          "idempotency_key": "<uuid>"
        }
      }
    """
    item = ImpactChangeItem.model_validate(
        {
            "asset_id": asset_id,
            "impact_code": impact_code,
            "level": level,
            **({"note": note} if note else {}),
        }
    )
    evt = WSImpactSetEvent(event="impact.set_level", payload=item)
    wire = evt.model_dump(mode="json")
    await WsHub.broadcast_json(wire)
    return wire


async def aemit_impact_set_levels(
    items: Sequence[Union[ImpactChangeItem, Dict]],
) -> Dict:
    """
    Emit a batched 'impact.set_levels' event validated by WSImpactSetListEvent.

    Input:
      items: sequence of ImpactChangeItem or dicts convertible to it.
             Each item must contain: asset_id, impact_code, level, (note?)

    Wire format (single message):
      {
        "event": "impact.set_levels",
        "payload": {
          "impacts": [ ImpactChangeItem, ... ]
        }
      }
    """
    normalized: List[ImpactChangeItem] = [
        it if isinstance(it, ImpactChangeItem) else ImpactChangeItem.model_validate(it)
        for it in items
    ]

    evt = WSImpactSetListEvent(
        event="impact.set_levels",
        impacts=normalized,  # schema uses root field 'impacts' inside payload
    )
    wire = evt.model_dump(mode="json")
    await WsHub.broadcast_json(wire)
    return wire
