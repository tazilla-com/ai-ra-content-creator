# File: workshops_ai/agents/utils/mappers.py
# -----------------------------------------------------------------------------
# Purpose
# Canonical adapters (mappers) between unified DTOs (api/schema/*) and
# the project's current runtime shapes used for:
#   - WebSocket events (UI-expected payloads for optimistic updates)
#   - ChangeSet submit to Tazila (send-now or future staging backend)
#
# Notes
# - Input validation is already done by Pydantic DTOs (AssetImpact, AssetRisk).
# - We keep the 0..4 scale as-is here; any mapping to domain values (e.g., 5,20,50,70,100)
#   should remain in the submit layer if it already exists there.
# - If an idempotency helper is available, we include a deterministic key; otherwise we skip it.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional

from workshops_ai.api.schema import AssetImpact, ImpactEvaluation, AssetRisk

# Optional idempotency helper (present in many setups)
try:
    from workshops_ai.agents.utils.changeset_builder import make_idempotency_key  # type: ignore
except Exception:  # pragma: no cover
    make_idempotency_key = None  # type: ignore[misc,assignment]


__all__ = [
    "map_asset_impact_to_ws_event",
    "map_asset_impact_to_changeset",
    "map_asset_risk_to_ws_event",
    "map_asset_risk_to_changeset",
]


# -----------------------------------------------------------------------------
# Impacts (AssetImpact)
# -----------------------------------------------------------------------------

def map_asset_impact_to_ws_event(dto: AssetImpact) -> Dict[str, Any]:
    """
    Convert AssetImpact DTO -> WebSocket payload in the UI-expected shape.
    No scale conversion (keeps 0..4).
    Example:
    {
      "assetId": "uuid|None",
      "assetName": "Customer Database",
      "impacts": [
        {"impactName": "Confidentiality", "impactLevel": 4},
        {"impactName": "Integrity",       "impactLevel": 3},
        {"impactName": "Availability",    "impactLevel": 3}
      ]
    }
    """
    return {
        "assetId": str(dto.assetId) if getattr(dto, "assetId", None) else None,
        "assetName": dto.assetName,
        "impacts": [
            {
                "impactName": it.impactName,
                "impactLevel": it.impactLevel,
            }
            for it in (dto.impacts or [])
        ],
    }


def map_asset_impact_to_changeset(
    dto: AssetImpact,
    org_id: str,
    *,
    include_idempotency: bool = True,
) -> Dict[str, Any]:
    """
    Convert AssetImpact DTO -> ChangeSet payload for Tazila submit.
    Keeps a runtime-compatible shape with today's submit layer.
    Example:
    {
      "org_id": "demo-org",
      "asset_id": "uuid|None",
      "asset_name": "Customer Database",
      "items": [
        {"impact_code": "Confidentiality", "level": 4},
        {"impact_code": "Integrity",       "level": 3},
        {"impact_code": "Availability",    "level": 3}
      ],
      "idempotency_key": "..."  # optional
    }
    """
    items: List[Dict[str, Any]] = [
        {"impact_code": it.impactName, "level": it.impactLevel}
        for it in (dto.impacts or [])
    ]

    payload: Dict[str, Any] = {
        "org_id": org_id,
        "asset_id": str(dto.assetId) if getattr(dto, "assetId", None) else None,
        "asset_name": dto.assetName,
        "items": items,
    }

    if include_idempotency and make_idempotency_key:
        # Stable, short-lived key to prevent duplicate applies on quick re-submits.
        # Seed: org_id + asset reference + ordered list of actions.
        payload["idempotency_key"] = make_idempotency_key(
            org_id,
            actions=items,
        )

    return payload


# -----------------------------------------------------------------------------
# Threats (AssetRisk)
# -----------------------------------------------------------------------------

def map_asset_risk_to_ws_event(dto: AssetRisk) -> Dict[str, Any]:
    """
    Convert AssetRisk DTO -> WebSocket payload in the UI-expected shape.
    No scale conversion (keeps 0..4).
    Example:
    {
      "assetId": "uuid|None",
      "assetName": "Backend API Server",
      "risks": [
        {"threatName": "Fire", "probability": 2},
        {"threatName": "Water, humidity", "probability": 1}
      ]
    }
    """
    return {
        "assetId": str(dto.assetId) if getattr(dto, "assetId", None) else None,
        "assetName": dto.assetName,
        "risks": [
            {
                "threatName": r.threatName,
                "probability": r.probability,
            }
            for r in (dto.risks or [])
        ],
    }


def map_asset_risk_to_changeset(
    dto: AssetRisk,
    org_id: str,
    *,
    include_idempotency: bool = True,
) -> Dict[str, Any]:
    """
    Convert AssetRisk DTO -> ChangeSet payload for Tazila submit.
    Keeps a runtime-compatible shape with today's submit layer.
    Example:
    {
      "org_id": "demo-org",
      "asset_id": "uuid|None",
      "asset_name": "Backend API Server",
      "items": [
        {"threat_code": "Fire", "probability": 2},
        {"threat_code": "Water, humidity", "probability": 1}
      ],
      "idempotency_key": "..."  # optional
    }
    """
    items: List[Dict[str, Any]] = [
        {"threat_code": r.threatName, "probability": r.probability}
        for r in (dto.risks or [])
    ]

    payload: Dict[str, Any] = {
        "org_id": org_id,
        "asset_id": str(dto.assetId) if getattr(dto, "assetId", None) else None,
        "asset_name": dto.assetName,
        "items": items,
    }

    if include_idempotency and make_idempotency_key:
        payload["idempotency_key"] = make_idempotency_key(
            org_id,
            actions=items,
        )

    return payload
