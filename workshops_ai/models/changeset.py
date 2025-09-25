# /app/models/changeset.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, StrictFloat, StrictInt, StrictStr, ConfigDict


# -----------------------------------------------------------------------------
# Item models (strict, fail-closed)
# -----------------------------------------------------------------------------

class ImpactItem(BaseModel):
    """Single impact level entry for a specific impact code."""
    model_config = ConfigDict(extra="forbid")
    impact_code: StrictStr
    level: StrictInt = Field(ge=0, le=4)
    note: Optional[str] = None


class RiskItem(BaseModel):
    """Single risk probability entry for a specific threat code."""
    model_config = ConfigDict(extra="forbid")
    threat_code: StrictStr
    probability: StrictInt = Field(ge=0, le=4)
    note: Optional[str] = None


class ControlUpsert(BaseModel):
    """Control proposal/upsert with optional expected coverage (0.0..1.0)."""
    model_config = ConfigDict(extra="forbid")
    control_code: StrictStr
    rationale: Optional[str] = None
    expected_coverage: Optional[StrictFloat] = Field(default=None, ge=0.0, le=1.0)


class ThreatUpsert(BaseModel):
    """
    Threat upsert payload; mirrors catalog semantics.
    - If `code` is provided, update/ensure; otherwise create (server may assign code).
    - `asset_type_codes` is a list of canonical EN asset-type codes.
    - Flags represent intentional/unintentional/environmental categorization.
    """
    model_config = ConfigDict(extra="forbid")
    code: Optional[StrictStr] = None
    name: StrictStr
    category: StrictStr
    asset_type_codes: List[StrictStr] = Field(default_factory=list)
    # flags
    intentional: bool = False
    unintentional: bool = False
    environmental: bool = False
    description: Optional[str] = None


# -----------------------------------------------------------------------------
# Action models (strict, fail-closed)
# -----------------------------------------------------------------------------

class ActionSetImpacts(BaseModel):
    """
    Set/merge impact levels for the given asset (or client-side reference).
    - `mode="merge"` updates/inserts provided items.
    - `mode="replace"` replaces all existing items with the provided list.
    """
    model_config = ConfigDict(extra="forbid")
    op: Literal["set_impacts"]
    mode: Literal["merge", "replace"] = "merge"
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None  # e.g. "client:<token>"
    items: List[ImpactItem]


class ActionSetRisks(BaseModel):
    """
    Set/merge risk probabilities for the given asset (or client-side reference).
    Probability is strictly 0..4 (already normalized before reaching backend).
    """
    model_config = ConfigDict(extra="forbid")
    op: Literal["set_risks"]
    mode: Literal["merge", "replace"] = "merge"
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None
    items: List[RiskItem]


class ActionUpsertControls(BaseModel):
    """Upsert a set of controls (per asset or client-side reference)."""
    model_config = ConfigDict(extra="forbid")
    op: Literal["upsert_controls"]
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None
    items: List[ControlUpsert]


class ActionRemoveControls(BaseModel):
    """Remove listed controls (by code) from the given asset/ref."""
    model_config = ConfigDict(extra="forbid")
    op: Literal["remove_controls"]
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None
    codes: List[StrictStr]


class ActionSetAssetMeta(BaseModel):
    """
    Update arbitrary asset metadata fields (server enforces allowed keys).
    Example fields: name, asset_type_uuid/asset_type_code, primary, CIA, ...
    """
    model_config = ConfigDict(extra="forbid")
    op: Literal["set_asset_meta"]
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None
    fields: Dict[str, Any]


class ActionCreateAsset(BaseModel):
    """
    Create a new asset; `client_token` lets the client correlate response → request.
    Example fields: name, asset_type_uuid/asset_type_code, primary, ...
    """
    model_config = ConfigDict(extra="forbid")
    op: Literal["create_asset"]
    client_token: StrictStr
    fields: Dict[str, Any]


class ActionUpsertThreats(BaseModel):
    """
    Upsert threats in org scope (not tied to a single asset).
    Each item corresponds to one threat (create/update).
    """
    model_config = ConfigDict(extra="forbid")
    op: Literal["upsert_threats"]
    items: List[ThreatUpsert]


# Union of all action types
Action = (
    ActionCreateAsset
    | ActionSetAssetMeta
    | ActionSetImpacts
    | ActionSetRisks
    | ActionUpsertControls
    | ActionRemoveControls
    | ActionUpsertThreats
)


# -----------------------------------------------------------------------------
# ChangeSet envelope (strict)
# -----------------------------------------------------------------------------

class ChangeSet(BaseModel):
    """
    Canonical batch envelope for mutating operations.
    - `schema_version` pins the contract version.
    - `idempotency_key` is required; server should reject/ignore duplicates.
    - `audit` must include tags/userId/timestamp/source as per project rules.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    idempotency_key: StrictStr
    org_id: StrictStr
    intent: Literal["preview", "apply"] = "apply"
    audit: Dict[str, Any]
    actions: List[Action]


__all__ = [
    "ImpactItem",
    "RiskItem",
    "ControlUpsert",
    "ThreatUpsert",
    "ActionSetImpacts",
    "ActionSetRisks",
    "ActionUpsertControls",
    "ActionRemoveControls",
    "ActionSetAssetMeta",
    "ActionCreateAsset",
    "ActionUpsertThreats",
    "Action",
    "ChangeSet",
]
