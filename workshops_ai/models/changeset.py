# File: workshops_ai/models/changeset.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Annotated, Union, Literal

from pydantic import (
    BaseModel,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    ConfigDict,
    model_validator,
)

# -----------------------------------------------------------------------------
# Audit envelope (strict, snake_case on the wire)
# -----------------------------------------------------------------------------

class Audit(BaseModel):
    """
    Unified audit envelope for all ChangeSets.
    Snake_case everywhere (including JSON field names).
    """
    model_config = ConfigDict(extra="forbid")

    user_id: StrictStr
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: StrictStr = "AI"
    tags: list[StrictStr] = Field(default_factory=list)

    org_id: Optional[StrictStr] = None
    session_id: Optional[StrictStr] = None
    request_id: Optional[StrictStr] = None
    batch_id: Optional[StrictStr] = None


# -----------------------------------------------------------------------------
# Item models (strict, fail-closed)
# -----------------------------------------------------------------------------

class ImpactItem(BaseModel):
    """Single impact level entry for a specific impact code (level 0..4)."""
    model_config = ConfigDict(extra="forbid")

    impact_code: StrictStr
    level: StrictInt = Field(ge=0, le=4)
    note: Optional[str] = None


class RiskItem(BaseModel):
    """Single risk probability entry for a specific threat code (0..4)."""
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
    asset_type_codes: list[StrictStr] = Field(default_factory=list)

    # flags
    intentional: bool = False
    unintentional: bool = False
    environmental: bool = False
    description: Optional[str] = None


class ImpactUpsert(BaseModel):
    """
    Impact upsert payload; mirrors catalog semantics.
    - If `code` is provided, update/ensure; otherwise create (server may assign code).
    - `category` is the canonical EN category for impacts (e.g., Availability).
    """
    model_config = ConfigDict(extra="forbid")

    code: Optional[StrictStr] = None
    name: StrictStr
    category: StrictStr
    description: Optional[str] = None


# -----------------------------------------------------------------------------
# Action models (strict, fail-closed) — FIX: op is Literal[...] for discriminator
# -----------------------------------------------------------------------------

class ActionSetImpacts(BaseModel):
    """
    Set/merge impact levels for the given asset (or client-side reference).
    - `mode="merge"` updates/inserts provided items.
    - `mode="replace"` replaces all existing items with the provided list.
    """
    model_config = ConfigDict(extra="forbid")

    op: Literal["set_impacts"] = "set_impacts"
    mode: Literal["merge", "replace"] = "merge"
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None  # e.g. "client:<token>"
    items: list[ImpactItem]

    @model_validator(mode="after")
    def _require_asset_selector(self) -> "ActionSetImpacts":
        if not (self.asset_id or self.asset_ref):
            raise ValueError("One of 'asset_id' or 'asset_ref' must be provided.")
        return self


class ActionSetRisks(BaseModel):
    """
    Set/merge risk probabilities for the given asset (or client-side reference).
    Probability is strictly 0..4 (already normalized before reaching backend).
    """
    model_config = ConfigDict(extra="forbid")

    op: Literal["set_risks"] = "set_risks"
    mode: Literal["merge", "replace"] = "merge"
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None
    items: list[RiskItem]

    @model_validator(mode="after")
    def _require_asset_selector(self) -> "ActionSetRisks":
        if not (self.asset_id or self.asset_ref):
            raise ValueError("One of 'asset_id' or 'asset_ref' must be provided.")
        return self


class ActionUpsertControls(BaseModel):
    """Upsert a set of controls (per asset or client-side reference)."""
    model_config = ConfigDict(extra="forbid")

    op: Literal["upsert_controls"] = "upsert_controls"
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None

    items: list[ControlUpsert]

    @model_validator(mode="after")
    def _require_asset_selector(self) -> "ActionUpsertControls":
        if not (self.asset_id or self.asset_ref):
            raise ValueError("One of 'asset_id' or 'asset_ref' must be provided.")
        return self


class ActionRemoveControls(BaseModel):
    """Remove listed controls (by code) from the given asset/ref."""
    model_config = ConfigDict(extra="forbid")

    op: Literal["remove_controls"] = "remove_controls"
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None

    codes: list[StrictStr]

    @model_validator(mode="after")
    def _require_asset_selector(self) -> "ActionRemoveControls":
        if not (self.asset_id or self.asset_ref):
            raise ValueError("One of 'asset_id' or 'asset_ref' must be provided.")
        return self


class ActionSetAssetMeta(BaseModel):
    """
    Update arbitrary asset metadata fields (server enforces allowed keys).
    Example fields: name, asset_type_uuid/asset_type_code, primary, CIA, ...
    """
    model_config = ConfigDict(extra="forbid")

    op: Literal["set_asset_meta"] = "set_asset_meta"
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None

    fields: dict[str, Any]

    @model_validator(mode="after")
    def _require_asset_selector(self) -> "ActionSetAssetMeta":
        if not (self.asset_id or self.asset_ref):
            raise ValueError("One of 'asset_id' or 'asset_ref' must be provided.")
        return self


class ActionCreateAsset(BaseModel):
    """
    Create a new asset; `client_token` lets the client correlate response → request.
    Example fields: name, asset_type_uuid/asset_type_code, primary, ...
    """
    model_config = ConfigDict(extra="forbid")

    op: Literal["create_asset"] = "create_asset"
    client_token: StrictStr
    fields: dict[str, Any]


class ActionUpsertThreats(BaseModel):
    """
    Upsert threats in org scope (not tied to a single asset).
    Each item corresponds to one threat (create/update).
    """
    model_config = ConfigDict(extra="forbid")

    op: Literal["upsert_threats"] = "upsert_threats"
    items: list[ThreatUpsert]


class ActionUpsertImpacts(BaseModel):
    """
    Upsert impacts in org scope (not tied to a single asset).
    Each item corresponds to one impact (create/update).
    """
    model_config = ConfigDict(extra="forbid")

    op: Literal["upsert_impacts"] = "upsert_impacts"
    items: list[ImpactUpsert]


Action = Annotated[
    Union[
        ActionCreateAsset,
        ActionSetAssetMeta,
        ActionSetImpacts,
        ActionSetRisks,
        ActionUpsertControls,
        ActionRemoveControls,
        ActionUpsertThreats,
        ActionUpsertImpacts,
    ],
    Field(discriminator="op"),
]


# -----------------------------------------------------------------------------
# ChangeSet envelope (strict)
# -----------------------------------------------------------------------------

class ChangeSet(BaseModel):
    """
    Canonical batch envelope for mutating operations.
    - `schema_version` pins the contract version.
    - `idempotency_key` is required; server should reject/ignore duplicates.
    - `audit` is a unified snake_case model with keys:
      user_id, ts, source, tags, org_id, session_id, request_id, batch_id
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    idempotency_key: StrictStr
    org_id: StrictStr
    intent: Literal["preview", "apply"] = "apply"

    actions: list[Action]

    audit: Audit


__all__ = [
    "Audit",
    "ImpactItem",
    "RiskItem",
    "ControlUpsert",
    "ThreatUpsert",
    "ImpactUpsert",
    "ActionSetImpacts",
    "ActionSetRisks",
    "ActionUpsertControls",
    "ActionRemoveControls",
    "ActionSetAssetMeta",
    "ActionCreateAsset",
    "ActionUpsertThreats",
    "ActionUpsertImpacts",
    "Action",
    "ChangeSet",
]
