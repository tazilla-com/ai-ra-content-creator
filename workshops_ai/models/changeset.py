# File: workshops_ai/models/changeset.py
from __future__ import annotations

import hashlib
import json
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
    TypeAdapter,
    field_serializer,
    field_validator,
)

# =============================================================================
# Optional op-name constants (handy for call-sites/tests; not used in Literals)
# =============================================================================

OP_CREATE_ASSET                     = "create_asset"
OP_SET_ASSET_META                   = "set_asset_meta"
OP_SET_IMPACTS                      = "set_impacts"
OP_SET_RISKS                        = "set_risks"
OP_UPSERT_CONTROLS                  = "upsert_controls"
OP_REMOVE_CONTROLS                  = "remove_controls"
OP_UPSERT_THREATS                   = "upsert_threats"
OP_UPSERT_IMPACTS                   = "upsert_impacts"
OP_ASSET_UPSERT_THREAT_PRESENCE     = "asset_upsert_threat_presence"

# =============================================================================
# CIA enums for normalization/validation in CreateAssetFields
# =============================================================================

CONF_LEVELS = {"PUBLIC", "INTERNAL", "PROTECTED", "STRICTLY_PROTECTED"}
TRI_LEVELS = {"LOW", "MEDIUM", "HIGH"}

# =============================================================================
# Audit envelope (strict, snake_case on the wire)
# =============================================================================

class Audit(BaseModel):
    """
    Unified audit envelope for all ChangeSets.
    Field names are snake_case in both Python and JSON.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    user_id: StrictStr
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: StrictStr = "AI"
    tags: list[StrictStr] = Field(default_factory=list)

    org_id: Optional[StrictStr] = None
    session_id: Optional[StrictStr] = None
    request_id: Optional[StrictStr] = None
    batch_id: Optional[StrictStr] = None

    # Ensure JSON-safe ISO 8601 serialization (even outside model_dump(mode="json"))
    @field_serializer("ts")
    def _ser_ts(self, v: datetime) -> str:
        return v.isoformat()

# =============================================================================
# Item models (strict, fail-closed)
# =============================================================================

class ImpactItem(BaseModel):
    """Single impact level entry for a specific impact code (level 0..4)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    impact_code: StrictStr
    level: StrictInt = Field(ge=0, le=4)
    note: Optional[str] = None

    @field_validator("impact_code")
    @classmethod
    def _trim_code(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("impact_code must be a non-empty string")
        return v2


class RiskItem(BaseModel):
    """Single risk probability entry for a specific threat code (0..4)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    threat_code: StrictStr
    probability: StrictInt = Field(ge=0, le=4)
    note: Optional[str] = None

    @field_validator("threat_code")
    @classmethod
    def _trim_code(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("threat_code must be a non-empty string")
        return v2


class ControlUpsert(BaseModel):
    """Control proposal/upsert with optional expected coverage (0.0..1.0)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    control_code: StrictStr
    rationale: Optional[str] = None
    expected_coverage: Optional[StrictFloat] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("control_code", "rationale")
    @classmethod
    def _trim_opt(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v2 = v.strip()
        return v2 or None

# =============================================================================
# Catalog upserts (org-scope, not tied to a single asset)
# =============================================================================

class ThreatUpsert(BaseModel):
    """
    Threat upsert payload; mirrors catalog semantics.
    - If `code` is provided, update/ensure; otherwise create (server may assign code).
    - `asset_type_codes` is a list of canonical EN asset-type codes.
    - Flags represent intentional/unintentional/environmental categorization.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    code: Optional[StrictStr] = None
    name: StrictStr
    category: StrictStr
    asset_type_codes: list[StrictStr] = Field(default_factory=list)

    # flags
    intentional: bool = False
    unintentional: bool = False
    environmental: bool = False
    description: Optional[str] = None

    # Trim + basic non-empty checks (backwards compatible)
    @field_validator("code", "name", "category", "description")
    @classmethod
    def _trim_fields(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return v
        v2 = v.strip()
        # Require non-empty for required fields
        if info.field_name in ("name", "category") and not v2:
            raise ValueError(f"{info.field_name} must be a non-empty string")
        # For optional fields, normalize empty -> None
        return v2 or None

    @model_validator(mode="after")
    def _normalize_asset_types(self) -> "ThreatUpsert":
        """
        Normalize asset_type_codes: trim, drop empties, preserve order, deduplicate.
        """
        if self.asset_type_codes:
            cleaned: list[str] = []
            seen: set[str] = set()
            for x in self.asset_type_codes:
                s = x.strip()
                if s and s not in seen:
                    seen.add(s)
                    cleaned.append(s)
            self.asset_type_codes = cleaned
        return self


class ImpactUpsert(BaseModel):
    """
    Impact upsert payload; mirrors catalog semantics.
    - If `code` is provided, update/ensure; otherwise create (server may assign code).
    - `category` is the canonical EN category for impacts (e.g., Availability).
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    code: Optional[StrictStr] = None
    name: StrictStr
    category: StrictStr
    description: Optional[str] = None
    weight: Optional[StrictFloat] = None

    # Trim + basic non-empty checks (backwards compatible)
    @field_validator("code", "name", "category", "description")
    @classmethod
    def _trim_fields(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return v
        v2 = v.strip()
        if info.field_name in ("name", "category") and not v2:
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return v2 or None

# =============================================================================
# Action models (strict, fail-closed) — discriminator is Literal[...] `op`
# =============================================================================

class ActionSetImpacts(BaseModel):
    """
    Set/merge impact levels for the given asset (or client-side reference).
    - `mode="merge"` updates/inserts provided items.
    - `mode="replace"` replaces all existing items with the provided list.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    op: Literal["set_impacts"] = OP_SET_IMPACTS
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
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    op: Literal["set_risks"] = OP_SET_RISKS
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
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    op: Literal["upsert_controls"] = OP_UPSERT_CONTROLS
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None

    items: list[ControlUpsert]

    @model_validator(mode="after")
    def _require_asset_selector(self) -> "ActionUpsertControls":
        if not (self.asset_id or self.asset_ref):
            raise ValueError("One of 'asset_id' or 'asset_ref' must be provided.")
        return self


class ActionRemoveControls(BaseModel):
    """Remove listed controls (by code) from the given asset/ref)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    op: Literal["remove_controls"] = OP_REMOVE_CONTROLS
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
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    op: Literal["set_asset_meta"] = OP_SET_ASSET_META
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None

    fields: dict[str, Any]

    @model_validator(mode="after")
    def _require_asset_selector(self) -> "ActionSetAssetMeta":
        if not (self.asset_id or self.asset_ref):
            raise ValueError("One of 'asset_id' or 'asset_ref' must be provided.")
        return self

# -----------------------------------------------------------------------------
# Typed create payload with optional isPrimary and conditional CIA requirement
# -----------------------------------------------------------------------------

class CreateAssetFields(BaseModel):
    """
    Strict payload for the create_asset operation.

    Policy:
      - Required: name, assetType.
      - Optional: isPrimary (bool), description, id.
      - CIA (confidentiality/integrity/availability) is required ONLY when
        isPrimary=True. Otherwise CIA must be None and is ignored by the server.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Required core fields
    name: StrictStr
    assetType: StrictStr

    # Optional flags/metadata
    isPrimary: Optional[bool] = None
    confidentiality: Optional[str] = None
    integrity: Optional[str] = None
    availability: Optional[str] = None
    description: Optional[str] = None

    # Optional client-supplied id/reference passthrough
    id: Optional[StrictStr] = None

    @field_validator("name", "assetType")
    @classmethod
    def _trim_required_nonempty(cls, v: str, info) -> str:
        """Trim and ensure required strings remain non-empty."""
        s = v.strip()
        if not s:
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return s

    @field_validator("confidentiality")
    @classmethod
    def _norm_conf(cls, v: Optional[str]) -> Optional[str]:
        """Normalize confidentiality to UPPER_SNAKE and validate against whitelist."""
        if v is None:
            return None
        x = v.strip().upper().replace("-", "_").replace(" ", "_")
        if x == "STRICTLYPROTECTED":
            x = "STRICTLY_PROTECTED"
        return x if x in CONF_LEVELS else None

    @field_validator("integrity", "availability")
    @classmethod
    def _norm_tri(cls, v: Optional[str]) -> Optional[str]:
        """Normalize integrity/availability to UPPER_SNAKE and validate against whitelist."""
        if v is None:
            return None
        x = v.strip().upper().replace("-", "_").replace(" ", "_")
        if x == "MEDIUM_":
            x = "MEDIUM"
        return x if x in TRI_LEVELS else None

    @model_validator(mode="after")
    def _gate_required_and_cia(self) -> "CreateAssetFields":
        """
        Enforce CIA only for primary assets; null out CIA for supporting assets
        or when isPrimary is not specified.
        """
        if self.isPrimary is True:
            missing = [k for k in ("confidentiality", "integrity", "availability") if getattr(self, k) in (None, "")]
            if missing:
                raise ValueError(f"primary asset requires CIA fields, missing: {', '.join(missing)}")
        else:
            # Supporting (or unspecified) assets must not carry CIA in the payload
            self.confidentiality = None
            self.integrity = None
            self.availability = None
        return self

class ActionCreateAsset(BaseModel):
    """
    Create a new asset; `client_token` lets the client correlate response → request.
    Example fields: name, asset_type_uuid/asset_type_code, primary, ...
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    op: Literal["create_asset"] = OP_CREATE_ASSET
    client_token: StrictStr
    # Typed & validated payload instead of an arbitrary dict
    fields: CreateAssetFields

    @field_validator("client_token")
    @classmethod
    def _non_empty_client_token(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("client_token must be a non-empty string")
        return v2


class ActionUpsertThreats(BaseModel):
    """
    Upsert threats in org scope (not tied to a single asset).
    Each item corresponds to one threat (create/update).
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    op: Literal["upsert_threats"] = OP_UPSERT_THREATS
    items: list[ThreatUpsert]


class ActionUpsertImpacts(BaseModel):
    """
    Upsert impacts in org scope (not tied to a single asset).
    Each item corresponds to one impact (create/update).
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    op: Literal["upsert_impacts"] = OP_UPSERT_IMPACTS
    items: list[ImpactUpsert]


class ActionAssetUpsertThreatPresence(BaseModel):
    """
    Ensure presence/absence of a given threat on a specific asset.
    - present=True  → add/ensure presence
    - present=False → remove presence (some backends may queue this to OUTBOX)
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    op: Literal["asset_upsert_threat_presence"] = OP_ASSET_UPSERT_THREAT_PRESENCE
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None
    threat_code: StrictStr
    present: bool = True
    note: Optional[str] = None

    @field_validator("threat_code")
    @classmethod
    def _trim_code(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("threat_code must be a non-empty string")
        return v2

    @field_validator("note")
    @classmethod
    def _trim_note(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v2 = v.strip()
        return v2 or None

    @model_validator(mode="after")
    def _require_asset_selector(self) -> "ActionAssetUpsertThreatPresence":
        if not (self.asset_id or self.asset_ref):
            raise ValueError("One of 'asset_id' or 'asset_ref' must be provided.")
        return self


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
        ActionAssetUpsertThreatPresence,  
    ],
    Field(discriminator="op"),
]

# =============================================================================
# ChangeSet envelope (strict) + deterministic builder
# =============================================================================

class ChangeSet(BaseModel):
    """
    Canonical batch envelope for mutating operations.
    - `schema_version` pins the contract version.
    - `idempotency_key` is required; server should reject/ignore duplicates.
    - `audit` is a unified snake_case model with keys:
      user_id, ts, source, tags, org_id, session_id, request_id, batch_id
    - `intent` controls semantics:
        * "preview" → dry-run/validate only (no writes)
        * "apply"   → perform writes (server-side)
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: Literal["1.0"] = "1.0"
    idempotency_key: StrictStr
    org_id: StrictStr
    intent: Literal["preview", "apply"] = "apply"

    actions: list[Action]
    audit: Audit

    @model_validator(mode="after")
    def _validate_non_empty(self) -> "ChangeSet":
        if not self.actions:
            raise ValueError("actions must not be empty")
        if not self.idempotency_key:
            raise ValueError("idempotency_key must not be empty")
        return self

# ---- Deterministic idempotency helpers --------------------------------------

def _canonical_dump(obj: Any) -> str:
    """
    Deterministic JSON dump:
    - sort keys
    - no whitespace variance
    - ensure_ascii=False (but content is typically ASCII)
    - IMPORTANT: use JSON mode for Pydantic models so datetimes etc. are safe
    """
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def _actions_digest(actions: list[Action]) -> str:
    """
    Create a stable digest of the action list that is insensitive to
    incidental formatting. Assumes items already normalized by the agent.
    """
    payload = [_canonical_dump(a) for a in actions]
    joined = "[" + ",".join(payload) + "]"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

def _make_idempotency_key(
    *,
    org_id: str,
    user_id: str,
    intent: Literal["preview", "apply"],
    actions: list[Action],
) -> str:
    """
    Build a stable idempotency key from (org, user, intent, actionsDigest).
    Intent is included so that preview/apply for the same actions do not collide.
    """
    digest = _actions_digest(actions)
    base = f"cs:v1:{org_id}:{user_id}:{intent}:{digest}"
    return base

# Lightweight helper for logging/tests (optional public utility)
def actions_digest_for_logging(actions: list[Action]) -> str:
    """Expose the normalized actions digest for logs/snapshots."""
    return _actions_digest(actions)

# ---- Public builder used by agents/graphs -----------------------------------

def build_changeset(
    *,
    org_id: str,
    user_id: str,
    actions: list[dict] | list[Action],
    intent: Literal["preview", "apply"] = "apply",
    idempotency_key: Optional[str] = None,
    session_id: Optional[StrictStr] = None,
    request_id: Optional[StrictStr] = None,
    batch_id: Optional[StrictStr] = None,
    source: str = "AI",
    tags: Optional[list[StrictStr]] = None,
) -> ChangeSet:
    """
    Deterministic ChangeSet builder.
    - Accepts plain dict actions or already-validated Action models.
    - Produces a stable idempotency_key if not provided.
    - Adds `preview` tag to audit.tags when intent == "preview" (non-breaking).
    """
    # Normalize actions into typed models with Pydantic v2 TypeAdapter
    action_adapter = TypeAdapter(list[Action])
    typed_actions: list[Action] = action_adapter.validate_python(actions)

    # Compute deterministic idempotency_key (separate for preview/apply)
    idem = idempotency_key or _make_idempotency_key(
        org_id=org_id, user_id=user_id, intent=intent, actions=typed_actions
    )

    # Prepare audit
    audit_tags = list(tags or [])
    if intent == "preview" and "preview" not in audit_tags:
        audit_tags.append("preview")

    audit = Audit(
        user_id=user_id,
        source=source,
        tags=audit_tags,
        org_id=org_id,
        session_id=session_id,
        request_id=request_id,
        batch_id=batch_id,
    )

    return ChangeSet(
        idempotency_key=idem,
        org_id=org_id,
        intent=intent,
        actions=typed_actions,
        audit=audit,
    )

# ---- Convenience builder for asset-scoped threat presence -------------------

def build_upsert_threat_presence(
    org_id: str,
    user_id: str,
    asset_id: str,
    threat_code: str,
    present: bool,
    note: Optional[str] = None,
) -> ChangeSet:
    """
    Build a ChangeSet that toggles presence of a specific threat on an asset.

    Intent:
      - present=True  → ensure the threat is linked to the asset
      - present=False → remove the threat link (CSV backend lacks hard delete,
                        so submit may fall back to OUTBOX — this is OK)

    Returns:
      ChangeSet with a single ActionAssetUpsertThreatPresence.
    """
    action = ActionAssetUpsertThreatPresence(
        asset_id=asset_id,
        threat_code=threat_code,
        present=bool(present),
        note=(note.strip() or None) if isinstance(note, str) else None,
    )
    return build_changeset(
        org_id=org_id,
        user_id=user_id,
        actions=[action],
        intent="apply",
    )

__all__ = [
    # constants
    "OP_CREATE_ASSET",
    "OP_SET_ASSET_META",
    "OP_SET_IMPACTS",
    "OP_SET_RISKS",
    "OP_UPSERT_CONTROLS",
    "OP_REMOVE_CONTROLS",
    "OP_UPSERT_THREATS",
    "OP_UPSERT_IMPACTS",
    "OP_ASSET_UPSERT_THREAT_PRESENCE",
    # models
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
    "CreateAssetFields",
    "ActionCreateAsset",
    "ActionUpsertThreats",
    "ActionUpsertImpacts",
    "ActionAssetUpsertThreatPresence",
    "Action",
    "ChangeSet",
    # helpers
    "actions_digest_for_logging",
    "build_changeset",
    "build_upsert_threat_presence",
]
