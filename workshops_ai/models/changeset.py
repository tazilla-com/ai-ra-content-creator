# File: workshops_ai/models/changeset.py
from __future__ import annotations

import hashlib
import json
from uuid import uuid4
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
OP_SET_ASSET_CIA                    = "set_asset_cia"

# =============================================================================
# CIA enums for normalization/validation in CreateAssetFields
# (Keep STRICT enums – only the values listed below are accepted.)
# =============================================================================

CONF_LEVELS = {"PUBLIC", "INTERNAL", "PROTECTED", "STRICTLY_PROTECTED"}
TRI_LEVELS  = {"LOW", "MEDIUM", "HIGH"}

# =============================================================================
# AssetType whitelist + synonyms
# -----------------------------------------------------------------------------
# - The backend schema has a FIXED set of allowed assetType values (EN only).
# - Incoming values from mappers/extractors may use "friendly" names
#   (e.g., "Web application", "Database"). We normalize such synonyms here.
# - If a value can't be normalized to a whitelisted code, validation fails
#   early inside CreateAssetFields (before reaching the backend).
# =============================================================================

ASSET_TYPE_CODES: set[str] = {
    "Application software", "Cloud services", "Data storage",
    "Electronic data/Business process", "Other", "Information system",
    "IoT", "Location", "Mobile devices", "Laptop", "Operating system",
    "Paper data", "PC", "Staff", "Portable data media",
    "Server infrastructure", "Network infrastructure",
    "Virtualization platform", "Equipment, machines",
}

# Case-insensitive synonyms -> canonical codes from ASSET_TYPE_CODES
ASSET_TYPE_SYNONYMS: dict[str, str] = {
    # common categories seen in mappers/extractors
    "web application": "Application software",
    "mobile application": "Application software",
    "monitoring system": "Application software",
    "devops toolchain": "Application software",

    "database": "Data storage",
    "data warehouse": "Data storage",

    "identity provider": "Cloud services",
    "idp": "Cloud services",
    # feel free to extend as needed:
    "it service": "Cloud services",
}

def normalize_asset_type(v: str) -> str:
    """
    Normalize a human/friendly assetType to the canonical catalog code.

    Rules:
      - preserve exact matches against ASSET_TYPE_CODES
      - try case-insensitive synonyms from ASSET_TYPE_SYNONYMS
      - otherwise return the original string (validator will reject later)

    Note:
      Keep values strictly in English; backend expects canonical EN labels.
    """
    s = (v or "").strip()
    if not s:
        return s
    if s in ASSET_TYPE_CODES:
        return s
    alt = ASSET_TYPE_SYNONYMS.get(s.lower())
    return alt if alt in ASSET_TYPE_CODES else s

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
            # Avoid mutation warnings under pydantic v2 validators
            object.__setattr__(self, "asset_type_codes", cleaned)
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

# --- CIA update (asset-scoped) ----------------------------------------------

class CIAUpdateFields(BaseModel):
    """
    CIA levels as numeric policy values:
      - confidentiality: 1..4
      - integrity:       1..3
      - availability:    1..3

    None means "clear the value". Zero must not appear here (callers should
    map it to None upstream if they use zero as a sentinel).
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    confidentiality: Optional[StrictInt] = Field(default=None, ge=1, le=4)
    integrity:       Optional[StrictInt] = Field(default=None, ge=1, le=3)
    availability:    Optional[StrictInt] = Field(default=None, ge=1, le=3)

    @model_validator(mode="after")
    def _at_least_one(self) -> "CIAUpdateFields":
        if (self.confidentiality is None and
            self.integrity       is None and
            self.availability    is None):
            raise ValueError("At least one of confidentiality/integrity/availability must be provided")
        return self


class ActionSetAssetCIA(BaseModel):
    """
    Set/clear CIA values for a given asset.
    - Zero values are not allowed (should be mapped to None by the caller).
    - Requires either asset_id or asset_ref.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    op: Literal["set_asset_cia"] = OP_SET_ASSET_CIA
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None
    fields: CIAUpdateFields

    @model_validator(mode="after")
    def _require_asset_selector(self) -> "ActionSetAssetCIA":
        if not (self.asset_id or self.asset_ref):
            raise ValueError("One of 'asset_id' or 'asset_ref' must be provided.")
        return self

# -----------------------------------------------------------------------------
# Typed create payload with conditional CIA policy (STRICT enums)
# -----------------------------------------------------------------------------

class CreateAssetFields(BaseModel):
    """
    Strict payload for the create_asset operation.

    Policy:
      - Required: name, assetType.
      - Optional: isPrimary (bool), description, id.
      - CIA (confidentiality/integrity/availability) is required ONLY when
        isPrimary=True. Otherwise CIA is forcibly set to None.
      - "Strict enums": only {PUBLIC, INTERNAL, PROTECTED, STRICTLY_PROTECTED}
        for confidentiality and {LOW, MEDIUM, HIGH} for integrity/availability
        are accepted (case/spacing normalized). Any other value is rejected
        for primary assets and nulled for supporting assets.
      - assetType must be one of the FIXED catalog values (EN only).
        Synonyms are normalized; unknown values raise a validation error.
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

    # --- Basic trimming for required strings ---------------------------------
    @field_validator("name", "assetType")
    @classmethod
    def _trim_required_nonempty(cls, v: str, info) -> str:
        """Trim and ensure required strings remain non-empty."""
        s = v.strip()
        if not s:
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return s

    # --- Enforce canonical assetType against the fixed code list -------------
    @field_validator("assetType")
    @classmethod
    def _asset_type_enum(cls, v: str) -> str:
        """
        Normalize + validate assetType against the fixed catalog.

        - exact match -> OK
        - synonym -> mapped to canonical code
        - otherwise -> raise a ValueError with clear message
        """
        s = normalize_asset_type(v)
        if s not in ASSET_TYPE_CODES:
            allowed = ", ".join(sorted(ASSET_TYPE_CODES))
            raise ValueError(f"assetType must be one of [{allowed}], got: {v!r}")
        return s

    # --- CIA normalization gates ---------------------------------------------
    @field_validator("confidentiality")
    @classmethod
    def _norm_conf(cls, v: Optional[str]) -> Optional[str]:
        """
        Normalize confidentiality to UPPER_SNAKE and validate against CONF_LEVELS.
        Return:
          - normalized value on success,
          - None if empty/None or not in whitelist (invalid).
        """
        if v is None:
            return None
        x = v.strip().upper().replace("-", "_").replace(" ", "_")
        if x == "STRICTLYPROTECTED":
            x = "STRICTLY_PROTECTED"
        return x if x in CONF_LEVELS else None

    @field_validator("integrity", "availability")
    @classmethod
    def _norm_tri(cls, v: Optional[str]) -> Optional[str]:
        """
        Normalize integrity/availability to UPPER_SNAKE and validate against TRI_LEVELS.
        Return normalized value or None (if empty/invalid).
        """
        if v is None:
            return None
        x = v.strip().upper().replace("-", "_").replace(" ", "_")
        if x == "MEDIUM_":  # tolerate a common typo
            x = "MEDIUM"
        return x if x in TRI_LEVELS else None

    @model_validator(mode="after")
    def _gate_required_and_cia(self) -> "CreateAssetFields":
        """
        Enforce CIA only for primary assets; null out CIA for supporting assets
        or when isPrimary is not specified.

        - If isPrimary is True:
            * all CIA must be present AND belong to strict enums;
              invalid/missing values trigger an error.
        - If isPrimary is False or None:
            * CIA is removed from payload (set to None) regardless of input.

        Note:
          Using Python None here ensures JSON serialization as `null`, which
          matches the backend schema requirement for supporting assets.
        """
        if self.isPrimary is True:
            missing = [k for k in ("confidentiality", "integrity", "availability") if getattr(self, k) in (None, "")]
            if missing:
                raise ValueError(f"primary asset requires CIA fields, missing: {', '.join(missing)}")
        else:
            # Supporting (or unspecified) assets must not carry CIA in the payload
            object.__setattr__(self, "confidentiality", None)
            object.__setattr__(self, "integrity", None)
            object.__setattr__(self, "availability", None)
        return self


class ActionCreateAsset(BaseModel):
    """
    Create a new asset; `client_token` lets the client correlate response → request.
    Example fields: name, asset_type_uuid/asset_type_code, primary, ...
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    op: Literal["create_asset"] = OP_CREATE_ASSET
    # Was: client_token: StrictStr
    # Now: generate a token if the caller didn't provide one
    client_token: StrictStr = Field(default_factory=lambda: uuid4().hex)
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
        ActionSetAssetCIA,
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
    "OP_SET_ASSET_CIA",
    "OP_SET_IMPACTS",
    "OP_SET_RISKS",
    "OP_UPSERT_CONTROLS",
    "OP_REMOVE_CONTROLS",
    "OP_UPSERT_THREATS",
    "OP_UPSERT_IMPACTS",
    "OP_ASSET_UPSERT_THREAT_PRESENCE",
    "ASSET_TYPE_CODES",
    "ASSET_TYPE_SYNONYMS",
    "normalize_asset_type",
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
    "ActionSetAssetCIA",
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
