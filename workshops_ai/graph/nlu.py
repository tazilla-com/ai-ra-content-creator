# File: workshops_ai/graph/nlu.py
"""
Shared NLU result schemas (Pydantic v2) for all agents/tabs.

Contract-first rules
- All models are strict with extra='forbid'.
- `action` MUST be a string (never an array). A before-validator coerces ["list"] -> "list"
  and rejects any other arrays (fixes smoke-test failures from LLM JSON).
- All `action` values are normalized to lowercase. Common aliases are mapped:
    * assets: "update"/"edit" -> "set_meta", "add"/"new" -> "create"
- Levels/probabilities are integers in [0, 4].
- Bulk payloads are strongly typed with UUID-verified asset_ids.
- Use these models to parse/validate LLM NLU JSON before routing to agent logic.

Additions in this revision
- No forced fallback to "list" in ensure_action_is_string (let clarifying logic handle it).
- AssetsNLU gains `question: Optional[str]` for explain flows.
- Multilingual (CZ/SK/EN) action aliases for more natural chat inputs.

Lock-branch helpers (NEW in this file)
- `should_fetch_asset_types(pending_fields)` returns False unless "assetType" is among
  the currently pending fields → prevents unnecessary catalog lookups.
- `force_asset_create_route(is_draft_in_progress)` keeps the "force to asset.create" route
  when a create draft is in progress.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.types import StrictStr, StrictBool

__all__ = [
    "Selection",
    "NewAssetNLU", "UpdateAssetNLU",
    "AssetsAction", "AssetsNLU",
    "ImpactSetItem", "ImpactsAction", "ImpactsBulkEntry", "ImpactsNLU",
    "ThreatSetItem", "ThreatsAction", "ThreatsBulkEntry", "ThreatsNLU",
    "Tab",
    "parse_nlu_for_tab",
    "ensure_action_is_string",
    # NEW helpers exported for lock-branch logic:
    "should_fetch_asset_types",
    "force_asset_create_route",
]

# ---------- Helpers ----------

def _coerce_uuid(v: Any) -> Optional[UUID]:
    if v is None or v == "":
        return None
    if isinstance(v, UUID):
        return v
    try:
        return UUID(str(v))
    except Exception as e:
        raise ValueError("asset_id must be a valid UUID") from e


def _int_in_range(v: Any, *, lo: int = 0, hi: int = 4, name: str = "value") -> int:
    try:
        iv = int(v)
    except Exception as e:
        raise ValueError(f"{name} must be an integer") from e
    if iv < lo or iv > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}]")
    return iv


def _to_lower_nonempty(v: Any, name: str) -> str:
    if v is None:
        raise ValueError(f"{name} is required")
    s = str(v).strip()
    if not s:
        raise ValueError(f"{name} cannot be empty")
    return s.lower()


# ---------- Base / common ----------

class _StrictModel(BaseModel):
    """Base model with strict config and helpful defaults."""
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
    )


class Selection(_StrictModel):
    """UI selection object (currently only asset_id is used)."""
    asset_id: Optional[UUID] = Field(default=None)

    @field_validator("asset_id", mode="before")
    @classmethod
    def _coerce_asset_id(cls, v: Any) -> Optional[UUID]:
        return _coerce_uuid(v)


# ---------- Assets tab ----------

AssetsAction = Literal["list", "get", "create", "set_meta", "explain", "list_types"]


class NewAssetNLU(_StrictModel):
    """Payload for creating a new asset via NLU (fail-closed)."""
    name: StrictStr
    assetType: StrictStr
    description: StrictStr | None = None
    isPrimary: StrictBool | None = None
    confidentiality: int | None = Field(default=None, ge=0, le=4)
    integrity: int | None = Field(default=None, ge=0, le=4)
    availability: int | None = Field(default=None, ge=0, le=4)
    org_id: StrictStr | None = None

    @field_validator("confidentiality", "integrity", "availability", mode="before")
    @classmethod
    def _cia(cls, v: Any) -> int | None:
        if v is None or v == "":
            return None
        return _int_in_range(v, name="level")


class UpdateAssetNLU(_StrictModel):
    """Partial update payload for an existing asset via NLU (fail-closed)."""
    asset_id: UUID
    name: StrictStr | None = None
    description: StrictStr | None = None
    isPrimary: StrictBool | None = None
    confidentiality: int | None = Field(default=None, ge=0, le=4)
    integrity: int | None = Field(default=None, ge=0, le=4)
    availability: int | None = Field(default=None, ge=0, le=4)
    assetType: StrictStr | None = None
    org_id: StrictStr | None = None

    @field_validator("asset_id", mode="before")
    @classmethod
    def _coerce_asset(cls, v: Any) -> UUID:
        a = _coerce_uuid(v)
        if a is None:
            raise ValueError("asset_id is required")
        return a

    @field_validator("confidentiality", "integrity", "availability", mode="before")
    @classmethod
    def _cia(cls, v: Any) -> int | None:
        if v is None or v == "":
            return None
        return _int_in_range(v, name="level")


class AssetsNLU(_StrictModel):
    """
    NLU result for the Assets agent.
    - `action`: intent ("list" | "get" | "create" | "set_meta" | "explain" | "list_types")
        Accepted aliases (EN + CZ/SK):
          - "update", "edit", "uprav", "zmen", "zmeniť"                -> "set_meta"
          - "add", "new", "create_asset", "vytvor", "vytvoř", "pridej" -> "create"
          - "show", "details", "ukaz", "ukaž", "zobraz", "detail"      -> "get"
          - "types", "typy", "dostupne_typy", "dostupné typy"          -> "list_types"
          - "explain", "vysvetli", "vysvětli"                          -> "explain"
          - "list", "seznam", "zoznam"                                 -> "list"
    - `selection`: target selection (e.g., selection.asset_id)
    - `new_asset`: data for creation when action=create
    - `update`:    data for partial update when action=set_meta
    - `question`:  optional question for explain
    """
    action: AssetsAction
    selection: Optional[Selection] = None
    new_asset: Optional[NewAssetNLU] = None
    update: Optional[UpdateAssetNLU] = None
    question: Optional[str] = None  # for explain

    @field_validator("action", mode="before")
    @classmethod
    def action_must_be_str(cls, v: Any) -> str:
        """
        Guard against the LLM emitting ["list"] instead of "list".
        Accept a single-item list[str] (coerce), reject any other arrays.
        Normalize to lowercase and map common aliases (EN + CZ/SK).
        """
        # Coerce from ["list"] shape
        if isinstance(v, list):
            if len(v) == 1 and isinstance(v[0], str):
                v = v[0]
            else:
                raise ValueError("action must be a string, not array")
        s = str(v).strip().lower()

        # Alias mapping (assets; EN + CZ/SK)
        if s in {"update", "edit", "uprav", "zmen", "zmeniť"}:
            return "set_meta"
        if s in {"add", "new", "create_asset", "vytvor", "vytvoř", "pridej", "pridaj"}:
            return "create"
        if s in {"show", "details", "ukaz", "ukaž", "zobraz", "detail"}:
            return "get"
        if s in {"types", "typy", "dostupne_typy", "dostupné typy"}:
            return "list_types"
        if s in {"explain", "vysvetli", "vysvětli"}:
            return "explain"
        if s in {"list", "seznam", "zoznam"}:
            return "list"

        # Pass through literal when already normalized
        return s

    def effective_asset_id(self) -> Optional[UUID]:
        """Prefer explicit `update.asset_id` (when present), then `selection.asset_id`."""
        if self.update and self.update.asset_id:
            return self.update.asset_id
        return self.selection.asset_id if self.selection else None


# ---------- Impacts tab ----------

class ImpactSetItem(_StrictModel):
    """Single impact assignment item for an asset."""
    impact_code: str = Field(min_length=1)
    level: int

    @field_validator("impact_code", mode="before")
    @classmethod
    def _impact_code_norm(cls, v: Any) -> str:
        # Normalize and ensure non-empty; catalog membership is validated server-side.
        return _to_lower_nonempty(v, "impact_code")

    @field_validator("level", mode="before")
    @classmethod
    def _lvl(cls, v: Any) -> int:
        return _int_in_range(v, name="level")


ImpactsAction = Literal["list", "get", "set", "bulk_set", "explain", "defs_create"]


class ImpactDefNLU(_StrictModel):
    """NLU payload to define (or upsert) an impact in the catalog."""
    name: StrictStr
    description: StrictStr
    weight: float
    impactCategory: Literal[
        "Availability", "Clients", "Confidentiality", "Finance",
        "Integrity", "Legislation", "Reputation", "People", "Other"
    ]


class ImpactsBulkEntry(_StrictModel):
    """One entry for bulk setting impacts on a single asset."""
    asset_id: UUID
    items: list[ImpactSetItem]

    @field_validator("asset_id", mode="before")
    @classmethod
    def _coerce_asset(cls, v: Any) -> UUID:
        a = _coerce_uuid(v)
        if a is None:
            raise ValueError("asset_id is required")
        return a


class ImpactsNLU(_StrictModel):
    """
    NLU result for the Impacts agent.
    - For "set":        provide `asset_id` and `items`.
    - For "bulk_set":   provide `entries`.
    - For "defs_create":provide `new_def`.
    - For "get":        provide `asset_id` (directly or via `selection.asset_id`).
    """
    action: ImpactsAction
    selection: Optional[Selection] = None

    # For 'defs_create'
    new_def: Optional[ImpactDefNLU] = None

    # For 'set'
    asset_id: Optional[UUID] = None
    items: Optional[list[ImpactSetItem]] = None

    # For 'bulk_set'
    entries: Optional[list[ImpactsBulkEntry]] = None

    # For 'explain'
    question: Optional[str] = None

    @field_validator("action", mode="before")
    @classmethod
    def action_must_be_str(cls, v: Any) -> str:
        if isinstance(v, list):
            if len(v) == 1 and isinstance(v[0], str):
                v = v[0]
            else:
                raise ValueError("action must be a string, not array")
        s = str(v).strip().lower()
        if s in {"create", "add", "new"}:
            return "defs_create"
        return s

    @field_validator("asset_id", mode="before")
    @classmethod
    def _coerce_asset(cls, v: Any) -> Optional[UUID]:
        return _coerce_uuid(v)

    def effective_asset_id(self) -> Optional[UUID]:
        """Prefer explicit `asset_id`; fallback to `selection.asset_id`."""
        return self.asset_id or (self.selection.asset_id if self.selection else None)


# ---------- Threats tab ----------

class ThreatSetItem(_StrictModel):
    """Single threat probability assignment item for an asset."""
    threat_code: str = Field(min_length=1)
    probability: int

    @field_validator("threat_code", mode="before")
    @classmethod
    def _threat_code_norm(cls, v: Any) -> str:
        # Normalize and ensure non-empty; catalog membership is enforced server-side.
        return _to_lower_nonempty(v, "threat_code")

    @field_validator("probability", mode="before")
    @classmethod
    def _prob(cls, v: Any) -> int:
        return _int_in_range(v, name="probability")


ThreatsAction = Literal["list", "get", "set", "bulk_set", "explain", "defs_create"]


class ThreatsBulkEntry(_StrictModel):
    """One entry for bulk setting threat probabilities on a single asset."""
    asset_id: UUID
    items: list[ThreatSetItem]

    @field_validator("asset_id", mode="before")
    @classmethod
    def _coerce_asset(cls, v: Any) -> UUID:
        a = _coerce_uuid(v)
        if a is None:
            raise ValueError("asset_id is required")
        return a


class ThreatDefNLU(_StrictModel):
    """NLU payload to define (or upsert) a threat in the catalog."""
    name: StrictStr
    description: StrictStr
    category: Literal[
        "Physical threats",
        "Natural threats",
        "Compromise of functions or services",
        "Human actions",
        "Technical failures and infrastructure disruptions",
        "Organizational threats",
        "Legal and regulatory threats",
        "Violation of privacy",
        "Other"
    ]
    unintentional: StrictBool = False
    intentional:   StrictBool = False
    environmental: StrictBool = False


class ThreatsNLU(_StrictModel):
    """
    NLU result for the Threats agent.
    - For "set":         provide `asset_id` and `items`.
    - For "bulk_set":    provide `entries`.
    - For "defs_create": provide `new_def`.
    - For "get":         provide `asset_id` (directly or via `selection.asset_id`).
    """
    action: ThreatsAction
    selection: Optional[Selection] = None

    # For 'defs_create'
    new_def: Optional[ThreatDefNLU] = None

    # For 'set'
    asset_id: Optional[UUID] = None
    items: Optional[list[ThreatSetItem]] = None

    # For 'bulk_set'
    entries: Optional[list[ThreatsBulkEntry]] = None

    # For 'explain'
    question: Optional[str] = None

    @field_validator("action", mode="before")
    @classmethod
    def action_must_be_str(cls, v: Any) -> str:
        if isinstance(v, list):
            if len(v) == 1 and isinstance(v[0], str):
                v = v[0].strip().lower()
            else:
                raise ValueError("action must be a string, not array")
        s = str(v).strip().lower()
        if s in {"create", "add", "new"}:
            return "defs_create"
        return s

    @field_validator("asset_id", mode="before")
    @classmethod
    def _coerce_asset(cls, v: Any) -> Optional[UUID]:
        return _coerce_uuid(v)

    def effective_asset_id(self) -> Optional[UUID]:
        """Prefer explicit `asset_id`; fallback to `selection.asset_id`."""
        return self.asset_id or (self.selection.asset_id if self.selection else None)


# ---------- Unified parsing helpers ----------

class Tab(str):
    ASSETS = "assets"
    IMPACTS = "impacts"
    THREATS = "threats"


def parse_nlu_for_tab(
    tab: str,
    data: dict[str, Any],
) -> Union[AssetsNLU, ImpactsNLU, ThreatsNLU]:
    """
    Parse an LLM-produced NLU JSON into a strict, tab-specific model.
    Use this right after the LLM call, before downstream agent logic.
    """
    t = (tab or "").strip().lower()
    if t == Tab.ASSETS:
        return AssetsNLU.model_validate(data)
    if t == Tab.IMPACTS:
        return ImpactsNLU.model_validate(data)
    if t == Tab.THREATS:
        return ThreatsNLU.model_validate(data)
    raise ValueError(f"Unsupported tab '{tab}'")


# ---------- Normalization shim (no longer forces a fallback) ----------

def ensure_action_is_string(nlu_dict: dict[str, Any], tab: str | None = None) -> dict[str, Any]:
    """
    Lightweight normalization shim for callers that still pass raw dicts.
    - Coerces ["list"] -> "list"
    - Lowercases string values
    - Applies basic alias mapping (EN + CZ/SK) for assets
    - Does NOT force an invalid/unknown action to "list" anymore
      (let higher layers do clarify-first instead).
    """
    v = nlu_dict.get("action", None)
    if isinstance(v, list):
        if len(v) == 1 and isinstance(v[0], str):
            nlu_dict["action"] = v[0].strip().lower()
        else:
            raise ValueError("action must be a string, not array")
    elif isinstance(v, str):
        nlu_dict["action"] = v.strip().lower()

    # alias mapping for convenience (mirrors AssetsNLU validator; EN + CZ/SK)
    aliases_assets = {
        # set_meta
        "update": "set_meta", "edit": "set_meta", "uprav": "set_meta", "zmen": "set_meta", "zmeniť": "set_meta",
        # create
        "add": "create", "new": "create", "create_asset": "create", "vytvor": "create", "vytvoř": "create",
        "pridej": "create", "pridaj": "create",
        # get
        "show": "get", "details": "get", "ukaz": "get", "ukaž": "get", "zobraz": "get", "detail": "get",
        # list_types
        "types": "list_types", "typy": "list_types", "dostupne_typy": "list_types", "dostupné typy": "list_types",
        # explain
        "explain": "explain", "vysvetli": "explain", "vysvětli": "explain",
        # list
        "list": "list", "seznam": "list", "zoznam": "list",
    }
    if (tab or "").strip().lower() == "assets":
        act = nlu_dict.get("action")
        if isinstance(act, str) and act in aliases_assets:
            nlu_dict["action"] = aliases_assets[act]

    return nlu_dict


# ---------- Lock-branch helpers (used by the agent layer) ----------

def should_fetch_asset_types(pending_fields: Optional[list[str]]) -> bool:
    """
    Return True iff the 'assetType' field is among the pending fields that the
    agent still needs to fill. This prevents hitting the catalog endpoint when
    we don't actually need asset types right now.

    - Matching is case-insensitive.
    - Accepts both "assetType" and "asset_type" just in case the caller uses
      snake_case internally for state keys.
    """
    if not pending_fields:
        return False
    wanted = {"assettype", "asset_type"}
    return any((f or "").strip().lower() in wanted for f in pending_fields)


def force_asset_create_route(is_draft_in_progress: bool) -> Optional[str]:
    """
    Keep the "force to asset.create" route while a create flow is in progress.
    If a draft create is active, the agent should continue routing to the
    'asset.create' operation regardless of incidental intents detected by NLU.

    Returns:
        "asset.create" when a draft is in progress, otherwise None.
    """
    return "asset.create" if is_draft_in_progress else None
