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
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.types import StrictStr, StrictBool

__all__ = [
    "Selection",
    "NewAssetNLU", "UpdateAssetNLU",
    "AssetsAction", "AssetsNLU",
    "ImpactSetItem", "ImpactsAction", "ImpactsBulkEntry", "ImpactsNLU",
    "ThreatSetItem", "ThreatsAction", "ThreatsBulkEntry", "ThreatsNLU",
    # "ControlsAction", "ControlsNLU",
    "Tab",
    "parse_nlu_for_tab",
    "ensure_action_is_string",
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
        Accepted aliases:
          - "update", "edit"        -> "set_meta"
          - "add", "new"            -> "create"
          - "create_asset"          -> "create"
          - "show", "details"       -> "get"
        Notes:
          - "list_types" is a special action to list available asset types
            (catalog) — it passes through unchanged.
    - `selection`: target selection (e.g., selection.asset_id)
    - `new_asset`: data for creation when action=create
    - `update`:    data for partial update when action=set_meta
    """
    action: AssetsAction
    selection: Optional[Selection] = None
    new_asset: Optional[NewAssetNLU] = None
    update: Optional[UpdateAssetNLU] = None

    @field_validator("action", mode="before")
    @classmethod
    def action_must_be_str(cls, v: Any) -> str:
        """
        Guard against the LLM emitting ["list"] instead of "list".
        Accept a single-item list[str] (coerce), reject any other arrays.
        Normalize to lowercase and map common aliases.
        """
        # Coerce from ["list"] shape
        if isinstance(v, list):
            if len(v) == 1 and isinstance(v[0], str):
                v = v[0]
            else:
                raise ValueError("action must be a string, not array")
        s = str(v).strip().lower()

        # Alias mapping (assets)
        if s in {"update", "edit"}:
            return "set_meta"
        if s in {"add", "new", "create_asset"}:
            return "create"
        if s in {"show", "details"}:
            return "get"

        # Allow "list_types" (and other supported literals) to pass unchanged
        return s



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

# ---------- Controls tab (placeholder for future) ----------

# ControlsAction = Literal["list", "get", "set", "explain"]


# class ControlsNLU(_StrictModel):
#     """
#     Minimal NLU result for the Controls agent (extend as needed).
#     """
#     action: ControlsAction
#     selection: Optional[Selection] = None
#     question: Optional[str] = None
#
#     @field_validator("action", mode="before")
#     @classmethod
#     def action_must_be_str(cls, v: Any) -> str:
#         if isinstance(v, list):
#             if len(v) == 1 and isinstance(v[0], str):
#                 return v[0].strip().lower()
#             raise ValueError("action must be a string, not array")
#         return str(v).strip().lower()


# ---------- Unified parsing helpers ----------

class Tab(str):
    ASSETS = "assets"
    IMPACTS = "impacts"
    THREATS = "threats"
    # CONTROLS = "controls"


def parse_nlu_for_tab(
    tab: str,
    data: dict[str, Any],
) -> Union[AssetsNLU, ImpactsNLU, ThreatsNLU]:  # ControlsNLU
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
    # if t == Tab.CONTROLS:
    #     return ControlsNLU.model_validate(data)
    raise ValueError(f"Unsupported tab '{tab}'")


def ensure_action_is_string(nlu_dict: dict[str, Any], tab: str | None = None) -> dict[str, Any]:
    v = nlu_dict.get("action", None)
    if isinstance(v, list):
        if len(v) == 1 and isinstance(v[0], str):
            nlu_dict["action"] = v[0].strip().lower()
        else:
            raise ValueError("action must be a string, not array")
    elif isinstance(v, str):
        nlu_dict["action"] = v.strip().lower()

    # alias mapping for convenience (mirrors AssetsNLU validator)
    aliases_assets = {
        "update": "set_meta",
        "edit": "set_meta",
        "add": "create",
        "new": "create",
        "create_asset": "create",
        "show": "get",
        "details": "get",
    }
    if (tab or "").strip().lower() == "assets":
        act = nlu_dict.get("action")
        if isinstance(act, str) and act in aliases_assets:
            nlu_dict["action"] = aliases_assets[act]

    allowed_by_tab = {
        "assets":   {"list", "get", "create", "set_meta", "explain", "list_types"},  # <- added list_types
        "impacts":  {"list", "get", "set", "bulk_set", "explain", "defs_create"},
        "threats":  {"list", "get", "set", "bulk_set", "explain", "defs_create"},
        # "controls": {"list", "get", "set", "explain"},
    }
    allowed_all = set().union(*allowed_by_tab.values())

    act = nlu_dict.get("action")
    t = (tab or "").strip().lower()
    allowed = allowed_by_tab.get(t, allowed_all)

    if act not in allowed:
        nlu_dict["action"] = "list"
    return nlu_dict
