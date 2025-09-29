# workshops_ai/graph/nlu.py
"""
Shared NLU result schemas (Pydantic v2) for all agents/tabs.

Contract-first rules
- All models are strict with extra='forbid'.
- `action` MUST be a string (never an array). A before-validator coerces ["list"] -> "list"
  and rejects any other arrays (fixes the smoke-test failure).
- Levels/probabilities are integers in [0, 4].
- Bulk payloads are strongly typed.
- Use these models to parse/validate LLM NLU JSON before routing to agent logic.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


# ---------- Helpers ----------

def _coerce_uuid(v: Any) -> Optional[UUID]:
    if v is None or v == "":
        return None
    if isinstance(v, UUID):
        return v
    try:
        return UUID(str(v))
    except Exception:
        raise ValueError("asset_id must be a valid UUID")

def _int_in_range(v: Any, *, lo: int = 0, hi: int = 4, name: str = "value") -> int:
    try:
        iv = int(v)
    except Exception:
        raise ValueError(f"{name} must be an integer")
    if iv < lo or iv > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}]")
    return iv


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

AssetsAction = Literal["list", "get", "create", "set_meta", "explain"]


class AssetsNLU(_StrictModel):
    """
    NLU result for the Assets agent.
    - encode user intent in `action`
    - optionally include target `asset_id` (for 'get', 'set_meta', etc.)
    """
    action: AssetsAction
    selection: Optional[Selection] = None
    # Additional parsed fields can be added here (e.g., meta updates).

    @field_validator("action", mode="before")
    @classmethod
    def action_must_be_str(cls, v: Any) -> str:
        """
        Guard against LLM emitting ["list"] instead of "list".
        Accept single-item list[str] (coerce), reject other arrays.
        """
        if isinstance(v, list):
            if len(v) == 1 and isinstance(v[0], str):
                return v[0]
            raise ValueError("action must be a string, not array")
        return v


# ---------- Impacts tab ----------

class ImpactSetItem(_StrictModel):
    impact_code: str = Field(min_length=1)
    level: int

    @field_validator("level", mode="before")
    @classmethod
    def _lvl(cls, v: Any) -> int:
        return _int_in_range(v, name="level")


ImpactsAction = Literal["list", "get", "set", "bulk_set", "explain"]


class ImpactsBulkEntry(_StrictModel):
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
    For 'set': provide `asset_id` and `items`.
    For 'bulk_set': provide `entries`.
    For 'get': provide `asset_id` (via selection or asset_id field).
    """
    action: ImpactsAction
    selection: Optional[Selection] = None

    # For 'set'
    asset_id: Optional[UUID] = None
    items: Optional[list[ImpactSetItem]] = None

    # For 'bulk_set'
    entries: Optional[list[ImpactsBulkEntry]] = None

    question: Optional[str] = None  # for 'explain'

    @field_validator("action", mode="before")
    @classmethod
    def action_must_be_str(cls, v: Any) -> str:
        if isinstance(v, list):
            if len(v) == 1 and isinstance(v[0], str):
                return v[0]
            raise ValueError("action must be a string, not array")
        return v

    @field_validator("asset_id", mode="before")
    @classmethod
    def _coerce_asset(cls, v: Any) -> Optional[UUID]:
        return _coerce_uuid(v)

    def effective_asset_id(self) -> Optional[UUID]:
        """Prefer explicit asset_id; fallback to selection.asset_id."""
        return self.asset_id or (self.selection.asset_id if self.selection else None)


# ---------- Threats tab ----------

class ThreatSetItem(_StrictModel):
    threat_code: str = Field(min_length=1)
    probability: int

    @field_validator("probability", mode="before")
    @classmethod
    def _prob(cls, v: Any) -> int:
        return _int_in_range(v, name="probability")


ThreatsAction = Literal["list", "get", "set", "bulk_set", "explain"]


class ThreatsBulkEntry(_StrictModel):
    asset_id: UUID
    items: list[ThreatSetItem]

    @field_validator("asset_id", mode="before")
    @classmethod
    def _coerce_asset(cls, v: Any) -> UUID:
        a = _coerce_uuid(v)
        if a is None:
            raise ValueError("asset_id is required")
        return a


class ThreatsNLU(_StrictModel):
    """
    NLU result for the Threats agent.
    For 'set': provide `asset_id` and `items`.
    For 'bulk_set': provide `entries`.
    For 'get': provide `asset_id` (via selection or asset_id field).
    """
    action: ThreatsAction
    selection: Optional[Selection] = None

    # For 'set'
    asset_id: Optional[UUID] = None
    items: Optional[list[ThreatSetItem]] = None

    # For 'bulk_set'
    entries: Optional[list[ThreatsBulkEntry]] = None

    question: Optional[str] = None  # for 'explain'

    @field_validator("action", mode="before")
    @classmethod
    def action_must_be_str(cls, v: Any) -> str:
        if isinstance(v, list):
            if len(v) == 1 and isinstance(v[0], str):
                return v[0]
            raise ValueError("action must be a string, not array")
        return v

    @field_validator("asset_id", mode="before")
    @classmethod
    def _coerce_asset(cls, v: Any) -> Optional[UUID]:
        return _coerce_uuid(v)

    def effective_asset_id(self) -> Optional[UUID]:
        """Prefer explicit asset_id; fallback to selection.asset_id."""
        return self.asset_id or (self.selection.asset_id if self.selection else None)


# ---------- Controls tab (placeholder for future) ----------

ControlsAction = Literal["list", "get", "set", "explain"]


class ControlsNLU(_StrictModel):
    """
    Minimal NLU result for the Controls agent (extend as needed).
    """
    action: ControlsAction
    selection: Optional[Selection] = None
    question: Optional[str] = None

    @field_validator("action", mode="before")
    @classmethod
    def action_must_be_str(cls, v: Any) -> str:
        if isinstance(v, list):
            if len(v) == 1 and isinstance(v[0], str):
                return v[0]
            raise ValueError("action must be a string, not array")
        return v


# ---------- Unified parsing helpers ----------

class Tab(str):
    ASSETS = "assets"
    IMPACTS = "impacts"
    THREATS = "threats"
    CONTROLS = "controls"


def parse_nlu_for_tab(tab: str, data: dict[str, Any]) -> AssetsNLU | ImpactsNLU | ThreatsNLU | ControlsNLU:
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
    if t == Tab.CONTROLS:
        return ControlsNLU.model_validate(data)
    raise ValueError(f"Unsupported tab '{tab}'")


def ensure_action_is_string(nlu_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Utility that can be applied pre-parse if needed in legacy flows.
    It in-place normalizes {"action": ["list"]} -> {"action": "list"}.
    """
    v = nlu_dict.get("action", None)
    if isinstance(v, list):
        if len(v) == 1 and isinstance(v[0], str):
            nlu_dict["action"] = v[0]
        else:
            raise ValueError("action must be a string, not array")
    return nlu_dict
