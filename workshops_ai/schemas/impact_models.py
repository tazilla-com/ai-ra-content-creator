# File: workshops_ai/schemas/impact_models.py
# -----------------------------------------------------------------------------
# Pydantic v2 models for the Impacts domain (SEND-NOW workflow).
# These models are used across the agent, WS hub, and Tazila client payloads.
# - ImpactUpdate: validated input (one change for one asset).
# - ImpactChangeItem: one item inside a ChangeSet sent to BE.
# - ImpactChangeSet: whole changeset for immediate submission (send-now).
# - WSImpactSetEvent: WS payload to update UI optimistically.
#
# Conventions
# - Canonical impact codes are EN (normalized by the agent/tooling).
# - Internal level scale is 0..4; methodical mapped values are [5,20,50,70,100].
# - Strict schemas (extra="forbid"), immutable once created (frozen=True).
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --- Severity scale mapping (kept here for schema-local validation) ------------

LEVEL_INDEX_TO_VALUE: list[int] = [5, 20, 50, 70, 100]
ALLOWED_LEVEL_INDEX = {0, 1, 2, 3, 4}
ALLOWED_LEVEL_VALUE = set(LEVEL_INDEX_TO_VALUE)


# --- Public models --------------------------------------------------------------

class ImpactUpdate(BaseModel):
    """
    Validated single impact change (one asset).
    Typically produced by an agent/tool before emit+submit.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    org_id: str
    user_id: str
    asset_id: str
    impact_code: str = Field(..., min_length=1, description="Canonical EN code/name")
    level: int = Field(..., ge=0, le=4, description="Internal index 0..4")
    note: str | None = Field(default=None, description="Optional rationale/remark")

    @field_validator("impact_code")
    @classmethod
    def _trim_code(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("impact_code must not be empty")
        return v


class ImpactChangeItem(BaseModel):
    """
    One entry of a ChangeSet sent to backend (send-now).
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str
    impact_code: str
    level_index: int = Field(..., ge=0, le=4)
    level_value: int = Field(..., description="One of 5/20/50/70/100")
    note: str | None = None

    @field_validator("impact_code")
    @classmethod
    def _trim_code(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("impact_code must not be empty")
        return v

    @field_validator("level_value")
    @classmethod
    def _check_value(cls, v: int) -> int:
        if v not in ALLOWED_LEVEL_VALUE:
            raise ValueError(f"level_value must be one of {sorted(ALLOWED_LEVEL_VALUE)}")
        return v


class ImpactChangeSet(BaseModel):
    """
    Whole ChangeSet payload (send-now) submitted to backend.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["impact.set_level"] = "impact.set_level"
    org_id: str
    items: list[ImpactChangeItem]
    idempotency_key: str = Field(..., description="UUIDv5 (string)")
    audit: dict[str, Any] = Field(
        ...,
        description='Audit envelope, e.g. {"source":"AI","user_id":"...","ts":"..."}',
    )

    @field_validator("items")
    @classmethod
    def _non_empty_items(cls, v: list[ImpactChangeItem]) -> list[ImpactChangeItem]:
        if not v:
            raise ValueError("items must contain at least one ImpactChangeItem")
        return v


class WSImpactSetEvent(BaseModel):
    """
    WebSocket payload for optimistic UI update when an impact level is set.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["impact.set_level"] = "impact.set_level"
    org_id: str
    asset_id: str
    impact_code: str
    level_index: int = Field(..., ge=0, le=4)
    level_value: int = Field(..., description="One of 5/20/50/70/100")
    note: str | None = None
    actor: str = Field(..., description="User ID who triggered the change")
    ts: str = Field(..., description="ISO 8601 timestamp")

    @field_validator("impact_code")
    @classmethod
    def _trim_code(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("impact_code must not be empty")
        return v

    @field_validator("level_value")
    @classmethod
    def _check_value(cls, v: int) -> int:
        if v not in ALLOWED_LEVEL_VALUE:
            raise ValueError(f"level_value must be one of {sorted(ALLOWED_LEVEL_VALUE)}")
        return v


__all__ = [
    "LEVEL_INDEX_TO_VALUE",
    "ImpactUpdate",
    "ImpactChangeItem",
    "ImpactChangeSet",
    "WSImpactSetEvent",
]
