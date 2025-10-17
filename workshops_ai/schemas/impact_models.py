# File: workshops_ai/schemas/impact_models.py
# -----------------------------------------------------------------------------
# Pydantic v2 models for the Impacts domain (SEND-NOW workflow).
# These models are used across the agent, WS hub, and Tazila client payloads.
# - ImpactUpdate: validated input (one change for one asset).
# - ImpactChangeItem: one item inside a ChangeSet sent to BE.
# - ImpactChangeSet: whole changeset for immediate submission (send-now).
# - WSImpactSetEvent: WS payload to update UI optimistically (single item).
# - WSImpactSetListEvent: WS payload to update UI optimistically (batch/list).
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


# --- Public models -------------------------------------------------------------

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
    WebSocket payload for optimistic UI update when an impact level is set (single item).
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


# --- Batch/list WS event -------------------------------------------------------

class _ImpactListEntry(BaseModel):
    """
    One entry in the batch/list WS payload for impacts.

    NOTE:
    - Field names follow the requested camelCase for the WS contract:
      * impactName (str)
      * impactLevel (int, 0..4)
    """
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    impact_name: str = Field(..., alias="impactName")
    impact_level: int = Field(..., alias="impactLevel", ge=0, le=4)

    @field_validator("impact_name")
    @classmethod
    def _trim_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("impactName must not be empty")
        return v


class _ImpactSetListPayload(BaseModel):
    """
    Inner 'payload' for the batch/list WS event.
    """
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    asset_id: str = Field(..., alias="assetId")
    asset_name: str | None = Field(default=None, alias="assetName")
    impacts: list[_ImpactListEntry]

    @field_validator("impacts")
    @classmethod
    def _non_empty_impacts(cls, v: list[_ImpactListEntry]) -> list[_ImpactListEntry]:
        if not v:
            raise ValueError("impacts must contain at least one entry")
        return v

    @field_validator("asset_id")
    @classmethod
    def _trim_asset_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("assetId must not be empty")
        return v

    @field_validator("asset_name")
    @classmethod
    def _trim_asset_name(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        return v or None


class WSImpactSetListEvent(BaseModel):
    """
    WebSocket payload for optimistic UI update when multiple impact levels are set.

    Envelope (top-level) validation:
      - event  : must be "impact.set_levels"
      - org_id : organization id
      - actor  : user id (originator)
      - ts     : ISO 8601 timestamp
      - payload: {_ImpactSetListPayload} with camelCase keys as requested
                 { assetId, assetName?, impacts: [{ impactName, impactLevel }, ...] }
    """
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,  # keep for potential .model_dump(by_alias=True)
    )

    event: Literal["impact.set_levels"] = "impact.set_levels"
    org_id: str
    actor: str
    ts: str
    payload: _ImpactSetListPayload

    @field_validator("org_id", "actor", "ts")
    @classmethod
    def _trim_envelope_strs(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("org_id/actor/ts must not be empty")
        return v


__all__ = [
    "LEVEL_INDEX_TO_VALUE",
    "ImpactUpdate",
    "ImpactChangeItem",
    "ImpactChangeSet",
    "WSImpactSetEvent",
    "WSImpactSetListEvent",
]
