# File: workshops_ai/schemas/risk_models.py
# -----------------------------------------------------------------------------
# Strict Pydantic v2 models for Threat risk updates (0..4 scale).
#
# Provides:
# - RiskUpdate         → minimal inbound shape (from tools / UI)
# - RiskChangeItem     → normalized item with idempotency key + timestamp
# - RiskChangeSet      → envelope for "risk.set_probability" batches
# - WSRiskSetEvent     → WS payload for optimistic UI updates
#
# Notes:
# - Probability is validated on the canonical 0..4 integer scale.
# - Strings like "low"/"medium"/"high"/"very_high" (and numeric strings) are
#   accepted and normalized to 1..4; "none"/"0" → 0.
# - Idempotency key is UUIDv5 derived from (asset_id, threat_code, probability).
#   This yields stable keys for identical updates and prevents duplicates.
# -----------------------------------------------------------------------------

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ------------------------------ Probability -----------------------------------

# Canonical 0..4 scale
Probability = int

_PROB_ALIASES = {
    "0": 0,
    "none": 0,
    "no": 0,
    "zero": 0,

    "1": 1,
    "low": 1,

    "2": 2,
    "medium": 2,
    "moderate": 2,

    "3": 3,
    "high": 3,

    "4": 4,
    "very_high": 4,
    "critical": 4,
}

def _normalize_probability(value: Any) -> Probability:
    """
    Accepts int/str and normalizes to an integer in [0, 4].
    Raises ValueError if the value is not recognized or out of range.
    """
    if isinstance(value, bool):
        # Disallow booleans even if they are ints underneath.
        raise ValueError("Probability must be an integer 0..4, not a boolean.")

    if isinstance(value, (int, float)):
        # Allow 0.0 .. 4.0 but require it to be an integer after cast.
        iv = int(value)
        if float(value) != float(iv):
            raise ValueError("Probability must be an integer 0..4.")
        if 0 <= iv <= 4:
            return iv
        raise ValueError("Probability must be within 0..4.")
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _PROB_ALIASES:
            return _PROB_ALIASES[key]
        # Try bare integer string
        try:
            iv = int(key)
        except ValueError as exc:
            raise ValueError(f"Unrecognized probability: {value!r}") from exc
        if 0 <= iv <= 4:
            return iv
        raise ValueError("Probability must be within 0..4.")
    raise ValueError("Unsupported probability type.")


# Stable UUIDv5 namespace string for this action kind
_UUID5_NAMESPACE_NAME = "workshops_ai:risk.set_probability:v1"


def _make_idempotency_key(asset_id: str, threat_code: str, probability: Probability) -> str:
    """
    Deterministic idempotency key (UUIDv5) for a given tuple.
    """
    name = f"{_UUID5_NAMESPACE_NAME}:{asset_id}:{threat_code}:{probability}"
    return str(uuid5(NAMESPACE_URL, name))


# ----------------------------- Public Schemas ---------------------------------

class RiskUpdate(BaseModel):
    """
    Minimal inbound update (tool/UI level).
    """
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(..., min_length=1)
    threat_code: str = Field(..., min_length=1)
    probability: Probability = Field(..., description="Integer 0..4")
    note: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("asset_id", "threat_code")
    @classmethod
    def _strip(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Value must not be empty.")
        return s

    @field_validator("probability")
    @classmethod
    def _validate_prob(cls, v: Any) -> Probability:
        return _normalize_probability(v)


class RiskChangeItem(RiskUpdate):
    """
    Normalized update with idempotency key and timestamp.
    """
    model_config = ConfigDict(extra="forbid")

    # ISO-like timestamp; generated if not provided
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Stable, deterministic key; generated from (asset_id, threat_code, probability)
    idempotency_key: str | None = Field(
        default=None, description="UUIDv5, stable for identical tuples"
    )

    @model_validator(mode="after")
    def _ensure_idem_key(self) -> "RiskChangeItem":
        if not self.idempotency_key:
            self.idempotency_key = _make_idempotency_key(
                asset_id=self.asset_id,
                threat_code=self.threat_code,
                probability=self.probability,
            )
        return self


class RiskChangeSet(BaseModel):
    """
    Envelope for a batch of risk.set_probability updates.
    """
    model_config = ConfigDict(extra="forbid")

    kind: Literal["risk.set_probability"] = "risk.set_probability"
    items: list[RiskChangeItem] = Field(..., min_length=1, max_length=200)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_unique_idem(self) -> "RiskChangeSet":
        # prevent duplicates within a single batch
        keys = [it.idempotency_key for it in self.items]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate idempotency_key detected in items.")
        return self

    def to_wire(self) -> dict[str, Any]:
        """
        Return a generic wire dict compatible with a common ChangeSet structure.
        """
        return {
            "kind": self.kind,
            "items": [it.model_dump() for it in self.items],
            "meta": self.meta,
        }


class WSRiskSetEvent(BaseModel):
    """
    WebSocket event for optimistic UI updates.
    """
    model_config = ConfigDict(extra="forbid")

    event: Literal["risk.set_probability"] = "risk.set_probability"
    payload: RiskChangeItem


__all__ = [
    "Probability",
    "RiskUpdate",
    "RiskChangeItem",
    "RiskChangeSet",
    "WSRiskSetEvent",
]
