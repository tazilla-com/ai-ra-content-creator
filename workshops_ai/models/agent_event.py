# File: workshops_ai/models/agent_event.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, StrictStr, ConfigDict, field_validator


def _utc_now() -> datetime:
    """Return current UTC time with timezone info (tz-aware)."""
    return datetime.now(timezone.utc)


class AgentEvent(BaseModel):
    """
    Canonical event envelope broadcast by agents over WsHub.

    Notes:
    - Fail-closed (extra fields are forbidden).
    - Timestamps are UTC and timezone-aware.
    - `data` is an agent-specific payload (validated by the emitting agent).
    - Minimal and stable schema; UIResponse.agent_events may echo these synchronously.
    """
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    event: StrictStr = Field(..., description='Event code, e.g. "asset.create" | "asset.set_meta"')
    org_id: StrictStr = Field(..., description="Tenant/org identifier")
    user_id: StrictStr = Field(..., description="Actor user identifier")
    asset_id: Optional[StrictStr] = Field(default=None, description="Related asset id, if applicable")
    data: dict[str, Any] = Field(default_factory=dict, description="Event-specific payload")
    ts: datetime = Field(default_factory=_utc_now, description="Event timestamp (UTC)")
    source: StrictStr = Field(default="AssetAgent", description="Emitter name")
    version: StrictStr = Field(default="v1", description="Schema version of the event")

    # --- Validators -----------------------------------------------------------

    @field_validator("event")
    @classmethod
    def _event_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("event must be a non-empty string")
        return v.strip()

    @field_validator("org_id", "user_id")
    @classmethod
    def _ids_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("identifier must be a non-empty string")
        return v.strip()

    # --- Helpers --------------------------------------------------------------

    def to_wire(self) -> dict[str, Any]:
        """
        JSON-safe dict intended for WebSocket/HTTP responses.
        Ensures ISO8601 timestamp with timezone in the output.
        """
        d = self.model_dump()
        ts = d.get("ts")
        if isinstance(ts, datetime):
            d["ts"] = ts.isoformat()
        return d


__all__ = ["AgentEvent"]
