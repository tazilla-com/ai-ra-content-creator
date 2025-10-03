# File: workshops_ai/api/schema/threat_tool.py
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# Reuse the client's canonical Threat model to derive the exact Literal[...] for category
from workshops_ai.api.schema.threat import Threat as ClientThreat

# This pulls the identical Literal[...] used by the client's Threat.category
ThreatCategory = ClientThreat.model_fields["category"].annotation  # <- stays in sync


class ThreatDefCreate(BaseModel):
    """
    Tool DTO for creating (or idempotently upserting) a threat definition.
    Types (incl. the ThreatCategory Literal) are reused from the client model.
    """
    name: str = Field(..., description="Unique threat name (already localized).")
    description: str = Field(..., description="Localized description of the threat.")
    category: ThreatCategory = Field(..., description="Canonical EN category (reused Literal from client).")
    unintentional: bool = Field(False, description="Threat is unintentional (human error, accident).")
    intentional: bool = Field(False, description="Threat is intentional (attack, sabotage).")
    environmental: bool = Field(False, description="Threat is environmental/natural (flood, fire, etc.).")

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name must not be empty")
        return v

    @field_validator("description")
    @classmethod
    def _desc_not_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("description must not be empty")
        return v


class ThreatDefUpdate(BaseModel):
    """
    Partial update for an existing threat definition, addressed by name.
    (Server may treat this as update-or-create if you prefer upsert semantics.)
    All fields optional except the target 'name'.
    """
    name: str = Field(..., description="Target threat name to update.")
    description: str | None = Field(None, description="New localized description.")
    category: ThreatCategory | None = Field(None, description="New canonical EN category.")
    unintentional: bool | None = Field(None, description="Set/unset unintentional flag.")
    intentional: bool | None = Field(None, description="Set/unset intentional flag.")
    environmental: bool | None = Field(None, description="Set/unset environmental flag.")

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name must not be empty")
        return v
