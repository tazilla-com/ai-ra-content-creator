# File: workshops_ai/api/schema/impact_tool.py
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# Reuse the client's canonical Impact model to derive the exact Literal[...] for impactCategory
from workshops_ai.api.schema.impact import Impact as ClientImpact

# This pulls the identical Literal[...] used by the client's Impact.impactCategory
ImpactCategory = ClientImpact.model_fields["impactCategory"].annotation  # <- stays in sync


class ImpactDefCreate(BaseModel):
    """
    Tool DTO for creating (or idempotently upserting) an impact definition.
    Types (incl. the ImpactCategory Literal) are reused from the client model.
    """
    name: str = Field(..., description="Unique impact name (already localized).")
    description: str = Field(..., description="Localized description of the impact.")
    impactCategory: ImpactCategory = Field(
        ..., description="Canonical EN category (reused Literal from client)."
    )

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


class ImpactDefUpdate(BaseModel):
    """
    Partial update for an existing impact definition, addressed by name.
    (Server can treat this as update-or-create if you want upsert semantics.)
    All fields are optional except the target 'name'.
    """
    name: str = Field(..., description="Target impact name to update.")
    description: str | None = Field(None, description="New localized description.")
    impactCategory: ImpactCategory | None = Field(
        None, description="New canonical EN category."
    )

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name must not be empty")
        return v
