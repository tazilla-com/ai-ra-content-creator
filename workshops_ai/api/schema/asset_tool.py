# File: workshops_ai/api/schema/asset_tool.py
from __future__ import annotations

from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator

# Reuse the client's canonical Asset model to derive the exact Literal[...] for assetType
from workshops_ai.api.schema.asset import Asset as ClientAsset

# This pulls the identical Literal[...] used by the client's Asset.assetType
AssetType = ClientAsset.model_fields["assetType"].annotation  # <- stays in sync

class AssetCreate(BaseModel):
    name: str = Field(..., description="Asset name (localized).")
    type: AssetType = Field(..., description="Asset type from fixed code list.")

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name must not be empty")
        return v

class AssetUpdate(BaseModel):
    # If your ids are guaranteed UUIDs, switch to UUID; otherwise keep str.
    asset_id: str = Field(..., description="Target asset identifier.")
    name: Optional[str] = Field(None, description="New asset name.")
    type: Optional[AssetType] = Field(None, description="New asset type.")

    @field_validator("name")
    @classmethod
    def _name_if_present_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("name must not be empty when provided")
        return v
