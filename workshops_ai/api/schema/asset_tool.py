# File: workshops_ai/api/schema/asset_tool.py


from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Reuse the client's canonical Asset model to derive the exact Literal[...] for enums,
# so we stay in lockstep with the client SDK/types.
from workshops_ai.api.schema.asset import Asset as ClientAsset

# This pulls the identical Literal[...] used by the client's Asset.assetType
AssetType = ClientAsset.model_fields["assetType"].annotation  # <- stays in sync

# Try to reuse CIA enum annotations from the client model if available; otherwise
# fall back to our own Literals. This makes the schema resilient if the client
# adds/removes variants.
try:
    ConfidentialityEnum = ClientAsset.model_fields["confidentiality"].annotation
except Exception:  # pragma: no cover - defensive fallback
    from typing import Literal
    ConfidentialityEnum = Literal["PUBLIC", "INTERNAL", "PROTECTED", "STRICTLY_PROTECTED"]

try:
    IntegrityEnum = ClientAsset.model_fields["integrity"].annotation
except Exception:  # pragma: no cover
    from typing import Literal
    IntegrityEnum = Literal["LOW", "MEDIUM", "HIGH"]

try:
    AvailabilityEnum = ClientAsset.model_fields["availability"].annotation
except Exception:  # pragma: no cover
    from typing import Literal
    AvailabilityEnum = Literal["LOW", "MEDIUM", "HIGH"]


class AssetCreate(BaseModel):
    """
    Contract-first DTO for the asset.create tool.

    IMPORTANT:
    - We require the caller to explicitly tell us whether the asset is primary.
      (No default True; we want the agent to ask.)
    - CIA (confidentiality, integrity, availability) are required only for primary assets.
      If isPrimary is False/None they must not be provided (we null them).
    """
    # Strict handling: reject unknown keys to avoid drifting payloads.
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    name: str = Field(..., description="Asset name (localized).")
    # Rename `type` -> `assetType` to match internal naming; accept legacy 'type' via alias.
    assetType: AssetType = Field(
        ...,
        alias="type",
        description="Asset type from fixed code list."
    )

    # NEW: explicit primary flag without default (the agent must ask the user).
    isPrimary: Optional[bool] = Field(
        None,
        description="Whether this is a primary (business-critical) asset."
    )

    # NEW: CIA fields — required iff isPrimary == True
    confidentiality: Optional[ConfidentialityEnum] = Field(
        None,
        description="Required if isPrimary=True. One of PUBLIC, INTERNAL, PROTECTED, STRICTLY_PROTECTED."
    )
    integrity: Optional[IntegrityEnum] = Field(
        None,
        description="Required if isPrimary=True. One of LOW, MEDIUM, HIGH."
    )
    availability: Optional[AvailabilityEnum] = Field(
        None,
        description="Required if isPrimary=True. One of LOW, MEDIUM, HIGH."
    )

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name must not be empty")
        return v

    @model_validator(mode="after")
    def _gate_required_and_cia(self) -> "AssetCreate":
        """
        Cross-field gate that aligns with the write-layer contract:

        - If isPrimary is True: all CIA must be present (non-null).
        - If isPrimary is False or None: CIA must be nulled (we do it here).
        """
        if self.isPrimary is True:
            missing = [k for k in ("confidentiality", "integrity", "availability")
                       if getattr(self, k) is None]
            if missing:
                raise ValueError(
                    f"Missing CIA fields for primary asset: {', '.join(missing)}"
                )
        else:
            # Supporting/unknown -> CIA not collected; normalize to None
            object.__setattr__(self, "confidentiality", None)
            object.__setattr__(self, "integrity", None)
            object.__setattr__(self, "availability", None)
        return self


class AssetUpdate(BaseModel):
    """
    Contract for asset metadata updates exposed via the tool layer.

    Note:
    - We keep updates conservative. Typically only `name` and (optionally) `assetType`
      should be changed here. CIA updates are intentionally NOT exposed here to avoid
      policy drift (use a dedicated tool if needed).
    """
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # If your ids are guaranteed UUIDs in the transport, switch to UUID here.
    asset_id: str = Field(..., description="Target asset identifier.")
    name: Optional[str] = Field(None, description="New asset name.")
    # Align naming with create: `assetType` (+ alias 'type' for backward compatibility).
    assetType: Optional[AssetType] = Field(
        None,
        alias="type",
        description="New asset type."
    )

    @field_validator("name")
    @classmethod
    def _name_if_present_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("name must not be empty when provided")
        return v
