# File: workshops_ai/schemas/ui_contract.py
# Purpose: Pydantic v2 models for the UI ↔ Agent chat contract.
# - Fail-closed (extra fields are forbidden)
# - Canonical EN codes for tabs; selection is per-tab (extensible)
# - Response staging object mirrors the global “staging → diff → explicit commit” rule
# - Strings are stripped; messages are non-empty, trimmed

from __future__ import annotations

from typing import Any, Dict, Literal, Optional
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    HttpUrl,
    field_validator,
)
from pydantic.config import ConfigDict

# Agent event model used for optional synchronous event echo in UIResponse
from workshops_ai.models.agent_event import AgentEvent

# ---------------------------------------------------------------------------
# Shared / aliases
# ---------------------------------------------------------------------------

ActiveTab = Literal["assets", "impacts", "threats", "controls"]
JSONDict = Dict[str, Any]


# ---------------------------------------------------------------------------
# Request (UI → Chat)
# ---------------------------------------------------------------------------

class Selection(BaseModel):
    """Per-tab selection. For `assets` we may get `asset_id`."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    asset_id: Optional[StrictStr] = Field(
        default=None,
        description="Selected asset identifier (if any).",
    )


class PolicyConfig(BaseModel):
    """Policy switches for runtime behavior."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    emit_changes_enabled: StrictBool = Field(
        default=True,
        description="If false, the agent must not emit WS events nor submit changesets (dry-run).",
    )


class OrganizationProfile(BaseModel):
    """
    Free-form organization profile. Intentionally allows arbitrary keys so that
    backend can pass rich org context without schema churn.
    """
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)


class ServiceItem(BaseModel):
    """Company service description used to enrich asset proposals."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: StrictStr
    description: Optional[StrictStr] = None
    sector: Optional[StrictStr] = None  # canonicalization is handled by the agent


class ExistingAssetIn(BaseModel):
    """
    Pre-existing asset snippet coming from the wizard/back-end, used as context
    for AssetAgent reasoning or deduplication checks.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: Optional[UUID] = None
    name: StrictStr
    description: Optional[StrictStr] = None
    isPrimary: Optional[bool] = None
    confidentiality: Optional[StrictStr] = None
    integrity: Optional[StrictStr] = None
    availability: Optional[StrictStr] = None
    assetType: Optional[StrictStr] = None
    org_id: Optional[StrictStr] = None


class WebsitePage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    url: HttpUrl
    content: Optional[StrictStr] = None
    page_type: Optional[StrictStr] = None  # e.g. "home", "services", "contact"


class WebsiteContent(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    base_url: HttpUrl
    pages: list[WebsitePage] = Field(default_factory=list)
    errors: list[StrictStr] = Field(default_factory=list)


class UIRequest(BaseModel):
    """
    Chat input contract from UI / wizard. Minimal required: org_id, active_tab.
    Everything else is optional here to keep backward compatibility; upstream
    wizard may require session_id/language on its side.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    org_id: StrictStr = Field(..., min_length=1, description="Organization identifier (tenant).")
    active_tab: ActiveTab = Field(..., description="Current UI tab guiding tool-gating.")
    selection: Selection = Field(default_factory=Selection, description="Tab-specific selection.")
    message: StrictStr = Field(
        default="",
        description="User message; can contain token filters like 'search:', 'type:', etc.",
    )
    staging: StrictBool = Field(
        default=False,
        description="If true, user expects staging semantics (used by non-asset agents).",
    )
    user_id: Optional[StrictStr] = Field(default=None, description="Authenticated user identifier.")

    # Wizard / context enrichment (all optional)
    session_id: Optional[StrictStr] = Field(default=None, description="Client session identifier.")
    language: Optional[StrictStr] = Field(
        default=None,
        description="IETF language tag like 'cs', 'sk', 'en-US'.",
    )
    number_of_assets: Optional[StrictInt] = Field(
        default=None, ge=1, le=1000,
        description="Agent hint for list/summary limits.",
    )
    user_instructions: Optional[StrictStr] = Field(
        default=None,
        description="Free-form user guidance for the agent.",
    )
    organization_profile: Optional[OrganizationProfile] = Field(
        default=None,
        description="Organization profile (free-form).",
    )
    services: Optional[list[ServiceItem]] = Field(
        default=None,
        description="List of company services to inform asset proposals.",
    )
    existing_assets: Optional[list[ExistingAssetIn]] = Field(
        default=None,
        description="Already known assets; used for deduplication/context.",
    )
    url: Optional[HttpUrl] = Field(default=None, description="Primary organization URL.")
    website_content: Optional[WebsiteContent] = Field(
        default=None,
        description="Fetched website content (pages, errors).",
    )
    policy: Optional[PolicyConfig] = Field(
        default=None,
        description="Runtime policy switches (e.g., dry-run emission gate).",
    )

    @field_validator("message")
    @classmethod
    def _message_not_none(cls, v: str) -> str:
        # Allow empty string (e.g., pure selection-driven actions), but never None
        return "" if v is None else v

    @field_validator("language")
    @classmethod
    def _normalize_lang(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if isinstance(v, str) else v


# ---------------------------------------------------------------------------
# Response (Chat → UI)
# ---------------------------------------------------------------------------

class StagingState(BaseModel):
    """
    Staging state returned to the UI.
    - `needs_commit`: whether the UI should offer an explicit commit action
    - `batch_id`: identifier of an open staging batch (if any)
    - `diff`: a machine/human-friendly diff preview (backend-defined shape)
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    needs_commit: StrictBool = Field(..., description="True if there are staged changes requiring commit.")
    batch_id: Optional[StrictStr] = Field(default=None, description="Open staging batch identifier.")
    diff: Optional[JSONDict] = Field(default=None, description="Staged diff preview.")


class UIResponse(BaseModel):
    """
    Chat output contract to UI.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    messages: list[StrictStr] = Field(
        ...,
        min_length=1,
        description="Assistant text lines to render in the chat.",
    )
    proposals: Optional[JSONDict] = Field(
        default=None,
        description="Structured payload for UI widgets (tables/cards/forms).",
    )
    staging: StagingState = Field(
        default_factory=lambda: StagingState(needs_commit=False),
        description="Staging state for commit/diff UX.",
    )
    agent_events: Optional[list[AgentEvent]] = Field(
        default=None,
        description="Optional synchronous echo of agent events returned with the response.",
    )

    @field_validator("messages")
    @classmethod
    def _trim_and_nonempty(cls, v: list[str]) -> list[str]:
        cleaned = [m.strip() for m in v if isinstance(m, str)]
        if not cleaned or any(m == "" for m in cleaned):
            raise ValueError("messages must contain at least one non-empty string")
        return cleaned


__all__ = [
    "ActiveTab",
    "Selection",
    "PolicyConfig",
    "OrganizationProfile",
    "ServiceItem",
    "ExistingAssetIn",
    "WebsitePage",
    "WebsiteContent",
    "UIRequest",
    "StagingState",
    "UIResponse",
]
