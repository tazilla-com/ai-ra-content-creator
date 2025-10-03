# File: workshops_ai/schemas/ui_contract.py
# -----------------------------------------------------------------------------
# Purpose
# Pydantic v2 models for the UI ↔ Agent chat contract.
#
# Design
# - Fail-closed (extra fields are forbidden).
# - Canonical EN codes for tabs; selection is per-tab (extensible).
# - Response staging object mirrors the global “staging → diff → explicit commit” rule.
# - Strings are stripped; messages can be strings OR JSON DTO dumps.
# - `policy.emit_changes_enabled` is preserved end-to-end.
# - `locale` is accepted (alongside `language`) to align with telemetry/manager usage.
# - Optional `history_token` allows clients to resume a LangGraph thread (checkpoint).
#
# Notes
# - Legacy local OrganizationProfile/ServiceItem models were removed to prevent drift.
#   We import canonical DTOs from `workshops_ai.api.schema` instead.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    HttpUrl,
    field_validator,
    ConfigDict,
)

# Agent event model used for optional synchronous event echo in UIResponse
from workshops_ai.models.agent_event import AgentEvent

# Canonical DTOs (deduplicate vs. local copies)
from workshops_ai.api.schema.organization import Organization as OrganizationProfile
from workshops_ai.api.schema.service import Service as ServiceItem

# ---------------------------------------------------------------------------
# Shared / aliases
# ---------------------------------------------------------------------------

ActiveTab = Literal["assets", "impacts", "threats", "controls"]
JSONDict = Dict[str, Any]
MessageItem = Union[StrictStr, JSONDict]  # <-- allow text or DTO dumps


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


class ExistingAssetIn(BaseModel):
    """
    Pre-existing asset snippet coming from the wizard/back-end, used as context
    for AssetAgent reasoning or deduplication checks.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: Optional[UUID] = None
    name: StrictStr
    description: Optional[StrictStr] = None
    isPrimary: Optional[StrictBool] = None
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
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

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
    # Added to align with telemetry/manager which already read `locale`
    locale: Optional[StrictStr] = Field(
        default=None,
        description="Locale hint similar to `language` (e.g., 'cs_CZ'); UI may send either.",
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
        description="Organization profile (canonical DTO).",
    )
    services: Optional[list[ServiceItem]] = Field(
        default=None,
        description="List of company services to inform asset proposals (canonical DTO).",
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

    # Optional cursor for LangGraph history/checkpoint resume (thread id / token)
    history_token: Optional[StrictStr] = Field(
        default=None,
        description="Opaque token to resume a conversation thread/history (checkpointer).",
    )

    @field_validator("message", mode="before")
    @classmethod
    def _message_not_none(cls, v: Any) -> str:
        # Allow empty string (e.g., pure selection-driven actions), but never None
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        # Hard fail for non-string inputs to preserve StrictStr semantics
        raise TypeError("message must be a string")

    @field_validator("language", "locale", mode="before")
    @classmethod
    def _normalize_lang(cls, v: Any) -> Optional[str]:
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


class UIResponsePolicy(BaseModel):
    """Policy hints for UI rendering/behavior on the response channel."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    emit_changes_enabled: StrictBool = Field(
        default=True,
        description="If true, UI may assume the agent emits WS events / submits changesets.",
    )


class UIResponse(BaseModel):
    """
    Chat output contract to UI.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    messages: list[MessageItem] = Field(
        ...,
        min_length=1,
        description="Assistant lines to render: either strings or JSON DTO dumps.",
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
    policy: Optional[UIResponsePolicy] = Field(
        default=None,
        description="Response-side policy switches and hints for the UI.",
    )

    @field_validator("messages")
    @classmethod
    def _validate_messages(cls, v: list[MessageItem]) -> list[MessageItem]:
        """
        - For strings: strip and ensure non-empty.
        - For dicts: accept as-is (DTO dumps).
        - Ensure at least one item remains after trimming.
        """
        cleaned: list[MessageItem] = []
        for item in v:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    cleaned.append(s)
            elif isinstance(item, dict):
                cleaned.append(item)
            else:
                # Should not happen given the union, but be defensive.
                raise ValueError("messages items must be string or JSON object")
        if not cleaned:
            raise ValueError("messages must contain at least one non-empty item")
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
    "UIResponsePolicy",
    "UIResponse",
]
