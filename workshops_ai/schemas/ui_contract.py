# File: workshops_ai/schemas/ui_contract.py
# -----------------------------------------------------------------------------
# Purpose
# Pydantic v2 models for the UI ↔ Agent chat contract.
#
# Design Principles
# - Fail-closed: extra fields are rejected to prevent accidental drift.
# - Canonical tab codes (EN) to keep routing simple and stable.
# - Per-tab selection (extensible) to capture the currently focused item.
# - Response supports:
#     * human-facing messages (strings or JSON DTO "message items"),
#     * structured "proposals" payload for rich UI widgets (tables/cards/forms),
#     * optional synchronous agent_events echo for live progress rendering,
#     * staging state (diff/commit UX),
#     * policy hints (e.g., whether emitting changes is enabled).
# - Request accepts both `language` and `locale` (aligned with telemetry/manager).
# - Optional `history_token` lets clients resume a LangGraph thread (checkpoint).
#
# Notes
# - Legacy local OrganizationProfile/ServiceItem models were removed to prevent
#   duplication; canonical DTOs are imported from workshops_ai.api.schema.
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

# Optional synchronous event echo (lets UI show tool progress inline with the reply)
from workshops_ai.models.agent_event import AgentEvent

# Canonical DTOs (single source of truth)
from workshops_ai.api.schema.organization import Organization as OrganizationProfile
from workshops_ai.api.schema.service import Service as ServiceItem

# ---------------------------------------------------------------------------
# Shared aliases
# ---------------------------------------------------------------------------

ActiveTab = Literal["assets", "impacts", "threats", "controls"]
JSONDict = Dict[str, Any]

# A message to render in the chat stream:
# - string for plain assistant text
# - JSON object for DTO-like items (e.g., "kind":"asset.list", ...), which the UI may render specially
MessageItem = Union[StrictStr, JSONDict]


# ---------------------------------------------------------------------------
# Request (UI → Chat)
# ---------------------------------------------------------------------------

class Selection(BaseModel):
    """
    Per-tab selection coming from the UI.
    Keep minimal and make it easy to extend per tab.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    asset_id: Optional[StrictStr] = Field(
        default=None,
        description="Selected asset identifier (if any).",
    )


class PolicyConfig(BaseModel):
    """
    Policy switches affecting runtime behavior of the agent.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    emit_changes_enabled: StrictBool = Field(
        default=True,
        description=(
            "If false, the agent must not emit WS events nor submit changesets. "
            "Useful for dry-run/local-preview modes."
        ),
    )


class ExistingAssetIn(BaseModel):
    """
    Pre-existing asset snippet forwarded from a wizard or backend to help the agent
    with grounding, deduplication, or contextual reasoning.
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
    """Single crawled page (if UI/wizard pre-fetched website content)."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    url: HttpUrl
    content: Optional[StrictStr] = None
    page_type: Optional[StrictStr] = None  # e.g., "home", "services", "contact"


class WebsiteContent(BaseModel):
    """Aggregate of crawled pages for the organization's website."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    base_url: HttpUrl
    pages: list[WebsitePage] = Field(default_factory=list)
    errors: list[StrictStr] = Field(default_factory=list)


class UIRequest(BaseModel):
    """
    Chat input contract from UI/wizard. Only `org_id` and `active_tab` are strictly
    required; the rest is optional to ease rollouts and integration. The upstream
    wizard can enforce stricter input rules if needed.
    """
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    # Core context
    org_id: StrictStr = Field(..., min_length=1, description="Tenant/organization identifier.")
    active_tab: ActiveTab = Field(..., description="Current UI tab guiding tool-gating and routing.")
    selection: Selection = Field(default_factory=Selection, description="Tab-specific selection (e.g., asset_id).")

    # User input message (can be empty for pure selection-driven actions)
    message: StrictStr = Field(
        default="",
        description="User message; may include token-like filters ('search:', 'type:', ...).",
    )

    # Staging hint for non-asset agents (assets rely on explicit changesets)
    staging: StrictBool = Field(
        default=False,
        description="If true, user expects staging semantics (used by some non-asset flows).",
    )

    user_id: Optional[StrictStr] = Field(
        default=None,
        description="User identifier (deprecated - use identity from auth instead)",
        deprecated=True
    )

    # Auth identity (populated by security middleware)
    identity: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Authentication identity from security layer",
        exclude=True  # Not exposed in API docs
    )

    # Wizard / enrichment (optional, tolerated to keep the contract flexible)
    session_id: Optional[StrictStr] = Field(default=None, description="Client session identifier.")
    language: Optional[StrictStr] = Field(
        default=None,
        description="IETF language tag, e.g., 'cs', 'sk', 'en-US'.",
    )
    # Accept `locale` because telemetry/manager already handle it; UI may send either
    locale: Optional[StrictStr] = Field(
        default=None,
        description="Locale hint similar to `language` (e.g., 'cs_CZ').",
    )
    number_of_assets: Optional[StrictInt] = Field(
        default=None, ge=1, le=1000,
        description="Agent hint for list/summary sizing.",
    )
    user_instructions: Optional[StrictStr] = Field(
        default=None,
        description="Free-form guidance for the agent (persistent within the session).",
    )
    organization_profile: Optional[OrganizationProfile] = Field(
        default=None,
        description="Canonical Organization DTO to ground reasoning.",
    )
    services: Optional[list[ServiceItem]] = Field(
        default=None,
        description="List of services (canonical DTO) to inform asset proposals.",
    )
    existing_assets: Optional[list[ExistingAssetIn]] = Field(
        default=None,
        description="Known assets to aid deduplication and context.",
    )
    url: Optional[HttpUrl] = Field(default=None, description="Primary organization URL.")
    website_content: Optional[WebsiteContent] = Field(
        default=None,
        description="Pre-fetched site content (pages & errors).",
    )
    policy: Optional[PolicyConfig] = Field(
        default=None,
        description="Runtime switches (e.g., disable changes emission for dry-runs).",
    )

    # Optional cursor for LangGraph history/checkpoint resume
    history_token: Optional[StrictStr] = Field(
        default=None,
        description="Opaque token used by the checkpointer to resume conversation threads.",
    )

    # -------------------------
    # Validators (Pydantic v2)
    # -------------------------

    @field_validator("message", mode="before")
    @classmethod
    def _message_not_none(cls, v: Any) -> str:
        """
        Normalize `message`:
        - allow empty string (selection-only actions),
        - never allow None (enforce string semantics),
        - hard-fail for non-string types to keep the contract strict.
        """
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        raise TypeError("message must be a string")

    @field_validator("language", "locale", mode="before")
    @classmethod
    def _normalize_lang(cls, v: Any) -> Optional[str]:
        """Trim language/locale if present; otherwise pass through."""
        return v.strip() if isinstance(v, str) else v


# ---------------------------------------------------------------------------
# Response (Chat → UI)
# ---------------------------------------------------------------------------

class StagingState(BaseModel):
    """
    Staging state returned to the UI:
    - needs_commit: should the UI display an explicit commit action?
    - batch_id: identifier of the current staging batch (if any)
    - diff: backend-defined diff preview (machine/human friendly)
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    needs_commit: StrictBool = Field(..., description="True if there are changes waiting for commit.")
    batch_id: Optional[StrictStr] = Field(default=None, description="Open staging batch identifier (if any).")
    diff: Optional[JSONDict] = Field(default=None, description="Staged diff preview (shape defined by backend).")


class UIResponsePolicy(BaseModel):
    """
    Policy hints flowing back to the UI alongside the response.
    Useful when the backend derives policy state dynamically and the UI should adapt.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    emit_changes_enabled: StrictBool = Field(
        default=True,
        description="If true, the UI may assume WS events/changeset submission are active.",
    )


class UIResponse(BaseModel):
    """
    Chat output contract delivered to the UI.
    The UI should:
      - Always render `messages` (strings or JSON DTOs).
      - Prefer `proposals` for structured widgets (tables/cards/forms).
      - Optionally show `agent_events` as a live activity log when present.
      - Use `staging` to display commit/diff UX where applicable.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    # 1) Human-facing stream:
    messages: list[MessageItem] = Field(
        ...,
        min_length=1,
        description="Assistant lines to render: strings or JSON DTO dumps.",
    )

    # 2) Structured payload for widgets (tables/cards/forms)
    #    NOTE: This is the canonical place where list/detail data for the UI lives.
    proposals: Optional[JSONDict] = Field(
        default=None,
        description="Structured payload for UI widgets; if present, the UI should render it."
    )

    # 3) Staging state (commit/diff UX)
    staging: StagingState = Field(
        default_factory=lambda: StagingState(needs_commit=False),
        description="Staging state for commit/diff UX."
    )

    # 4) Optional synchronous echo of agent events (progress transparency)
    agent_events: Optional[list[AgentEvent]] = Field(
        default=None,
        description="Agent tool events echoed with the response for live progress rendering."
    )

    # 5) Response-side policy hints
    policy: Optional[UIResponsePolicy] = Field(
        default=None,
        description="Response policy hints for the UI (e.g., whether changes emission is enabled)."
    )

    # -------------------------
    # Validators & helpers
    # -------------------------

    @field_validator("messages")
    @classmethod
    def _validate_messages(cls, v: list[MessageItem]) -> list[MessageItem]:
        """
        Messages clean-up:
        - Strip strings and drop empty ones.
        - Accept dicts (DTOs) as-is.
        - Ensure at least one non-empty item remains.
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
                # Defensive guard: union should already guarantee str|dict.
                raise ValueError("messages items must be string or JSON object")
        if not cleaned:
            raise ValueError("messages must contain at least one non-empty item")
        return cleaned

    # Factory helpers used by graphs/routers to produce consistent envelopes.
    @staticmethod
    def ok(tab: str, op: str, result: dict) -> "UIResponse":
        """
        Build a success response:
        - `messages` carry a compact DTO announcing the operation.
        - `proposals` carry the same structured payload for widget rendering.
        """
        return UIResponse(
            messages=[{"kind": "success", "tab": tab, "op": op, "result": result}],
            proposals={"tab": tab, "op": op, "result": result},
        )

    @staticmethod
    def error(tab: str, op: str, error: str, details: Optional[list] = None) -> "UIResponse":
        """
        Build an error response:
        - Provide both a message-item and `proposals` with the error shape so the UI can render either.
        """
        return UIResponse(
            messages=[{"kind": "error", "tab": tab, "op": op, "error": error, "details": details}],
            proposals={"tab": tab, "op": op, "error": error, "details": details},
        )

    @staticmethod
    def submitted(tab: str, op: str, changeset: dict, submit: dict, result: dict) -> "UIResponse":
        """
        Build a 'submitted' response:
        - Include changeset echo + submission outcome.
        - Mirror the same structure in `proposals` for widget consumption.
        """
        return UIResponse(
            messages=[{
                "kind": "submitted",
                "tab": tab,
                "op": op,
                "changeset": changeset,
                "submit": submit,
                "result": result
            }],
            proposals={
                "tab": tab,
                "op": op,
                "changeset": changeset,
                "submit": submit,
                "result": result
            },
        )


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
