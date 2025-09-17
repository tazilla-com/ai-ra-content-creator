# File: workshops_ai/schemas/ui_contract.py
# Purpose: Pydantic v2 models for the UI ↔ Agent chat contract.
# - Fail-closed (extra fields are forbidden)
# - Canonical EN codes for tabs; selection is per-tab (extensible)
# - Response staging object mirrors the global “staging → diff → explicit commit” rule
# - Strings are stripped; messages are non-empty, trimmed

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, StrictBool, StrictStr, field_validator
from pydantic.config import ConfigDict

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


class UIRequest(BaseModel):
    """
    Chat input contract from UI.
    Example:
    {
      "org_id": "acme",
      "active_tab": "assets",
      "selection": {"asset_id": "asset-123"},
      "message": "list search:db type:Server",
      "staging": false,
      "user_id": "u-42"
    }
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
    user_id: StrictStr = Field(..., min_length=1, description="Authenticated user identifier.")

    @field_validator("message")
    @classmethod
    def _message_not_none(cls, v: str) -> str:
        # Allow empty string (e.g., pure selection-driven actions), but never None
        return "" if v is None else v


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
    Example:
    {
      "messages": ["Nalezeno aktiv: 3 (hledáno: \"db\")."],
      "proposals": {"assets": [...]},
      "staging": {"needs_commit": false}
    }
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
    "UIRequest",
    "StagingState",
    "UIResponse",
]
