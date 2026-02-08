"""
Unified session state model.

Redis Key: {session_id} (no prefix!)
TTL: 8 hours (28800 seconds)

Used by:
- Wizard: stores wizard step data and scraped content
- Agent: stores organization context, assets, conversation history
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator

from workshops_ai.schemas.ws_protocol import (
    Organization,
    ConversationMessage,
)

__all__ = ["SessionState", "RecentOperation"]


class RecentOperation(BaseModel):
    """
    Single operation in recent history for agent context.

    Tracks what the agent did in previous turns to provide continuity
    even when message history is trimmed.
    """
    turn: int = Field(..., description="Conversation turn number")
    action: Literal["created", "updated", "deleted"] = Field(
        ...,
        description="Type of operation performed"
    )
    assets: list[str] = Field(
        ...,
        description="Names of assets affected by this operation"
    )
    details: Optional[str] = Field(
        None,
        description="Human-readable details (e.g., 'Updated: confidentiality, integrity')"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this operation occurred"
    )


class SessionState(BaseModel):
    """
    Unified session state for both Wizard and Agent.

    WIZARD FIELDS:
    - wizard_steps: Wizard step data (services, assets, company, etc.)
    - wizard_metadata: Scraped content, cached data, etc.

    AGENT FIELDS:
    - organization: Organization context (set by wizard)
    - conversation: Last 20 messages / 10 turns (FIFO)
    - busy/busy_since: Agent execution lock

    NOTE: Assets live in wizard_steps["assets"]["assets"] (single source of truth).

    Both wizard and agent share the SAME session!
    """

    # Core identity
    session_id: str

    # ─────────────────────────────────────────────────────────────
    # WIZARD FIELDS
    # ─────────────────────────────────────────────────────────────
    wizard_steps: dict[str, Any] = Field(
        default_factory=dict,
        description="Wizard step data: {'services': {...}, 'assets': {...}, 'company': {...}}"
    )
    wizard_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Wizard metadata: website_content, scraped_data, etc."
    )

    # ─────────────────────────────────────────────────────────────
    # AGENT FIELDS (workshop_id comes from WS query param, not stored)
    # ─────────────────────────────────────────────────────────────
    busy: bool = False
    busy_since: Optional[datetime] = None

    organization: Optional[Organization] = None

    conversation: list[ConversationMessage] = Field(
        default_factory=list,
        description="Last 20 messages / 10 turns, FIFO (for UI display)"
    )

    recent_operations: list[RecentOperation] = Field(
        default_factory=list,
        description="Last 5 operations for agent context. FIFO."
    )

    # ─────────────────────────────────────────────────────────────
    # Timestamps
    # ─────────────────────────────────────────────────────────────
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("conversation")
    @classmethod
    def limit_conversation(cls, v: list[ConversationMessage]) -> list[ConversationMessage]:
        """Keep only last 20 messages (10 turns = 10 user + 10 assistant), FIFO."""
        return v[-20:] if len(v) > 20 else v

    @field_validator("recent_operations")
    @classmethod
    def limit_recent_operations(cls, v: list[RecentOperation]) -> list[RecentOperation]:
        """Keep only last 5 operations (FIFO)."""
        return v[-5:] if len(v) > 5 else v
