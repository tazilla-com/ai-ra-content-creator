# workshops_ai/agents/session/models.py
"""
Pydantic models for agent session state.

Design Philosophy:
- Session ID is the single source of truth (no org_id/user_id in URLs)
- org_id and user_id are stored in session (extracted from auth token)
- Educational approach: drafts include reasoning, dependencies, compliance notes
- Context memory: track industry, preferences, conversation history
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field

__all__ = ["AgentSessionState", "AssetDraft", "DraftStatus"]


# -----------------------------------------------------------------------------
# Enums & Literals
# -----------------------------------------------------------------------------

DraftStatus = Literal["draft", "preview", "confirmed", "submitted"]


# -----------------------------------------------------------------------------
# Asset Draft Model
# -----------------------------------------------------------------------------

class AssetDraft(BaseModel):
    """
    Draft asset being created or edited by the agent.

    Educational Features:
    - reasoning: Why these specific values were chosen
    - dependencies: Related assets that should exist
    - compliance_notes: Industry-specific requirements
    - created_from: Provenance tracking (scraping, manual, recommendation)
    """

    # Core asset fields (matching Tazilla schema)
    name: str = Field(..., min_length=1, description="Asset name")
    assetType: str | None = Field(
        None,
        description="Asset type. MUST be one of: 'Application software', 'Cloud services', 'Data storage', "
                    "'Electronic data/Business process', 'Information system', 'IoT', 'Location', "
                    "'Mobile devices', 'Laptop', 'Operating system', 'Paper data', 'PC', 'Staff', "
                    "'Portable data media', 'Server infrastructure', 'Network infrastructure', "
                    "'Virtualization platform', 'Equipment, machines', 'Other'. "
                    "Use EXACT value, do NOT translate."
    )
    isPrimary: bool | None = Field(None, description="Is this a primary asset?")
    confidentiality: str | None = Field(
        None,
        description="Confidentiality level. MUST be one of: 'PUBLIC', 'INTERNAL', 'PROTECTED', 'STRICTLY_PROTECTED'. "
                    "Use EXACT value, do NOT translate."
    )
    integrity: str | None = Field(
        None,
        description="Integrity level. MUST be one of: 'LOW', 'MEDIUM', 'HIGH'. "
                    "Use EXACT value, do NOT translate."
    )
    availability: str | None = Field(
        None,
        description="Availability level. MUST be one of: 'LOW', 'MEDIUM', 'HIGH'. "
                    "Use EXACT value, do NOT translate."
    )
    description: str | None = Field(None, description="Asset description")

    # Educational features (NEW - agent-specific)
    reasoning: str | None = Field(
        None,
        description="Agent's explanation for why these values were chosen"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Related asset names/IDs that this asset depends on or relates to"
    )
    compliance_notes: str | None = Field(
        None,
        description="Industry-specific compliance requirements (GDPR, HIPAA, etc.)"
    )

    # Metadata
    status: DraftStatus = Field(
        default="draft",
        description="Current status of this draft"
    )
    created_from: str | None = Field(
        None,
        description="Source: 'scraping' | 'manual' | 'recommendation' | 'batch'"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this draft was created"
    )

    # Optional Tazilla fields (for updates)
    asset_id: str | None = Field(None, description="Tazilla asset ID (if updating existing)")


# -----------------------------------------------------------------------------
# Session State Model
# -----------------------------------------------------------------------------

class AgentSessionState(BaseModel):
    """
    Agent session state stored in Redis.

    Key Design:
    - session_id is the primary identifier (in Redis key)
    - org_id and user_id are stored here (extracted from auth token on first access)
    - No need to pass org_id/user_id in WebSocket URL - they come from session!

    Redis Key: asset:agent:session:{session_id}
    TTL: 8 hours (28800 seconds)
    """

    # Identity (extracted from auth token, NOT from URL!)
    session_id: str = Field(..., description="Session identifier")
    org_id: str = Field(..., description="Organization ID (from auth token)")
    user_id: str = Field(..., description="User ID (from auth token)")
    language: str = Field(default="en", description="User language (cs/sk/en)")

    # Context Memory (learned from conversation)
    industry: str | None = Field(
        None,
        description="Company industry (healthcare, finance, etc.)"
    )
    company_name: str | None = Field(
        None,
        description="Company name (from organization profile)"
    )
    preferences: dict[str, Any] = Field(
        default_factory=dict,
        description="User preferences (e.g., default CIA levels, naming conventions)"
    )

    # Working Drafts (assets being created/edited)
    drafts: list[AssetDraft] = Field(
        default_factory=list,
        description="Current draft assets (before confirmation)"
    )

    # Current Selection & Context (from UI)
    selected_asset_id: str | None = Field(
        None,
        description="Currently selected asset ID from UI (for edits)"
    )
    existing_assets: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Existing assets from UI (for context and edits)"
    )

    # History Tracking
    created_asset_names: list[str] = Field(
        default_factory=list,
        description="Names of assets created in this session (for conversation context)"
    )
    conversation_summary: str | None = Field(
        None,
        description="Brief summary of conversation so far (for context)"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When session was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    last_activity: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last user activity (for TTL tracking)"
    )

    # Optional: Website scraping cache
    scraped_content: str | None = Field(
        None,
        description="Cached website content from scraping"
    )
    scraped_url: str | None = Field(
        None,
        description="URL that was scraped"
    )
