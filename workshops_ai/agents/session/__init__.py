# workshops_ai/agents/session
"""
Agent session management using Redis.

Main exports:
- AgentSessionState: Session state model
- AssetDraft: Draft asset being created/edited
- get_agent_session(): Load session from Redis
- save_agent_session(): Save session to Redis
- update_agent_session(): Atomically update session fields
"""
from __future__ import annotations

from workshops_ai.agents.session.models import (
    AgentSessionState,
    AssetDraft,
    DraftStatus,
)
from workshops_ai.agents.session.storage import (
    get_agent_session,
    save_agent_session,
    update_agent_session,
    clear_agent_session,
)

__all__ = [
    "AgentSessionState",
    "AssetDraft",
    "DraftStatus",
    "get_agent_session",
    "save_agent_session",
    "update_agent_session",
    "clear_agent_session",
]
