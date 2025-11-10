# workshops_ai/agents/session/storage.py
"""
Redis storage for agent sessions.

Design:
- Session ID only (no org_id/user_id in function params - they're in session state)
- Redis key pattern: asset:agent:session:{session_id}
- TTL: 8 hours (same as wizard sessions)
- Graceful degradation: logs warning if Redis unavailable

Usage:
    # Create new session (org_id/user_id from auth token)
    session = AgentSessionState(
        session_id="ABC123",
        org_id="org_456",  # from JWT
        user_id="user_789"  # from JWT
    )
    await save_agent_session(session)

    # Load session (only need session_id!)
    session = await get_agent_session("ABC123")
    if session:
        print(session.org_id)  # org_456
        print(session.user_id)  # user_789
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from workshops_ai.app.services.redis_pool import get_redis
from workshops_ai.agents.session.models import AgentSessionState

logger = logging.getLogger(__name__)

__all__ = [
    "get_agent_session",
    "save_agent_session",
    "update_agent_session",
    "clear_agent_session",
]

# Redis key prefix for agent sessions
REDIS_KEY_PREFIX = "asset:agent:session"

# Default TTL: 8 hours (same as wizard)
DEFAULT_TTL_SECONDS = 8 * 60 * 60


# -----------------------------------------------------------------------------
# Redis Key Helpers
# -----------------------------------------------------------------------------

def _make_key(session_id: str) -> str:
    """Build Redis key for session."""
    return f"{REDIS_KEY_PREFIX}:{session_id}"


# -----------------------------------------------------------------------------
# CRUD Operations
# -----------------------------------------------------------------------------

async def get_agent_session(session_id: str) -> AgentSessionState | None:
    """
    Load agent session from Redis.

    Args:
        session_id: Session identifier (only thing we need!)

    Returns:
        AgentSessionState if found, None otherwise

    Note:
        - Returns None if Redis unavailable
        - Returns None if session doesn't exist
        - Returns None if session data is corrupted
        - org_id and user_id are INSIDE the session state (not in function params!)
    """
    redis = await get_redis()
    if not redis:
        logger.warning(
            "Redis unavailable, cannot load agent session",
            extra={"session_id": session_id}
        )
        return None

    try:
        key = _make_key(session_id)
        data = await redis.get(key)

        if not data:
            logger.debug(
                "Agent session not found",
                extra={"session_id": session_id, "redis_key": key}
            )
            return None

        # Deserialize JSON → Pydantic model
        session_dict = json.loads(data)
        session = AgentSessionState.model_validate(session_dict)

        logger.debug(
            "Agent session loaded",
            extra={
                "session_id": session_id,
                "org_id": session.org_id,
                "user_id": session.user_id,
                "drafts_count": len(session.drafts),
            }
        )
        return session

    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse agent session JSON",
            extra={"session_id": session_id, "error": str(e)}
        )
        return None
    except Exception as e:
        logger.error(
            "Failed to load agent session",
            extra={"session_id": session_id, "error": str(e)}
        )
        return None


async def save_agent_session(
    session: AgentSessionState,
    ttl_seconds: int | None = None
) -> bool:
    """
    Save agent session to Redis.

    Args:
        session: Session state to save
        ttl_seconds: Optional TTL override (default: 8 hours)

    Returns:
        True if saved successfully, False otherwise

    Note:
        - Automatically updates session.updated_at and session.last_activity
        - Graceful degradation: logs warning if Redis unavailable
    """
    redis = await get_redis()
    if not redis:
        logger.warning(
            "Redis unavailable, cannot save agent session",
            extra={"session_id": session.session_id}
        )
        return False

    try:
        # Update timestamps
        now = datetime.utcnow()
        session.updated_at = now
        session.last_activity = now

        # Serialize to JSON
        key = _make_key(session.session_id)
        data = session.model_dump_json()
        ttl = ttl_seconds or DEFAULT_TTL_SECONDS

        # Save to Redis with TTL
        await redis.set(key, data, ex=ttl)

        logger.debug(
            "Agent session saved",
            extra={
                "session_id": session.session_id,
                "org_id": session.org_id,
                "user_id": session.user_id,
                "drafts_count": len(session.drafts),
                "ttl_seconds": ttl,
            }
        )
        return True

    except Exception as e:
        logger.error(
            "Failed to save agent session",
            extra={"session_id": session.session_id, "error": str(e)}
        )
        return False


async def update_agent_session(
    session_id: str,
    updates: dict[str, Any]
) -> bool:
    """
    Atomically update specific fields in agent session.

    Args:
        session_id: Session identifier
        updates: Dict of fields to update

    Returns:
        True if updated successfully, False otherwise

    Example:
        await update_agent_session("ABC123", {
            "industry": "healthcare",
            "company_name": "Hospital XYZ"
        })

    Note:
        - Loads current session, applies updates, saves back
        - NOT truly atomic (race conditions possible)
        - For atomic operations, use Redis transactions
    """
    # Load current session
    session = await get_agent_session(session_id)
    if not session:
        logger.warning(
            "Cannot update agent session - not found",
            extra={"session_id": session_id}
        )
        return False

    # Apply updates
    try:
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
            else:
                logger.warning(
                    "Ignoring unknown field in session update",
                    extra={"session_id": session_id, "field": key}
                )

        # Save back
        return await save_agent_session(session)

    except Exception as e:
        logger.error(
            "Failed to update agent session",
            extra={"session_id": session_id, "error": str(e)}
        )
        return False


async def clear_agent_session(session_id: str) -> bool:
    """
    Delete agent session from Redis.

    Args:
        session_id: Session identifier

    Returns:
        True if deleted successfully, False otherwise

    Note:
        - Idempotent: returns True even if session doesn't exist
        - Useful for logout, session reset, testing
    """
    redis = await get_redis()
    if not redis:
        logger.warning(
            "Redis unavailable, cannot clear agent session",
            extra={"session_id": session_id}
        )
        return False

    try:
        key = _make_key(session_id)
        await redis.delete(key)

        logger.info(
            "Agent session cleared",
            extra={"session_id": session_id}
        )
        return True

    except Exception as e:
        logger.error(
            "Failed to clear agent session",
            extra={"session_id": session_id, "error": str(e)}
        )
        return False
