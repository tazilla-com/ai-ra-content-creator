"""
Unified session storage operations.

Redis Key Pattern: {session_id} (NO prefix!)
TTL: 8 hours (configurable)

Example keys:
  - "abc-123" → full session state
  - NOT "session:abc-123" (we removed the prefix!)
  - NOT "asset:agent:session:abc-123" (old pattern, removed!)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

from workshops_ai.app.services.redis_pool import get_redis
from workshops_ai.session.models import SessionState

logger = logging.getLogger(__name__)

__all__ = [
    "get_session",
    "save_session",
    "update_session",
    "clear_session",
    "clear_all_sessions",
    "validate_session_id",
]

# Session ID validation pattern
SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,40}$')


def _get_default_ttl() -> int:
    """
    Get default session TTL from settings.

    Returns configurable value from WORKSHOPS_AI_WIZARD__SESSION_TTL_HOURS.
    Defaults to 8 hours if not set.
    """
    try:
        from workshops_ai.settings import get_settings
        settings = get_settings()
        return settings.wizard.session_ttl_hours * 60 * 60
    except Exception:
        # Fallback to 8 hours if settings unavailable
        return 8 * 60 * 60


def validate_session_id(session_id: str) -> bool:
    """
    Validate session ID format.

    Args:
        session_id: Session identifier

    Returns:
        True if valid format, False otherwise
    """
    return bool(SESSION_ID_PATTERN.match(session_id))


async def get_session(session_id: str) -> SessionState | None:
    """
    Load session from Redis.

    Args:
        session_id: Session identifier

    Returns:
        SessionState if found, None otherwise

    Note:
        - Returns None if Redis unavailable
        - Returns None if session doesn't exist
        - Returns None if session data is corrupted
    """
    redis = await get_redis()
    if not redis:
        logger.warning(
            "Redis unavailable, cannot load session",
            extra={"session_id": session_id}
        )
        return None

    try:
        # Direct key lookup (no prefix!)
        data = await redis.get(session_id)

        if not data:
            logger.debug(
                "Session not found",
                extra={"session_id": session_id}
            )
            return None

        # Deserialize JSON → Pydantic model
        session_dict = json.loads(data)
        session = SessionState.model_validate(session_dict)

        logger.debug(
            "Session loaded",
            extra={
                "session_id": session_id,
                "has_wizard_data": bool(session.wizard_steps),
                "has_agent_data": bool(session.organization or session.wizard_steps.get("assets", {}).get("assets")),
            }
        )
        return session

    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse session JSON",
            extra={"session_id": session_id, "error": str(e)}
        )
        return None
    except Exception as e:
        logger.error(
            "Failed to load session",
            extra={"session_id": session_id, "error": str(e)}
        )
        return None


async def save_session(
    session: SessionState,
    ttl_seconds: int | None = None
) -> bool:
    """
    Save session to Redis.

    Args:
        session: Session state to save
        ttl_seconds: Optional TTL override (default: 8 hours)

    Returns:
        True if saved successfully, False otherwise

    Note:
        - Automatically updates session.updated_at
        - Graceful degradation: logs warning if Redis unavailable
    """
    redis = await get_redis()
    if not redis:
        logger.warning(
            "Redis unavailable, cannot save session",
            extra={"session_id": session.session_id}
        )
        return False

    try:
        # Update timestamp
        now = datetime.utcnow()
        session.updated_at = now

        # Serialize to JSON
        data = session.model_dump_json()
        ttl = ttl_seconds or _get_default_ttl()

        # Save to Redis with TTL (direct key, no prefix!)
        await redis.set(session.session_id, data, ex=ttl)

        logger.debug(
            "Session saved",
            extra={
                "session_id": session.session_id,
                "wizard_steps_count": len(session.wizard_steps),
                "conversation_length": len(session.conversation),
                "ttl_seconds": ttl,
            }
        )
        return True

    except Exception as e:
        logger.error(
            "Failed to save session",
            extra={"session_id": session.session_id, "error": str(e)}
        )
        return False


async def update_session(
    session_id: str,
    updates: dict[str, Any]
) -> bool:
    """
    Atomically update specific fields in session.

    Args:
        session_id: Session identifier
        updates: Dict of fields to update

    Returns:
        True if updated successfully, False otherwise

    Example:
        await update_session("abc-123", {
            "organization": {...},
        })

    Note:
        - Loads current session, applies updates, saves back
        - NOT truly atomic (race conditions possible)
        - For atomic operations, use Redis transactions
    """
    # Load current session
    session = await get_session(session_id)
    if not session:
        logger.warning(
            "Cannot update session - not found",
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
        return await save_session(session)

    except Exception as e:
        logger.error(
            "Failed to update session",
            extra={"session_id": session_id, "error": str(e)}
        )
        return False


async def clear_session(session_id: str) -> bool:
    """
    Delete session from Redis.

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
            "Redis unavailable, cannot clear session",
            extra={"session_id": session_id}
        )
        return False

    try:
        # Delete direct key (no prefix!)
        await redis.delete(session_id)

        logger.info(
            "Session cleared",
            extra={"session_id": session_id}
        )
        return True

    except Exception as e:
        logger.error(
            "Failed to clear session",
            extra={"session_id": session_id, "error": str(e)}
        )
        return False


async def clear_all_sessions(
    pattern: str | None = None,
    exclude_patterns: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Delete multiple sessions from Redis.

    DANGEROUS: Use with caution! This can delete many sessions at once.

    Args:
        pattern: Session ID pattern to match (supports wildcards)
                 Examples: "test-*", "*-staging", None (all sessions)
        exclude_patterns: List of patterns to exclude from deletion
                          Examples: ["prod-*", "user-*"]
        dry_run: If True, only list sessions without deleting

    Returns:
        Dict with operation results:
        {
            "deleted": 10,
            "skipped": 2,
            "session_ids": ["abc-123", "xyz-789", ...],
            "excluded": ["prod-001"],
            "dry_run": False
        }

    Examples:
        # Delete all test sessions
        await clear_all_sessions(pattern="test-*")

        # Delete all sessions except production
        await clear_all_sessions(exclude_patterns=["prod-*", "user-*"])

        # Preview what would be deleted (dry run)
        result = await clear_all_sessions(pattern="staging-*", dry_run=True)
        print(f"Would delete: {result['session_ids']}")

    WARNING:
        - No confirmation prompt! Use dry_run=True first
        - Cannot be undone
        - May delete active user sessions
    """
    redis = await get_redis()
    if not redis:
        logger.error("Redis unavailable, cannot clear sessions")
        return {
            "deleted": 0,
            "skipped": 0,
            "session_ids": [],
            "excluded": [],
            "dry_run": dry_run,
            "error": "Redis unavailable",
        }

    try:
        # Find all session keys matching pattern
        search_pattern = pattern if pattern else "*"
        all_keys = await redis.keys(search_pattern)

        # Filter out keys that don't look like session IDs
        # (avoid deleting LangGraph checkpointer keys, etc.)
        session_keys = [
            k for k in all_keys
            if validate_session_id(k) and not k.startswith("lg:ckpt:")
        ]

        # Apply exclusion patterns
        excluded = []
        if exclude_patterns:
            import fnmatch
            filtered_keys = []
            for key in session_keys:
                should_exclude = any(
                    fnmatch.fnmatch(key, exclude_pattern)
                    for exclude_pattern in exclude_patterns
                )
                if should_exclude:
                    excluded.append(key)
                else:
                    filtered_keys.append(key)
            session_keys = filtered_keys

        if not session_keys:
            logger.info(
                "No sessions found to delete",
                extra={"pattern": search_pattern, "exclude_patterns": exclude_patterns}
            )
            return {
                "deleted": 0,
                "skipped": 0,
                "session_ids": [],
                "excluded": excluded,
                "dry_run": dry_run,
            }

        if dry_run:
            logger.info(
                "DRY RUN: Would delete sessions",
                extra={"count": len(session_keys), "session_ids": session_keys}
            )
            return {
                "deleted": 0,
                "skipped": 0,
                "session_ids": session_keys,
                "excluded": excluded,
                "dry_run": True,
            }

        # Delete sessions in batch
        if session_keys:
            deleted_count = await redis.delete(*session_keys)
        else:
            deleted_count = 0

        logger.warning(
            "Bulk session deletion completed",
            extra={
                "deleted": deleted_count,
                "pattern": search_pattern,
                "excluded_count": len(excluded),
            }
        )

        return {
            "deleted": deleted_count,
            "skipped": 0,
            "session_ids": session_keys,
            "excluded": excluded,
            "dry_run": False,
        }

    except Exception as e:
        logger.error(
            "Failed to clear sessions",
            extra={"pattern": pattern, "error": str(e)}
        )
        return {
            "deleted": 0,
            "skipped": 0,
            "session_ids": [],
            "excluded": [],
            "dry_run": dry_run,
            "error": str(e),
        }
