# workshops_ai/wizard/session.py
"""
Wizard session management using Redis.

Storage pattern:
  Key: {session_id}
  Value: JSON dict with all session data (wizard steps + scraped content)
  TTL: Configurable (default 8 hours, see WORKSHOPS_AI_WIZARD__SESSION_TTL_HOURS)

Graceful degradation: If Redis unavailable, wizard continues without state persistence.
"""
from __future__ import annotations

from workshops_ai.app.services.redis_pool import get_redis
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Session ID format: 22 characters of base64url (from secrets.token_urlsafe(16))
# Example: nOB3VEiMhk8AyxQvz9m_8A
SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{22}$')


def validate_session_id(session_id: str) -> bool:
    """
    Validate session ID format.

    Args:
        session_id: Session ID to validate

    Returns:
        True if valid format, False otherwise
    """
    return bool(SESSION_ID_PATTERN.match(session_id))


async def get_wizard_state(session_id: str) -> dict:
    """
    Load wizard state from Redis.

    Returns empty dict if:
    - Redis unavailable
    - No wizard state exists yet
    - Error loading data

    Args:
        session_id: Session ID

    Returns:
        Dict containing all wizard step data
    """
    redis = await get_redis()
    if not redis:
        logger.warning(
            "Redis unavailable, wizard state not loaded",
            extra={"session_id": session_id}
        )
        return {}

    try:
        data = await redis.get(session_id)
        return json.loads(data) if data else {}
    except Exception as e:
        logger.error(
            "Failed to load wizard state",
            extra={"session_id": session_id, "error": str(e)}
        )
        return {}


async def save_wizard_state(session_id: str, state: dict) -> None:
    """
    Save wizard state to Redis with configurable TTL.

    Graceful degradation: Logs warning if Redis unavailable.

    Args:
        session_id: Session ID
        state: Full wizard state dict
    """
    from workshops_ai.settings import get_settings

    redis = await get_redis()
    if not redis:
        logger.warning(
            "Redis unavailable, wizard state not persisted",
            extra={"session_id": session_id}
        )
        return

    try:
        settings = get_settings()
        ttl_seconds = settings.wizard.session_ttl_hours * 60 * 60
        await redis.set(session_id, json.dumps(state), ex=ttl_seconds)
    except Exception as e:
        logger.error(
            "Failed to save wizard state",
            extra={"session_id": session_id, "error": str(e)}
        )


async def get_wizard_step(session_id: str, step_name: str) -> Optional[dict]:
    """
    Get data from specific wizard step.

    Args:
        session_id: Session ID
        step_name: Step name (e.g., "services", "assets")

    Returns:
        Step data dict or None if not found
    """
    state = await get_wizard_state(session_id)
    return state.get(step_name)


async def save_wizard_step(session_id: str, step_name: str, data: dict) -> None:
    """
    Save data for specific wizard step.

    Args:
        session_id: Session ID
        step_name: Step name (e.g., "services", "assets")
        data: Step data to save
    """
    state = await get_wizard_state(session_id)
    state[step_name] = data
    await save_wizard_state(session_id, state)


async def update_session_data(session_id: str, updates: dict) -> None:
    """
    Atomically update session data fields.

    Loads current state, applies updates, saves back.
    Useful for updating individual fields (e.g., website_content) without
    affecting wizard step data.

    Args:
        session_id: Session ID
        updates: Dict of fields to update
    """
    state = await get_wizard_state(session_id)
    state.update(updates)
    await save_wizard_state(session_id, state)


async def clear_wizard_state(session_id: str) -> None:
    """
    Clear wizard state (reset wizard).

    Args:
        session_id: Session ID
    """
    redis = await get_redis()
    if not redis:
        logger.warning(
            "Redis unavailable, cannot clear wizard state",
            extra={"session_id": session_id}
        )
        return

    try:
        await redis.delete(session_id)
        logger.info(
            "Wizard state cleared",
            extra={"session_id": session_id}
        )
    except Exception as e:
        logger.error(
            "Failed to clear wizard state",
            extra={"session_id": session_id, "error": str(e)}
        )
