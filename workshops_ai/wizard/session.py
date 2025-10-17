# workshops_ai/wizard/session.py
"""
Wizard session management using Redis.

Storage pattern:
  Key: wizard:{org_id}:{user_id}
  Value: JSON dict with step data
  TTL: Configurable (default 8 hours, see WORKSHOPS_AI_WIZARD__SESSION_TTL_HOURS)

Graceful degradation: If Redis unavailable, wizard continues without state persistence.
"""
from __future__ import annotations

from workshops_ai.app.services.redis_pool import get_redis
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def get_wizard_state(org_id: str, user_id: str) -> dict:
    """
    Load wizard state for user from Redis.

    Returns empty dict if:
    - Redis unavailable
    - No wizard state exists yet
    - Error loading data

    Args:
        org_id: Organization ID
        user_id: User ID

    Returns:
        Dict containing all wizard step data
    """
    redis = await get_redis()
    if not redis:
        logger.warning(
            "Redis unavailable, wizard state not loaded",
            extra={"org_id": org_id, "user_id": user_id}
        )
        return {}

    try:
        key = f"wizard:{org_id}:{user_id}"
        data = await redis.get(key)
        return json.loads(data) if data else {}
    except Exception as e:
        logger.error(
            "Failed to load wizard state",
            extra={"org_id": org_id, "user_id": user_id, "error": str(e)}
        )
        return {}


async def save_wizard_state(org_id: str, user_id: str, state: dict) -> None:
    """
    Save wizard state to Redis with configurable TTL.

    Graceful degradation: Logs warning if Redis unavailable.

    Args:
        org_id: Organization ID
        user_id: User ID
        state: Full wizard state dict
    """
    from workshops_ai.settings import get_settings

    redis = await get_redis()
    if not redis:
        logger.warning(
            "Redis unavailable, wizard state not persisted",
            extra={"org_id": org_id, "user_id": user_id}
        )
        return

    try:
        settings = get_settings()
        ttl_seconds = settings.wizard.session_ttl_hours * 60 * 60
        key = f"wizard:{org_id}:{user_id}"
        await redis.set(key, json.dumps(state), ex=ttl_seconds)
    except Exception as e:
        logger.error(
            "Failed to save wizard state",
            extra={"org_id": org_id, "user_id": user_id, "error": str(e)}
        )


async def get_wizard_step(org_id: str, user_id: str, step_name: str) -> Optional[dict]:
    """
    Get data from specific wizard step.

    Args:
        org_id: Organization ID
        user_id: User ID
        step_name: Step name (e.g., "services", "assets")

    Returns:
        Step data dict or None if not found
    """
    state = await get_wizard_state(org_id, user_id)
    return state.get(step_name)


async def save_wizard_step(org_id: str, user_id: str, step_name: str, data: dict) -> None:
    """
    Save data for specific wizard step.

    Args:
        org_id: Organization ID
        user_id: User ID
        step_name: Step name (e.g., "services", "assets")
        data: Step data to save
    """
    state = await get_wizard_state(org_id, user_id)
    state[step_name] = data
    await save_wizard_state(org_id, user_id, state)


async def clear_wizard_state(org_id: str, user_id: str) -> None:
    """
    Clear wizard state (reset wizard).

    Args:
        org_id: Organization ID
        user_id: User ID
    """
    redis = await get_redis()
    if not redis:
        logger.warning(
            "Redis unavailable, cannot clear wizard state",
            extra={"org_id": org_id, "user_id": user_id}
        )
        return

    try:
        key = f"wizard:{org_id}:{user_id}"
        await redis.delete(key)
        logger.info(
            "Wizard state cleared",
            extra={"org_id": org_id, "user_id": user_id}
        )
    except Exception as e:
        logger.error(
            "Failed to clear wizard state",
            extra={"org_id": org_id, "user_id": user_id, "error": str(e)}
        )
