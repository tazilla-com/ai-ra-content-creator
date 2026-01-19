"""
Wizard session management - uses unified session.

All wizard data is stored in SessionState.wizard_steps and SessionState.wizard_metadata.
This module provides wizard-specific convenience functions that wrap the unified session.
"""

from __future__ import annotations

from workshops_ai.session import (
    get_session,
    save_session,
    validate_session_id,
    SessionState,
)
from workshops_ai.session.storage import SESSION_ID_PATTERN
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "SESSION_ID_PATTERN",
    "validate_session_id",
    "get_wizard_state",
    "get_wizard_metadata",
    "save_wizard_state",
    "get_wizard_step",
    "save_wizard_step",
    "update_session_data",
    "clear_wizard_state",
]


async def get_wizard_state(session_id: str) -> dict:
    """
    Load wizard state from unified session.

    Returns empty dict if:
    - Redis unavailable
    - No wizard state exists yet
    - Error loading data

    Args:
        session_id: Session ID

    Returns:
        Dict containing all wizard step data
    """
    session = await get_session(session_id)
    if not session:
        logger.warning(
            "Session not found, returning empty wizard state",
            extra={"session_id": session_id}
        )
        return {}

    return session.wizard_steps


async def get_wizard_metadata(session_id: str) -> dict:
    """
    Load wizard metadata from unified session.

    Returns empty dict if:
    - Redis unavailable
    - No session exists yet
    - Error loading data

    Args:
        session_id: Session ID

    Returns:
        Dict containing wizard metadata (e.g., website_content, website_url)
    """
    session = await get_session(session_id)
    if not session:
        logger.debug(
            "Session not found, returning empty wizard metadata",
            extra={"session_id": session_id}
        )
        return {}

    return session.wizard_metadata


async def save_wizard_state(session_id: str, state: dict) -> None:
    """
    Save wizard state to unified session.

    Creates session if it doesn't exist, otherwise updates wizard_steps.

    Args:
        session_id: Session ID
        state: Full wizard state dict
    """
    session = await get_session(session_id)

    if not session:
        # Create new session
        session = SessionState(
            session_id=session_id,
            wizard_steps=state,
        )
        logger.info(
            "Created new session for wizard",
            extra={"session_id": session_id}
        )
    else:
        # Update existing session
        session.wizard_steps = state

    await save_session(session)


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
    session = await get_session(session_id)

    if not session:
        session = SessionState(
            session_id=session_id,
            wizard_steps={step_name: data},
        )
        logger.info(
            "Created new session for wizard step",
            extra={"session_id": session_id, "step": step_name}
        )
    else:
        session.wizard_steps[step_name] = data

    await save_session(session)


async def update_session_data(session_id: str, updates: dict) -> None:
    """
    Update wizard metadata (e.g., website_content, scraped_data).

    Uses wizard_metadata field in unified session.

    Args:
        session_id: Session ID
        updates: Dict of fields to update
    """
    session = await get_session(session_id)

    if not session:
        session = SessionState(
            session_id=session_id,
            wizard_metadata=updates,
        )
        logger.info(
            "Created new session for wizard metadata",
            extra={"session_id": session_id}
        )
    else:
        session.wizard_metadata.update(updates)

    await save_session(session)


async def clear_wizard_state(session_id: str) -> None:
    """
    Clear wizard state (reset wizard).

    Only clears wizard fields, preserves agent data.

    Args:
        session_id: Session ID
    """
    session = await get_session(session_id)

    if not session:
        logger.warning(
            "Cannot clear wizard state - session not found",
            extra={"session_id": session_id}
        )
        return

    # Clear wizard data but keep agent data
    session.wizard_steps = {}
    session.wizard_metadata = {}

    await save_session(session)

    logger.info(
        "Wizard state cleared",
        extra={"session_id": session_id}
    )
