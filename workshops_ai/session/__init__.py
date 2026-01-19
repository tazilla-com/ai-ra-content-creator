"""
Unified session management for wizard and agent.

Single source of truth for session state stored in Redis.
Used by both wizard workflow and AI agent conversations.
"""

from workshops_ai.session.models import SessionState, RecentOperation
from workshops_ai.session.storage import (
    get_session,
    save_session,
    update_session,
    clear_session,
    clear_all_sessions,
    validate_session_id,
)
from workshops_ai.session.utils import (
    get_action_details_from_patches,
    add_recent_operation,
)

__all__ = [
    "SessionState",
    "RecentOperation",
    "get_session",
    "save_session",
    "update_session",
    "clear_session",
    "clear_all_sessions",
    "validate_session_id",
    "get_action_details_from_patches",
    "add_recent_operation",
]
