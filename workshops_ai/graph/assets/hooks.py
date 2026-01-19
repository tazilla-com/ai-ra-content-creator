"""
LangGraph hooks for asset workshop agent.

Hooks are called at specific points in the agent execution lifecycle:
- pre_model_hook: Called before LLM invocation to trim message history
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import RemoveMessage
from langchain_core.messages.utils import trim_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES

if TYPE_CHECKING:
    from workshops_ai.graph.assets.state import AgentState

logger = logging.getLogger(__name__)


def pre_model_hook(state: AgentState) -> dict:
    """
    Trim message history before sending to LLM.

    This prevents token overflow by keeping only recent messages.
    Uses LangGraph's RemoveMessage to actually trim the checkpointer,
    since we have SessionState.conversation for UI history.

    Strategy:
    - Keep most recent messages up to max_tokens
    - Start on human message (don't cut off mid-exchange)
    - End on human or tool message (complete turn)

    Args:
        state: Current AgentState

    Returns:
        Dict with "messages" key containing trimmed message list

    Note:
        We use Option 2 (actually trim checkpointer) because:
        - SessionState.conversation stores last 10 for UI
        - No need to keep full history in expensive Redis checkpointer
        - Saves memory and keeps checkpointer lean
    """
    from workshops_ai.settings import get_settings

    settings = get_settings()
    max_tokens = settings.manager.max_history_tokens

    messages = state.get("messages", [])

    if not messages:
        return {}

    # Count tokens approximately (fast heuristic)
    def count_tokens_approximately(msgs: list) -> int:
        """Approximate token count: 4 chars ≈ 1 token."""
        total_chars = sum(len(str(m.content)) for m in msgs)
        return total_chars // 4

    try:
        trimmed_messages = trim_messages(
            messages,
            strategy="last",                    # Keep most recent
            token_counter=count_tokens_approximately,
            max_tokens=max_tokens,
            start_on="human",                   # Start on human message
            end_on=("human", "tool"),           # End on human or tool message
        )

        # Calculate how many messages were trimmed
        original_count = len(messages)
        trimmed_count = len(trimmed_messages)
        removed_count = original_count - trimmed_count

        if removed_count > 0:
            logger.info(
                f"Trimmed {removed_count} messages from history "
                f"({original_count} → {trimmed_count})",
                extra={
                    "session_id": state.get("session_id"),
                    "original_count": original_count,
                    "trimmed_count": trimmed_count,
                    "removed_count": removed_count,
                    "max_tokens": max_tokens,
                }
            )

        # Option 2: Actually overwrite checkpointer history
        # Use REMOVE_ALL_MESSAGES then add trimmed messages
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + trimmed_messages
        }

    except Exception as e:
        logger.error(
            f"Failed to trim messages: {e}",
            extra={"session_id": state.get("session_id"), "error": str(e)}
        )
        # On error, return empty dict (don't trim)
        return {}
