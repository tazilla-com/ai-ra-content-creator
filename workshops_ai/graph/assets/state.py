"""
AgentState TypedDict for LangGraph asset workshop agent.

This state is created fresh for each request and managed by LangGraph's checkpointer.
Dynamic fields (organization, assets, etc.) are replaced every message from ChatPayload.
"""

from __future__ import annotations

from typing import Annotated, Optional, Literal
from typing_extensions import TypedDict
from operator import add

from langchain_core.messages import BaseMessage


class RecentOperationDict(TypedDict):
    """
    Recent operation for agent context (TypedDict version for state).

    Note: This is a dict representation of session.RecentOperation.
    We use TypedDict instead of Pydantic here because LangGraph state must be serializable.
    """
    turn: int
    action: Literal["created", "updated", "deleted"]
    assets: list[str]
    details: Optional[str]
    timestamp: str  # ISO format datetime string


class AgentState(TypedDict):
    """
    LangGraph state for asset workshop agent.

    Created fresh for each request. LangGraph checkpointer manages persistence
    between nodes within a single request.

    FIELD BEHAVIORS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ACCUMULATED (managed by LangGraph):
    - messages: Annotated[list[BaseMessage], add]
      → LangGraph appends new messages to existing list
      → Trimmed by pre_model_hook before LLM call to prevent token overflow

    REPLACED (from ChatPayload every message):
    - organization, input_assets, selected_assets, valid_asset_types
      → Fresh data from Tazilla on each turn
      → Ensures agent always sees current state

    ACCUMULATED/OUTPUT (managed by LangGraph):
    - generated_assets, tools_used: Annotated with 'add' operator
      → Accumulated across turns within same checkpoint thread
      → Provides full history of tools used during multi-turn asset creation flow

    RESET (flow control):
    - needs_clarification, clarification_question, error
      → Cleared at start of each request
      → Filled by agent during execution

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    # ═══════════════════════════════════════════════════════════════════════
    # CONVERSATION (accumulated, trimmed by pre_model_hook)
    # ═══════════════════════════════════════════════════════════════════════
    messages: Annotated[list[BaseMessage], add]

    # ═══════════════════════════════════════════════════════════════════════
    # DYNAMIC CONTEXT (replaced every message from ChatPayload)
    # ═══════════════════════════════════════════════════════════════════════

    # Organization context
    organization: Optional[dict]  # From ChatPayload.organization

    # Assets context
    input_assets: list[dict]       # All current assets from ChatPayload
    selected_assets: list[dict]    # Filtered: assets where selected=True

    # Valid values (from Tazilla API catalog, can change)
    valid_asset_types: list[str]   # Asset type catalog

    # Recent operations for context (from SessionState)
    recent_operations: list[RecentOperationDict]  # Last 5 operations

    # ═══════════════════════════════════════════════════════════════════════
    # CURRENT REQUEST
    # ═══════════════════════════════════════════════════════════════════════
    user_message: str              # Current user message
    model: str                     # LLM model for this request
    session_id: str                # Link back to SessionState
    workshop_id: str               # "assets" | "impacts" | "threats" | "controls"

    # ═══════════════════════════════════════════════════════════════════════
    # AGENT OUTPUT (reset each request)
    # ═══════════════════════════════════════════════════════════════════════
    generated_assets: Annotated[list[dict], add]   # New/updated assets from agent (accumulated)
    tools_used: Annotated[list[str], add]          # Tools that were called (accumulated)

    # ═══════════════════════════════════════════════════════════════════════
    # FLOW CONTROL
    # ═══════════════════════════════════════════════════════════════════════
    remaining_steps: int           # Steps remaining for agent execution
    needs_clarification: bool      # Agent needs more info
    clarification_question: Optional[str]  # Question to ask user
    error: Optional[str]           # Error message if failed
