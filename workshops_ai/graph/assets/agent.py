"""
Asset Agent using create_react_agent with ToolRuntime pattern.
"""

from __future__ import annotations

import logging

from langgraph.prebuilt import create_react_agent

from workshops_ai.graph.assets.state import AgentState
from workshops_ai.graph.assets.context import AssetAgentContext
from workshops_ai.graph.assets.prompts import build_system_prompt
from workshops_ai.graph.assets.hooks import pre_model_hook
from workshops_ai.graph.assets.tools import (
    lookup_knowledge,
    scrape_company_info,
    list_assets,
    create_assets,
    update_assets,
)
from workshops_ai.llm.client import make_llm_client_from_settings
from workshops_ai.settings import get_settings

logger = logging.getLogger(__name__)


def create_asset_agent(
    checkpointer=None,
    model_override: str | None = None
):
    """
    Create ReAct agent for asset management.

    Args:
        checkpointer: Optional checkpointer (defaults to Redis checkpointer)
        model_override: Optional model name override

    Returns:
        Compiled LangGraph agent
    """
    settings = get_settings()

    llm = make_llm_client_from_settings(
        settings,
        override_model=model_override
    )

    tools = [
        lookup_knowledge,
        scrape_company_info,
        list_assets,
        create_assets,
        update_assets,
    ]

    if checkpointer is None:
        # Use MemorySaver - we don't need persistent checkpointing because
        # messages accumulate via conversation_history parameter across turns
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    from workshops_ai.graph.assets.prompts import ASSET_AGENT_SYSTEM_PROMPT

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_schema=AgentState,
        prompt=ASSET_AGENT_SYSTEM_PROMPT,  # Plain string, not callable!
        # pre_model_hook=pre_model_hook,  # Temporarily disable to isolate issue
        checkpointer=checkpointer,
    )

    return agent


async def invoke_asset_agent(
    message: str,
    session_id: str,
    workshop_id: str,
    input_assets: list[dict] | None = None,
    selected_assets: list[dict] | None = None,
    valid_asset_types: list[str] | None = None,
    recent_operations: list[dict] | None = None,
    conversation_history: list[dict] | None = None,
    model: str | None = None,
    checkpointer = None,
):
    """
    Invoke asset agent with message.

    Args:
        message: User message
        session_id: Session ID
        workshop_id: Workshop ID
        input_assets: Current assets
        selected_assets: Selected assets
        valid_asset_types: Valid asset types from catalog
        recent_operations: Recent operations for context
        conversation_history: Previous conversation messages
        model: Optional model override
        checkpointer: Optional checkpointer (defaults to MemorySaver)

    Returns:
        Agent state with generated_assets, tools_used, messages, etc.

    Note:
        Organization context is retrieved by agent using scrape_company_info tool
        from session.wizard_metadata.website_content
    """
    from langchain_core.messages import HumanMessage, AIMessage

    logger.info(
        "Invoking asset agent",
        extra={
            "session_id": session_id,
            "workshop_id": workshop_id,
            "message_length": len(message),
            "input_assets_count": len(input_assets or []),
            "selected_assets_count": len(selected_assets or []),
        }
    )

    agent = create_asset_agent(
        checkpointer=checkpointer,
        model_override=model
    )

    # Build messages from conversation history
    messages = []

    # Convert conversation history to LangChain messages
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    # Add current user message
    messages.append(HumanMessage(content=message))

    logger.info(f"Agent will process {len(messages)} messages (history + current)")

    initial_state: AgentState = {
        "messages": messages,
        # organization retrieved via scrape_company_info tool from wizard_metadata.website_content
        "input_assets": input_assets or [],
        "selected_assets": selected_assets or [],
        "valid_asset_types": valid_asset_types or [],
        "recent_operations": recent_operations or [],
        "user_message": message,
        "model": model or "gpt-4o",
        "session_id": session_id,
        "workshop_id": workshop_id,
        "generated_assets": [],
        "tools_used": [],  # Will be extracted from messages in ws_router
        "remaining_steps": 25,
        "needs_clarification": False,
        "clarification_question": None,
        "error": None,
    }

    result = await agent.ainvoke(
        initial_state,
        config={
            "configurable": {
                "thread_id": f"{workshop_id}:{session_id}",
            }
        },
    )

    logger.info(
        "Asset agent completed",
        extra={
            "session_id": session_id,
            "generated_assets_count": len(result.get("generated_assets", [])),
            "tools_used": result.get("tools_used", []),
            "has_error": result.get("error") is not None,
            "needs_clarification": result.get("needs_clarification", False),
        }
    )

    return result
