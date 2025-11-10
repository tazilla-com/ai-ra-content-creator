# workshops_ai/graph/assets/agent_tools_factory.py
"""
Tool factory for binding session_id to agent tools.

Problem:
- Tools need session_id to load/save session state
- LangGraph ReAct agent doesn't easily pass extra context to tools
- User message contains session_id, but tools can't access it easily

Solution:
- Use RunnableLambda to wrap tools with session_id injection
- Parse session_id from message content (temporary solution)
- Future: Use proper context variables when LangGraph supports it

Design:
- Tools receive session_id automatically
- Agent doesn't need to worry about session_id
- Clean separation of concerns
"""
from __future__ import annotations

import re
import logging
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableLambda

from workshops_ai.agents.tools import (
    asset_create_batch,
    asset_update_with_diff,
    confirm_and_submit,
    show_draft_preview,
    clear_drafts,
    scrape_company_website,
    detect_asset_dependencies,
    recommend_assets_for_context,
)

logger = logging.getLogger(__name__)

__all__ = ["get_bound_tools"]


# -----------------------------------------------------------------------------
# Session ID Extraction
# -----------------------------------------------------------------------------

def _extract_session_id_from_message(message_content: str) -> str | None:
    """
    Extract session_id from message content.

    Format: [SESSION_ID: abc123]

    Args:
        message_content: User message content

    Returns:
        session_id if found, None otherwise
    """
    match = re.search(r'\[SESSION_ID:\s*([^\]]+)\]', message_content)
    if match:
        return match.group(1).strip()
    return None


# -----------------------------------------------------------------------------
# Tool Binding
# -----------------------------------------------------------------------------

def _bind_session_id(tool: StructuredTool, session_id: str) -> StructuredTool:
    """
    Bind session_id to tool using partial application.

    Args:
        tool: Original tool
        session_id: Session ID to bind

    Returns:
        Tool with session_id pre-filled

    Note:
        This creates a new tool instance with session_id in default args.
        The tool can still be called without session_id (uses bound value).
    """
    # Get original function (use coroutine for async tools, func for sync)
    original_func = tool.coroutine if tool.coroutine else tool.func

    if not original_func:
        logger.error(
            f"Tool {tool.name} has no callable function (both .func and .coroutine are None)",
            extra={"tool_name": tool.name}
        )
        raise ValueError(f"Tool {tool.name} has no callable function")

    # Create wrapper that injects session_id
    async def wrapper(*args, **kwargs):
        logger.debug(
            f"Tool wrapper called: {tool.name}",
            extra={
                "session_id": session_id,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()) if kwargs else [],
            }
        )

        try:
            # LangGraph sends Pydantic model in kwargs['input'], not args[0]!
            input_obj = None

            # Check kwargs first (LangGraph pattern)
            if 'input' in kwargs and hasattr(kwargs['input'], 'session_id'):
                input_obj = kwargs['input']
            # Fallback: check args[0] (direct call pattern)
            elif args and hasattr(args[0], 'session_id'):
                input_obj = args[0]

            if input_obj:
                # Pydantic v2 models are immutable, so we need to create a new instance
                # Convert to dict, update session_id, create new instance
                input_dict = input_obj.model_dump()
                input_dict['session_id'] = session_id

                # Create new instance of the same type
                new_input = type(input_obj)(**input_dict)

                logger.info(
                    f"Tool wrapper: injected session_id into {type(input_obj).__name__}",
                    extra={"session_id": session_id, "tool": tool.name}
                )

                # Call with new input (replace in kwargs or args)
                if 'input' in kwargs:
                    kwargs['input'] = new_input
                    result = await original_func(*args, **kwargs)
                else:
                    result = await original_func(new_input, *args[1:], **kwargs)

                logger.info(
                    f"Tool wrapper: {tool.name} returned successfully",
                    extra={"session_id": session_id, "result_type": type(result).__name__}
                )

                return result
            else:
                # No Pydantic model found - this shouldn't happen!
                logger.error(
                    f"Tool wrapper: no Pydantic model with session_id found",
                    extra={
                        "tool": tool.name,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    }
                )
                raise ValueError(
                    f"Tool {tool.name}: no Pydantic model with session_id found in args or kwargs"
                )
        except Exception as e:
            logger.error(
                f"Tool wrapper failed: {tool.name}",
                extra={
                    "session_id": session_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True
            )
            raise

    # Create new tool with wrapper (mark as async with coroutine=True)
    return StructuredTool(
        name=tool.name,
        description=tool.description,
        func=wrapper,
        coroutine=wrapper,  # Tell LangGraph this is async
        args_schema=tool.args_schema,
    )


def get_bound_tools(session_id: str) -> list[StructuredTool]:
    """
    Get all agent tools with session_id bound.

    Args:
        session_id: Session ID to bind to all tools

    Returns:
        List of tools with session_id pre-filled

    Example:
        tools = get_bound_tools("ABC123")
        agent = create_react_agent(model=llm, tools=tools)
        # Tools will automatically use session_id="ABC123"
    """
    base_tools = [
        asset_create_batch,
        asset_update_with_diff,
        confirm_and_submit,
        show_draft_preview,
        clear_drafts,
        scrape_company_website,
        detect_asset_dependencies,
        recommend_assets_for_context,
    ]

    bound_tools = [_bind_session_id(tool, session_id) for tool in base_tools]

    logger.debug(
        "Created bound tools",
        extra={"session_id": session_id, "tool_count": len(bound_tools)}
    )

    return bound_tools
