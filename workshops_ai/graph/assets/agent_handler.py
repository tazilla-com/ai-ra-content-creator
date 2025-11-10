# workshops_ai/graph/assets/agent_handler.py
"""
Asset agent handler with WebSocket progress streaming.

Design:
- LangGraph ReAct agent with tool calling
- Progress streaming (NOT token streaming - that's expensive!)
- Session-based state (Redis, not checkpointer)
- Educational approach with reasoning
- Confirmation required before writing to Tazilla

Streaming Strategy:
- NO token-by-token streaming (expensive, slow)
- YES progress events (cheap, meaningful)
  * "Analyzing request..."
  * "Scraping website..."
  * "Creating draft assets..."
  * "Preview ready"
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.graph.factory import GraphDeps
from workshops_ai.agents.session import (
    AgentSessionState,
    get_agent_session,
    save_agent_session,
)
from workshops_ai.graph.assets.agent_prompts import build_system_prompt
from workshops_ai.graph.assets.agent_tools_factory import get_bound_tools
from workshops_ai.app.services.ws_hub import WsHub

logger = logging.getLogger(__name__)

__all__ = ["run_asset_agent"]


# -----------------------------------------------------------------------------
# Agent Handler
# -----------------------------------------------------------------------------

async def run_asset_agent(
    ui_req: UIRequest,
    deps: GraphDeps,
    conn_id: str | None = None,
) -> UIResponse:
    """
    Run conversational asset agent with progress streaming.

    Args:
        ui_req: Unified request (from WebSocket or HTTP)
        deps: Graph dependencies (llm_client, backend, etc.)
        conn_id: WebSocket connection ID for progress streaming (if WS)

    Returns:
        UIResponse with messages, proposals (preview), agent_events

    Streaming Behavior:
        - If conn_id provided: sends progress events via WsHub
        - Progress events: "Analyzing...", "Scraping...", "Creating drafts...", etc.
        - NO token streaming (too expensive and slow)
        - Final response returned via UIResponse

    Flow:
        1. Load/create session from Redis
        2. Send progress: "Loading session..."
        3. Build contextualized system prompt
        4. Create LangGraph ReAct agent with tools
        5. Send progress: "Analyzing your request..."
        6. Run agent (with tool calling)
        7. Send progress for each tool execution
        8. Extract final response
        9. Save session
        10. Return UIResponse
    """
    try:
        # Progress: Loading session
        await _send_progress(conn_id, ui_req.org_id, "Loading session...", 10)

        # 1. Load or create session
        session = await get_agent_session(ui_req.session_id)

        if not session:
            # Create new session
            # Extract org_id and user_id from auth (these should come from request context)
            session = AgentSessionState(
                session_id=ui_req.session_id,
                org_id=ui_req.org_id,
                user_id=ui_req.user_id or "anonymous",
                language=ui_req.language or "en",
            )
            logger.info(
                "Created new agent session",
                extra={"session_id": session.session_id, "org_id": session.org_id}
            )

            # CRITICAL: Save new session to Redis IMMEDIATELY so tools can access it!
            await save_agent_session(session)
            logger.info(
                "Saved new session to Redis",
                extra={"session_id": session.session_id}
            )

        # 2. Enrich session with context from UI request
        if ui_req.organization_profile:
            session.industry = ui_req.organization_profile.industry
            session.company_name = ui_req.organization_profile.name

        # Update selection and existing assets for edit context
        if ui_req.selection and ui_req.selection.asset_id:
            session.selected_asset_id = ui_req.selection.asset_id

        if ui_req.existing_assets:
            # Convert ExistingAssetIn models to dicts for storage
            session.existing_assets = [asset.model_dump() for asset in ui_req.existing_assets]

        # Save enriched session
        await save_agent_session(session)

        # Progress: Building agent
        await _send_progress(conn_id, ui_req.org_id, "Preparing AI agent...", 20)

        # 3. Build system prompt with session context
        system_prompt = build_system_prompt(
            language=session.language,
            industry=session.industry,
            company_name=session.company_name,
            created_assets_count=len(session.created_asset_names),
            drafts_count=len(session.drafts),
            selected_asset_id=session.selected_asset_id,
            existing_assets_count=len(session.existing_assets),
        )

        # 4. Create LangGraph agent with session-bound tools
        bound_tools = get_bound_tools(session.session_id)

        agent = create_react_agent(
            model=deps.llm_client,
            tools=bound_tools,
            prompt=system_prompt,  # System prompt as string
            checkpointer=deps.checkpointer,  # Redis checkpointer for conversation history
        )

        # Progress: Analyzing request
        await _send_progress(conn_id, ui_req.org_id, "Analyzing your request...", 30)

        # 5. Prepare messages
        # System prompt is already set in agent creation (prompt parameter)
        # Just add the current user message - checkpointer handles history
        messages = [
            HumanMessage(content=ui_req.message),
        ]

        # 6. Run agent with thread_id (session_id) for conversation continuity
        logger.info(
            "Running asset agent",
            extra={
                "session_id": session.session_id,
                "message_preview": ui_req.message[:100],
            }
        )

        # Execute agent with thread_id to maintain conversation history
        # Checkpointer will load previous messages and append new ones
        result = await agent.ainvoke(
            {"messages": messages},
            config={"configurable": {"thread_id": f"asset_agent_{session.session_id}"}}
        )

        # Progress: Processing complete
        await _send_progress(conn_id, ui_req.org_id, "Processing response...", 90)

        # 7. Extract final response, proposals, and agent events
        final_message = _extract_final_message(result)
        agent_events = _extract_agent_events(result)
        proposals = _extract_proposals(result)

        # 8. Reload session to get latest state from tools
        # (tools modify session independently, so we need to reload)
        session = await get_agent_session(ui_req.session_id) or session

        # Progress: Done
        await _send_progress(conn_id, ui_req.org_id, "Complete", 100)

        # 9. Build preview if drafts exist (and no proposals from tool)
        if not proposals and session.drafts:
            proposals = {
                "drafts": [
                    {
                        "name": d.name,
                        "assetType": d.assetType,
                        "isPrimary": d.isPrimary,
                        "reasoning": d.reasoning,
                    }
                    for d in session.drafts
                ],
            }

        # 10. Add debug info to proposals (visible in UI)
        if proposals is None:
            proposals = {}

        proposals["_debug"] = {
            "session_id": session.session_id,
            "drafts_count": len(session.drafts),
            "created_assets_count": len(session.created_asset_names),
            "created_assets": session.created_asset_names[-5:] if session.created_asset_names else [],
            "language": session.language,
            "industry": session.industry,
        }

        # 11. Return UIResponse
        ui_response = UIResponse(
            messages=[final_message],
            proposals=proposals,
            agent_events=agent_events,
        )

        logger.info(
            "Asset agent completed",
            extra={
                "session_id": session.session_id,
                "drafts_count": len(session.drafts),
                "has_proposals": proposals is not None,
                "proposals_keys": list(proposals.keys()) if proposals else [],
                "response_length": len(final_message),
                "agent_events_count": len(agent_events),
            }
        )

        return ui_response

    except Exception as e:
        logger.error(
            "Asset agent failed",
            extra={
                "session_id": ui_req.session_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            exc_info=True,
        )

        await _send_progress(
            conn_id,
            ui_req.org_id,
            f"Error: {str(e)}",
            -1  # Error indicator
        )

        return UIResponse(
            messages=[f"Sorry, I encountered an error: {str(e)}"],
            proposals=None,
        )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

async def _send_progress(
    conn_id: str | None,
    org_id: str,
    message: str,
    percent: int
) -> None:
    """
    Send progress event via WebSocket.

    Args:
        conn_id: WebSocket connection ID (if None, no-op)
        org_id: Organization ID for routing
        message: Progress message
        percent: Progress percentage (0-100, or -1 for error)

    Note:
        This is NOT token streaming! Just meaningful progress updates.
    """
    if not conn_id:
        return  # No WebSocket, skip

    try:
        await WsHub.broadcast_json({
            "event": "agent.progress",
            "payload": {
                "conn_id": conn_id,
                "org_id": org_id,
                "message": message,
                "percent": percent,
                "timestamp": None,  # Will be added by WsHub if needed
            }
        })
    except Exception as e:
        logger.warning(
            "Failed to send progress event",
            extra={"conn_id": conn_id, "error": str(e)}
        )


def _extract_agent_events(agent_result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract tool calls as agent events.

    Args:
        agent_result: Result from agent.ainvoke()

    Returns:
        List of agent events (tool calls)
    """
    all_events = []

    try:
        messages = agent_result.get("messages", [])

        for msg in messages:
            # Check for AIMessage with tool_calls
            if isinstance(msg, dict):
                msg_type = msg.get("type")
                tool_calls = msg.get("tool_calls", [])
            elif hasattr(msg, "type"):
                msg_type = msg.type
                tool_calls = getattr(msg, "tool_calls", [])
            else:
                continue

            if msg_type == "ai" and tool_calls:
                for tc in tool_calls:
                    tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    tool_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})

                    if tool_name:
                        all_events.append({
                            "tool": tool_name,
                            "args": tool_args if isinstance(tool_args, dict) else {},
                        })

    except Exception as e:
        logger.warning(
            "Failed to extract agent events",
            extra={"error": str(e)}
        )

    return all_events


def _extract_proposals(agent_result: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extract proposals from tool results (e.g., assets from confirm_and_submit).

    Args:
        agent_result: Result from agent.ainvoke()

    Returns:
        Dictionary for UIResponse.proposals or None

    Note:
        Returns LAST match (most recent tool result) to handle:
        - asset_create_batch returns {"drafts": [...]}
        - confirm_and_submit returns {"assets": [...]}
        We want the final state (assets if confirmed, drafts if not)
    """
    try:
        messages = agent_result.get("messages", [])
        last_proposal = None

        for msg in messages:
            # Check if it's a ToolMessage
            is_tool_msg = False
            content = None

            if isinstance(msg, dict):
                is_tool_msg = msg.get("type") == "tool"
                content = msg.get("content")
            elif hasattr(msg, "type") and msg.type == "tool":
                is_tool_msg = True
                content = msg.content if hasattr(msg, "content") else None

            if not is_tool_msg or not content:
                continue

            # Parse tool result
            try:
                if isinstance(content, str):
                    import json
                    result = json.loads(content)
                else:
                    result = content

                # Extract assets if present (from confirm_and_submit) - PRIORITY
                if isinstance(result, dict) and "assets" in result:
                    last_proposal = {"assets": result["assets"]}

                # Extract drafts if present (from asset_create_batch preview)
                elif isinstance(result, dict) and "drafts" in result:
                    last_proposal = {"drafts": result["drafts"]}

            except Exception:
                continue

        return last_proposal

    except Exception as e:
        logger.warning(
            "Failed to extract proposals",
            extra={"error": str(e)}
        )

    return None


def _extract_final_message(agent_result: dict[str, Any]) -> str:
    """
    Extract final assistant message from agent result.

    Args:
        agent_result: Result from agent.ainvoke()

    Returns:
        Final message content as string
    """
    try:
        messages = agent_result.get("messages", [])
        if not messages:
            return "No response from agent"

        # Get last message
        last_msg = messages[-1]

        # Extract content (handle different message types)
        if isinstance(last_msg, dict):
            return last_msg.get("content", str(last_msg))
        elif hasattr(last_msg, "content"):
            return last_msg.content
        else:
            return str(last_msg)

    except Exception as e:
        logger.error(
            "Failed to extract final message",
            extra={"error": str(e)},
            exc_info=True
        )
        return "Error extracting response"
