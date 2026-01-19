from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from workshops_ai.settings import get_settings
from workshops_ai.schemas.ws_protocol import (
    ClientMessage,
    ChatPayload,
    HelloPayload,
    ErrorPayload,
    HistorySyncPayload,
    WorkshopId,
    ConversationMessage,
)
from workshops_ai.session import SessionState, get_session, save_session
from workshops_ai.app.security import verify_ws_auth
from workshops_ai.graph.assets.patches import compute_patches_auto

try:
    from workshops_ai.app.services.ws_hub import WsHub
except Exception:
    class WsHub:
        @staticmethod
        async def register(ws: WebSocket) -> None: ...
        @staticmethod
        async def unregister(ws: WebSocket) -> None: ...

try:
    from workshops_ai.app.middlewares.rate_limit import allow_ws
except Exception:
    async def allow_ws(event: str, key: str, per_minute: int) -> bool:
        return True

try:
    from workshops_ai.app.main import get_request_id
except Exception:
    def get_request_id() -> str:
        return "n/a"

router = APIRouter()
logger = logging.getLogger(__name__)

_settings = get_settings()
WS_TIMEOUT_SEC = _settings.ws.timeout_sec


async def _ws_send(ws: WebSocket, event_type: str, payload: dict[str, Any] | None = None) -> None:
    """Send message to client with error handling"""
    try:
        await ws.send_json({"type": event_type, "payload": payload or {}})
    except WebSocketDisconnect:
        return
    except RuntimeError as e:
        if "Unexpected ASGI message" in str(e) or "once a close message has been sent" in str(e):
            return
        raise


async def _ws_error(
    ws: WebSocket,
    message: str,
    code: str,
    *,
    session_id: Optional[str] = None,
    details: Any = None,
) -> None:
    """Send error message to client"""
    payload = ErrorPayload(message=message, code=code, details=details)
    await _ws_send(ws, "error", payload.model_dump())
    logger.warning(
        f"WS error sent: {code} - {message}",
        extra={"session_id": session_id, "code": code, "details": details}
    )


async def _build_agent_state(
    session: SessionState,
    chat_payload: ChatPayload,
) -> dict:
    """
    Build initial AgentState from SessionState + ChatPayload.

    Returns dict that will be passed to agent.ainvoke().
    """
    from langchain_core.messages import HumanMessage
    from workshops_ai.wizard.prompt_context import PromptContextBuilder

    # Get valid asset types from catalog (cached 5-min)
    builder = PromptContextBuilder(session_id=session.session_id)
    asset_types_data = await builder._get_cached_catalog(
        "asset_types",
        builder.provider.load_asset_type_catalog
    )
    valid_asset_types = [at["name"] for at in asset_types_data]

    # chat_payload.assets is source of truth (includes all assets, user changes, etc)
    # FALLBACK: If chat_payload.assets is empty, try using session.wizard_steps.assets.assets
    # This is useful for testing with testui that doesn't send assets in payload
    assets_to_use = chat_payload.assets
    logger.info(f"chat_payload.assets: {len(chat_payload.assets)} assets")
    if not assets_to_use:
        # Try to get assets from wizard_steps (created during asset generation)
        wizard_steps = getattr(session, 'wizard_steps', {})
        logger.info(f"wizard_steps type: {type(wizard_steps)}, keys: {wizard_steps.keys() if isinstance(wizard_steps, dict) else 'not a dict'}")
        assets_step = wizard_steps.get('assets', {}) if wizard_steps else {}
        logger.info(f"assets_step type: {type(assets_step)}, keys: {assets_step.keys() if isinstance(assets_step, dict) else 'not a dict'}")
        wizard_assets = assets_step.get('assets', []) if assets_step else []
        logger.info(f"wizard_assets count: {len(wizard_assets) if wizard_assets else 0}")
        if wizard_assets:
            logger.info(f"chat_payload.assets is empty, using {len(wizard_assets)} assets from wizard_steps")
            assets_to_use = wizard_assets
        else:
            logger.warning(f"No assets in wizard_steps for session {session.session_id}")

    # Convert all assets to dict format
    all_assets = [
        a.model_dump() if hasattr(a, 'model_dump') else a
        for a in assets_to_use
    ]

    # Filter selected assets (user wants to actively work on these)
    selected_assets = [
        a.model_dump() if hasattr(a, 'model_dump') else a
        for a in assets_to_use
        if getattr(a, 'selected', False) if hasattr(a, 'selected') else a.get('selected', False)
    ]

    # Convert recent_operations to dict format for AgentState
    recent_operations_dict = [
        {
            "turn": op.turn,
            "action": op.action,
            "assets": op.assets,
            "details": op.details,
            "timestamp": op.timestamp.isoformat(),
        }
        for op in session.recent_operations
    ]

    # Build initial state
    initial_state = {
        # New message
        "messages": [HumanMessage(content=chat_payload.message)],

        # From ChatPayload (fresh every message - source of truth)
        "input_assets": all_assets,  # ALL assets for context (avoid duplicates)
        "selected_assets": selected_assets,  # Only selected assets (user wants to work on)

        # From catalog cache
        "valid_asset_types": valid_asset_types,

        # From SessionState
        "recent_operations": recent_operations_dict,

        # Current request
        "user_message": chat_payload.message,
        "model": chat_payload.model,
        "session_id": session.session_id,
        "workshop_id": session.workshop_id or "assets",

        # Output (empty, will be filled by agent)
        "generated_assets": [],
        "tools_used": [],
        "needs_clarification": False,
        "clarification_question": None,
        "error": None,
    }

    return initial_state


async def handle_chat(
    ws: WebSocket,
    session: SessionState,
    chat_payload: ChatPayload,
) -> None:
    """Handle CHAT message"""
    logger.info(f"handle_chat called for session {session.session_id}, busy={session.busy}")
    if session.busy:
        logger.warning(f"Session {session.session_id} is busy, rejecting request")
        await _ws_error(ws, "Agent is busy, please wait", "BUSY", session_id=session.session_id)
        return

    if _settings.ratelimit and _settings.ratelimit.enabled:
        rate_ok = await allow_ws("chat", session.session_id, _settings.ratelimit.per_minute)
        if not rate_ok:
            await _ws_error(
                ws,
                "Too many requests, please slow down",
                "RATE_LIMITED",
                session_id=session.session_id
            )
            return

    # Set busy flag
    logger.info(f"Setting busy=True for session {session.session_id}")
    session.busy = True
    await save_session(session)
    logger.info(f"Saved session {session.session_id} with busy=True")

    try:
        # Update session with latest data from frontend (source of truth)
        # FALLBACK: If chat_payload.assets is empty (e.g., during testui testing),
        # preserve existing session.assets instead of overwriting with empty list
        if chat_payload.assets:
            session.assets = chat_payload.assets
        elif not session.assets:
            # If both are empty, check wizard_steps as fallback
            wizard_steps = getattr(session, 'wizard_steps', {})
            assets_step = wizard_steps.get('assets', {}) if wizard_steps else {}
            wizard_assets = assets_step.get('assets', []) if assets_step else []
            if wizard_assets:
                logger.info(f"Using {len(wizard_assets)} assets from wizard_steps as session.assets")
                session.assets = wizard_assets
        await save_session(session)

        await _ws_send(ws, "progress", {"status": "running", "session_id": session.session_id})

        # Build initial state
        initial_state = await _build_agent_state(session, chat_payload)

        # DEBUG: Log assets being passed to agent
        input_assets = initial_state.get("input_assets", [])
        logger.info(f"Sending {len(input_assets)} assets to agent: {[a.get('name') for a in input_assets]}")

        # Invoke agent with conversation history
        from workshops_ai.graph.assets.agent import invoke_asset_agent

        # Convert session conversation to dict format for agent
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in session.conversation
        ]
        logger.info(f"Passing {len(conversation_history)} conversation messages to agent")

        result = await invoke_asset_agent(
            message=chat_payload.message,
            session_id=session.session_id,
            workshop_id=session.workshop_id or "assets",
            # No organization - agent will call scrape_company_info to get wizard_metadata.website_content
            input_assets=initial_state.get("input_assets"),
            selected_assets=initial_state.get("selected_assets"),
            valid_asset_types=initial_state.get("valid_asset_types"),
            recent_operations=initial_state.get("recent_operations"),
            conversation_history=conversation_history,
            model=chat_payload.model,
        )

        # Extract tools from messages (ToolMessage has tool name)
        from langchain_core.messages import ToolMessage, AIMessage
        tools_used = []
        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage) and hasattr(msg, "name"):
                tools_used.append(msg.name)

        # DEBUG: Log last 6 messages to see tool calls
        messages = result.get("messages", [])
        if len(messages) > 6:
            debug_messages = messages[-6:]
        else:
            debug_messages = messages

        for i, msg in enumerate(debug_messages):
            msg_type = type(msg).__name__
            if isinstance(msg, ToolMessage):
                logger.info(f"Message {i}: {msg_type} - tool={msg.name}, content_preview={msg.content[:200] if msg.content else 'None'}")
            elif isinstance(msg, AIMessage):
                has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
                logger.info(f"Message {i}: {msg_type} - has_tool_calls={has_tool_calls}, content_preview={str(msg.content)[:200] if msg.content else 'None'}")
            else:
                logger.info(f"Message {i}: {msg_type} - content_preview={str(msg.content)[:200] if hasattr(msg, 'content') else 'No content'}")

        # Extract results
        logger.info(f"Agent result keys: {list(result.keys())}")
        logger.info(f"Generated assets count: {len(result.get('generated_assets', []))}")
        logger.info(f"Tools used from messages: {tools_used}")

        # DEBUG: Log full agent result (temporary)
        import json as json_module
        debug_result = {
            'generated_assets': result.get('generated_assets'),
            'tools_used': result.get('tools_used'),
            'needs_clarification': result.get('needs_clarification'),
            'clarification_question': result.get('clarification_question'),
            'error': result.get('error'),
            'remaining_steps': result.get('remaining_steps'),
            'model': result.get('model'),
            'workshop_id': result.get('workshop_id'),
            'messages_count': len(result.get('messages', [])),
        }
        logger.info("FULL AGENT RESULT: " + json_module.dumps(debug_result, ensure_ascii=False, indent=2))

        generated_assets = result.get("generated_assets", [])
        # tools_used already extracted from messages above
        error = result.get("error")
        needs_clarification = result.get("needs_clarification", False)
        clarification_question = result.get("clarification_question")

        # Extract final AI message
        final_message = _extract_final_message(result)

        # Reload session to get updates from tools
        session = await get_session(session.session_id) or session

        # CRITICAL: Clear busy flag before saving conversation
        session.busy = False

        # NOTE: We don't save generated_assets to session.assets here!
        # Frontend handles persistence and sends all assets in chat_payload.assets
        # which we already assign to session.assets at line 192.

        # Update session conversation history with assistant response
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        session.conversation.append(ConversationMessage(
            role="user",
            content=chat_payload.message,
            timestamp=now
        ))
        session.conversation.append(ConversationMessage(
            role="assistant",
            content=final_message,
            timestamp=now
        ))

        # DEBUG: Log state before save
        logger.info(f"About to save session {session.session_id}: busy={session.busy}, assets_count={len(session.assets)}")
        try:
            await save_session(session)
            logger.info(f"Saved session {session.session_id} with busy={session.busy}")
        except Exception as save_ex:
            logger.error(f"EXCEPTION during save_session: {save_ex}", exc_info=True)
            raise

        # DEBUG: Verify save by reloading
        logger.info(f"Starting verification for session {session.session_id}")
        try:
            verification_session = await get_session(session.session_id)
            if verification_session:
                logger.info(f"Verified session {session.session_id} in Redis: busy={verification_session.busy}, assets_count={len(verification_session.assets)}")
            else:
                logger.error(f"CRITICAL: Session {session.session_id} not found in Redis after save!")
        except Exception as verify_ex:
            logger.error(f"EXCEPTION during verification: {verify_ex}", exc_info=True)

        # Send response based on agent result
        if error:
            await _ws_error(
                ws,
                error,
                "AGENT_ERROR",
                session_id=session.session_id,
                details=error
            )
        elif needs_clarification:
            await _ws_send(ws, "clarification", {
                "question": clarification_question,
                "session_id": session.session_id
            })
        elif generated_assets:
            patches = compute_patches_auto(
                original_assets=chat_payload.assets,
                generated_assets=generated_assets
            )

            await _ws_send(ws, "assets_generated", {
                "assets": generated_assets,
                "patches": patches,
                "tools_used": tools_used,
                "message": final_message,
                "session_id": session.session_id
            })
        else:
            await _ws_send(ws, "message", {
                "content": final_message,
                "session_id": session.session_id
            })

        await _ws_send(ws, "progress", {"status": "done", "session_id": session.session_id})

    except asyncio.TimeoutError:
        await _ws_error(
            ws,
            f"Request timed out after {WS_TIMEOUT_SEC}s",
            "TIMEOUT",
            session_id=session.session_id
        )
    except WebSocketDisconnect:
        raise
    except Exception as e:
        logger.exception(f"Agent error for session {session.session_id}: {e}")
        await _ws_error(
            ws,
            f"Agent execution failed: {str(e)}",
            "AGENT_ERROR",
            session_id=session.session_id,
            details=str(e)
        )
    finally:
        # Ensure busy flag is cleared even on error/timeout
        # (happy path already clears it above)
        try:
            fresh_session = await get_session(session.session_id)
            if fresh_session and fresh_session.busy:
                fresh_session.busy = False
                await save_session(fresh_session)
                logger.warning(f"Cleared busy flag in finally block for session {session.session_id} (error/timeout path)")
        except Exception as e:
            logger.error(f"Failed to clear busy flag in finally for session {session.session_id}: {e}")


async def handle_sync_history(ws: WebSocket, session: SessionState) -> None:
    """Handle SYNC_HISTORY message"""
    payload = HistorySyncPayload(
        messages=session.conversation,
        total_messages=len(session.conversation),
        returned_messages=len(session.conversation)
    )
    await _ws_send(ws, "history_sync", payload.model_dump())
    logger.debug(
        f"History synced: {len(session.conversation)} messages",
        extra={"session_id": session.session_id}
    )


@router.websocket("/ws")
async def ws_endpoint(
    ws: WebSocket,
    session_id: str = Query(..., description="Session ID"),
    workshop_id: WorkshopId = Query(..., description="Agent type"),
) -> None:
    """WebSocket endpoint for AI agent communication"""

    auth_ok = await verify_ws_auth(ws)
    if not auth_ok:
        return

    await ws.accept()
    await WsHub.register(ws)

    rid = get_request_id() or "n/a"

    session = await get_session(session_id)
    if not session:
        session = SessionState(
            session_id=session_id,
            workshop_id=workshop_id,
        )
        await save_session(session)
        logger.info(
            f"Created new session",
            extra={
                "session_id": session_id,
                "workshop_id": workshop_id,
                "request_id": rid
            }
        )
    else:
        logger.info(
            f"Restored session",
            extra={
                "session_id": session_id,
                "workshop_id": workshop_id,
                "request_id": rid
            }
        )

    hello_payload = HelloPayload(
        session_id=session_id,
        workshop_id=workshop_id,
        message="Hello! I'm ready to help with your assets."
    )
    await _ws_send(ws, "hello", hello_payload.model_dump())

    try:
        while True:
            try:
                raw = await ws.receive_json()
            except WebSocketDisconnect:
                logger.info(f"Client disconnected", extra={"session_id": session_id})
                break
            except Exception as e:
                code = getattr(e, "code", None)
                if code in (1000, 1006):
                    logger.info(f"WS client disconnected ({code})", extra={"session_id": session_id})
                    break
                logger.warning(f"WS receive error: {e}", extra={"session_id": session_id})
                continue

            try:
                client_msg = ClientMessage.model_validate(raw)
            except ValidationError as ve:
                await _ws_error(
                    ws,
                    "Invalid message format",
                    "VALIDATION_ERROR",
                    session_id=session_id,
                    details=ve.errors()
                )
                continue
            except Exception as e:
                await _ws_error(
                    ws,
                    "Invalid message format",
                    "VALIDATION_ERROR",
                    session_id=session_id,
                    details=str(e)
                )
                continue

            if client_msg.type == "ping":
                await _ws_send(ws, "pong", {})
                continue

            if client_msg.type == "sync_history":
                await handle_sync_history(ws, session)
                continue

            if client_msg.type == "chat":
                if not client_msg.payload:
                    await _ws_error(
                        ws,
                        "Missing payload for chat message",
                        "VALIDATION_ERROR",
                        session_id=session_id
                    )
                    continue

                try:
                    chat_payload = ChatPayload.model_validate(client_msg.payload)
                except ValidationError as ve:
                    await _ws_error(
                        ws,
                        "Invalid chat payload",
                        "VALIDATION_ERROR",
                        session_id=session_id,
                        details=ve.errors()
                    )
                    continue

                # CRITICAL: Reload session from Redis before handling each message
                # to get the latest busy flag state
                session = await get_session(session_id) or session
                logger.info(f"Reloaded session {session_id} before chat, busy={session.busy}")

                await handle_chat(ws, session, chat_payload)
                continue

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"Unexpected error in WebSocket handler: {e}", extra={"session_id": session_id})
    finally:
        try:
            await WsHub.unregister(ws)
        except Exception:
            pass
        logger.info(f"WebSocket closed", extra={"session_id": session_id})


def _extract_final_message(agent_result: dict) -> str:
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

        last_msg = messages[-1]

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
