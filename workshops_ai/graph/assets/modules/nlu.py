# workshops_ai/graph/assets/modules/nlu.py

from __future__ import annotations
from typing import cast
from workshops_ai.graph.factory import GraphDeps
from workshops_ai.graph.agent_node import agent_decide, AgentDecision
from .state import AssetState, Intent
from .text_utils import looks_czech, looks_like_greeting, is_yes_like, is_no_like
from .text_utils import looks_like_create, looks_like_explain, mentions_types
from .catalog_utils import extract_catalog_type_from_text
from .history import history_tail


def should_fetch_asset_types(pending_fields) -> bool:
    """
    Fetch asset types only when we are explicitly filling the 'assetType' field.
    This keeps the question→answer flow focused and avoids noisy UI jumps.
    """
    return bool(pending_fields and "assetType" in pending_fields)


async def nlu(deps: GraphDeps, state: AssetState) -> AssetState:
    raw = (state.message or "").strip()
    state.intent = None

    # Lightweight locale auto-hinting: if message looks Czech and locale is not set to CS, switch to cs-CZ.
    if raw and looks_czech(raw):
        if not state.locale or not state.locale.lower().startswith("cs"):
            state.locale = "cs-CZ"

    # Greeting outside of a pending flow: reply as agent and reset transient state.
    if looks_like_greeting(raw):
        if not (state.pending_fields and len(state.pending_fields) > 0):
            state.create_flow = False
            state.pending_fields = None
            state.create_draft = None
            state.actions = None
            state.once_flags = None
            state.intent = cast(Intent, "agent_reply")
            return state

    # Respect explicit action pre-routing (e.g., from UI buttons).
    if state.action in ("list", "get", "create", "update_meta", "explain", "list_types", "agent_reply"):
        state.intent = cast(Intent, state.action)
        if state.action == "list_types" and not state.create_flow:
            # If types listing isn't part of the create flow, clear pending fields to avoid confusion.
            state.pending_fields = None
        return state

    sel = state.selection or {}

    # Short yes/no outside create flow → plain agent reply (confirmation/decline).
    if not state.create_flow and not (state.pending_fields and len(state.pending_fields) > 0):
        if is_yes_like(raw) or is_no_like(raw):
            state.intent = cast(Intent, "agent_reply")
            return state

    # Inside create flow or if we have pending fields to fill.
    if state.create_flow or (state.pending_fields and len(state.pending_fields) > 0):
        # If nothing is currently pending, route based on simple intents.
        if not (state.pending_fields and len(state.pending_fields) > 0):
            if is_yes_like(raw):
                state.intent = cast(Intent, "create")
                return state
            if is_no_like(raw):
                # Abort current create flow cleanly.
                state.create_flow = False
                state.pending_fields = None
                state.create_draft = None
                state.intent = cast(Intent, "agent_reply")
                return state
            if looks_like_explain(raw):
                state.intent = cast(Intent, "explain")
                return state
            state.intent = cast(Intent, "agent_reply")
            return state

        # We are actively filling fields; only fetch catalog types if assetType is actually pending.
        if state.pending_fields and "assetType" in state.pending_fields:
            types: list[str] = []
            if should_fetch_asset_types(state.pending_fields):
                try:
                    types = [str(x) for x in (deps.backend_port.list_asset_types() or [])]
                except Exception:
                    types = []
            matched = extract_catalog_type_from_text(raw, types)
            if matched:
                state.create_draft = state.create_draft or {}
                state.create_draft["assetType"] = matched
                state.pending_fields = [f for f in (state.pending_fields or []) if f != "assetType"]
                state.intent = cast(Intent, "create")
                return state

        if looks_like_explain(raw):
            state.intent = cast(Intent, "explain")
            return state

        # Default inside-flow routing: proceed with create step handling.
        state.intent = cast(Intent, "create")
        return state

    # Outside flow: user likely expressed creation intent ("create …").
    create_intent = looks_like_create(raw)
    if create_intent:
        # Do NOT fetch types here; let the create flow prompt for type when needed.
        state.create_draft = state.create_draft or {}
        state.create_flow = True
        state.intent = cast(Intent, "create")
        return state

    # Outside flow generic type matching: intentionally skip fetching types to avoid UI noise.
    matched = None  # We deliberately avoid list_asset_types() here.
    if matched:
        state.create_draft = state.create_draft or {}
        state.create_draft["assetType"] = matched
        state.create_flow = True
        state.intent = cast(Intent, "create")
        return state

    # Agent planner decision (no unnecessary type fetching).
    allowed_actions: list[str] = ["llm.explain", "asset.list", "asset.get", "asset.update_meta"]
    if create_intent or matched or mentions_types(raw):
        allowed_actions += ["asset.create", "asset.list_types"]

    decision: AgentDecision = await agent_decide(
        deps=deps,
        tab="assets",
        message=raw,
        language=state.locale or "en",
        history=history_tail(state, 5),
        selection=sel,
        allowed_actions=allowed_actions,
        one_action_only=True,
    )

    tool_to_intent = {
        "asset.list": "list",
        "asset.list_types": "list_types",
        "asset.create": "create",
        "asset.update_meta": "update_meta",
        "llm.explain": "explain",
        "asset.get": "get",
    }
    chosen = tool_to_intent.get((decision.action or "").strip())
    state.intent = cast(Intent, chosen or "agent_reply")

    # If a bare list_types was chosen outside create flow, avoid leaving stale pending fields.
    if state.intent == "list_types" and not state.create_flow:
        state.pending_fields = None

    # Pass-through extra LLM question (if any).
    if getattr(decision, "args", None):
        args = decision.args or {}
        if "question" in args and isinstance(args["question"], str):
            state.question = args["question"]

    return state
