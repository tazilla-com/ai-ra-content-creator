# File: workshops_ai/graph/assets/modules/nlu.py
# -----------------------------------------------------------------------------
# High-level NLU router for the Assets agent (unified "update" flow).
#
# Responsibilities:
# - Map a user utterance to an internal intent (list/get/create/update_meta/explain/list_types).
# - Prefer human-readable names over raw IDs; IDs still supported downstream.
# - Keep catalog-backed list_types out of the way unless needed (create/type slot).
# - Never block CSV-backed operations by catalog failures.
#
# Selection fallback (detail → update QoL):
# - Use Manager's session_state.current_asset as a fallback target ONLY for update-like
#   intents and only if the user didn't explicitly specify another asset this turn
#   AND there is no explicit UI selection. Never apply to list/get/create.
# -----------------------------------------------------------------------------

from __future__ import annotations

import re
from typing import cast

from workshops_ai.graph.factory import GraphDeps
from workshops_ai.graph.agent_node import agent_decide, AgentDecision
from .state import AssetState, Intent
from .text_utils import (
    looks_czech,
    looks_like_greeting,
    is_yes_like,
    is_no_like,
    looks_like_create,
    looks_like_explain,
    mentions_types,
    extract_name_from_text,  # used to detect explicit target mention
)

# ----------------------------- Czech heuristics ------------------------------


def _cs_wants_list(raw: str) -> bool:
    """
    Heuristic for Czech 'list' intent.
    Triggers on verbs like "vypiš/ukaž", noun "seznam", and quantifiers "všechny/...".
    """
    text = (raw or "").lower()
    return bool(
        re.search(r"\b(vypi[sš]|uk[aá]z|seznam|v[sš]echny|v[sš]echna|v[sš]echno)\b", text)
    )


def _cs_wants_get(raw: str) -> bool:
    """Heuristic for Czech 'get detail' intent: look for the keyword 'detail'."""
    text = (raw or "").lower()
    return "detail" in text


def _cs_wants_update(raw: str) -> bool:
    """Heuristic for Czech 'update' intent: verbs 'změň/přejmenuj/uprav/aktualizuj/nastav'."""
    text = (raw or "").lower()
    return bool(re.search(r"\b(zm[eě]ň|p[řr]ejmenuj|uprav|aktualizuj|nastav)\b", text))


# ----------------------------- Target detection ------------------------------


_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


def _message_mentions_explicit_target(raw: str) -> bool:
    """
    True if message specifies a concrete target asset:
      - contains a UUID-like token, or
      - contains an extractable quoted/name-like segment.
    """
    s = (raw or "").strip()
    if not s:
        return False
    if _UUID_RE.search(s) is not None:
        return True
    name = (extract_name_from_text(s) or "").strip()
    return bool(name)


# ----------------------------- Type-list intent -------------------------------


def _wants_type_list(raw: str) -> bool:
    """
    True iff the user explicitly asks to see the catalog of asset types.
    Designed to be deterministic for Czech/Slovak/English phrasing.
    """
    if not raw:
        return False
    t = raw.strip().lower()
    # Czech/Slovak cues
    if re.search(r"\b(seznam|typy|druhy|kategorie)\b", t):
        return True
    # English cues
    if re.search(r"\b(list|show|display)\b", t) and ("type" in t or "types" in t):
        return True
    # Common free-form phrases
    if "dej mi pros" in t or "seznam typ" in t:
        return True
    return False


# ------------------------------- NLU routine ---------------------------------


async def nlu(deps: GraphDeps, state: AssetState) -> AssetState:
    """
    Top-level NLU for the Assets tab.

    Core rules:
    - asset.list / asset.get / asset.update_meta operate on existing assets (CSV-backed).
      Do NOT preflight asset.list_types for them.
    - asset.list_types is a catalog helper primarily for create/explain; it must not block others.
    - During create flow, fetch catalog types only when 'assetType' is pending AND the user actually asked to see the list.
      If the user provides a concrete type string, keep intent 'create' so the create node can parse it.
    """
    raw = (state.message or "").strip()
    state.intent = None  # reset per-turn intent

    # --- Lightweight locale auto-hint (switch to cs-CZ if text looks Czech) ---
    if raw and looks_czech(raw):
        if not state.locale or not state.locale.lower().startswith("cs"):
            state.locale = "cs-CZ"

    # --- Greetings (outside any slot-filling flow) ----------------------------
    if looks_like_greeting(raw):
        # Only reset flows if we're not mid slot-filling (prevents accidental aborts)
        if not (state.pending_fields and len(state.pending_fields) > 0) and not state.create_flow:
            # Reset transient flow state on greeting
            state.create_flow = False
            state.pending_fields = None
            state.create_draft = None
            state.actions = None
            state.once_flags = None
        state.intent = cast(Intent, "agent_reply")
        return state

    # --- Respect explicit UI action overrides (incl. BC for set_meta) ---------
    if state.action in (
        "list",
        "get",
        "create",
        "update_meta",
        "set_meta",      # backward compatibility → update_meta
        "explain",
        "list_types",
        "agent_reply",
    ):
        mapped = "update_meta" if state.action == "set_meta" else state.action
        state.intent = cast(Intent, mapped)

        # If list_types is triggered outside create flow, avoid stale slot state
        if mapped == "list_types" and not state.create_flow:
            state.pending_fields = None

        # Apply selection fallback only for update-like intents
        if state.intent == "update_meta":
            _apply_selection_fallback_for_update(state, raw)
        return state

    sel = state.selection or {}

    # --- Short yes/no outside a form flow → agent reply -----------------------
    if not state.create_flow and not (state.pending_fields and len(state.pending_fields) > 0):
        if is_yes_like(raw) or is_no_like(raw):
            state.intent = cast(Intent, "agent_reply")
            return state

    # --- Create flow (slot filling) ------------------------------------------
    if state.create_flow or (state.pending_fields and len(state.pending_fields) > 0):
        # IMPORTANT:
        # If 'assetType' is pending, route to list_types ONLY WHEN the user actually asked to see the list.
        # Otherwise, keep intent 'create' so the create node can parse the provided type string.
        if "assetType" in (state.pending_fields or []):
            if _wants_type_list(raw) or mentions_types(raw):
                state.intent = cast(Intent, "list_types")
                return state
            state.intent = cast(Intent, "create")
            return state

        # Auxiliary flow controls
        if is_yes_like(raw):
            # Let the create node capture "OK/yes" confirmations in non-staging mode.
            state.intent = cast(Intent, "create")
            return state
        if is_no_like(raw):
            # Graceful abort of create flow
            state.create_flow = False
            state.pending_fields = None
            state.create_draft = None
            state.intent = cast(Intent, "agent_reply")
            return state
        if looks_like_explain(raw):
            state.intent = cast(Intent, "explain")
            return state

        # Default: continue the create flow
        state.intent = cast(Intent, "create")
        return state

    # --- Outside flow: detect explicit create intention -----------------------
    create_intent = looks_like_create(raw)
    if create_intent:
        # Do not prefetch types here; the create flow will prompt for assetType when needed.
        state.create_draft = state.create_draft or {}
        state.create_flow = True
        state.intent = cast(Intent, "create")
        return state

    # --- Czech deterministic mapping for list/get/update (more reliable than planner)
    if looks_czech(raw):
        if _cs_wants_list(raw):
            state.intent = cast(Intent, "list")
            return state
        if _cs_wants_get(raw):
            state.intent = cast(Intent, "get")
            return state
        if _cs_wants_update(raw):
            state.intent = cast(Intent, "update_meta")
            _apply_selection_fallback_for_update(state, raw)
            return state

    # --- Agent planner decision (tool gating) --------------------------------
    # Always allow CSV-backed reads/writes:
    allowed_actions: list[str] = [
        "llm.explain",
        "asset.list",
        "asset.get",
        "asset.update_meta",
    ]
    # Widen to create & list_types only when relevant
    if create_intent or mentions_types(raw):
        allowed_actions += ["asset.create", "asset.list_types"]

    decision: AgentDecision = await agent_decide(
        deps=deps,
        tab="assets",
        message=raw,
        language=state.locale or "en",
        history=history_tail(state, 5),
        selection=sel,
        allowed_actions=allowed_actions,
        one_action_only=True,  # predictable: one tool per turn
    )

    # Map chosen planner tool → internal intent
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

    # If list_types was chosen outside create flow, avoid leaving stale pending fields
    if state.intent == "list_types" and not state.create_flow:
        state.pending_fields = None

    # Apply selection fallback only for update-like intents
    if state.intent == "update_meta":
        _apply_selection_fallback_for_update(state, raw)

    # Optional: forward planner's extra question (if supplied)
    if getattr(decision, "args", None):
        args = decision.args or {}
        if "question" in args and isinstance(args["question"], str):
            state.question = args["question"]

    return state


# --------------------------- Selection fallback ------------------------------


def _apply_selection_fallback_for_update(state: AssetState, raw_message: str) -> None:
    """
    Mutate state.selection ONLY for update-like intents when:
      - there is NO explicit UI selection (asset_id/name absent)
      - the message does NOT appear to specify a new target (uuid or extractable name)
      - Manager supplied a current_asset with an id (from prior detail/selection turn)

    Guarantees smooth “detail → update” while avoiding stale selection leaking elsewhere.
    """
    try:
        sel = state.selection or {}
        has_ui_target = bool(sel.get("asset_id") or sel.get("asset_name"))
        if has_ui_target:
            return  # honor explicit UI selection

        # If the user names a concrete target in the message, do not override
        if _message_mentions_explicit_target(raw_message):
            return

        cur = getattr(state, "current_asset", None)
        if isinstance(cur, dict):
            cid = cur.get("asset_id") or cur.get("id")
            cname = cur.get("name")
            if cid:
                # Populate selection for downstream update node
                state.selection = sel or {}
                state.selection["asset_id"] = str(cid)
                if cname:
                    state.selection["asset_name"] = str(cname)
    except Exception:
        # Best-effort; never break NLU on fallback logic
        return


# ----------------------------- History helper --------------------------------


def history_tail(state: AssetState, n: int) -> list[str]:
    """
    Return up to the last n history items (messages) from state.history if present.
    Local shim to avoid importing a heavier helper here.
    """
    try:
        hist = getattr(state, "history", None)
        if isinstance(hist, list):
            return [str(x) for x in hist[-n:]]
    except Exception:
        pass
    return []
