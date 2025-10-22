# File: workshops_ai/graph/assets/asset_graph.py
from __future__ import annotations

"""
LangGraph wiring for the *Assets* tab.

This file is intentionally thin: it only composes the graph from small
modules and exposes the public runners. All business logic lives in
`workshops_ai/graph/assets/modules/*`, while tool-facing runner
entrypoints (used by the app/manager) live in `.../assets/tools/runners.py`.

Directory layout this file expects:

assets/
  asset_graph.py            
  modules/                  
    state.py
    nlu.py
    nodes_read.py
    nodes_create.py
    nodes_update.py
    explain.py
    reply.py
    write_utils.py
    ... (helpers)
  tools/
    runners.py              
"""

from typing import Callable

# LangGraph
try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("LangGraph must be installed to use asset_graph.") from e

# Types / deps
from workshops_ai.graph.factory import GraphDeps

# ---- Core state (Pydantic model) -------------------------------------------
from .modules.state import AssetState

# ---- Nodes & planner (internal modules) ------------------------------------
# Planner / NLU: inspects the incoming turn and chooses an intent.
from .modules.nlu import nlu as _nlu

# Read-only nodes
from .modules.nodes_read import (
    assets_list as _assets_list,
    asset_get_or_preview as _asset_get_or_preview,
    list_asset_types as _list_asset_types,
)

# Create / Update nodes (separated)
from .modules.nodes_create import create_asset as _create_asset
from .modules.nodes_update import update_meta as _update_meta, set_meta as _set_meta

# Explain (LLM/RAG) and small-talk reply
from .modules.explain import explain_assets_async as _explain_assets_async
from .modules.reply import agent_reply as _agent_reply

# ---- Public runners (tool-surface) -----------------------------------------
# These are used by the app/manager to execute single-turn helpers.
from .tools.runners import run_create, run_update_meta, run_set_meta, run_explain


def create_graph(deps: GraphDeps):
    """
    Build and compile the Assets LangGraph.

    The graph has a single entry ("nlu") which decides the `intent`
    and then routes to an action node. Write operations are submitted
    inline inside their nodes (no recursive state), therefore most
    nodes terminate at END. A legacy "submit_changeset" node is kept
    for backward compatibility but is normally unused.
    """
    graph = StateGraph(AssetState)

    # --- Async adapters around modular functions ----------------------------
    async def _nlu_node(state: AssetState) -> AssetState:
        # Planner: mutate `state.intent` and any slot-filling flags
        return await _nlu(deps, state)

    async def _explain_node(state: AssetState) -> AssetState:
        # LLM / RAG explain path
        return await _explain_assets_async(deps, state)

    async def _submit_node(state: AssetState) -> AssetState:
        """
        Legacy submit path. Normally unused because create/update submit inline.
        Kept only so older edges won't break.
        """
        from .modules.write_utils import apply_actions_async as _apply  # lazy import
        acts = state.actions or []
        return await _apply(deps=deps, state=state, acts=acts)

    # --- Nodes ---------------------------------------------------------------
    graph.add_node("nlu", _nlu_node)
    graph.add_node("route", lambda s: s)  # trivial pass-through gate

    # Business/action nodes
    graph.add_node("agent.reply", _agent_reply(deps))
    graph.add_node("asset.list", _assets_list(deps))
    graph.add_node("asset.list_types", _list_asset_types(deps))
    graph.add_node("asset.get", _asset_get_or_preview(deps))
    graph.add_node("asset.create", _create_asset(deps))
    graph.add_node("asset.update_meta", _update_meta(deps))
    graph.add_node("asset.set_meta", _set_meta(deps))  # alias; routed z intentu 'set_meta'
    graph.add_node("submit_changeset", _submit_node)
    graph.add_node("llm.explain", _explain_node)

    # --- Edges & routing -----------------------------------------------------
    graph.set_entry_point("nlu")
    graph.add_edge("nlu", "route")

    def _route(state: AssetState) -> str:
        """
        Map planner intent → node name.
        Defaults to a friendly `agent.reply`.
        """
        intent = (state.intent or "agent_reply").strip()
        if intent == "list":
            return "asset.list"
        if intent == "list_types":
            return "asset.list_types"
        if intent == "get":
            return "asset.get"
        if intent == "create":
            return "asset.create"
        if intent == "update_meta":
            return "asset.update_meta"
        if intent == "set_meta":
            return "asset.set_meta"
        if intent == "explain":
            return "llm.explain"
        return "agent.reply"

    graph.add_conditional_edges(
        "route",
        _route,
        {
            "agent.reply": "agent.reply",
            "asset.list": "asset.list",
            "asset.list_types": "asset.list_types",
            "asset.get": "asset.get",
            "asset.create": "asset.create",
            "asset.update_meta": "asset.update_meta",
            "asset.set_meta": "asset.set_meta",
            "llm.explain": "llm.explain",
        },
    )

    # After create/update we end (writes are submitted inline).
    # The old "submit_changeset" edge remains harmlessly for compatibility.
    graph.add_conditional_edges("asset.create", lambda s: "END", {"END": END, "submit_changeset": "submit_changeset"})
    graph.add_edge("asset.update_meta", END)
    graph.add_edge("asset.set_meta", END)

    # All read-only / helper nodes end the turn.
    for node in ("agent.reply", "asset.list", "asset.list_types", "asset.get", "llm.explain", "submit_changeset"):
        graph.add_edge(node, END)

    # Compile with the shared checkpointer from deps.
    return graph.compile(checkpointer=deps.checkpointer)


__all__ = [
    # State + builder
    "AssetState",
    "create_graph",
    # Public runners
    "run_create",
    "run_update_meta",
    "run_set_meta",
    "run_explain",
]
