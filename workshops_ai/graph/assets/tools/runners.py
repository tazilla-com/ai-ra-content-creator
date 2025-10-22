# workshops_ai/graph/assets/tools/runners.py

from __future__ import annotations
from typing import Optional
from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.graph.factory import GraphDeps
from workshops_ai.graph.assets.modules.state import AssetState

# NOTE:
# Runners are intentionally thin. They forward UI requests into the graph,
# where all logic (NLU, slot-filling, validation, writes) is implemented.
# Keep runners minimal to avoid duplication of business rules.

async def run_create(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    """Create flow is fully handled by the graph (NLU + nodes)."""
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

async def run_update_meta(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    """Unified update flow (name/description + CIA w/ primary gating) handled in the graph."""
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

# ---------------------------------------------------------------------------
# Backward-compat shim:
# Some routers/environments still import `run_set_meta`. We no longer expose a
# separate "set_meta" tool; CIA updates are covered by the unified update flow.
# To prevent ImportError and keep older routes working, we alias to update.
# ---------------------------------------------------------------------------
async def run_set_meta(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    """Deprecated: kept as a compatibility alias. Delegates to run_update_meta."""
    return await run_update_meta(deps=deps, ui_req=ui_req)

async def run_explain(*, deps: GraphDeps, text: str, locale: Optional[str] = None) -> dict:
    """Explain intent runs the explain node directly (no UIRequest wrapper needed)."""
    state = AssetState(
        org_id=getattr(getattr(deps, "settings", object()), "org_id", "demo-org"),
        active_tab="assets",
        message=text,
        question=text,
        locale=locale or "en",
        agent_events=[],
    )
    from workshops_ai.graph.assets.modules.explain import explain_assets_async
    state = await explain_assets_async(deps, state)
    return state.result or {"ok": False, "error": "no_result"}
