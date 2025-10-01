# File: workshops_ai/routers/threats.py
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Body, status, Query, Header
from fastapi.responses import JSONResponse
from pydantic import RootModel, ValidationError

from workshops_ai.app.security import require_auth
from workshops_ai.tools.telemetry import telemetry_log

from workshops_ai.graph.factory import GraphDeps, make_default_deps
from workshops_ai.graph.threat_graph import create_graph, ThreatState

# Unified DTO for write requests
from workshops_ai.api.schema.threat import AssetRisk

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/graph", tags=["threats"])

_GRAPH_COMPILED = None  # type: ignore[var-annotated]
_GRAPH_DEPS: Optional[GraphDeps] = None


def _get_graph_deps(request: Request) -> GraphDeps:
    """
    Compose GraphDeps from app state; pass through all injected components so
    CSV backend and search catalogs are wired.
    """
    global _GRAPH_DEPS
    if _GRAPH_DEPS is not None:
        return _GRAPH_DEPS

    app_state = getattr(request.app, "state", None)

    prebuilt: Any = getattr(app_state, "graph_deps", None) if app_state else None
    if isinstance(prebuilt, GraphDeps):
        _GRAPH_DEPS = prebuilt
        return _GRAPH_DEPS

    _GRAPH_DEPS = make_default_deps(
        tab="threats",
        settings=getattr(app_state, "settings", None),
        backend_port=getattr(app_state, "backend_port", None),     # <-- CSV backend here
        ws_hub=getattr(app_state, "ws_hub", None),
        make_retriever=getattr(app_state, "make_retriever", None), # <-- catalog search here
        llm=getattr(app_state, "llm", None),
        memory_get=getattr(app_state, "memory_get", None),
        memory_set=getattr(app_state, "memory_set", None),
    )
    return _GRAPH_DEPS


def _get_compiled_graph(deps: GraphDeps):
    global _GRAPH_COMPILED
    if _GRAPH_COMPILED is not None:
        return _GRAPH_COMPILED
    try:
        _GRAPH_COMPILED = create_graph(deps)
        logger.info("threats_router: threats graph compiled")
    except Exception as e:  # pragma: no cover
        logger.exception("Failed to compile threats graph: %s", e)
        raise
    return _GRAPH_COMPILED


class _LooseThreatInput(RootModel[dict[str, Any]]):
    """Loose wrapper for docs; we validate with ThreatState."""
    pass


# --------------------------------------------------------------------------------------
# Generic graph endpoint (accepts full ThreatState as dict)
# --------------------------------------------------------------------------------------
@router.post("/threats", response_class=JSONResponse, summary="Execute THREATS graph")
async def run_threats_graph(
    request: Request,
    payload: dict = Body(...),
    _auth=Depends(require_auth),
):
    deps = _get_graph_deps(request)
    graph = _get_compiled_graph(deps)

    try:
        telemetry_log(
            "graph.threats.invoke",
            {
                "org_id": payload.get("org_id"),
                "action": payload.get("action"),
                "has_selection": bool((payload.get("selection") or {}).get("asset_id")),
            },
        )
    except Exception:
        pass

    try:
        state_in = ThreatState.model_validate(payload)
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Invalid threats graph input", "errors": ve.errors()},
        ) from ve

    cfg: dict[str, Any] = {"configurable": {}}
    user = state_in.user_id or "anon"
    cfg["configurable"]["thread_id"] = f"{state_in.org_id}:{deps.tab}:{user}"
    checkpoint_ns = getattr(deps, "checkpoint_ns", None)
    if checkpoint_ns:
        cfg["configurable"]["checkpoint_ns"] = checkpoint_ns

    try:
        out = await graph.ainvoke(state_in.model_dump(), config=cfg)
    except Exception as e:
        logger.exception("threats graph execution failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Threats graph execution failed", "error": str(e)},
        ) from e

    try:
        state_out = ThreatState.model_validate(out if isinstance(out, dict) else out)
    except Exception:
        return JSONResponse(content={"raw": out}, status_code=status.HTTP_200_OK)

    result = state_out.result or {
        "messages": ["No result."],
        "proposals": None,
        "staging": {"needs_commit": False, "batch_id": None, "diff": None},
        "policy": {"emit_changes_enabled": True},
    }
    return JSONResponse(content=result, status_code=status.HTTP_200_OK)


# --------------------------------------------------------------------------------------
# DTO endpoint (request body is the unified AssetRisk)
# For OpenAPI consistency and stricter validation on write.
# It wraps the validated DTO into a minimal graph state and triggers action="set".
# --------------------------------------------------------------------------------------
@router.post(
    "/threats/set",
    response_class=JSONResponse,
    summary="Set a single risk probability (DTO body: AssetRisk) via THREATS graph",
)
async def set_single_risk_via_graph(
    request: Request,
    dto: AssetRisk,                                # <- strict request body (OpenAPI shows the DTO)
    org_id: str = Query(..., description="Organization ID"),
    user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    _auth=Depends(require_auth),
):
    deps = _get_graph_deps(request)
    graph = _get_compiled_graph(deps)

    # Prepare minimal state for the graph. We forward the validated DTO as plain dict.
    state_payload: dict[str, Any] = {
        "org_id": org_id,
        "user_id": user_id,
        "active_tab": "threats",
        "action": "set",
        # The graph (threat_graph) validates as AssetRisk and uses mappers.
        "payload": dto.model_dump(),
        # Keep selection.asset_id for downstream reads or UI hints.
        "selection": {"asset_id": dto.asset_id},
    }

    try:
        telemetry_log(
            "graph.threats.set",
            {"org_id": org_id, "asset_id": dto.asset_id, "threat_code": dto.threat_code, "probability": dto.probability},
        )
    except Exception:
        pass

    # Validate state against ThreatState (single entry point shape)
    try:
        state_in = ThreatState.model_validate(state_payload)
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Invalid threats set input", "errors": ve.errors()},
        ) from ve

    cfg: dict[str, Any] = {"configurable": {}}
    user = state_in.user_id or "anon"
    cfg["configurable"]["thread_id"] = f"{state_in.org_id}:{deps.tab}:{user}"
    checkpoint_ns = getattr(deps, "checkpoint_ns", None)
    if checkpoint_ns:
        cfg["configurable"]["checkpoint_ns"] = checkpoint_ns

    try:
        out = await graph.ainvoke(state_in.model_dump(), config=cfg)
    except Exception as e:
        logger.exception("threats graph (set) execution failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Threats graph (set) execution failed", "error": str(e)},
        ) from e

    try:
        state_out = ThreatState.model_validate(out if isinstance(out, dict) else out)
    except Exception:
        return JSONResponse(content={"raw": out}, status_code=status.HTTP_200_OK)

    result = state_out.result or {
        "messages": ["No result."],
        "proposals": None,
        "staging": {"needs_commit": False, "batch_id": None, "diff": None},
        "policy": {"emit_changes_enabled": True},
    }
    return JSONResponse(content=result, status_code=status.HTTP_200_OK)


@router.get("/threats/ping", summary="Ping threats graph router")
async def ping_threats_router(_auth=Depends(require_auth)):
    return {"status": "ok", "router": "threats"}
