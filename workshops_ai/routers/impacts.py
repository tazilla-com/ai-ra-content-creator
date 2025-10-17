# File: workshops_ai/routers/impacts.py
# -----------------------------------------------------------------------------
# Impacts Router (SSOT-aligned)
#
# What changed in this version
# - ✅ Still validates tool payloads strictly against the central ToolRegistry.
# - ✅ READ endpoints remain local (catalog + levels.for_asset).
# - 🔁 WRITE paths (impact.set_level, impact.bulk_set) no longer handcraft
#   confirmations / submit flows. Instead, they delegate to thin graph runners
#   (run_set_level / run_bulk_set) that go through the Impact LangGraph and
#   produce Asset-like confirmations (short LLM confirm + {"kind":"changeset"}
#   + {"kind":"submit_result"}). This unifies UX/telemetry with the Asset graph.
# - ℹ️  impact.explain already uses a graph runner — kept as-is.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from workshops_ai.app.security import require_auth
from workshops_ai.tools.telemetry import telemetry_log

# Backend + GraphDeps (DI like assets)
from workshops_ai.backend.port import BackendPort
from workshops_ai.backend.csv_backend import CsvBackend
from workshops_ai.graph.factory import GraphDeps
from workshops_ai.graph.factory import get_graph_deps  # same helper used by assets/threats

# UI contract
from workshops_ai.schemas.ui_contract import UIRequest, UIResponse

# SSOT: central registry + strict args
from workshops_ai.agents.tools.registry import (
    ToolRegistry,
    ImpactListArgs,
    ImpactLevelsForAssetArgs,
    ImpactSetLevelArgs,
    ImpactBulkSetArgs,
    ImpactExplainArgs,
)

# Thin graph runners (Impact graph) — single source of submit/confirm behavior
from workshops_ai.graph.impact_graph import (
    run_explain_impacts,
    run_set_level,
    run_bulk_set,
)

logger = logging.getLogger(__name__)

# Public read-only catalog router
router = APIRouter(prefix="/api/impacts", tags=["impacts"])
# Unified tool-first dispatcher (LLM tools)
graph_router = APIRouter(prefix="/graph/impacts", tags=["graph:impacts"])


# -----------------------------------------------------------------------------
# Helpers (kept; reused)
# -----------------------------------------------------------------------------
def _clamp_0_4(v: int) -> int:
    return max(0, min(4, v))


async def _run_async(func, *args, **kwargs):
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


def _get_backend_from_app(request: Request) -> BackendPort:
    app_state = getattr(request.app, "state", None)
    prebuilt = getattr(app_state, "backend_port", None) if app_state else None
    if prebuilt is not None:
        return cast(BackendPort, prebuilt)
    return CsvBackend()  # dev fallback


async def _resolve_asset_id(
    backend: BackendPort, *, org_id: Optional[str], asset_id: Optional[str], asset_name: Optional[str]
) -> str:
    if asset_id:
        return str(asset_id)
    if not asset_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Either assetId or assetName must be provided"},
        )
    resolver = getattr(backend, "asset_id_by_name", None)
    if not callable(resolver):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "assetId missing and backend cannot resolve ID by assetName"},
        )
    try:
        resolved = await _run_async(resolver, org_id=org_id, name=asset_name)
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to resolve asset by name", "error": str(ex)},
        ) from ex
    if not resolved:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"message": "Asset not found"})
    return str(resolved)


def _current_level_from_rows(rows: List[Dict[str, Any]], impact_name: str) -> Optional[int]:
    try:
        target = impact_name.strip()
        for r in rows or []:
            code = str(r.get("impact_code") or r.get("code") or r.get("name") or "").strip()
            if code == target:
                lv = r.get("level")
                if isinstance(lv, int):
                    return _clamp_0_4(lv)
                if isinstance(lv, str) and lv.strip().isdigit():
                    return _clamp_0_4(int(lv.strip()))
    except Exception:
        return None
    return None


# -----------------------------------------------------------------------------
# Public read-only route (catalog)
# -----------------------------------------------------------------------------
class ImpactLite(BaseModel):
    name: str
    description: Optional[str] = None
    weight: Optional[float] = None
    impactCategory: Optional[str] = None


class ImpactDefsListResponse(BaseModel):
    items: List[ImpactLite]
    total: int


@router.get(
    "/defs",
    response_model=ImpactDefsListResponse,
    response_class=JSONResponse,
    summary="List impact definitions (catalog)",
)
async def list_impact_defs(
    request: Request,
    org_id: Optional[str] = Query(None, description="Organization ID (if backend uses tenant catalogs)"),
    q: Optional[str] = Query(None, description="Optional free-text filter (backend-dependent)"),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    _auth=Depends(require_auth),
) -> ImpactDefsListResponse:
    backend: BackendPort = _get_backend_from_app(request)
    try:
        telemetry_log("rest.impacts.defs.list", {"org_id": org_id, "q": q, "limit": limit, "offset": offset})
    except Exception:
        pass

    try:
        defs = await _run_async(getattr(backend, "impact_defs_list"), org_id, q)  # type: ignore[misc]
    except AttributeError:
        raise HTTPException(status_code=501, detail={"message": "Backend does not support impact_defs_list()"})
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Failed to list impact definitions", "error": str(ex)}) from ex

    items: List[ImpactLite] = []
    for d in defs or []:
        obj = d.model_dump() if hasattr(d, "model_dump") else dict(d)
        items.append(
            ImpactLite(
                name=str(obj.get("name") or ""),
                description=obj.get("description"),
                weight=(obj.get("weight") if isinstance(obj.get("weight"), (int, float)) else None),
                impactCategory=obj.get("impactCategory"),
            )
        )
    sliced = items[offset : offset + limit]
    return ImpactDefsListResponse(items=sliced, total=len(items))


# -----------------------------------------------------------------------------
# Tool-first DTOs (for responses)
# -----------------------------------------------------------------------------
class ToolImpactLevelItem(BaseModel):
    impactName: str
    impactLevel: int = Field(..., ge=0, le=4)


class ToolImpactLevelsResponse(BaseModel):
    asset_id: str
    items: List[ToolImpactLevelItem]


class ImpactSetResultItem(BaseModel):
    asset_id: str
    impactName: str
    level: int = Field(..., ge=0, le=4)
    previous: Optional[int] = Field(default=None, ge=0, le=4)


class ImpactSetResult(BaseModel):
    results: List[ImpactSetResultItem]


class GraphToolCall(BaseModel):
    op: Literal[
        "impact.list",
        "impact.levels.for_asset",
        "impact.set_level",
        "impact.bulk_set",
        "impact.explain",
    ]
    args: dict = Field(default_factory=dict)
    ui: dict = Field(default_factory=dict)  # passthrough org_id/user_id/session if provided


# -----------------------------------------------------------------------------
# Unified tool-first dispatcher (SSOT; mirrors assets/threats)
# -----------------------------------------------------------------------------
@graph_router.post(
    "",
    response_class=JSONResponse,
    summary="[tool] Unified impacts tools (SSOT): list/levels.for_asset/set_level/bulk_set/explain",
)
async def graph_impacts_entrypoint(
    call: GraphToolCall = Body(...),
    deps: GraphDeps = Depends(get_graph_deps),
    _auth=Depends(require_auth),
):
    op = call.op
    args = call.args or {}
    ui = call.ui or {}

    try:
        telemetry_log("graph.impacts.dispatch", {"op": op})
    except Exception:
        pass

    # Strict SSOT validation (single source of truth)
    try:
        validated = ToolRegistry.validate_call(op, args)
    except ValidationError as ve:
        return UIResponse.error(tab="impacts", op=op, error=f"Invalid payload for {op}", details=ve.errors())

    backend: BackendPort = deps.backend
    org_id: str = ui.get("org_id") or "demo-org"
    user_id: str = ui.get("user_id") or "demo-user"
    locale: Optional[str] = ui.get("locale")

    # ---------------------------
    # READ-ONLY: impact.list
    # ---------------------------
    if op == "impact.list":
        _ = cast(ImpactListArgs, validated)
        try:
            defs = await _run_async(getattr(backend, "impact_defs_list"), org_id, None)  # type: ignore[misc]
        except AttributeError:
            return UIResponse.error(tab="impacts", op=op, error="Backend does not support impact_defs_list()")
        except Exception as ex:
            return UIResponse.error(tab="impacts", op=op, error=f"Failed to list impact definitions: {ex!s}")

        items: List[ImpactLite] = []
        for d in defs or []:
            obj = d.model_dump() if hasattr(d, "model_dump") else dict(d)
            items.append(
                ImpactLite(
                    name=str(obj.get("name") or ""),
                    description=obj.get("description"),
                    weight=(obj.get("weight") if isinstance(obj.get("weight"), (int, float)) else None),
                    impactCategory=obj.get("impactCategory"),
                )
            )
        return UIResponse.ok(tab="impacts", op=op, result={"items": items, "total": len(items)})

    # ---------------------------
    # READ: impact.levels.for_asset
    # ---------------------------
    if op == "impact.levels.for_asset":
        v = cast(ImpactLevelsForAssetArgs, validated)
        asset_id = await _resolve_asset_id(
            backend, org_id=org_id, asset_id=v.assetId, asset_name=v.assetName
        )
        try:
            rows = await _run_async(getattr(backend, "asset_get_impacts"), asset_id)  # type: ignore[misc]
        except AttributeError:
            return UIResponse.error(tab="impacts", op=op, error="Backend does not support asset_get_impacts()")
        except Exception as ex:
            return UIResponse.error(tab="impacts", op=op, error=f"Failed to fetch impacts for asset: {ex!s}")

        items: List[ToolImpactLevelItem] = []
        for r in rows or []:
            name = str(r.get("impact_code") or r.get("code") or r.get("name") or "")
            lv = r.get("level")
            if isinstance(lv, int):
                items.append(ToolImpactLevelItem(impactName=name, impactLevel=_clamp_0_4(lv)))
            elif isinstance(lv, str) and lv.strip().isdigit():
                items.append(ToolImpactLevelItem(impactName=name, impactLevel=_clamp_0_4(int(lv.strip()))))
        return UIResponse.ok(tab="impacts", op=op, result={"asset_id": str(asset_id), "items": items})

    # ---------------------------
    # WRITE (delegate to graph runner): impact.set_level
    # ---------------------------
    if op == "impact.set_level":
        v = cast(ImpactSetLevelArgs, validated)
        asset_id = await _resolve_asset_id(
            backend, org_id=org_id, asset_id=v.assetId, asset_name=v.assetName
        )

        # Build a UIRequest that tells the Impact graph to run the 'set' path directly.
        ui_req = UIRequest(
            tab="impacts",
            org_id=org_id,
            user_id=user_id,
            locale=locale,
            # Provide selection for consistency with UI + graph behavior
            selection={"asset_id": asset_id},
            # Direct intent + typed payload (graph bypasses planner and goes straight to node)
            action="set",
            set_payload={
                "asset_id": asset_id,
                "items": [{"impact_code": v.impactName, "level": _clamp_0_4(int(v.impactLevel))}],
            },
        )
        # Delegate to the graph runner (handles submit, outbox fallback, confirmation)
        return await run_set_level(deps=deps, ui_req=ui_req)

    # ---------------------------
    # WRITE (delegate to graph runner): impact.bulk_set
    # ---------------------------
    if op == "impact.bulk_set":
        v = cast(ImpactBulkSetArgs, validated)
        asset_id = await _resolve_asset_id(
            backend, org_id=org_id, asset_id=v.assetId, asset_name=v.assetName
        )

        entries = [{
            "asset_id": asset_id,
            "items": [{"impact_code": it.impactName, "level": _clamp_0_4(int(it.impactLevel))} for it in v.items],
        }]

        ui_req = UIRequest(
            tab="impacts",
            org_id=org_id,
            user_id=user_id,
            locale=locale,
            selection={"asset_id": asset_id},
            action="bulk_set",
            bulk_payload={"entries": entries},
        )
        return await run_bulk_set(deps=deps, ui_req=ui_req)

    # ---------------------------
    # EXPLAIN (delegate to graph runner): impact.explain
    # ---------------------------
    if op == "impact.explain":
        v = cast(ImpactExplainArgs, validated)
        ui_req = UIRequest(
            tab="impacts",
            org_id=org_id,
            user_id=user_id,
            locale=locale,
            action="explain",
            message=v.impactName,  # the graph will use message/question as the query
        )
        return await run_explain_impacts(deps=deps, ui_req=ui_req)

    # Unknown op (shouldn’t happen due to registry validation)
    return UIResponse.error(tab="impacts", op=op, error=f"Unsupported op: {op}")


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------
__all__ = ["router", "graph_router"]
