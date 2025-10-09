# File: workshops_ai/routers/threats.py
from __future__ import annotations


# ---------------------------
# Unified like assets/impacts; ChangeSet + outbox; RAG→LLM explain)
# ---------------------------
import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from workshops_ai.graph.threat_graph import run_explain
from workshops_ai.app.security import require_auth
from workshops_ai.tools.telemetry import telemetry_log

# Backend + GraphDeps (DI consistent with other routers)
from workshops_ai.backend.port import BackendPort
from workshops_ai.backend.csv_backend import CsvBackend
from workshops_ai.graph.factory import GraphDeps, get_graph_deps


# UI contract (same DTO used across routers)
from workshops_ai.schemas.ui_contract import UIRequest, UIResponse

# SSOT: central registry + strict args
from workshops_ai.agents.tools.registry import (
    ToolRegistry,
    ThreatListArgs,
    ThreatSetProbabilityArgs,
    ThreatBulkSetArgs,
    ThreatExplainArgs,
)

logger = logging.getLogger(__name__)

# Public read-only catalog router
router = APIRouter(prefix="/api/threats", tags=["threats"])
# Unified tool-first dispatcher
graph_router = APIRouter(prefix="/graph/threats", tags=["graph:threats"])


# ---------------------------
# Helpers
# ---------------------------
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


def _current_probability_from_rows(rows: List[Dict[str, Any]], threat_name: str) -> Optional[int]:
    """
    backend.threats_for_asset(asset_id) rows may contain:
      - threat_code | code | name
      - probability | level (0..4)
    """
    try:
        target = threat_name.strip()
        for r in rows or []:
            code = str(r.get("threat_code") or r.get("code") or r.get("name") or "").strip()
            if code == target:
                lv = r.get("probability") or r.get("level")
                if isinstance(lv, int):
                    return _clamp_0_4(lv)
                if isinstance(lv, str) and lv.strip().isdigit():
                    return _clamp_0_4(int(lv.strip()))
    except Exception:
        return None
    return None


# ---------------------------
# Public read-only route (catalog)
# ---------------------------
class ThreatLite(BaseModel):
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    unintentional: Optional[bool] = None
    intentional: Optional[bool] = None
    environmental: Optional[bool] = None


class ThreatDefsListResponse(BaseModel):
    items: List[ThreatLite]
    total: int


@router.get(
    "/defs",
    response_model=ThreatDefsListResponse,
    response_class=JSONResponse,
    summary="List threat definitions (catalog)",
)
async def list_threat_defs(
    request: Request,
    org_id: Optional[str] = Query(None, description="Organization ID (if backend uses tenant catalogs)"),
    q: Optional[str] = Query(None, description="Optional free-text filter (backend-dependent)"),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    _auth=Depends(require_auth),
) -> ThreatDefsListResponse:
    backend: BackendPort = _get_backend_from_app(request)
    try:
        telemetry_log("rest.threats.defs.list", {"org_id": org_id, "q": q, "limit": limit, "offset": offset})
    except Exception:
        pass

    try:
        defs = await _run_async(getattr(backend, "threat_defs_list"), org_id, q)  # type: ignore[misc]
    except AttributeError:
        raise HTTPException(status_code=501, detail={"message": "Backend does not support threat_defs_list()"})
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Failed to list threat definitions", "error": str(ex)}) from ex

    items: List[ThreatLite] = []
    for d in defs or []:
        obj = d.model_dump() if hasattr(d, "model_dump") else dict(d)
        items.append(
            ThreatLite(
                name=str(obj.get("name") or ""),
                description=obj.get("description"),
                category=obj.get("category"),
                unintentional=bool(obj.get("unintentional", False)),
                intentional=bool(obj.get("intentional", False)),
                environmental=bool(obj.get("environmental", False)),
            )
        )
    sliced = items[offset : offset + limit]
    return ThreatDefsListResponse(items=sliced, total=len(items))


# ---------------------------
# Tool-first DTOs (for responses)
# ---------------------------
class ThreatSetResultItem(BaseModel):
    asset_id: str
    threatName: str
    level: int = Field(..., ge=0, le=4)
    previous: Optional[int] = Field(default=None, ge=0, le=4)


class ThreatSetResult(BaseModel):
    results: List[ThreatSetResultItem]


class GraphToolCall(BaseModel):
    op: Literal[
        "threat.list",
        "threat.set_probability",
        "threat.bulk_set",
        "threat.explain",
    ]
    args: dict = Field(default_factory=dict)
    ui: dict = Field(default_factory=dict)  # passthrough org_id/user_id/session if provided


# ---------------------------
# Unified tool-first dispatcher (SSOT; mirrors assets/impacts)
# ---------------------------
@graph_router.post(
    "",
    response_class=JSONResponse,
    summary="[tool] Unified threats tools (SSOT): list/set_probability/bulk_set/explain (RAG→LLM)",
)
async def graph_threats_entrypoint(
    call: GraphToolCall = Body(...),
    deps: GraphDeps = Depends(get_graph_deps),
    _auth=Depends(require_auth),
):
    op = call.op
    args = call.args or {}
    ui = call.ui or {}

    try:
        telemetry_log("graph.threats.dispatch", {"op": op})
    except Exception:
        pass

    # Strict SSOT validation (use tab scoping if your registry expects it)
    try:
        validated = ToolRegistry.validate_call(op, args)
        # or: ToolRegistry.validate_call(tab="threats", op=op, args=args)
    except ValidationError as ve:
        return UIResponse.error(tab="threats", op=op, error=f"Invalid payload for {op}", details=ve.errors())

    backend: BackendPort = deps.backend
    org_id: str = ui.get("org_id") or "demo-org"
    user_id: str = ui.get("user_id") or "demo-user"

    # ---------------------------
    # READ-ONLY: threat.list
    # ---------------------------
    if op == "threat.list":
        _ = cast(ThreatListArgs, validated)
        try:
            defs = await _run_async(getattr(backend, "threat_defs_list"), org_id, None)  # type: ignore[misc]
        except AttributeError:
            return UIResponse.error(tab="threats", op=op, error="Backend does not support threat_defs_list()")
        except Exception as ex:
            return UIResponse.error(tab="threats", op=op, error=f"Failed to list threat definitions: {ex!s}")

        items: List[ThreatLite] = []
        for d in defs or []:
            obj = d.model_dump() if hasattr(d, "model_dump") else dict(d)
            items.append(
                ThreatLite(
                    name=str(obj.get("name") or ""),
                    description=obj.get("description"),
                    category=obj.get("category"),
                    unintentional=bool(obj.get("unintentional", False)),
                    intentional=bool(obj.get("intentional", False)),
                    environmental=bool(obj.get("environmental", False)),
                )
            )
        return UIResponse.ok(tab="threats", op=op, result={"items": items, "total": len(items)})

    # ---------------------------
    # WRITE (ChangeSet→submit_or_outbox): threat.set_probability
    # ---------------------------
    if op == "threat.set_probability":
        v = cast(ThreatSetProbabilityArgs, validated)
        asset_id = await _resolve_asset_id(
            backend, org_id=org_id, asset_id=v.assetId, asset_name=v.assetName
        )

        # Best-effort previous read
        try:
            prev_rows = await _run_async(getattr(backend, "threats_for_asset"), asset_id)  # type: ignore[misc]
        except Exception:
            prev_rows = []

        previous = _current_probability_from_rows(prev_rows, v.threatName)
        lvl = _clamp_0_4(int(v.probability))

        # Build and submit ChangeSet (aligned with impacts/assets pattern)
        changeset = deps.changes.build_set_threat_probability(
            org_id=org_id,
            user_id=user_id,
            asset_id=asset_id,
            threat_code=v.threatName,  # registry keeps canonical code/name in threatName
            probability=lvl,
            note=None,
        )
        submit_res = await deps.submit_or_outbox(changeset)

        return UIResponse.submitted(
            tab="threats",
            op=op,
            changeset=changeset,
            submit=submit_res,
            result={"results": [ThreatSetResultItem(asset_id=asset_id, threatName=v.threatName, level=lvl, previous=previous)]},
        )

    # ---------------------------
    # WRITE (ChangeSet→submit_or_outbox): threat.bulk_set
    # ---------------------------
    if op == "threat.bulk_set":
        v = cast(ThreatBulkSetArgs, validated)
        asset_id = await _resolve_asset_id(
            backend, org_id=org_id, asset_id=v.assetId, asset_name=v.assetName
        )

        # Best-effort previous
        try:
            prev_rows = await _run_async(getattr(backend, "threats_for_asset"), asset_id)  # type: ignore[misc]
        except Exception:
            prev_rows = []

        entries = []
        results: List[ThreatSetResultItem] = []
        for it in v.items:
            lvl = _clamp_0_4(int(it.probability))
            entries.append({"threat_code": it.threatName, "probability": lvl, "note": None})
            results.append(
                ThreatSetResultItem(
                    asset_id=asset_id,
                    threatName=it.threatName,
                    level=lvl,
                    previous=_current_probability_from_rows(prev_rows, it.threatName),
                )
            )

        changeset = deps.changes.build_bulk_set_threat_probabilities(
            org_id=org_id, user_id=user_id, asset_id=asset_id, entries=entries
        )
        submit_res = await deps.submit_or_outbox(changeset)

        return UIResponse.submitted(
            tab="threats",
            op=op,
            changeset=changeset,
            submit=submit_res,
            result={"results": results},
        )

    # ---------------------------
    # EXPLAIN (RAG→LLM fallback): threat.explain
    # ---------------------------
async def run_explain_threats(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    # Backed by manager so máme jednotné chování (RAG→LLM fallback už řeší explain path uvnitř).
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)



# ---------------------------
# Exports
# ---------------------------
__all__ = ["router", "graph_router"]
