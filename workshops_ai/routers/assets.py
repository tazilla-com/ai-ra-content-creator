# File: workshops_ai/routers/assets.py
# -----------------------------------------------------------------------------
# Assets Router
#
# Responsibilities
# - Public, read-only REST for assets (client apps, dashboards).
# - Unified tool-first endpoint (/graph/assets) for the LLM agent to call.
#
# Design
# - Read-only REST: returns canonical API DTOs (workshops_ai.api.schema.asset.Asset).
# - Tool-first (/graph/assets): validates strictly against ToolRegistry (SSOT) and
#   dispatches to the asset graph runners for create/update/explain. One tool per call.
# - No hardcoded conversational strings here — the LLM + graph handle slot-asking
#   and confirmations. This router only validates, routes, and shapes responses.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import inspect
import uuid
from typing import Any, List, Optional, cast, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from workshops_ai.app.security import require_auth
from workshops_ai.backend.port import BackendPort
from workshops_ai.backend.csv_backend import CsvBackend
from workshops_ai.tools.telemetry import telemetry_log

# Graph runners (LLM-first behavior for selected ops)
from workshops_ai.graph.asset_graph import run_explain, run_create, run_update_meta
from workshops_ai.graph.factory import GraphDeps

# Canonical public DTOs for assets (client contracts)
from workshops_ai.api.schema.asset import Asset

# SSOT (tool args) from the canonical registry
from workshops_ai.agents.tools.registry import (
    ToolRegistry,
    AssetListArgs,
    AssetCreateArgs,        # imported for Literal parity & potential future use
    AssetUpdateMetaArgs,
    AssetExplainArgs,
)

# UI contract (used when delegating to graph runners)
from workshops_ai.schemas.ui_contract import UIRequest

# Routers:
# - `router`        → /api/assets (public read-only)
# - `graph_router`  → /graph/assets (tool-first SSOT endpoint for the agent)
router = APIRouter(prefix="/api/assets", tags=["assets"])
graph_router = APIRouter(prefix="/graph/assets", tags=["graph:assets"])

# -----------------------------------------------------------------------------
# Response models (public read-only)
# -----------------------------------------------------------------------------

class ListAssetsResponse(BaseModel):
    items: List[Asset]
    count: int


class AssetImpactItem(BaseModel):
    impact_code: str = Field(..., description="Impact code from state (canonical or tenant-specific)")
    level: int = Field(..., ge=0, le=4, description="0..4 scale stored in state")
    note: Optional[str] = Field(None, description="Optional note from state")


class AssetRiskItem(BaseModel):
    threat_code: str = Field(..., description="Threat code from state (canonical or tenant-specific)")
    probability: int = Field(..., ge=0, le=4, description="0..4 probability scale stored in state")
    note: Optional[str] = Field(None, description="Optional note from state")

# -----------------------------------------------------------------------------
# Tool-first (graph) lightweight models (agent-oriented)
# -----------------------------------------------------------------------------

class ToolAssetLite(BaseModel):
    asset_id: str
    name: str
    type: Optional[str] = None


class ToolListAssetsResponse(BaseModel):
    items: list[ToolAssetLite]
    total: int


class ToolAssetUpdateResponse(BaseModel):
    asset_id: str
    changed: dict = Field(default_factory=dict)


# Unified tool call request/response wrapper for /graph/assets
class GraphToolCall(BaseModel):
    op: Literal["asset.list", "asset.create", "asset.update_meta", "asset.explain"]
    args: dict = Field(default_factory=dict)

# -----------------------------------------------------------------------------
# DI helpers
# -----------------------------------------------------------------------------

def get_backend(request: Request) -> BackendPort:
    """
    Prefer a prebuilt backend living on app.state (shared across requests).
    Fall back to CsvBackend() if none was injected into the app.
    """
    app_state = getattr(request.app, "state", None)
    prebuilt = getattr(app_state, "backend_port", None) if app_state else None
    if prebuilt is not None:
        return cast(BackendPort, prebuilt)
    return CsvBackend()  # safe fallback for local/dev usage


def get_deps(request: Request) -> GraphDeps:
    """Obtain the shared GraphDeps instance (required for graph runners)."""
    app_state = getattr(request.app, "state", None)
    deps: GraphDeps | None = getattr(app_state, "deps", None) or getattr(request.state, "deps", None)
    if deps is None:
        raise HTTPException(status_code=500, detail={"message": "GraphDeps (app.state.deps) is not configured"})
    return deps

# -----------------------------------------------------------------------------
# Utilities (robust normalization to canonical Asset DTO)
# -----------------------------------------------------------------------------

_CONF_INDEX_TO_LABEL = {1: "PUBLIC", 2: "INTERNAL", 3: "PROTECTED", 4: "STRICTLY_PROTECTED"}
_IA_INDEX_TO_LABEL = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}

TA_ASSET = TypeAdapter(Asset)

def _as_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on", "ano"}:
            return True
        if s in {"0", "false", "no", "n", "off", "ne"}:
            return False
    return None

def _first_nonempty(*vals: Any) -> Any | None:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None

def _as_uuid_str_or_none(v: Any) -> str | None:
    """
    Try to coerce to UUID string; return None if not a valid UUID.
    The public Asset DTO tolerates Optional id for non-UUID states.
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return str(uuid.UUID(s))
    except Exception:
        return None

def _num_conf_to_label(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return _CONF_INDEX_TO_LABEL.get(int(v))
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"none", "null"}:
            return None
        return s.upper().replace("-", "_")
    return None

def _num_ia_to_label(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return _IA_INDEX_TO_LABEL.get(int(v))
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"none", "null"}:
            return None
        return s.upper()
    return None

def _to_asset(item: Any) -> Asset:
    """
    Convert backend item (dict or Pydantic model) into canonical Asset DTO.
    - Never inject non-UUID strings into `id`; invalid UUIDs become None.
    - Do not include isPrimary if unknown → let client's default apply.
    - Raise a 500 if name/assetType are missing (data contract violation).
    """
    data = item.model_dump() if hasattr(item, "model_dump") else dict(item)

    raw_id = _first_nonempty(data.get("id"), data.get("asset_id"))
    name = str(_first_nonempty(data.get("name"), data.get("title"), "")).strip()
    a_type = _first_nonempty(data.get("assetType"), data.get("type"), data.get("asset_type"))

    if not name:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Backend returned an asset without name", "raw": {"id": raw_id}},
        )

    if not a_type:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Backend returned an asset without assetType", "raw": {"id": raw_id, "name": name}},
        )

    payload: dict[str, Any] = {
        "id": _as_uuid_str_or_none(raw_id),
        "name": name,
        "description": _first_nonempty(data.get("description"), data.get("desc")),
        "assetType": str(a_type),
        "org_id": _first_nonempty(data.get("org_id"), data.get("orgId")),
    }

    is_primary = _as_bool(_first_nonempty(data.get("isPrimary"), data.get("primary"), data.get("is_primary")))
    if is_primary is not None:
        payload["isPrimary"] = is_primary

    conf = _num_conf_to_label(_first_nonempty(data.get("confidentiality"), data.get("c")))
    integ = _num_ia_to_label(_first_nonempty(data.get("integrity"), data.get("i")))
    avail = _num_ia_to_label(_first_nonempty(data.get("availability"), data.get("a")))

    if conf is not None:
        payload["confidentiality"] = conf
    if integ is not None:
        payload["integrity"] = integ
    if avail is not None:
        payload["availability"] = avail

    try:
        return TA_ASSET.validate_python(payload)
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Failed to coerce backend asset to client Asset schema",
                "errors": ve.errors(),
                "payload": payload,
            },
        ) from ve

async def _run_async(func, *args, **kwargs):
    """Run sync backends on a worker thread; await coroutines directly."""
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)

def _shape_tool_asset(item: Any) -> ToolAssetLite:
    """Compress backend item into an agent-friendly lite shape."""
    data = item.model_dump() if hasattr(item, "model_dump") else dict(item)
    asset_id = str(_first_nonempty(data.get("id"), data.get("asset_id")))
    name = str(_first_nonempty(data.get("name"), data.get("title"), ""))
    a_type = _first_nonempty(data.get("assetType"), data.get("type"), data.get("asset_type"))
    return ToolAssetLite(asset_id=asset_id, name=name, type=(str(a_type) if a_type else None))

# -----------------------------------------------------------------------------
# Public read-only routes (/api/assets)
# -----------------------------------------------------------------------------

@router.get(
    "",
    response_model=ListAssetsResponse,
    response_class=JSONResponse,
    summary="List assets (filtered)",
)
async def list_assets(
    request: Request,
    org_id: str = Query(..., description="Organization identifier"),
    search: str | None = Query(None, description="Free-text search in name/description"),
    asset_type: str | None = Query(None, alias="type", description="Asset type (EN code from catalog)"),
    is_primary: bool | None = Query(None, description="Filter primary assets"),
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> ListAssetsResponse:
    try:
        telemetry_log(
            "rest.assets.list",
            {"org_id": org_id, "search": search, "type": asset_type, "is_primary": is_primary, "request_path": str(request.url)},
        )
    except Exception:
        pass

    try:
        items = await _run_async(backend.assets_list, org_id, search, asset_type, is_primary)
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Invalid filters", "errors": ve.errors()},
        ) from ve
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to list assets", "error": str(ex)},
        ) from ex

    out_items = [_to_asset(a) for a in items]
    return ListAssetsResponse(items=out_items, count=len(out_items))

@router.get(
    "/{asset_id}",
    response_model=Asset,
    response_class=JSONResponse,
    summary="Get asset by ID",
)
async def get_asset(
    request: Request,
    asset_id: str,
    org_id: str = Query(..., description="Organization identifier"),
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> Asset:
    try:
        telemetry_log("rest.assets.get", {"org_id": org_id, "asset_id": asset_id, "request_path": str(request.url)})
    except Exception:
        pass

    try:
        item = await _run_async(backend.asset_get, asset_id)
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to retrieve asset", "asset_id": asset_id, "error": str(ex)},
        ) from ex

    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"message": "Asset not found", "asset_id": asset_id})

    return _to_asset(item)

@router.get(
    "/{asset_id}/impacts",
    response_model=list[AssetImpactItem],
    response_class=JSONResponse,
    summary="List impacts for a given asset (state)",
)
async def get_asset_impacts(
    request: Request,
    asset_id: str,
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> list[AssetImpactItem]:
    """
    Reads impacts directly from state/asset_impacts.csv via CsvBackend.
    No catalog validation here; state is the source of truth.
    """
    try:
        telemetry_log("rest.assets.impacts", {"asset_id": asset_id, "request_path": str(request.url)})
    except Exception:
        pass

    if not asset_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"message": "asset_id is required"})

    try:
        rows = await _run_async(backend.asset_get_impacts, asset_id)
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to fetch impacts", "asset_id": asset_id, "error": str(ex)},
        ) from ex

    return [AssetImpactItem(**r) for r in rows]

@router.get(
    "/{asset_id}/risks",
    response_model=list[AssetRiskItem],
    response_class=JSONResponse,
    summary="List risks for a given asset (state)",
)
async def get_asset_risks(
    request: Request,
    asset_id: str,
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> list[AssetRiskItem]:
    """
    Reads risks directly from state/asset_risks.csv via CsvBackend.
    No catalog validation here; state is the source of truth.
    """
    try:
        telemetry_log("rest.assets.risks", {"asset_id": asset_id, "request_path": str(request.url)})
    except Exception:
        pass

    if not asset_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"message": "asset_id is required"})

    try:
        rows = await _run_async(backend.asset_get_risks, asset_id)
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to fetch risks", "asset_id": asset_id, "error": str(ex)},
        ) from ex

    return [AssetRiskItem(**r) for r in rows]

# -----------------------------------------------------------------------------
# Unified tool-first route (/graph/assets) — SSOT endpoint with strict validation
# -----------------------------------------------------------------------------

@graph_router.post(
    "",
    response_class=JSONResponse,
    summary="[tool] Unified assets tools (SSOT): list/create/update_meta/explain",
)
async def graph_assets_entrypoint(
    request: Request,
    call: GraphToolCall,
    org_id: str = Query(..., description="Organization identifier"),
    backend: BackendPort = Depends(get_backend),
    deps: GraphDeps = Depends(get_deps),
    _auth=Depends(require_auth),
):
    """
    Single SSOT-aligned endpoint for all asset tools.
    Strictly validates arguments via ToolRegistry to ensure schema parity.
      - asset.list        → handled here (backend list)
      - asset.update_meta → delegated to graph runner (slot-driven + ChangeSet)
      - asset.create      → delegated to graph runner (slot-driven + ChangeSet)
      - asset.explain     → delegated to graph runner (RAG/LLM)
    """
    op = call.op
    args = call.args or {}

    try:
        telemetry_log("graph.assets.dispatch", {"org_id": org_id, "op": op})
    except Exception:
        pass

    # Validate against SSOT (raises ValidationError on mismatch)
    try:
        validated_model = ToolRegistry.validate_call(op, args)
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": f"Invalid payload for {op}", "errors": ve.errors()},
        ) from ve

    # Common request metadata for graph runners
    headers = dict(request.headers)
    user_id = headers.get("x-user-id") or "demo-user"

    # Dispatch per op
    if op == "asset.list":
        v: AssetListArgs = cast(AssetListArgs, validated_model)
        return await _handle_asset_list(org_id=org_id, q=(v.query or None), backend=backend)

    if op == "asset.update_meta":
        # Delegate to slot-driven graph runner (the graph will ask questions if needed)
        v: AssetUpdateMetaArgs = cast(AssetUpdateMetaArgs, validated_model)
        ui_resp = await run_update_meta(
            deps=deps,
            ui_req=UIRequest(
                org_id=org_id,
                user_id=user_id,
                active_tab="assets",
                body={"op": op, "args": v.model_dump(mode="json")},
                agent_events={"subscribed": False, "mode": "raw_json"},
                emit_changes_enabled=True,
            ),
        )
        return JSONResponse(ui_resp.model_dump(mode="json"), status_code=200)

    if op == "asset.create":
        # Delegate to slot-driven graph runner (the graph will iterate questions and then apply)
        v: AssetCreateArgs = cast(AssetCreateArgs, validated_model)  # type: ignore[assignment]
        ui_resp = await run_create(
            deps=deps,
            ui_req=UIRequest(
                org_id=org_id,
                user_id=user_id,
                active_tab="assets",
                body={"op": op, "args": v.model_dump(mode="json") if hasattr(v, "model_dump") else args},
                agent_events={"subscribed": False, "mode": "raw_json"},
                emit_changes_enabled=True,
            ),
        )
        return JSONResponse(ui_resp.model_dump(mode="json"), status_code=200)

    if op == "asset.explain":
        # Our asset graph exposes run_explain(deps, text, locale) returning a dict UI payload.
        # AssetExplainArgs = AssetSelector (assetId | assetName). Build a natural question.
        v: AssetExplainArgs = cast(AssetExplainArgs, validated_model)
        asset_id = getattr(v, "assetId", None)
        asset_name = getattr(v, "assetName", None)
        # Prefer name when available; otherwise use the id
        question = f"Explain asset '{asset_name}'." if asset_name else f"Explain asset with id '{asset_id}'."
        # Locale: take Accept-Language best-effort; otherwise EN
        accept_lang = headers.get("accept-language") or "en"
        payload = await run_explain(deps=deps, text=question, locale=accept_lang)
        return JSONResponse(payload, status_code=200)

    # Should never happen (Literal + registry validation)
    raise HTTPException(status_code=400, detail={"message": f"Unknown op '{op}'"})

# -----------------------------------------------------------------------------
# Handlers for tools (internal helpers)
# -----------------------------------------------------------------------------

async def _handle_asset_list(*, org_id: str, q: Optional[str], backend: BackendPort) -> ToolListAssetsResponse:
    try:
        items = await _run_async(backend.assets_list, org_id, q, None, None)
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Failed to list assets", "error": str(ex)}) from ex
    return ToolListAssetsResponse(items=[_shape_tool_asset(a) for a in (items or [])], total=len(items or []))

# -----------------------------------------------------------------------------
# Export both routers; app/main.py should mount them separately:
#   app.include_router(assets.router)
#   app.include_router(assets.graph_router)
# -----------------------------------------------------------------------------

__all__ = ["router", "graph_router"]
