# File: workshops_ai/routers/assets.py
# -----------------------------------------------------------------------------
# REST (read-only) for Assets state:
#   - GET /api/assets?org_id=&search=&type=&is_primary=
#   - GET /api/assets/{asset_id}?org_id=
#   - GET /api/assets/{asset_id}/impacts
#   - GET /api/assets/{asset_id}/risks
#
# Tool-first endpoints for agent use:
#   - POST /graph/assets/create         (AssetCreate → {asset_id, existed})
#   - PUT  /graph/assets/update         (AssetUpdate → {asset_id, changed})
#   - GET  /graph/assets/list           (q/limit/offset → {items,total})
#
# Principles
# - Returns the canonical API DTO `workshops_ai.api.schema.asset.Asset` for assets.
# - Per-asset impacts/risks return lightweight DTOs defined here (state is source of truth).
# - Read-only; all writes go via graph/tool endpoints (create/update) or other routers.
# - Non-blocking: sync CSV backend wrapped via asyncio.to_thread where needed.
# - Auth via centralized `require_auth`.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import inspect
import uuid
from typing import Any, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from workshops_ai.app.security import require_auth
from workshops_ai.backend.port import BackendPort
from workshops_ai.backend.csv_backend import CsvBackend
from workshops_ai.tools.telemetry import telemetry_log

# Canonical public DTOs for assets (client contracts)
from workshops_ai.api.schema.asset import Asset 
from workshops_ai.api.schema.asset_tool import AssetCreate, AssetUpdate 

# Routers:
# - `router` stays on /api/assets (read-only public API)
# - `graph_router` mounts /graph/assets (LLM tool-first minimal endpoints)
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
# Tool-first (graph) response models (minimalistic for LLM tools)
# -----------------------------------------------------------------------------

class ToolAssetLite(BaseModel):
    asset_id: str
    name: str
    type: Optional[str] = None


class ToolListAssetsResponse(BaseModel):
    items: list[ToolAssetLite]
    total: int


class ToolAssetCreateResponse(BaseModel):
    asset_id: str
    name: str
    existed: bool = False


class ToolAssetUpdateResponse(BaseModel):
    asset_id: str
    changed: dict = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# DI helper
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
    NOTE: The public Asset DTO must tolerate Optional id for non-UUID states.
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


# --- replace _to_asset and helpers with this slightly safer version ---

def _to_asset(item: Any) -> Asset:
    """
    Convert backend item (dict or Pydantic model) into canonical Asset DTO.
    - Never inject non-UUID strings into `id`; invalid UUIDs become None.
    - Do not include isPrimary if unknown → let client's default (True) apply.
    - Do not pass empty assetType → raise 500 with clear diagnostics (or fallback if agreed).
    """
    data = item.model_dump() if hasattr(item, "model_dump") else dict(item)

    raw_id = _first_nonempty(data.get("id"), data.get("asset_id"))
    name = str(_first_nonempty(data.get("name"), data.get("title"), "")).strip()
    a_type = _first_nonempty(data.get("assetType"), data.get("type"), data.get("asset_type"))

    # Hard guards for required fields in client's schema
    if not name:
        # You can also choose to silently skip the record; explicit error is clearer during integration.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Backend returned an asset without name", "raw": {"id": raw_id}},
        )

    if not a_type:
        # If you prefer a fallback, replace the raise with: a_type = "Other"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Backend returned an asset without assetType", "raw": {"id": raw_id, "name": name}},
        )

    payload: dict[str, Any] = {
        "id": _as_uuid_str_or_none(raw_id),  # Optional[UUID] – string OK, Pydantic přeparsuje
        "name": name,
        "description": _first_nonempty(data.get("description"), data.get("desc")),
        # isPrimary: set only if we can parse it; otherwise omit to keep client's default True
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
            detail={"message": "Failed to coerce backend asset to client Asset schema", "errors": ve.errors(), "payload": payload},
        ) from ve


async def _run_async(func, *args, **kwargs):
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


def _shape_tool_asset(item: Any) -> ToolAssetLite:
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
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"message": "Invalid filters", "errors": ve.errors()}) from ve
    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"message": "Failed to list assets", "error": str(ex)}) from ex

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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"message": "Failed to retrieve asset", "asset_id": asset_id, "error": str(ex)}) from ex

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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"message": "Failed to fetch impacts", "asset_id": asset_id, "error": str(ex)}) from ex

    # Backend already normalizes shape; just coerce to response model
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"message": "Failed to fetch risks", "asset_id": asset_id, "error": str(ex)}) from ex

    return [AssetRiskItem(**r) for r in rows]


# -----------------------------------------------------------------------------
# Tool-first routes (/graph/assets) — explicit, minimal, idempotent
# -----------------------------------------------------------------------------

@graph_router.get(
    "/list",
    response_model=ToolListAssetsResponse,
    response_class=JSONResponse,
    summary="[tool] List assets for selection",
)
async def graph_list_assets(
    request: Request,
    org_id: str = Query(..., description="Organization identifier"),
    q: Optional[str] = Query(None, description="Free-text search"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> ToolListAssetsResponse:
    try:
        telemetry_log("graph.assets.list", {"org_id": org_id, "q": q, "limit": limit, "offset": offset})
    except Exception:
        pass

    try:
        items = await _run_async(backend.assets_list, org_id, q, None, None)
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Failed to list assets", "error": str(ex)}) from ex

    sliced = (items or [])[offset : offset + limit]
    return ToolListAssetsResponse(items=[_shape_tool_asset(a) for a in sliced], total=len(items or []))


@graph_router.post(
    "/create",
    response_model=ToolAssetCreateResponse,
    response_class=JSONResponse,
    summary="[tool] Create asset (idempotent by name within org)",
)
async def graph_create_asset(
    request: Request,
    payload: AssetCreate,
    org_id: str = Query(..., description="Organization identifier"),
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> ToolAssetCreateResponse:
    """
    Idempotent create by (org_id, name):
    - if an asset with the same name exists in org -> return that asset_id, existed=True
    - else create and return existed=False
    """
    try:
        telemetry_log("graph.assets.create.start", {"org_id": org_id, "name": payload.name, "type": payload.type})
    except Exception:
        pass

    # 1) Try to find by name (case-insensitive) inside org
    try:
        existing = await _run_async(backend.assets_list, org_id, payload.name, None, None)
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Lookup failed before create", "error": str(ex)}) from ex

    found_item = None
    if existing:
        name_lc = payload.name.strip().lower()
        for it in existing:
            it_name = str(_first_nonempty(getattr(it, "name", None), getattr(it, "title", None), "")).strip().lower()
            if it_name == name_lc:
                found_item = it
                break

    if found_item is not None:
        shaped = _shape_tool_asset(found_item)
        try:
            telemetry_log("graph.assets.create.idempotent_hit", {"org_id": org_id, "asset_id": shaped.asset_id})
        except Exception:
            pass
        return ToolAssetCreateResponse(asset_id=shaped.asset_id, name=shaped.name, existed=True)

    # 2) Create when not found
    try:
        # Expect backend to support: asset_create(org_id, name, type)
        created = await _run_async(backend.asset_create, org_id, payload.name, payload.type)
    except AttributeError:
        raise HTTPException(status_code=501, detail={"message": "Backend does not support asset_create()"})
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Failed to create asset", "error": str(ex)}) from ex

    shaped = _shape_tool_asset(created)
    try:
        telemetry_log("graph.assets.create.success", {"org_id": org_id, "asset_id": shaped.asset_id})
    except Exception:
        pass

    # Manager will set current_asset from tool events; routers remain stateless.
    return ToolAssetCreateResponse(asset_id=shaped.asset_id, name=shaped.name, existed=False)


@graph_router.put(
    "/update",
    response_model=ToolAssetUpdateResponse,
    response_class=JSONResponse,
    summary="[tool] Update asset fields",
)
async def graph_update_asset(
    request: Request,
    payload: AssetUpdate,
    org_id: str = Query(..., description="Organization identifier"),
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> ToolAssetUpdateResponse:
    """
    Update only provided fields; returns a compact 'changed' map.
    """
    asset_id = str(payload.asset_id)
    try:
        telemetry_log("graph.assets.update.start", {"org_id": org_id, "asset_id": asset_id})
    except Exception:
        pass

    # Fetch before (for diff)
    before = None
    try:
        before = await _run_async(backend.asset_get, asset_id)
    except Exception:
        before = None

    # Apply update
    try:
        # Expect backend to support: asset_update(asset_id, *, name=?, type=?)
        updated = await _run_async(
            backend.asset_update,
            asset_id,
            name=(payload.name if payload.name is not None else None),
            type=(payload.type if payload.type is not None else None),
        )
    except AttributeError:
        raise HTTPException(status_code=501, detail={"message": "Backend does not support asset_update()"})
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Failed to update asset", "error": str(ex)}) from ex

    # Build minimal "changed"
    after = updated
    changed: dict = {}
    try:
        b = before.model_dump() if hasattr(before, "model_dump") else (dict(before) if before else {})
        a = after.model_dump() if hasattr(after, "model_dump") else dict(after)

        def _val(d: dict, *keys: str) -> Any:
            return _first_nonempty(*(d.get(k) for k in keys))

        if before is not None:
            if _val(b, "name", "title") != _val(a, "name", "title"):
                changed["name"] = {"from": _val(b, "name", "title"), "to": _val(a, "name", "title")}
            if _val(b, "type", "assetType", "asset_type") != _val(a, "type", "assetType", "asset_type"):
                changed["type"] = {"from": _val(b, "type", "assetType", "asset_type"), "to": _val(a, "type", "assetType", "asset_type")}
        else:
            # If we couldn't fetch before, just report keys that were provided
            for k in ("name", "type"):
                if getattr(payload, k) is not None:
                    changed[k] = {"to": getattr(payload, k)}
    except Exception:
        pass

    try:
        telemetry_log("graph.assets.update.success", {"org_id": org_id, "asset_id": asset_id, "changed_keys": list(changed.keys())})
    except Exception:
        pass

    return ToolAssetUpdateResponse(asset_id=asset_id, changed=changed)


# -----------------------------------------------------------------------------
# Export both routers; app/main.py should mount them separately:
#   app.include_router(assets.router)
#   app.include_router(assets.graph_router)
# -----------------------------------------------------------------------------

__all__ = ["router", "graph_router"]
