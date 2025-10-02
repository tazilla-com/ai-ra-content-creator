# File: workshops_ai/routers/assets.py
# -----------------------------------------------------------------------------
# REST (read-only) for Assets state:
#   - GET /api/assets?org_id=&search=&type=&is_primary=
#   - GET /api/assets/{asset_id}?org_id=
#   - GET /api/assets/{asset_id}/impacts
#   - GET /api/assets/{asset_id}/risks
#
# Principles
# - Returns the canonical API DTO `workshops_ai.api.schema.asset.Asset` for assets.
# - Per-asset impacts/risks return lightweight DTOs defined here (state is source of truth).
# - Read-only; all writes go via graph routers (impacts/threats).
# - Non-blocking: sync CSV backend wrapped via asyncio.to_thread where needed.
# - Auth via centralized `require_auth`.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import inspect
import uuid
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from workshops_ai.app.security import require_auth
from workshops_ai.backend.port import BackendPort
from workshops_ai.backend.csv_backend import CsvBackend
from workshops_ai.tools.telemetry import telemetry_log

# Canonical public DTO for assets
from workshops_ai.api.schema.asset import Asset

router = APIRouter(prefix="/api/assets", tags=["assets"])

# -----------------------------------------------------------------------------
# Response models
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
# DI helper
# -----------------------------------------------------------------------------

def get_backend() -> BackendPort:
    # For MVP we instantiate CsvBackend; its internal caches are mtime-aware.
    return CsvBackend()  # type: ignore[return-value]

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

def _to_asset(item: Any) -> Asset:
    """
    Convert backend item (dict or Pydantic model) into canonical Asset DTO.
    We never inject a non-UUID string into `id`; invalid UUIDs become None.
    """
    data = item.model_dump() if hasattr(item, "model_dump") else dict(item)

    raw_id = _first_nonempty(data.get("id"), data.get("asset_id"))
    payload = {
        "id": _as_uuid_str_or_none(raw_id),  # keep None for non-UUID ids
        "name": str(_first_nonempty(data.get("name"), data.get("title"), "")),
        "description": _first_nonempty(data.get("description"), data.get("desc")),
        "isPrimary": _as_bool(_first_nonempty(data.get("isPrimary"), data.get("primary"), data.get("is_primary"))),
        "assetType": str(_first_nonempty(data.get("assetType"), data.get("type"), data.get("asset_type"), "")),
        "org_id": _first_nonempty(data.get("org_id"), data.get("orgId")),
    }

    conf = _num_conf_to_label(_first_nonempty(data.get("confidentiality"), data.get("c")))
    integ = _num_ia_to_label(_first_nonempty(data.get("integrity"), data.get("i")))
    avail = _num_ia_to_label(_first_nonempty(data.get("availability"), data.get("a")))

    if conf is not None:
        payload["confidentiality"] = conf
    if integ is not None:
        payload["integrity"] = integ
    if avail is not None:
        payload["availability"] = avail

    return TA_ASSET.validate_python(payload)

async def _run_async(func, *args, **kwargs):
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)

# -----------------------------------------------------------------------------
# Routes
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
    type: str | None = Query(None, alias="type", description="Asset type (EN code from catalog)"),
    is_primary: bool | None = Query(None, description="Filter primary assets"),
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> ListAssetsResponse:
    try:
        telemetry_log(
            "rest.assets.list",
            {"org_id": org_id, "search": search, "type": type, "is_primary": is_primary, "request_path": str(request.url)},
        )
    except Exception:
        pass

    try:
        items = await _run_async(backend.assets_list, org_id, search, type, is_primary)
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
