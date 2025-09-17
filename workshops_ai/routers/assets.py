# File: workshops_ai/routers/assets.py
# REST (read-only) for Assets:
#   - GET /api/assets?org_id=&search=&type=&is_primary=
#   - GET /api/assets/{asset_id}?org_id=
#
# Principles
# - Contract-first: strict Pydantic v2 response models (public DTO returns CIA as 0..4 scale).
# - No writes here (MVP is read-only).
# - Non-blocking: sync CSV backend wrapped via asyncio.to_thread.
# - Stable keys (EN codes) – do not localize catalog codes in responses.
# - Consistent error model (delegated to global handlers in main app).

from __future__ import annotations

import asyncio
import inspect
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from workshops_ai.backend.port import BackendPort
from workshops_ai.backend.csv_backend import CsvBackend  # MVP backend
from workshops_ai.tools.telemetry import telemetry_log

router = APIRouter(prefix="/api/assets", tags=["assets"])


# ------------------------------------------------------------------------------
# Response models (strict, public surface)
# ------------------------------------------------------------------------------

class AssetOut(BaseModel):
    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str | None = None
    isPrimary: bool | None = None
    confidentiality: int | None = Field(default=None, ge=0, le=4)
    integrity: int | None = Field(default=None, ge=0, le=4)
    availability: int | None = Field(default=None, ge=0, le=4)
    assetType: str = Field(..., min_length=1)  # EN canonical code from catalog
    org_id: str | None = None


class ListAssetsResponse(BaseModel):
    items: List[AssetOut]
    count: int


# ------------------------------------------------------------------------------
# DI helpers
# ------------------------------------------------------------------------------

def get_backend() -> BackendPort:
    # For MVP we return a new CsvBackend instance; swap to a singleton/factory if needed.
    return CsvBackend()  # type: ignore[return-value]


# ------------------------------------------------------------------------------
# Utilities (robust normalization)
# ------------------------------------------------------------------------------

def _as_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    # unknown/ambiguous → None
    return None


# canonical → numeric scale helpers
# confidentiality: PUBLIC(1), INTERNAL(2), PROTECTED(3), STRICTLY_PROTECTED(4)
_CONF_MAP = {
    "public": 1,
    "internal": 2,
    "protected": 3,
    "strictly_protected": 4,
    "strictly-protected": 4,
    "strictlyprotected": 4,
}

# integrity/availability: LOW(1), MEDIUM(2), HIGH(3)
_IA_MAP = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


def _canon_to_scale_conf(v: Any) -> int | None:
    """
    Normalize confidentiality to 0..4:
      - None -> None
      - int  -> clamp to 0..4
      - str  -> map canonical to 1..4 (None if unknown)
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        iv = int(v)
        if iv < 0:
            iv = 0
        if iv > 4:
            iv = 4
        return iv
    if isinstance(v, str):
        key = v.strip().lower()
        if key == "" or key in {"none", "null"}:
            return None
        mapped = _CONF_MAP.get(key)
        return mapped if mapped is not None else None
    return None


def _canon_to_scale_ia(v: Any) -> int | None:
    """
    Normalize integrity/availability to 0..4 (UI contract still allows 4; we clamp HIGH to 3):
      - None -> None
      - int  -> clamp to 0..4
      - str  -> map canonical to 1..3 (HIGH=3)
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        iv = int(v)
        if iv < 0:
            iv = 0
        if iv > 4:
            iv = 4
        return iv
    if isinstance(v, str):
        key = v.strip().lower()
        if key == "" or key in {"none", "null"}:
            return None
        mapped = _IA_MAP.get(key)
        return mapped if mapped is not None else None
    return None


def _first_nonempty(*vals: Any) -> Any | None:
    for v in vals:
        if v is None:
            continue
        # treat empty strings as empty
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None


def _to_asset_out(item: Any) -> AssetOut:
    """
    Map backend Asset (Pydantic model or dict-like) to public DTO.
    - Accepts both canonical CIA strings and numeric scales.
    - Accepts 'assetType' or 'type' or 'asset_type' as the source of the type code.
    """
    data = item.model_dump() if hasattr(item, "model_dump") else dict(item)

    # ID, name, description, org
    _id = str(_first_nonempty(data.get("id"), data.get("asset_id"), ""))
    _name = str(_first_nonempty(data.get("name"), data.get("title"), ""))
    _desc_raw = _first_nonempty(data.get("description"), data.get("desc"))
    _desc = str(_desc_raw) if _desc_raw is not None else None

    # isPrimary (robust)
    _is_primary = _as_bool(
        _first_nonempty(data.get("isPrimary"), data.get("primary"), data.get("is_primary"))
    )

    # CIA normalization (accept canonical or numeric)
    _conf = _canon_to_scale_conf(_first_nonempty(data.get("confidentiality"), data.get("c")))
    _inte = _canon_to_scale_ia(_first_nonempty(data.get("integrity"), data.get("i")))
    _avail = _canon_to_scale_ia(_first_nonempty(data.get("availability"), data.get("a")))

    # Asset type code (EN, canonical)
    _atype = str(
        _first_nonempty(
            data.get("assetType"),
            data.get("type"),
            data.get("asset_type"),
            "",
        )
    )

    _org = _first_nonempty(data.get("org_id"), data.get("orgId"))

    return AssetOut(
        id=_id,
        name=_name,
        description=_desc,
        isPrimary=_is_primary,
        confidentiality=_conf,
        integrity=_inte,
        availability=_avail,
        assetType=_atype,
        org_id=(str(_org) if _org is not None else None),
    )


async def _run_async(func, *args, **kwargs):
    """
    Await coroutine function directly or run sync work in a thread.
    Fixes a prior bug where the function was invoked twice.
    """
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------

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
) -> ListAssetsResponse:
    """
    Returns a filtered list of assets for the given organization.
    - Filtering by `search`, `type`, `is_primary` is optional.
    - Response is capped by backend (or the caller can paginate later via API evolution).
    """
    # Telemetry (non-blocking)
    try:
        telemetry_log(
            "rest.assets.list",
            {
                "org_id": org_id,
                "search": search,
                "type": type,
                "is_primary": is_primary,
                "request_path": str(request.url),
            },
        )
    except Exception:
        pass

    try:
        items = await _run_async(backend.assets_list, org_id, search, type, is_primary)
    except ValidationError as ve:
        # Backend validated and rejected something (rare for list); bubble up as 422.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Invalid filters", "errors": ve.errors()},
        ) from ve
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to list assets", "error": str(ex)},
        ) from ex

    # Map to public DTO
    out_items = [_to_asset_out(a) for a in items]
    return ListAssetsResponse(items=out_items, count=len(out_items))


@router.get(
    "/{asset_id}",
    response_model=AssetOut,
    response_class=JSONResponse,
    summary="Get asset by ID",
)
async def get_asset(
    request: Request,
    asset_id: str,
    org_id: str = Query(..., description="Organization identifier"),
    backend: BackendPort = Depends(get_backend),
) -> AssetOut:
    """
    Returns a single asset by its ID.
    - org_id is currently unused by CSV backend for direct get, but kept for compatibility
      with prospective API backends (multi-tenant).
    """
    # Telemetry (non-blocking)
    try:
        telemetry_log(
            "rest.assets.get",
            {"org_id": org_id, "asset_id": asset_id, "request_path": str(request.url)},
        )
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Asset not found", "asset_id": asset_id},
        )

    return _to_asset_out(item)
