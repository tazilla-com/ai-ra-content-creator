# File: workshops_ai/routers/assets.py
# -----------------------------------------------------------------------------
# REST (read-only) for Assets:
#   - GET /api/assets?org_id=&search=&type=&is_primary=
#   - GET /api/assets/{asset_id}?org_id=
#
# Principles
# - Contract-first: returns the canonical API DTO `workshops_ai.api.schema.asset.Asset`.
# - No writes here (MVP is read-only).
# - Non-blocking: sync CSV backend wrapped via asyncio.to_thread.
# - Stable keys (EN codes) – do not localize catalog codes in responses.
# - Consistent error model (delegated to global handlers in main app).
# - Auth: API-key/JWT via centralized settings (same behavior as /chat).
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import inspect
import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from workshops_ai.app.security import require_auth  # unified auth (API-key/JWT)
from workshops_ai.backend.port import BackendPort
from workshops_ai.backend.csv_backend import CsvBackend  # MVP backend
from workshops_ai.tools.telemetry import telemetry_log

# Canonical public DTO
from workshops_ai.api.schema.asset import Asset

router = APIRouter(prefix="/api/assets", tags=["assets"])


# -----------------------------------------------------------------------------
# Response models (public surface returns canonical Asset DTO)
# -----------------------------------------------------------------------------

class ListAssetsResponse(BaseModel):
    items: List[Asset]
    count: int


# -----------------------------------------------------------------------------
# DI helpers
# -----------------------------------------------------------------------------

def get_backend() -> BackendPort:
    # For MVP we return a new CsvBackend instance; swap to a singleton/factory if needed.
    return CsvBackend()  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# Utilities (robust normalization)
# -----------------------------------------------------------------------------

# numeric CIA → ISO labels (for legacy backends that still emit numbers)
# confidentiality: PUBLIC(1), INTERNAL(2), PROTECTED(3), STRICTLY_PROTECTED(4)
_CONF_INDEX_TO_LABEL = {
    1: "PUBLIC",
    2: "INTERNAL",
    3: "PROTECTED",
    4: "STRICTLY_PROTECTED",
}
# integrity/availability: LOW(1), MEDIUM(2), HIGH(3)
_IA_INDEX_TO_LABEL = {
    1: "LOW",
    2: "MEDIUM",
    3: "HIGH",
}

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
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    # unknown/ambiguous → None
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
    Try to coerce input to a canonical UUID string.
    Return None if value is empty/invalid. We never pass non-UUID strings
    into the client-provided Asset DTO (keeps their contract intact).
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
    """
    Normalize confidentiality to ISO label:
      - 1..4 → PUBLIC/INTERNAL/PROTECTED/STRICTLY_PROTECTED
      - 0/None/"" → None (field omitted)
      - string label → upper() and pass-through
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        iv = int(v)
        return _CONF_INDEX_TO_LABEL.get(iv)
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"none", "null"}:
            return None
        return s.upper().replace("-", "_")
    return None


def _num_ia_to_label(v: Any) -> str | None:
    """
    Normalize integrity/availability to ISO label:
      - 1..3 → LOW/MEDIUM/HIGH
      - 0/None/"" → None (field omitted)
      - string label → upper() pass-through (LOW/MEDIUM/HIGH)
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        iv = int(v)
        return _IA_INDEX_TO_LABEL.get(iv)
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"none", "null"}:
            return None
        return s.upper()
    return None


def _to_asset(item: Any) -> Asset:
    """
    Convert a backend item (dict or Pydantic model) into canonical `Asset`.
    Accepts multiple legacy key variants and numeric CIA scales and normalizes them.

    IMPORTANT:
    - We NEVER pass a non-UUID string into the client-provided Asset DTO `id`.
      If the incoming ID is not a valid UUID → we emit `None` for `id`.
    """
    data = item.model_dump() if hasattr(item, "model_dump") else dict(item)

    # Base fields
    raw_id = _first_nonempty(data.get("id"), data.get("asset_id"))
    payload = {
        "id": _as_uuid_str_or_none(raw_id),  # enforce UUID-or-None for client DTO
        "name": str(_first_nonempty(data.get("name"), data.get("title"), "")),
        "description": _first_nonempty(data.get("description"), data.get("desc")),
        "isPrimary": _as_bool(_first_nonempty(data.get("isPrimary"), data.get("primary"), data.get("is_primary"))),
        "assetType": str(_first_nonempty(data.get("assetType"), data.get("type"), data.get("asset_type"), "")),
        "org_id": _first_nonempty(data.get("org_id"), data.get("orgId")),
    }

    # CIA (normalize numeric → ISO labels if needed)
    conf_src = _first_nonempty(data.get("confidentiality"), data.get("c"))
    integ_src = _first_nonempty(data.get("integrity"), data.get("i"))
    avail_src = _first_nonempty(data.get("availability"), data.get("a"))

    conf = _num_conf_to_label(conf_src)
    integ = _num_ia_to_label(integ_src)
    avail = _num_ia_to_label(avail_src)

    if conf is not None:
        payload["confidentiality"] = conf
    if integ is not None:
        payload["integrity"] = integ
    if avail is not None:
        payload["availability"] = avail

    # Validate & coerce into canonical Asset DTO
    return TA_ASSET.validate_python(payload)


async def _run_async(func, *args, **kwargs):
    """
    Await coroutine function directly or run sync work in a thread.
    """
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
    _auth=Depends(require_auth),  # enforce unified auth
) -> ListAssetsResponse:
    """
    Returns a filtered list of assets for the given organization.
    - Filtering by `search`, `type`, `is_primary` is optional.
    - Response uses canonical `Asset` DTO (ISO labels for CIA).
    """
    # Telemetry (best-effort, non-blocking)
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
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Invalid filters", "errors": ve.errors()},
        ) from ve
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to list assets", "error": str(ex)},
        ) from ex

    # Normalize to canonical Asset DTO
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
    _auth=Depends(require_auth),  # enforce unified auth
) -> Asset:
    """
    Returns a single asset by its ID (canonical `Asset` DTO).
    - org_id is currently unused by CSV backend for direct get, but kept for compatibility
      with prospective API backends (multi-tenant).
    """
    # Telemetry (best-effort, non-blocking)
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

    return _to_asset(item)
