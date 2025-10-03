# File: workshops_ai/routers/threats.py
from __future__ import annotations

import logging
from typing import Any, Optional, List, Dict

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Body,
    status,
    Query,
    Header,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from workshops_ai.app.security import require_auth
from workshops_ai.tools.telemetry import telemetry_log

# Graph bits (kept for the generic endpoint)
from workshops_ai.graph.factory import GraphDeps, make_default_deps
from workshops_ai.graph.threat_graph import create_graph, ThreatState

# Client DTOs (unchanged; must match client contract)
from workshops_ai.api.schema.threat import Threat, Risk, AssetRisk
from workshops_ai.api.schema.threat_tool import ThreatDefCreate, ThreatDefUpdate

# Backend DI
from workshops_ai.backend.port import BackendPort
from workshops_ai.backend.csv_backend import CsvBackend

logger = logging.getLogger(__name__)

# We expose explicit tool-first endpoints under /graph/threats/...
router = APIRouter(prefix="/graph", tags=["threats"])

_GRAPH_COMPILED = None  # type: ignore[var-annotated]
_GRAPH_DEPS: Optional[GraphDeps] = None


# -----------------------------------------------------------------------------
# DI helpers
# -----------------------------------------------------------------------------
def get_backend(request: Request) -> BackendPort:
    """Prefer app.state.backend_port; fallback to CsvBackend for local/dev."""
    app_state = getattr(request.app, "state", None)
    prebuilt = getattr(app_state, "backend_port", None) if app_state else None
    if prebuilt is not None:
        return prebuilt  # type: ignore[return-value]
    return CsvBackend()  # type: ignore[return-value]


def _get_graph_deps(request: Request) -> GraphDeps:
    """
    Compose GraphDeps once per process. Prefer a prebuilt deps bundle if the app
    stashed it on app.state.graph_deps. Otherwise, create defaults while
    passing through the app's injected components (CSV backend, WS hub, retriever, etc.).
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
        backend_port=getattr(app_state, "backend_port", None),
        ws_hub=getattr(app_state, "ws_hub", None),
        make_retriever=getattr(app_state, "make_retriever", None),
        llm=getattr(app_state, "llm", None),
        memory_get=getattr(app_state, "memory_get", None),
        memory_set=getattr(app_state, "memory_set", None),
    )
    return _GRAPH_DEPS


def _get_compiled_graph(deps: GraphDeps):
    """Compile the threats graph once per process and reuse it."""
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


# -----------------------------------------------------------------------------
# Tool-first models (responses kept minimal for LLM tools)
# -----------------------------------------------------------------------------
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


class ThreatSetResultItem(BaseModel):
    asset_id: str
    threatName: str
    level: int  # final normalized 0..4
    previous: Optional[int] = None


class ThreatSetResult(BaseModel):
    results: List[ThreatSetResultItem]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _clamp_0_4(v: int) -> int:
    return max(0, min(4, v))


def _get_current_level_from_rows(rows: List[Dict[str, Any]], name: str) -> Optional[int]:
    """
    rows: backend.threats_for_asset(asset_id) -> [{threat_code|name, probability|level, ...}, ...]
    name: target threat name (exact match)
    """
    try:
        for r in rows or []:
            rc = str(
                r.get("threat_code") or r.get("code") or r.get("name") or ""
            ).strip()
            if rc == name:
                lv = r.get("probability") or r.get("level")
                if isinstance(lv, int):
                    return _clamp_0_4(lv)
                if isinstance(lv, str) and lv.strip().isdigit():
                    return _clamp_0_4(int(lv.strip()))
                return None
    except Exception:
        return None
    return None


# -----------------------------------------------------------------------------
# Tool-first endpoints
#   - GET  /graph/threats/defs/list
#   - POST /graph/threats/defs/create
#   - POST /graph/threats/set      (bulk according to AssetRisk)
# -----------------------------------------------------------------------------
@router.get(
    "/threats/defs/list",
    response_model=ThreatDefsListResponse,
    response_class=JSONResponse,
    summary="[tool] List threat definitions (catalog)",
)
async def list_threat_defs(
    request: Request,
    org_id: str = Query(..., description="Organization ID"),
    q: Optional[str] = Query(None, description="Free-text search"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> ThreatDefsListResponse:
    try:
        telemetry_log("graph.threats.defs.list", {"org_id": org_id, "q": q, "limit": limit, "offset": offset})
    except Exception:
        pass

    # Expect a backend method like: threat_defs_list(org_id, q)
    try:
        defs = await backend.threat_defs_list(org_id, q)  # type: ignore[attr-defined]
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


@router.post(
    "/threats/defs/create",
    response_model=ThreatLite,
    response_class=JSONResponse,
    summary="[tool] Create a threat definition (idempotent by name)",
)
async def create_threat_def(
    request: Request,
    payload: ThreatDefCreate,   # <- was Threat
    org_id: str = Query(..., description="Organization ID"),
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> ThreatLite:
    try:
        telemetry_log("graph.threats.defs.create.start", {"org_id": org_id, "name": payload.name})
    except Exception:
        pass

    # Idempotent lookup by name
    try:
        defs = await backend.threat_defs_list(org_id, payload.name)  # type: ignore[attr-defined]
    except AttributeError:
        raise HTTPException(status_code=501, detail={"message": "Backend does not support threat_defs_list()"})
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Catalog lookup failed", "error": str(ex)}) from ex

    for d in defs or []:
        existing_name = getattr(d, "name", None) or (d.get("name") if isinstance(d, dict) else None)
        if str(existing_name or "").strip().lower() == payload.name.strip().lower():
            obj = d.model_dump() if hasattr(d, "model_dump") else dict(d)
            return ThreatLite(
                name=str(obj.get("name") or payload.name),
                description=obj.get("description"),
                category=obj.get("category"),
                unintentional=obj.get("unintentional"),
                intentional=obj.get("intentional"),
                environmental=obj.get("environmental"),
            )

    # Create if not exists
    try:
        # Prefer a single-arg create API to pass the pydantic model through.
        try:
            created = await backend.threat_defs_create(org_id=org_id, threat=payload)  # type: ignore[attr-defined]
        except TypeError:
            # Fallback to expanded call if backend expects primitives.
            created = await backend.threat_defs_create(  # type: ignore[attr-defined]
                org_id=org_id,
                name=payload.name,
                description=payload.description,
                category=payload.category,
                unintentional=payload.unintentional,
                intentional=payload.intentional,
                environmental=payload.environmental,
            )
    except AttributeError:
        raise HTTPException(status_code=501, detail={"message": "Backend does not support threat_defs_create()"})
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Failed to create threat definition", "error": str(ex)}) from ex

    obj = created.model_dump() if hasattr(created, "model_dump") else dict(created)
    try:
        telemetry_log("graph.threats.defs.create.success", {"org_id": org_id, "name": payload.name})
    except Exception:
        pass
    return ThreatLite(
        name=str(obj.get("name") or payload.name),
        description=obj.get("description"),
        category=obj.get("category"),
        unintentional=obj.get("unintentional"),
        intentional=obj.get("intentional"),
        environmental=obj.get("environmental"),
    )

@router.put(
    "/threats/defs/update",
    response_model=ThreatLite,
    response_class=JSONResponse,
    summary="[tool] Update a threat definition (by name)",
)
async def update_threat_def(
    request: Request,
    payload: ThreatDefUpdate,
    org_id: str = Query(..., description="Organization ID"),
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> ThreatLite:
    # Ensure it exists (404 if not)
    try:
        defs = await backend.threat_defs_list(org_id, payload.name)  # type: ignore[attr-defined]
    except AttributeError:
        raise HTTPException(status_code=501, detail={"message": "Backend does not support threat_defs_list()"})
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Catalog lookup failed", "error": str(ex)}) from ex

    if not any((getattr(d, "name", None) or d.get("name")) == payload.name for d in (defs or [])):
        raise HTTPException(status_code=404, detail={"message": "Threat definition not found"})

    # Update
    try:
        try:
            updated = await backend.threat_defs_update(org_id=org_id, threat=payload)  # type: ignore[attr-defined]
        except TypeError:
            updated = await backend.threat_defs_update(  # type: ignore[attr-defined]
                org_id=org_id,
                name=payload.name,
                description=payload.description,
                category=payload.category,
                unintentional=payload.unintentional,
                intentional=payload.intentional,
                environmental=payload.environmental,
            )
    except AttributeError:
        raise HTTPException(status_code=501, detail={"message": "Backend does not support threat_defs_update()"})
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Failed to update threat definition", "error": str(ex)}) from ex

    obj = updated.model_dump() if hasattr(updated, "model_dump") else dict(updated)
    return ThreatLite(
        name=str(obj.get("name") or payload.name),
        description=obj.get("description"),
        category=obj.get("category") or payload.category,
        unintentional=obj.get("unintentional") if payload.unintentional is None else payload.unintentional,
        intentional=obj.get("intentional") if payload.intentional is None else payload.intentional,
        environmental=obj.get("environmental") if payload.environmental is None else payload.environmental,
    )
    
@router.post(
    "/threats/set",
    response_model=ThreatSetResult,
    response_class=JSONResponse,
    summary="[tool] Set threat probabilities (bulk for a single asset via AssetRisk)",
)
async def set_threat_level(
    request: Request,
    payload: AssetRisk,                       # client's bulk model
    org_id: str = Query(..., description="Organization ID"),
    user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> ThreatSetResult:
    """
    Tool-first bulk setter:
      - Accepts AssetRisk (assetId | assetName + list of Risk items).
      - Resolves asset_id (by id or by name).
      - Reads current levels to provide 'previous'.
      - Writes via backend and returns a compact per-item result list.
    """
    # Resolve asset_id
    asset_id: Optional[str]
    if payload.assetId:
        asset_id = str(payload.assetId)
    else:
        # Provide a backend helper to resolve ID by name inside org
        try:
            resolver = getattr(backend, "asset_id_by_name", None)
            if not resolver:
                raise AttributeError("asset_id_by_name not available on backend")
            asset_id = await resolver(org_id=org_id, name=payload.assetName)  # type: ignore[misc]
        except AttributeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "assetId missing and backend cannot resolve ID by assetName"},
            )
        except Exception as ex:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"message": "Failed to resolve asset by name", "error": str(ex)},
            ) from ex

    if not asset_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"message": "Asset not found"})

    # Read current rows (best-effort) to compute 'previous'
    try:
        rows = await backend.threats_for_asset(asset_id)  # type: ignore[attr-defined]
    except Exception:
        rows = []

    results: List[ThreatSetResultItem] = []
    for r in payload.risks:
        name = r.threatName.strip()
        prev = _get_current_level_from_rows(rows, name)
        level = _clamp_0_4(int(r.probability))  # Pydantic already constrains 0..4

        # Write through backend: prefer asset_threat_set, fallback to asset_risk_set
        try:
            setter = getattr(backend, "asset_threat_set", None) or getattr(backend, "asset_risk_set", None)
            if not setter:
                raise AttributeError("No asset_threat_set/asset_risk_set on backend")
            await setter(asset_id=asset_id, code=name, level=level)  # type: ignore[misc]
        except AttributeError:
            raise HTTPException(
                status_code=501,
                detail={"message": "Backend does not support asset_threat_set()/asset_risk_set()"},
            )
        except Exception as ex:
            raise HTTPException(
                status_code=500,
                detail={"message": f"Failed to set threat '{name}' level", "error": str(ex)},
            ) from ex

        results.append(
            ThreatSetResultItem(asset_id=asset_id, threatName=name, level=level, previous=prev)
        )

    try:
        telemetry_log(
            "graph.threats.set.success",
            {
                "org_id": org_id,
                "asset_id": asset_id,
                "count": len(results),
                "names": [it.threatName for it in results],
            },
        )
    except Exception:
        pass

    return ThreatSetResult(results=results)


# -----------------------------------------------------------------------------
# Generic graph endpoint (kept; accepts full ThreatState as dict)
# -----------------------------------------------------------------------------
@router.post("/threats", response_class=JSONResponse, summary="Execute THREATS graph (generic)")
async def run_threats_graph(
    request: Request,
    payload: dict = Body(...),
    _auth=Depends(require_auth),
):
    deps = _get_graph_deps(request)
    graph = _get_compiled_graph(deps)

    # Telemetry is best-effort; shouldn't block the request if it fails.
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

    # Validate request body against the graph's state model
    try:
        state_in = ThreatState.model_validate(payload)
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Invalid threats graph input", "errors": ve.errors()},
        ) from ve

    # Threading & checkpointing config so the graph can keep short-term state per user/tab
    cfg: dict[str, Any] = {"configurable": {}}
    user = state_in.user_id or "anon"
    cfg["configurable"]["thread_id"] = f"{state_in.org_id}:{deps.tab}:{user}"
    checkpoint_ns = getattr(deps, "checkpoint_ns", None)
    if checkpoint_ns:
        cfg["configurable"]["checkpoint_ns"] = checkpoint_ns

    # Execute the graph
    try:
        out = await graph.ainvoke(state_in.model_dump(), config=cfg)
    except Exception as e:
        logger.exception("threats graph execution failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Threats graph execution failed", "error": str(e)},
        ) from e

    # Normalize to ThreatState (if graph returns a plain dict)
    try:
        state_out = ThreatState.model_validate(out if isinstance(out, dict) else out)
    except Exception:
        return JSONResponse(content={"raw": out}, status_code=status.HTTP_200_OK)

    # Unified result envelope expected by the UI
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
