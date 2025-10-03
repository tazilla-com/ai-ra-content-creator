# File: workshops_ai/routers/impacts.py
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
from workshops_ai.graph.impact_graph import create_graph, ImpactState

# Client DTOs (unchanged; must match client contract)
from workshops_ai.api.schema.impact import ImpactEvaluation, AssetImpact
from workshops_ai.api.schema.impact_tool import ImpactDefCreate, ImpactDefUpdate

# Backend DI
from workshops_ai.backend.port import BackendPort
from workshops_ai.backend.csv_backend import CsvBackend

logger = logging.getLogger(__name__)

# We expose explicit tool-first endpoints under /graph/impacts/...
router = APIRouter(prefix="/graph", tags=["impacts"])

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
        tab="impacts",
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
    """Compile the impacts graph once per process and reuse it."""
    global _GRAPH_COMPILED
    if _GRAPH_COMPILED is not None:
        return _GRAPH_COMPILED
    try:
        _GRAPH_COMPILED = create_graph(deps)
        logger.info("impacts_router: impacts graph compiled")
    except Exception as e:  # pragma: no cover
        logger.exception("Failed to compile impacts graph: %s", e)
        raise
    return _GRAPH_COMPILED


# -----------------------------------------------------------------------------
# Tool-first models (responses kept minimal for LLM tools)
# -----------------------------------------------------------------------------
class ImpactLite(BaseModel):
    name: str
    description: Optional[str] = None
    weight: Optional[float] = None
    impactCategory: Optional[str] = None


class ImpactDefsListResponse(BaseModel):
    items: List[ImpactLite]
    total: int


class ImpactSetResultItem(BaseModel):
    asset_id: str
    impactName: str
    level: int  # final normalized 0..4
    previous: Optional[int] = None


class ImpactSetResult(BaseModel):
    results: List[ImpactSetResultItem]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _clamp_0_4(v: int) -> int:
    return max(0, min(4, v))


def _get_current_level_from_rows(rows: List[Dict[str, Any]], name: str) -> Optional[int]:
    """
    rows: backend.asset_get_impacts(asset_id) -> [{impact_code|name, level, ...}, ...]
    name: target impact name (exact match)
    """
    try:
        for r in rows or []:
            rc = str(
                r.get("impact_code") or r.get("code") or r.get("name") or ""
            ).strip()
            if rc == name:
                lv = r.get("level")
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
#   - GET  /graph/impacts/defs/list
#   - POST /graph/impacts/defs/create
#   - POST /graph/impacts/set      (bulk according to AssetImpact)
# -----------------------------------------------------------------------------
@router.get(
    "/impacts/defs/list",
    response_model=ImpactDefsListResponse,
    response_class=JSONResponse,
    summary="[tool] List impact definitions (catalog)",
)
async def list_impact_defs(
    request: Request,
    org_id: str = Query(..., description="Organization ID"),
    q: Optional[str] = Query(None, description="Free-text search"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> ImpactDefsListResponse:
    try:
        telemetry_log("graph.impacts.defs.list", {"org_id": org_id, "q": q, "limit": limit, "offset": offset})
    except Exception:
        pass

    # Expect a backend method like: impact_defs_list(org_id, q)
    try:
        defs = await backend.impact_defs_list(org_id, q)  # type: ignore[attr-defined]
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


@router.post(
    "/impacts/defs/create",
    response_model=ImpactLite,
    response_class=JSONResponse,
    summary="[tool] Create an impact definition (idempotent by name)",
)
async def create_impact_def(
    request: Request,
    payload: ImpactDefCreate,                      # <- changed
    org_id: str = Query(..., description="Organization ID"),
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> ImpactLite:
    try:
        telemetry_log("graph.impacts.defs.create.start", {"org_id": org_id, "name": payload.name})
    except Exception:
        pass

    # Idempotent lookup by name
    try:
        defs = await backend.impact_defs_list(org_id, payload.name)  # type: ignore[attr-defined]
    except AttributeError:
        raise HTTPException(status_code=501, detail={"message": "Backend does not support impact_defs_list()"})
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Catalog lookup failed", "error": str(ex)}) from ex

    for d in defs or []:
        existing_name = getattr(d, "name", None) or (d.get("name") if isinstance(d, dict) else None)
        if str(existing_name or "").strip().lower() == payload.name.strip().lower():
            obj = d.model_dump() if hasattr(d, "model_dump") else dict(d)
            return ImpactLite(
                name=str(obj.get("name") or payload.name),
                description=obj.get("description"),
                weight=(obj.get("weight") if isinstance(obj.get("weight"), (int, float)) else payload.weight),
                impactCategory=obj.get("impactCategory") or payload.impactCategory,
            )

    # Create if not exists
    try:
        try:
            created = await backend.impact_defs_create(org_id=org_id, impact=payload)  # type: ignore[attr-defined]
        except TypeError:
            created = await backend.impact_defs_create(  # type: ignore[attr-defined]
                org_id=org_id,
                name=payload.name,
                description=payload.description,
                weight=payload.weight,
                impactCategory=payload.impactCategory,
            )
    except AttributeError:
        raise HTTPException(status_code=501, detail={"message": "Backend does not support impact_defs_create()"})
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Failed to create impact definition", "error": str(ex)}) from ex

    obj = created.model_dump() if hasattr(created, "model_dump") else dict(created)
    try:
        telemetry_log("graph.impacts.defs.create.success", {"org_id": org_id, "name": payload.name})
    except Exception:
        pass
    return ImpactLite(
        name=str(obj.get("name") or payload.name),
        description=obj.get("description"),
        weight=(obj.get("weight") if isinstance(obj.get("weight"), (int, float)) else payload.weight),
        impactCategory=obj.get("impactCategory") or payload.impactCategory,
    )
    
@router.put(
    "/impacts/defs/update",
    response_model=ImpactLite,
    response_class=JSONResponse,
    summary="[tool] Update an impact definition (by name)",
)
async def update_impact_def(
    request: Request,
    payload: ImpactDefUpdate,
    org_id: str = Query(..., description="Organization ID"),
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> ImpactLite:
    try:
        telemetry_log("graph.impacts.defs.update.start", {"org_id": org_id, "name": payload.name})
    except Exception:
        pass

    # Optional: verify it exists first (gives 404 instead of creating)
    try:
        defs = await backend.impact_defs_list(org_id, payload.name)  # type: ignore[attr-defined]
    except AttributeError:
        raise HTTPException(status_code=501, detail={"message": "Backend does not support impact_defs_list()"})
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Catalog lookup failed", "error": str(ex)}) from ex

    if not any((getattr(d, "name", None) or d.get("name")) == payload.name for d in (defs or [])):
        raise HTTPException(status_code=404, detail={"message": "Impact definition not found"})

    # Update
    try:
        try:
            updated = await backend.impact_defs_update(org_id=org_id, impact=payload)  # type: ignore[attr-defined]
        except TypeError:
            updated = await backend.impact_defs_update(  # type: ignore[attr-defined]
                org_id=org_id,
                name=payload.name,
                description=payload.description,
                weight=payload.weight,
                impactCategory=payload.impactCategory,
            )
    except AttributeError:
        raise HTTPException(status_code=501, detail={"message": "Backend does not support impact_defs_update()"})
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"message": "Failed to update impact definition", "error": str(ex)}) from ex

    obj = updated.model_dump() if hasattr(updated, "model_dump") else dict(updated)
    try:
        telemetry_log("graph.impacts.defs.update.success", {"org_id": org_id, "name": payload.name})
    except Exception:
        pass
    return ImpactLite(
        name=str(obj.get("name") or payload.name),
        description=obj.get("description"),
        weight=(obj.get("weight") if isinstance(obj.get("weight"), (int, float)) else payload.weight),
        impactCategory=obj.get("impactCategory") or payload.impactCategory,
    )

@router.post(
    "/impacts/set",
    response_model=ImpactSetResult,
    response_class=JSONResponse,
    summary="[tool] Set impact levels (bulk for a single asset via AssetImpact)",
)
async def set_impact_levels(
    request: Request,
    payload: AssetImpact,                   # client's bulk model
    org_id: str = Query(..., description="Organization ID"),
    user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    backend: BackendPort = Depends(get_backend),
    _auth=Depends(require_auth),
) -> ImpactSetResult:
    """
    Tool-first bulk setter:
      - Accepts AssetImpact (assetId | assetName + list of ImpactEvaluation items).
      - Resolves asset_id (by id or by name).
      - Reads current levels to provide 'previous'.
      - Writes via backend and returns a compact per-item result list.
    """
    # Resolve asset_id
    asset_id: Optional[str]
    if payload.assetId:
        asset_id = str(payload.assetId)
    else:
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
        rows = await backend.asset_get_impacts(asset_id)  # type: ignore[attr-defined]
    except Exception:
        rows = []

    results: List[ImpactSetResultItem] = []
    for ev in payload.impacts:
        name = ev.impactName.strip()
        prev = _get_current_level_from_rows(rows, name)
        level = _clamp_0_4(int(ev.impactLevel))  # Pydantic already constrains 0..4

        # Write through backend: expect asset_impact_set(asset_id, code=name, level)
        try:
            _ = await backend.asset_impact_set(asset_id=asset_id, code=name, level=level)  # type: ignore[attr-defined]
        except AttributeError:
            raise HTTPException(
                status_code=501,
                detail={"message": "Backend does not support asset_impact_set()"},
            )
        except Exception as ex:
            raise HTTPException(
                status_code=500,
                detail={"message": f"Failed to set impact '{name}' level", "error": str(ex)},
            ) from ex

        results.append(
            ImpactSetResultItem(asset_id=asset_id, impactName=name, level=level, previous=prev)
        )

    try:
        telemetry_log(
            "graph.impacts.set.success",
            {"org_id": org_id, "asset_id": asset_id, "count": len(results), "names": [it.impactName for it in results]},
        )
    except Exception:
        pass

    return ImpactSetResult(results=results)


# -----------------------------------------------------------------------------
# Generic graph endpoint (kept; accepts full ImpactState as dict)
# -----------------------------------------------------------------------------
@router.post("/impacts", response_class=JSONResponse, summary="Execute IMPACTS graph (generic)")
async def run_impacts_graph(
    request: Request,
    payload: dict = Body(...),
    _auth=Depends(require_auth),
):
    deps = _get_graph_deps(request)
    graph = _get_compiled_graph(deps)

    try:
        telemetry_log(
            "graph.impacts.invoke",
            {
                "org_id": payload.get("org_id"),
                "action": payload.get("action"),
                "has_selection": bool((payload.get("selection") or {}).get("asset_id")),
            },
        )
    except Exception:
        pass

    try:
        state_in = ImpactState.model_validate(payload)
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Invalid impacts graph input", "errors": ve.errors()},
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
        logger.exception("impacts graph execution failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Impacts graph execution failed", "error": str(e)},
        ) from e

    try:
        state_out = ImpactState.model_validate(out if isinstance(out, dict) else out)
    except Exception:
        return JSONResponse(content={"raw": out}, status_code=status.HTTP_200_OK)

    result = state_out.result or {
        "messages": ["No result."],
        "proposals": None,
        "staging": {"needs_commit": False, "batch_id": None, "diff": None},
        "policy": {"emit_changes_enabled": True},
    }
    return JSONResponse(content=result, status_code=status.HTTP_200_OK)


@router.get("/impacts/ping", summary="Ping impacts graph router")
async def ping_impacts_router(_auth=Depends(require_auth)):
    return {"status": "ok", "router": "impacts"}
