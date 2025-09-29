# File: workshops_ai/graph/manager.py
# -----------------------------------------------------------------------------
# Unified LangGraph router (assets | impacts | threats).
# - route_chat(UIRequest) -> UIResponse
# - Graph factory + checkpointer (stateful)
# - Per-tab timeouts from settings.manager.*
# - Emit-only policy (optimistic WS + async submit in graphs)
# - Dependency factory call is signature-aware to avoid kwarg mismatches.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Final, Optional, cast

from pydantic import ValidationError

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.tools.telemetry import telemetry_log
from workshops_ai.settings import get_settings

from workshops_ai.graph.factory import (  # type: ignore
    GraphDeps,
    make_default_deps,
    build_graph,
    ainvoke_with_timeout,
)

_SETTINGS = get_settings()

# --- Timeouts -----------------------------------------------------------------
_AGENT_TIMEOUT_SEC: Final[int] = int(_SETTINGS.manager.agent_timeout_sec)
_THREATS_TIMEOUT_SEC: Final[int] = int(_SETTINGS.manager.threats_timeout_sec or _AGENT_TIMEOUT_SEC)

_SUPPORTED_TABS: Final[set[str]] = {"assets", "impacts", "threats"}

# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------
async def route_chat(req: UIRequest) -> UIResponse:
    _audit_in(req)

    # Defensive validation
    if not isinstance(req, UIRequest):
        try:
            req = UIRequest.model_validate(cast(Any, req))
        except ValidationError:
            return _fail("Input does not match UIRequest contract.", None)

    tab = (req.active_tab or "").strip().lower()
    if tab not in _SUPPORTED_TABS:
        return UIResponse(
            messages=[f'Tab "{req.active_tab}" is not supported. Use assets, impacts, threats.'],
            proposals=None,
            staging={"needs_commit": False},
            policy={"emit_changes_enabled": True},
        )

    # Build dependencies (backend, ws hub, retriever, checkpointer, settings).
    deps = _deps(tab)  # signature-aware DI builder  :contentReference[oaicite:9]{index=9}

    # Remember org-scoped context (best-effort via Redis ephemeral memory).
    await _remember_context(tab, req, deps)

    # --- Build graph -----------------------------------------------------------
    try:
        graph = build_graph(tab, deps)
    except Exception as e:
        telemetry_log("manager.build_graph_failed", {"tab": tab, "error": str(e)})
        return _fail("Failed to initialize agent graph.", req, details=str(e))

    # --- Inputs for the graph (do NOT pass the whole UIRequest) ---------------
    inputs = {
        "org_id": req.org_id,
        "active_tab": tab,
        "message": req.message,
        "selection": (
            req.selection.model_dump()
            if hasattr(req.selection, "model_dump")
            else dict(getattr(req, "selection", {}) or {})
        ),
        "user_id": req.user_id,
        "staging": req.staging,
        "language": getattr(req, "language", None),
    }

    # --- Per-tab timeout & LangGraph config -----------------------------------
    timeout = _timeout_for_tab(tab)
    cfg = {
        "configurable": {
            "thread_id": f"{req.org_id}:{tab}:{req.user_id}",
            "checkpoint_ns": getattr(deps.settings, "checkpoint_ns", "lg:ckpt"),
        }
    }  # ensure stateful threads + namespacing  :contentReference[oaicite:10]{index=10}

    # --- Execute ---------------------------------------------------------------
    try:
        result = await ainvoke_with_timeout(graph, inputs, timeout=timeout, config=cfg)
    except asyncio.TimeoutError:
        return _fail(f"{tab.capitalize()}Agent execution timed out.", req)
    except Exception as e:
        telemetry_log("manager.graph_invoke_error", {"tab": tab, "error": str(e)})
        return _fail("Agent execution failed.", req, details=str(e))

    # --- Outbound validation + emit-only policy --------------------------------
    try:
        resp = result if isinstance(result, UIResponse) else UIResponse.model_validate(result)
        pol = dict(getattr(resp, "policy", {}) or {})
        pol["emit_changes_enabled"] = True
        resp.policy = pol  # type: ignore[attr-defined]

        await _remember_post_run(tab, req, resp, deps)
        return resp
    except ValidationError as ve:
        telemetry_log("manager.response_validation_error", {"tab": tab, "errors": ve.errors()})
        return _fail("Graph response failed UIResponse validation.", req, details=ve.errors())

# -----------------------------------------------------------------------------
# DI helpers (signature-aware to avoid kwarg mismatch if factory evolves)
# -----------------------------------------------------------------------------
def _deps(tab: str) -> GraphDeps:
    """
    Build GraphDeps via the factory. We introspect make_default_deps(...) and only
    pass the keywords it actually supports (prevents runtime kwarg errors).  :contentReference[oaicite:11]{index=11}
    """
    backend = _resolve_backend_port()
    ws_hub = _resolve_ws_hub()
    make_retriever = _resolve_retriever_factory()

    candidate_kwargs: dict[str, Any] = {
        "tab": tab,
        "settings": _SETTINGS,
        "backend_port": backend,
        "ws_hub": ws_hub,
        "make_retriever": make_retriever,
    }
    try:
        sig = inspect.signature(make_default_deps)
        supported = set(sig.parameters.keys())
        safe_kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported}
    except Exception:
        safe_kwargs = {k: v for k, v in candidate_kwargs.items() if k != "tazila_client"}

    return make_default_deps(**safe_kwargs)  # type: ignore[call-arg]

def _resolve_backend_port() -> Any | None:
    try:
        from workshops_ai.backend.provider import get_backend_port  # type: ignore
        return get_backend_port()
    except Exception:
        pass
    try:
        from workshops_ai.backend.csv_backend import get_backend_port  # type: ignore
        return get_backend_port()
    except Exception:
        pass
    try:
        from workshops_ai.backend.port import backend_port  # type: ignore
        return backend_port
    except Exception:
        pass
    return None

def _resolve_ws_hub():
    try:
        from workshops_ai.app.services.ws_hub import WsHub  # type: ignore
        return WsHub
    except Exception:
        return None

def _resolve_retriever_factory():
    try:
        from workshops_ai.rag.retriever import make_retriever  # type: ignore
        return make_retriever
    except Exception:
        return None

def _timeout_for_tab(tab: str) -> int:
    if tab == "threats":
        return _THREATS_TIMEOUT_SEC
    return _AGENT_TIMEOUT_SEC

# -----------------------------------------------------------------------------
# Lightweight audit + memory context (best-effort; graphs remain functional without it)
# -----------------------------------------------------------------------------
def _audit_in(req: UIRequest) -> None:
    try:
        payload = req.model_dump() if hasattr(req, "model_dump") else cast(Any, req)
        telemetry_log("manager.route_chat_in", {"req": payload})
    except Exception:
        pass

def _fail(msg: str, req: Optional[UIRequest], details: Any = None) -> UIResponse:
    if req:
        telemetry_log("manager.fail", {"msg": msg, "details": details})
    return UIResponse(
        messages=[msg],
        proposals=None,
        staging={"needs_commit": False, "batch_id": None, "diff": None},
        policy={"emit_changes_enabled": True},
    )

async def _remember_context(tab: str, req: UIRequest, deps: GraphDeps) -> None:
    """Store a few org-scoped hints to help cross-tab flows."""
    try:
        mset = deps.memory_set
        if not mset:
            return
        await mset(f"{req.org_id}:{tab}:last_user", req.user_id, ttl_sec=3600)
    except Exception:
        pass

async def _remember_post_run(tab: str, req: UIRequest, resp: UIResponse, deps: GraphDeps) -> None:
    try:
        telemetry_log("manager.route_chat_out", {
            "tab": tab,
            "org_id": req.org_id,
            "user_id": req.user_id,
            "messages": resp.messages[:3] if resp.messages else [],
        })
    except Exception:
        pass
