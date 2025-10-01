# File: workshops_ai/graph/manager.py
# -----------------------------------------------------------------------------
# Unified LangGraph router (assets | impacts | threats).
# - route_chat(UIRequest) -> UIResponse
# - Graph factory + checkpointer (stateful)
# - Per-tab timeouts from settings (namespaced or flat; robust fallback)
# - Emit-only policy normalization (emit_changes_enabled=True)
# - Signature-aware DI to avoid kwarg mismatches when factory evolves
# - Strict I/O validation (Pydantic v2), fail-closed with friendly messages
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Final, Optional, cast

from pydantic import ValidationError

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.tools.telemetry import telemetry_log
from workshops_ai.settings import get_settings

# Graph factory & helpers
from workshops_ai.graph.factory import (  # type: ignore
    GraphDeps,
    make_default_deps,
    build_graph,
    ainvoke_with_timeout,
)

_SETTINGS = get_settings()

# -----------------------------------------------------------------------------
# Supported tabs
# -----------------------------------------------------------------------------
_SUPPORTED_TABS: Final[set[str]] = {"assets", "impacts", "threats"}


# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------
async def route_chat(req: UIRequest | dict[str, Any]) -> UIResponse:
    """Main router: validates input, builds deps/graph, invokes with timeout, normalizes output."""
    _audit_in(req)

    # 1) Defensive validation (coerce dict → UIRequest)
    if not isinstance(req, UIRequest):
        try:
            req = UIRequest.model_validate(cast(Any, req))
        except ValidationError as ve:
            return _fail("Input does not match UIRequest contract.", None, details=ve.errors())

    tab = (req.active_tab or "").strip().lower()
    if tab not in _SUPPORTED_TABS:
        return UIResponse(
            messages=[f'Tab "{req.active_tab}" is not supported. Use assets, impacts, or threats.'],
            proposals=None,
            staging={"needs_commit": False, "batch_id": None, "diff": None},
            policy={"emit_changes_enabled": True},
        )

    # 2) Build DI deps (backend, ws hub, retriever, settings)
    deps = _deps(tab)

    # 3) Best-effort: remember context for cross-tab flows
    await _remember_context(tab, req, deps)

    # 4) Build graph
    try:
        graph = build_graph(tab, deps)
    except Exception as e:
        telemetry_log("manager.build_graph_failed", {"tab": tab, "error": str(e)})
        return _fail("Failed to initialize agent graph.", req, details=str(e))

    # 5) Prepare inputs for the graph (dict-based)
    def _maybe(obj: Any, name: str) -> Any:
        return getattr(obj, name, None)

    inputs: dict[str, Any] = {
        "org_id": req.org_id,
        "active_tab": tab,
        "message": req.message,
        "selection": (
            req.selection.model_dump()
            if hasattr(req.selection, "model_dump")
            else dict(getattr(req, "selection", {}) or {})
        ),
        "user_id": req.user_id,
        "staging": bool(req.staging),
        "language": _maybe(req, "language"),
        "locale": _maybe(req, "locale"),
        "session_id": _maybe(req, "session_id"),
        # ----- Pass-through optionals so graphs can use inline data/seeds -----
        "number_of_assets": _maybe(req, "number_of_assets"),
        "user_instructions": _maybe(req, "user_instructions"),
        "organization_profile": _maybe(req, "organization_profile"),
        "services": _maybe(req, "services"),
        "existing_assets": _maybe(req, "existing_assets"),  # <— important for Assets tab
        "url": _maybe(req, "url"),
        "website_content": _maybe(req, "website_content"),
        "policy": _maybe(req, "policy"),
        "history_token": _maybe(req, "history_token"),
    }

    # 6) Per-tab timeout & LangGraph config
    timeout = _timeout_for_tab(tab)
    cfg = _make_langgraph_config(org_id=req.org_id, tab=tab, user_id=req.user_id, deps=deps)

    # 7) Execute graph
    try:
        raw_result = await ainvoke_with_timeout(graph, inputs, timeout=timeout, config=cfg)
    except asyncio.TimeoutError:
        return _fail(f"{tab.capitalize()}Agent execution timed out.", req)
    except Exception as e:
        telemetry_log("manager.graph_invoke_error", {"tab": tab, "error": repr(e)})
        return _fail("Agent execution failed.", req, details=str(e))

    # 8) Normalize & validate UIResponse
    try:
        resp = _ensure_uiresponse(raw_result)
        # Ensure policy.emit_changes_enabled = True (emit-only mode)
        pol = dict(getattr(resp, "policy", {}) or {})
        pol["emit_changes_enabled"] = True
        resp.policy = pol  # type: ignore[attr-defined]

        # Ensure staging block exists (some graphs may omit)
        if not getattr(resp, "staging", None):
            resp.staging = {"needs_commit": False, "batch_id": None, "diff": None}  # type: ignore[attr-defined]

        await _remember_post_run(tab, req, resp, deps)
        return resp
    except ValidationError as ve:
        telemetry_log("manager.response_validation_error", {"tab": tab, "errors": ve.errors()})
        return _fail("Graph response failed UIResponse validation.", req, details=ve.errors())


# -----------------------------------------------------------------------------
# DI helpers (signature-aware to avoid kwarg mismatches if factory evolves)
# -----------------------------------------------------------------------------
def _deps(tab: str) -> GraphDeps:
    """
    Build GraphDeps via the factory. We introspect make_default_deps(...) and only
    pass the keywords it actually supports (prevents runtime kwarg errors).
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
        # Extremely defensive: pass all we have
        safe_kwargs = dict(candidate_kwargs)

    return make_default_deps(**safe_kwargs)  # type: ignore[call-arg]


def _resolve_backend_port() -> Any | None:
    # Try provider (prod), then CSV, then legacy port symbol
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


# -----------------------------------------------------------------------------
# UIResponse normalization
# -----------------------------------------------------------------------------
def _ensure_uiresponse(raw: Any) -> UIResponse:
    """
    Accepts:
      - UIResponse instance
      - dict compatible with UIResponse
      - state dict containing a nested {"result": dict}
    Produces a validated UIResponse, or raises ValidationError.
    """
    payload = raw
    # Unwrap common pattern: {"result": {...}}
    if isinstance(payload, dict) and "result" in payload and isinstance(payload["result"], dict):
        payload = payload["result"]

    if isinstance(payload, UIResponse):
        return payload

    # Some graphs may return only messages or partial dicts; coerce carefully
    if isinstance(payload, dict):
        # Ensure required top-level fields exist before validation
        payload = {
            "messages": payload.get("messages") or [],
            "proposals": payload.get("proposals"),
            "staging": payload.get("staging") or {"needs_commit": False, "batch_id": None, "diff": None},
            "policy": payload.get("policy") or {"emit_changes_enabled": True},
            # Optionally pass-through agent_events if present
            **({"agent_events": payload["agent_events"]} if "agent_events" in payload else {}),
        }
        return UIResponse.model_validate(payload)

    # Last resort: wrap into a minimal UIResponse
    return UIResponse(
        messages=[str(payload)],
        proposals=None,
        staging={"needs_commit": False, "batch_id": None, "diff": None},
        policy={"emit_changes_enabled": True},
    )


# -----------------------------------------------------------------------------
# LangGraph config (checkpointer requires configurable keys)
# -----------------------------------------------------------------------------
def _make_langgraph_config(*, org_id: str, tab: str, user_id: Optional[str], deps: GraphDeps) -> dict:
    """
    Ensure we always pass the required configurables for stateful graphs:
      - thread_id: unique per org/tab/user
      - checkpoint_ns: namespace for the checkpointer (from settings, fallback 'lg:ckpt')
    """
    uid = (user_id or "anon").strip() or "anon"
    ns = getattr(deps.settings, "checkpoint_ns", None) or "lg:ckpt"
    return {
        "configurable": {
            "thread_id": f"{org_id}:{tab}:{uid}",
            "checkpoint_ns": ns,
        }
    }


# -----------------------------------------------------------------------------
# Timeouts (robust against both namespaced and flat settings)
# -----------------------------------------------------------------------------
def _timeout_for_tab(tab: str) -> int:
    """
    Tries (in order):
      1) settings.manager.threats_timeout_sec / agent_timeout_sec
      2) flat: settings.agent_timeout_sec_threats / agent_timeout_sec_impacts / agent_timeout_sec_assets
      3) chat timeout as a fallback (settings.chat.timeout_sec)
      4) hard default 120s
    """
    def _get(*names: str, default: Optional[int] = None) -> Optional[int]:
        cur: Any = _SETTINGS
        for name in names[:-1]:
            cur = getattr(cur, name, None)
        return cast(Optional[int], getattr(cur, names[-1], default) if cur is not None else default)

    if tab == "threats":
        return (
            _get("manager", "threats_timeout_sec")
            or _get("manager", "agent_timeout_sec")
            or _get("agent_timeout_sec_threats")
            or _get("agent_timeout_sec")
            or _get("chat", "timeout_sec")
            or 120
        )
    if tab == "impacts":
        return (
            _get("manager", "agent_timeout_sec")
            or _get("agent_timeout_sec_impacts")
            or _get("agent_timeout_sec")
            or _get("chat", "timeout_sec")
            or 120
        )
    # assets/default
    return (
        _get("manager", "agent_timeout_sec")
        or _get("agent_timeout_sec_assets")
        or _get("agent_timeout_sec")
        or _get("chat", "timeout_sec")
        or 120
    )


# -----------------------------------------------------------------------------
# Lightweight audit + memory context (best-effort; graphs remain functional)
# -----------------------------------------------------------------------------
def _audit_in(req: UIRequest | dict[str, Any]) -> None:
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
    """Store a few org-scoped hints to help cross-tab flows (if Redis is configured)."""
    try:
        mset = getattr(deps, "memory_set", None)
        if not mset:
            return
        await mset(f"{req.org_id}:{tab}:last_user", req.user_id or "anon", ttl_sec=3600)
    except Exception:
        pass


async def _remember_post_run(tab: str, req: UIRequest, resp: UIResponse, deps: GraphDeps) -> None:
    try:
        telemetry_log(
            "manager.route_chat_out",
            {
                "tab": tab,
                "org_id": req.org_id,
                "user_id": req.user_id,
                "messages": (resp.messages or [])[:3],
            },
        )
    except Exception:
        pass
