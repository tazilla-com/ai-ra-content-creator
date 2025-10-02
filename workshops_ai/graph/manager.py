# File: workshops_ai/graph/manager.py
# -----------------------------------------------------------------------------
# Unified LangGraph router (assets | impacts | threats).
# - route_chat(UIRequest) -> UIResponse
# - Graph factory + checkpointer (stateful)
# - Per-tab timeouts from settings (namespaced or flat; robust fallback)
# - Emit-only policy normalization (emit_changes_enabled=True)
# - Signature-aware DI to avoid kwarg mismatches when factory evolves
# - Strict I/O validation (Pydantic v2), fail-closed with friendly messages
# - Telemetry passthrough for tool events (tool.call.*) if agent emits agent_events
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Final, Mapping, Optional, cast

from pydantic import ValidationError

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.models.agent_event import AgentEvent as WireAgentEvent
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

    # 2) Build DI deps (backend, ws hub, retriever, settings, submit_changeset, llm if configured)
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
        "existing_assets": _maybe(req, "existing_assets"),  # important for Assets tab
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

    # 7a) Telemetry passthrough for tool events (if present in raw_result)
    _telemetry_from_raw_tool_events(raw_result, tab=tab, org_id=req.org_id, user_id=req.user_id)

    # 8) Normalize & validate UIResponse (incl. agent_events → canonical wrap)
    try:
        resp = _ensure_uiresponse(
            raw_result,
            tab=tab,
            org_id=req.org_id,
            user_id=req.user_id,
            asset_id=(getattr(req, "selection", None) and getattr(req.selection, "asset_id", None)),
        )

        # 8a) Telemetry passthrough for tool events (if agent_events carried through UIResponse)
        _telemetry_from_response_tool_events(resp, tab=tab, org_id=req.org_id, user_id=req.user_id)

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
        # Note: LLM and submit_changeset can be injected here
        # if you wire them in settings/provider later.
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
# UIResponse normalization (wraps telemetry-like dicts → canonical AgentEvent)
# -----------------------------------------------------------------------------
def _ensure_uiresponse(
    raw: Any,
    *,
    tab: str,
    org_id: str,
    user_id: Optional[str],
    asset_id: Optional[str],
) -> UIResponse:
    """
    Accepts:
      - UIResponse instance
      - dict compatible with UIResponse
      - state dict containing a nested {"result": dict}
    Produces a validated UIResponse, wrapping agent_events (if they are 'light'
    telemetry dicts) into canonical AgentEvent envelopes.
    """
    payload = raw

    # Unwrap common pattern: {"result": {...}}
    if isinstance(payload, Mapping) and "result" in payload and isinstance(payload["result"], Mapping):
        payload = payload["result"]

    if isinstance(payload, UIResponse):
        # Best-effort ensure canonical agent_events inside an already-typed UIResponse
        if getattr(payload, "agent_events", None):
            payload.agent_events = _normalize_agent_events(  # type: ignore[attr-defined]
                cast(list, payload.agent_events),
                tab=tab,
                org_id=org_id,
                user_id=user_id,
                asset_id=asset_id,
            )
        return payload

    if isinstance(payload, Mapping):
        base = {
            "messages": payload.get("messages") or [],
            "proposals": payload.get("proposals"),
            "staging": payload.get("staging") or {"needs_commit": False, "batch_id": None, "diff": None},
            "policy": payload.get("policy") or {"emit_changes_enabled": True},
        }
        # Normalize agent_events if present
        if "agent_events" in payload and isinstance(payload["agent_events"], list):
            base["agent_events"] = _normalize_agent_events(
                cast(list, payload["agent_events"]),
                tab=tab,
                org_id=org_id,
                user_id=user_id,
                asset_id=asset_id,
            )
        return UIResponse.model_validate(base)

    # Last resort: wrap into a minimal UIResponse
    return UIResponse(
        messages=[str(payload)],
        proposals=None,
        staging={"needs_commit": False, "batch_id": None, "diff": None},
        policy={"emit_changes_enabled": True},
    )


def _normalize_agent_events(
    events: list[Any],
    *,
    tab: str,
    org_id: str,
    user_id: Optional[str],
    asset_id: Optional[str],
) -> list[dict[str, Any]]:
    """
    Convert a possibly heterogeneous list of events into canonical AgentEvent dicts:
      - If an item already looks like a WireAgentEvent (has 'event'), keep it.
      - If it's a "light" telemetry dict (event_type/tool/ok/...),
        wrap it into WireAgentEvent(event='agent.tool_event', data=item).
    """
    source = {"assets": "AssetAgent", "impacts": "ImpactAgent", "threats": "ThreatAgent"}.get(tab, "Manager")
    out: list[dict[str, Any]] = []
    for e in events:
        if isinstance(e, Mapping) and "event" in e:
            # Assume it's already canonical or close enough; let Pydantic validate later if needed.
            out.append(cast(dict[str, Any], dict(e)))
            continue
        if isinstance(e, Mapping):
            ae = WireAgentEvent(
                event="agent.tool_event",
                org_id=org_id,
                user_id=user_id,
                asset_id=asset_id,
                data=dict(e),
                source=source,
            )
            out.append(ae.model_dump(mode="json"))
    return out


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
# Telemetry passthrough for tool.call.* events
# -----------------------------------------------------------------------------
def _telemetry_from_raw_tool_events(raw: Any, *, tab: str, org_id: str, user_id: Optional[str]) -> None:
    """
    Forward tool.call.* events from a raw graph result/envelope to telemetry.
    Accepts both:
      1) "light" telemetry dicts ({"event_type": "...", "tool": "...", ...})
      2) canonical AgentEvent envelopes ({"event": "agent.tool_event", "data": {...}})
    Never raises; telemetry must not break the main flow.
    """
    try:
        events: Any = None
        if isinstance(raw, Mapping):
            events = raw.get("agent_events")
            if events is None:
                result = raw.get("result")
                if isinstance(result, Mapping):
                    events = result.get("agent_events")

        if not isinstance(events, list):
            return

        for ev in events:
            try:
                if not isinstance(ev, Mapping):
                    continue

                # Case A: canonical AgentEvent envelope → unwrap "data"
                if ev.get("event") == "agent.tool_event" and isinstance(ev.get("data"), Mapping):
                    _maybe_log_tool_event(ev["data"], tab=tab, org_id=org_id, user_id=user_id)
                    continue

                # Case B: "light" telemetry dict → use directly
                _maybe_log_tool_event(ev, tab=tab, org_id=org_id, user_id=user_id)

            except Exception:
                continue
    except Exception:
        return


def _telemetry_from_response_tool_events(resp: UIResponse, *, tab: str, org_id: str, user_id: Optional[str]) -> None:
    """
    If the normalized UIResponse carries agent_events, forward tool.call.* events.
    """
    try:
        events = getattr(resp, "agent_events", None)
        if not isinstance(events, list):
            return
        for ev in events:
            try:
                if isinstance(ev, Mapping) and ev.get("event") == "agent.tool_event" and isinstance(ev.get("data"), Mapping):
                    _maybe_log_tool_event(ev["data"], tab=tab, org_id=org_id, user_id=user_id)
                elif isinstance(ev, Mapping):
                    _maybe_log_tool_event(ev, tab=tab, org_id=org_id, user_id=user_id)
            except Exception:
                continue
    except Exception:
        pass


def _maybe_log_tool_event(ev: Any, *, tab: str, org_id: str, user_id: Optional[str]) -> None:
    """
    Accepts a dict-like event and logs tool.call.* telemetry if event_type matches.
    Expected minimal schema:
      {
        "event_type": "tool.call.start|success|fail",
        "tool": "list_impacts" | "set_risk_probability" | ...,
        "duration_ms": 12,     # optional
        "ok": true/false,      # optional
        "meta": {...}          # optional
      }
    """
    try:
        if not isinstance(ev, Mapping):
            return
        etype = str(ev.get("event_type") or "")
        if not etype.startswith("tool.call."):
            return
        payload = {
            "tab": tab,
            "org_id": org_id,
            "user_id": (user_id or "anon"),
            "event_type": etype,
            "tool": ev.get("tool"),
            "duration_ms": ev.get("duration_ms"),
            "ok": ev.get("ok"),
            "meta": ev.get("meta"),
        }
        telemetry_log("agent.tool_event", payload)
    except Exception:
        pass


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
