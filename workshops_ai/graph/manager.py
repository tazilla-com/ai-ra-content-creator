# File: workshops_ai/graph/manager.py
# Purpose: Unified LangGraph router with conversation window, language autodetection,
# clarify-first policy, and light cross-domain orchestration (suggest-next-step).
# - Validates UIRequest → invokes tab graph with timeout → validates UIResponse.
# - Maintains a short rolling history (user/assistant) per org:tab:user:thread.
# - Adds thread-scoped session state with current_asset + MRU list and exposes it to graphs.
# - Performs one-time clarification handoff (agent asks once; manager avoids loops by storing a flag).
# - Normalizes policy to emit-only (emit_changes_enabled=True) unless staging flips it.
# - Passes compact policy + history into prompts (system.md) every turn.

from __future__ import annotations

import asyncio
import inspect
import re
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

# Conversation memory (Redis-like KV adapter)
from workshops_ai.graph.conversation_memory import (
    RollingWindowMemory,
    ChatTurn,
    MemoryConfig,
)

# Policy knobs (preview/apply, clarify-first, history, timeouts)
from workshops_ai.graph.policies import (
    TabName,
    GlobalPolicy,
    PolicyOverrides,
    EffectivePolicy,
    make_effective_policy,
    as_prompt_vars,
)

_SETTINGS = get_settings()

# Supported tabs
_SUPPORTED_TABS: Final[set[str]] = {"assets", "impacts", "threats"}

# Namespaces for memory
_CHAT_NS: Final[str] = "chat"
_SESSION_NS: Final[str] = "session"

# Keys for session state
_SESSION_STATE_KEY: Final[str] = "thread_state"
# Structure stored under _SESSION_STATE_KEY:
# {
#   "current_asset": {"asset_id": str, "name": str | None},
#   "recent_assets": [{"asset_id": str, "name": str | None}, ...],  # MRU (max N)
#   "once_flags": {"asked_for_asset": bool}
# }
_MRU_MAX: Final[int] = 5


# -----------------------------------------------------------------------------#
# Public entrypoint
# -----------------------------------------------------------------------------#
async def route_chat(req: UIRequest | dict[str, Any]) -> UIResponse:
    """Main router: validates input, builds deps/graph, injects history & policy, invokes with timeout, normalizes output."""
    _audit_in(req)

    # 1) Defensive validation (coerce dict → UIRequest)
    if not isinstance(req, UIRequest):
        try:
            req = UIRequest.model_validate(cast(Any, req))
        except ValidationError as ve:
            return _fail("Input does not match UIRequest contract.", None, details=ve.errors())

    tab = (req.active_tab or "").strip().lower()
    if tab not in _SUPPORTED_TABS:
        return _tab_not_supported(req)

    # 2) Build DI deps (backend, ws hub, retriever, memory, llm, settings, submit_changeset)
    deps = _deps(tab)

    # 3) Compute effective policy (respect staging flip)
    eff = make_effective_policy(
        tab=TabName(tab),  # values are "assets"|"impacts"|"threats"
        global_policy=GlobalPolicy(),  # default flags; can be overridden via overrides later if needed
        tab_policy=None,
        overrides=None,
        staging=bool(req.staging),
    )

    # 4) Compose thread scope + conversation memory (rolling window)
    uid = (req.user_id or "anon").strip() or "anon"
    thread_id = f"{req.org_id}:{tab}:{uid}"
    mem = RollingWindowMemory(
        memory_get=getattr(deps, "memory_get", None),
        memory_set=getattr(deps, "memory_set", None),
        memory_delete=getattr(deps, "memory_delete", None),
        config=MemoryConfig(max_messages=eff.max_history_messages, ttl_seconds=eff.history_ttl_seconds, namespace=_CHAT_NS),
    )
    turns = await mem.read(org_id=req.org_id, user_id=req.user_id, thread_id=thread_id, namespace=_CHAT_NS)
    history_msgs = RollingWindowMemory.to_llm_messages(turns, drop_tool_output=True)

    # 4a) Load thread session state (current_asset + MRU + once_flags)
    session_state = await _load_session_state(deps, org_id=req.org_id, user_id=req.user_id, thread_id=thread_id)
    # If UI selection was explicitly provided, treat it as an asset switch (highest priority)
    selection_asset_id = getattr(getattr(req, "selection", None), "asset_id", None)
    if selection_asset_id:
        session_state = _session_set_current_asset(session_state, asset_id=str(selection_asset_id), name=getattr(req.selection, "asset_name", None))

    # 5) Language autodetection (cs/sk/en)
    language = _autodetect_language(req.message, req.language, req.locale)

    # 6) Prepare inputs for the graph (dict-based; minimal + stable)
    selection_dump = (
        req.selection.model_dump()
        if hasattr(req.selection, "model_dump")
        else dict(getattr(req, "selection", {}) or {})
    )
    inputs: dict[str, Any] = {
        "org_id": req.org_id,
        "active_tab": tab,
        "message": req.message,
        "selection": selection_dump,
        "user_id": req.user_id,
        "staging": bool(req.staging),
        "language": language,
        "locale": (req.locale or language),
        "session_id": getattr(req, "session_id", None),
        # Conversational context & policy (consumed by base system prompt)
        "history": history_msgs[: eff.max_history_messages],
        "policy": as_prompt_vars(eff),
        # Thread-scoped session state provided to agents
        "current_asset": session_state.get("current_asset"),
        "recent_assets": session_state.get("recent_assets", []),
        "once_flags": session_state.get("once_flags", {}),
        # Pass-through optional context (no strict requirement)
        "history_token": getattr(req, "history_token", None),
        "organization_profile": getattr(req, "organization_profile", None),
        "services": getattr(req, "services", None),
        "existing_assets": getattr(req, "existing_assets", None),
        "user_instructions": getattr(req, "user_instructions", None),
    }

    # 7) Per-tab timeout & LangGraph config
    timeout = _timeout_for_tab(tab)
    cfg = _make_langgraph_config(org_id=req.org_id, tab=tab, user_id=req.user_id, deps=deps)

    # 8) Execute graph
    try:
        raw_result = await ainvoke_with_timeout(build_graph(tab, deps), inputs, timeout=timeout, config=cfg)
    except asyncio.TimeoutError:
        return _fail(f"{tab.capitalize()}Agent execution timed out.", req)
    except Exception as e:
        telemetry_log("manager.graph_invoke_error", {"tab": tab, "error": repr(e)})
        return _fail("Agent execution failed.", req, details=str(e))

    # 8a) Telemetry passthrough for tool events (if present in raw_result)
    _telemetry_from_raw_tool_events(raw_result, tab=tab, org_id=req.org_id, user_id=req.user_id)

    # 8b) Update session state from tool events (assets.create / explicit select)
    session_state = _maybe_update_current_asset_from_result(session_state, raw_result)
    # Ensure "asked_for_asset" once flag is not sticky forever: clear after any user message following the question.
    # We consider any non-empty user request as the "next turn" after the agent asked.
    if session_state.get("once_flags", {}).get("asked_for_asset"):
        # If the user provided an asset via selection or agent resolved it (we now have current_asset), clear the flag.
        if session_state.get("current_asset"):
            session_state["once_flags"]["asked_for_asset"] = False

    # 9) Normalize & validate UIResponse
    try:
        resp = _ensure_uiresponse(
            raw_result,
            tab=tab,
            org_id=req.org_id,
            user_id=req.user_id,
            asset_id=(getattr(req, "selection", None) and getattr(req.selection, "asset_id", None)),
        )

        # 9a) Forward tool events in normalized response
        _telemetry_from_response_tool_events(resp, tab=tab, org_id=req.org_id, user_id=req.user_id)

        # 9b) Enforce emit-only policy normalization (still honoring staging inside graphs)
        pol = dict(getattr(resp, "policy", {}) or {})
        pol["emit_changes_enabled"] = True
        resp.policy = pol  # type: ignore[attr-defined]

        # 9c) Ensure staging block exists
        if not getattr(resp, "staging", None):
            resp.staging = {"needs_commit": False, "batch_id": None, "diff": None}  # type: ignore[attr-defined]

        # 10) Light cross-domain orchestration (suggest-next-step only; no multi-write in one turn)
        _maybe_add_orchestration_suggestion(resp, tab=tab, language=language)

        # 11) Persist conversation window (user → assistant)
        await _persist_history(mem, req, resp, thread_id)

        # 12) Persist session state (current_asset + MRU + once_flags)
        await _save_session_state(deps, org_id=req.org_id, user_id=req.user_id, thread_id=thread_id, state=session_state)

        return resp
    except ValidationError as ve:
        telemetry_log("manager.response_validation_error", {"tab": tab, "errors": ve.errors()})
        return _fail("Graph response failed UIResponse validation.", req, details=ve.errors())


# -----------------------------------------------------------------------------#
# DI helpers (signature-aware to avoid kwarg mismatches if factory evolves)
# -----------------------------------------------------------------------------#
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
        # LLM & submit_changeset can be injected here via settings/provider if available.
    }
    try:
        sig = inspect.signature(make_default_deps)
        supported = set(sig.parameters.keys())
        safe_kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported}
    except Exception:
        safe_kwargs = dict(candidate_kwargs)

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


# -----------------------------------------------------------------------------#
# UIResponse normalization (wrap telemetry-like dicts → canonical AgentEvent)
# -----------------------------------------------------------------------------#
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


# -----------------------------------------------------------------------------#
# LangGraph config (checkpointer requires configurable keys)
# -----------------------------------------------------------------------------#
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


# -----------------------------------------------------------------------------#
# Timeouts (robust against both namespaced and flat settings)
# -----------------------------------------------------------------------------#
def _timeout_for_tab(tab: str) -> int:
    """
    Tries (in order):
      1) settings.manager.threats_timeout_sec / agent_timeout_sec
      2) flat: settings.agent_timeout_sec_threats / _impacts / _assets
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


# -----------------------------------------------------------------------------#
# Telemetry passthrough for tool.call.* events
# -----------------------------------------------------------------------------#
def _telemetry_from_raw_tool_events(raw: Any, *, tab: str, org_id: str, user_id: Optional[str]) -> None:
    """Forward tool.call.* events from a raw graph result/envelope to telemetry."""
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
                if ev.get("event") == "agent.tool_event" and isinstance(ev.get("data"), Mapping):
                    _maybe_log_tool_event(ev["data"], tab=tab, org_id=org_id, user_id=user_id)
                    continue
                _maybe_log_tool_event(ev, tab=tab, org_id=org_id, user_id=user_id)
            except Exception:
                continue
    except Exception:
        return


def _telemetry_from_response_tool_events(resp: UIResponse, *, tab: str, org_id: str, user_id: Optional[str]) -> None:
    """If the normalized UIResponse carries agent_events, forward tool.call.* events."""
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


# -----------------------------------------------------------------------------#
# Lightweight audit and minimal memory hints
# -----------------------------------------------------------------------------#
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


def _tab_not_supported(req: UIRequest) -> UIResponse:
    return UIResponse(
        messages=[f'Tab "{req.active_tab}" is not supported. Use assets, impacts, or threats.'],
        proposals=None,
        staging={"needs_commit": False, "batch_id": None, "diff": None},
        policy={"emit_changes_enabled": True},
    )


async def _persist_history(mem: RollingWindowMemory, req: UIRequest, resp: UIResponse, thread_id: str) -> None:
    """Append the user message and the first assistant reply to the rolling window."""
    try:
        first_reply = (resp.messages or [None])[0]
        new_msgs: list[tuple[str, str]] = []
        if req.message:
            new_msgs.append(("user", req.message))
        if first_reply:
            new_msgs.append(("assistant", str(first_reply)))
        if new_msgs:
            await mem.append(
                org_id=req.org_id,
                user_id=req.user_id,
                thread_id=thread_id,
                namespace=_CHAT_NS,
                new_messages=[cast(tuple[str, str], m) for m in new_msgs],
            )
    except Exception:
        # History must never break main flow
        pass


# -----------------------------------------------------------------------------#
# Cross-domain orchestration (suggest-next-step, no multi-write in a single turn)
# -----------------------------------------------------------------------------#
def _maybe_add_orchestration_suggestion(resp: UIResponse, *, tab: str, language: str) -> None:
    """
    Inspect agent_events to detect key operations (e.g., asset created) and add a
    localized suggestion for the next step in another domain (impacts → threats).
    This preserves the "one step per turn" rule while guiding the user forward.
    """
    try:
        events = getattr(resp, "agent_events", None)
        if not isinstance(events, list) or not events:
            return

        # Heuristic: look for successful 'assets_create' (tool.call.success)
        created = False
        for ev in events:
            if not isinstance(ev, Mapping):
                continue
            data = ev.get("data") if isinstance(ev.get("data"), Mapping) else ev
            etype = str(data.get("event_type") or "")
            tool = str(data.get("tool") or "")
            ok = data.get("ok", True)
            if etype == "tool.call.success" and tool in ("assets_create", "create_asset", "asset_create") and ok:
                created = True
                break

        if not created:
            return

        # Localized lightweight suggestion
        if language.startswith("cs"):
            suggestion = "Chceš rovnou nastavit dopady (impacts) a navrhnout hrozby (threats) pro nový asset?"
        elif language.startswith("sk"):
            suggestion = "Chceš hneď nastaviť dopady (impacts) a navrhnúť hrozby (threats) pre nový asset?"
        else:
            suggestion = "Do you want to set impacts and suggest threats for the new asset now?"

        msgs = list(resp.messages or [])
        msgs.append(suggestion)
        resp.messages = msgs  # type: ignore[attr-defined]
    except Exception:
        # Orchestration suggestion is best-effort
        pass


# -----------------------------------------------------------------------------#
# Language autodetection (very lightweight; prefers explicit req.language/locale)
# -----------------------------------------------------------------------------#
_CZ_CHARS = "ěščřžýáíéůúďťňĚŠČŘŽÝÁÍÉŮÚĎŤŇ"
_SK_CHARS = "äôľĺŕÄÔĽĹŔ"

def _autodetect_language(message: Optional[str], explicit: Optional[str], locale: Optional[str]) -> str:
    """
    Choose 'cs' | 'sk' | 'en' with a tiny heuristic:
    - If explicit language is provided and in {cs, sk, en} → use it.
    - Else if locale startswith(cs|sk|en) → use it.
    - Else detect by presence of Czech/Slovak diacritics.
    - Fallback to 'en'.
    """
    if explicit and explicit.lower() in {"cs", "sk", "en"}:
        return explicit.lower()
    if locale:
        l = locale.lower()
        if l.startswith(("cs", "sk", "en")):
            return l[:2]
    text = (message or "").strip()
    if any(ch in text for ch in _CZ_CHARS):
        return "cs"
    if any(ch in text for ch in _SK_CHARS):
        return "sk"
    return "en"


# -----------------------------------------------------------------------------#
# Session state helpers (current_asset + MRU + one-time clarify flags)
# -----------------------------------------------------------------------------#
async def _load_session_state(deps: GraphDeps, *, org_id: str, user_id: Optional[str], thread_id: str) -> dict:
    """Load session state dict from KV. Returns a normalized structure."""
    try:
        if not getattr(deps, "memory_get", None):
            return {"current_asset": None, "recent_assets": [], "once_flags": {"asked_for_asset": False}}
        raw = await deps.memory_get(org_id=org_id, user_id=user_id, thread_id=thread_id, namespace=_SESSION_NS, key=_SESSION_STATE_KEY)  # type: ignore[arg-type]
        if not isinstance(raw, Mapping):
            return {"current_asset": None, "recent_assets": [], "once_flags": {"asked_for_asset": False}}
        state = dict(raw)
    except Exception:
        state = {}
    # normalize
    current = state.get("current_asset") if isinstance(state.get("current_asset"), Mapping) else None
    recent = state.get("recent_assets") if isinstance(state.get("recent_assets"), list) else []
    once = state.get("once_flags") if isinstance(state.get("once_flags"), Mapping) else {"asked_for_asset": False}
    return {"current_asset": current, "recent_assets": recent[:_MRU_MAX], "once_flags": {"asked_for_asset": bool(once.get("asked_for_asset", False))}}


async def _save_session_state(deps: GraphDeps, *, org_id: str, user_id: Optional[str], thread_id: str, state: dict) -> None:
    """Persist session state dict into KV."""
    try:
        if not getattr(deps, "memory_set", None):
            return
        await deps.memory_set(org_id=org_id, user_id=user_id, thread_id=thread_id, namespace=_SESSION_NS, key=_SESSION_STATE_KEY, value=state)  # type: ignore[arg-type]
    except Exception:
        # session persistence must not break the main flow
        pass


def _session_set_current_asset(state: dict, *, asset_id: str, name: Optional[str]) -> dict:
    """Set current_asset and push into MRU, trimming to MRU_MAX."""
    state = dict(state or {})
    cur = {"asset_id": asset_id, "name": name}
    state["current_asset"] = cur
    mru = [i for i in state.get("recent_assets", []) if isinstance(i, Mapping)]
    # drop duplicates of the same id
    mru = [i for i in mru if str(i.get("asset_id")) != asset_id]
    mru.insert(0, cur)
    state["recent_assets"] = mru[:_MRU_MAX]
    # Once we have an asset, clear the once-clarify flag
    once = dict(state.get("once_flags", {}))
    once["asked_for_asset"] = False
    state["once_flags"] = once
    return state


def _maybe_update_current_asset_from_result(state: dict, raw_result: Any) -> dict:
    """
    Inspect raw_result/agent_events for:
      - successful assets_create (with result.asset_id/name)
      - any explicit asset selection event emitted by agents/router
    Update state.current_asset & MRU accordingly.
    """
    try:
        # locate events
        events: list[Any] = []
        if isinstance(raw_result, Mapping):
            ev = raw_result.get("agent_events")
            if isinstance(ev, list):
                events = ev
            elif isinstance(raw_result.get("result"), Mapping) and isinstance(raw_result["result"].get("agent_events"), list):
                events = raw_result["result"]["agent_events"]  # type: ignore[index]

        # pass 1: look for "agent.tool_event" envelopes
        for ev in events:
            data = ev
            if isinstance(ev, Mapping) and ev.get("event") == "agent.tool_event" and isinstance(ev.get("data"), Mapping):
                data = ev["data"]

            if not isinstance(data, Mapping):
                continue

            etype = str(data.get("event_type") or "")
            tool = str(data.get("tool") or "")

            if etype == "tool.call.success" and tool in {"assets_create", "create_asset", "asset_create"}:
                # Try common shapes for result
                res = data.get("result") if isinstance(data.get("result"), Mapping) else data.get("meta")
                if isinstance(res, Mapping):
                    asset_id = res.get("asset_id") or res.get("id")
                    name = res.get("name")
                    if asset_id:
                        return _session_set_current_asset(state, asset_id=str(asset_id), name=(str(name) if name else None))

            # Optional: some flows may emit a synthetic "asset.selected" event
            if etype == "asset.selected":
                asset_id = data.get("asset_id")
                name = data.get("name")
                if asset_id:
                    return _session_set_current_asset(state, asset_id=str(asset_id), name=(str(name) if name else None))
    except Exception:
        pass

    return state
