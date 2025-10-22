# File: workshops_ai/graph/manager.py
"""
Unified Graph Manager
---------------------
- Validates UIRequest → builds deps (incl. llm_client) → invokes the tab graph with timeout.
- Maintains a short rolling conversation history (user/assistant) per org:tab:user thread.
- Tracks thread session state (current_asset + MRU + once_flags [+ language]) exposed to graphs.
- Clarify-first policy lives in agent/tool layer; manager only forwards effective policy vars.
- IMPORTANT: Does NOT clobber graph policy on return (fixes emit_changes_enabled staging bug).

Enhancements in this version:
- Language lock for CS/SK/EN with **SK as the default**:
  * Resolution order per turn: explicit req.language → session language → autodetect → SK fallback.
  * Normalizes and pins `req.language` and `req.locale` (cs-CZ / sk-SK / en-US) for downstream prompts.
  * Session language is persisted in KV and reused on subsequent turns of the thread.
- Confirmation detector and next-step orchestration text prefer SK as the fallback language.
- **Graph runner cache:** graphs are compiled once per (tab, version) and reused across requests,
  dramatically reducing per-turn overhead. Bump `WSAI_GRAPH_VERSION` to force a rebuild.
"""

from __future__ import annotations

import asyncio
import inspect
import json            # JSON persistence for session state in KV
import ast             # Back-compat parser for legacy str(dict) payloads
from typing import Any, Final, Mapping, Optional, cast

from pydantic import ValidationError
from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.models.agent_event import AgentEvent as WireAgentEvent
from workshops_ai.tools.telemetry import telemetry_log
from workshops_ai.settings import get_settings

# Graph factory & helpers (centralized DI: llm_client, memory, checkpointer, config builder)
from workshops_ai.graph.factory import (
    GraphDeps,
    make_default_deps,
    build_graph,          # returns a compiled LangGraph runnable for a given tab
    ainvoke_with_timeout,
    build_graph_config,   # unified config builder (thread_id/lang/checkpoint_ns)
)

# Conversation memory (rolling window; KV-backed where available)
from workshops_ai.graph.conversation_memory import (
    RollingWindowMemory,
    MemoryConfig,
)

# Policy knobs (preview/apply, clarify-first, history, timeouts)
from workshops_ai.graph.policies import (
    TabName,
    GlobalPolicy,
    EffectivePolicy,
    make_effective_policy,
    as_prompt_vars,
)

# Single Source of Truth for tools — the canonical registry
from workshops_ai.agents.tools.registry import ToolRegistry


_SETTINGS = get_settings()

# Supported tabs
_SUPPORTED_TABS: Final[set[str]] = {"assets", "impacts", "threats"}

# -----------------------------------------------------------------------------
# Compiled-graph cache
# -----------------------------------------------------------------------------
# Bump this when you change graph definitions/nodes so the runners are recompiled.
WSAI_GRAPH_VERSION: Final[str] = "2025.10-iter+ckpt"

# Process-local cache: (version:tab) → compiled runner
_GRAPHS: dict[str, Any] = {}

# Per-key async locks to prevent duplicate compiles under bursty concurrency
_GRAPH_LOCKS: dict[str, asyncio.Lock] = {}


async def _get_or_build_runner(tab: str, deps: GraphDeps) -> Any:
    """
    Return a compiled graph runnable for the given tab.
    Compile once per (WSAI_GRAPH_VERSION, tab) and reuse across requests/users.
    This is safe because per-turn state (thread_id, language, checkpoint namespace) is supplied
    through LangGraph 'config', and the checkpointer comes from DI when the graph was compiled.
    """
    cache_key = f"{WSAI_GRAPH_VERSION}:{tab}"
    runner = _GRAPHS.get(cache_key)
    if runner is not None:
        return runner

    lock = _GRAPH_LOCKS.setdefault(cache_key, asyncio.Lock())
    async with lock:
        # Double-check after acquiring the lock in case another coroutine built it meanwhile.
        runner = _GRAPHS.get(cache_key)
        if runner is None:
            # Note: deps.checkpointer is wired by factory. We expect a stable instance (LRU/provider);
            # the per-turn config still sets thread_id & language so state doesn't bleed across users.
            runner = build_graph(tab, deps)
            _GRAPHS[cache_key] = runner
    return runner


# Namespaces for memory
_CHAT_NS: Final[str] = "chat"
_SESSION_NS: Final[str] = "session"

# Keys for session state (stored in KV)
_SESSION_STATE_KEY: Final[str] = "thread_state"
# Structure example:
# {
#   "current_asset": {"asset_id": str, "name": str | None},
#   "recent_assets": [{"asset_id": str, "name": str | None}, ...],  # MRU (max N)
#   "once_flags": {"asked_for_asset": bool},
#   "language": "cs|sk|en"   # ← persisted language lock for the thread
# }
_MRU_MAX: Final[int] = 5


# =============================================================================
# Public entrypoint
# =============================================================================
async def route_chat(req: UIRequest | dict[str, Any]) -> UIResponse:
    """Main router: validates input, builds deps/graph (with llm_client), injects history & policy, runs with timeout."""
    _audit_in(req)

    # --- Debug trace (request) ---
    try:
        from workshops_ai.telemetry_debug import start_trace  # lightweight & no-op safe
        _trace = start_trace((req.model_dump() if hasattr(req, "model_dump") else dict(req)))
    except Exception:
        _trace = None

    # 1) Defensive validation
    if not isinstance(req, UIRequest):
        try:
            req = UIRequest.model_validate(cast(Any, req))
        except ValidationError as ve:
            return _fail("Input does not match UIRequest contract.", None, details=ve.errors())

    tab = (req.active_tab or "").strip().lower()
    if tab not in _SUPPORTED_TABS:
        return _tab_not_supported(req)

    # 2) Build DI deps (backend, ws hub, retriever, memory, llm_client, settings, submit_changeset)
    deps = _deps(tab)

    # Propagate the trace to DI so that nodes (decide) can use it
    try:
        setattr(deps, "telemetry_trace", _trace)
    except Exception:
        pass

    # 3) Compute effective policy (respects incoming staging → may imply preview_required)
    eff: EffectivePolicy = make_effective_policy(
        tab=TabName(tab),
        global_policy=GlobalPolicy(),  # defaults; override path exists if needed
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
        config=MemoryConfig(
            max_messages=eff.max_history_messages,
            ttl_seconds=eff.history_ttl_seconds,
            namespace=_CHAT_NS,
        ),
    )
    turns = await mem.read(org_id=req.org_id, user_id=req.user_id, thread_id=thread_id, namespace=_CHAT_NS)
    history_msgs = RollingWindowMemory.to_llm_messages(turns, drop_tool_output=True)

    # 4a) Load/merge thread session state (current_asset + MRU + once_flags + language)
    session_state = await _load_session_state(deps, org_id=req.org_id, user_id=req.user_id, thread_id=thread_id)

    # UI selection explicitly overrides current asset
    selection_asset_id = getattr(getattr(req, "selection", None), "asset_id", None)
    if selection_asset_id:
        session_state = _session_set_current_asset(
            session_state,
            asset_id=str(selection_asset_id),
            name=getattr(req, "selection", None) and getattr(req.selection, "asset_name", None),
        )

    # 5) LANGUAGE LOCK (cs/sk/en) with SK as the default
    # Resolution order: explicit req.language → session_state["language"] → autodetect → SK
    session_lang = (session_state.get("language") or "").lower() if isinstance(session_state, dict) else None
    explicit_lang = _normalize_lang_code(getattr(req, "language", None))
    if explicit_lang in {"cs", "sk", "en"}:
        language = explicit_lang
    elif session_lang in {"cs", "sk", "en"}:
        language = session_lang
    else:
        language = _autodetect_language(req.message, req.language, req.locale) or "sk"

    # Persist language in session state and reflect it back to the request
    session_state["language"] = language
    req.language = language  # ensure the config builder picks it up

    # Normalize/pin locale to a stable value if not already coherent
    if not _normalize_lang_code(req.locale):
        req.locale = _default_locale_for(language)

    # 5a) Per-turn write-gate flag (not persisted): enable one-time write if the user confirms this turn
    # Start from session once_flags (carry other one-time hints) and add "one_shot_write" when confirmation is detected
    per_turn_flags = dict(session_state.get("once_flags", {}))
    if _looks_like_confirm_create(req.message, language):
        per_turn_flags["one_shot_write"] = True

    # Optional: allow UI to explicitly permit a one-time write in this turn.
    allow_once = getattr(req, "allow_write_once", None)
    if isinstance(allow_once, bool) and allow_once:
        per_turn_flags["one_shot_write"] = True

    # 6) Prepare SSOT tool specs for this tab (registry → prompt+tooling)
    tools_json = ToolRegistry.as_prompt_json(cast(Any, tab))              # JSON string for prompts
    allowed_functions = ToolRegistry.as_openai_functions(cast(Any, tab))  # OpenAI-style function list

    # 7) Prepare inputs for the graph (dict-based; minimal + stable)
    selection_dump = (
        req.selection.model_dump()
        if hasattr(req, "selection") and getattr(req, "selection") is not None and hasattr(req.selection, "model_dump")
        else dict(getattr(req, "selection", {}) or {})
    )

    inputs: dict[str, Any] = {
        "org_id": req.org_id,
        "active_tab": tab,
        "message": req.message,
        "user_id": req.user_id,
        "staging": bool(req.staging),
        "language": language,                 # "cs|sk|en"
        "locale": (req.locale or language),   # pinned locale
        "session_id": getattr(req, "session_id", None),
        "history": history_msgs[: eff.max_history_messages],
        "policy": as_prompt_vars(eff),
        "current_asset": session_state.get("current_asset"),
        "recent_assets": session_state.get("recent_assets", []),
        "once_flags": per_turn_flags,
        "history_token": getattr(req, "history_token", None),
        "organization_profile": getattr(req, "organization_profile", None),
        "services": getattr(req, "services", None),
        "existing_assets": getattr(req, "existing_assets", None),
        "user_instructions": getattr(req, "user_instructions", None),
        "manager_ctx": {
            "tools_json": tools_json,
            "allowed_functions": allowed_functions,
        },
        "tools_json": tools_json,
    }

    # Only add "selection" if it contains a concrete asset_id (prevents accidental wipe downstream)
    if isinstance(selection_dump, dict):
        _aid = str(selection_dump.get("asset_id") or "").strip()
        if _aid:
            inputs["selection"] = selection_dump

    # 8) Per-tab timeout & LangGraph config (via factory.build_graph_config)
    timeout = _timeout_for_tab(tab)
    cfg = build_graph_config(req, deps)  # single source of truth for configurable.thread_id/lang/checkpoint_ns)

    # 9) Execute graph (graphs get deps.llm_client via DI)
    try:
        runner = await _get_or_build_runner(tab, deps)
        raw_result = await ainvoke_with_timeout(runner, inputs, timeout=timeout, config=cfg)
    except asyncio.TimeoutError:
        return _fail(f"{tab.capitalize()}Agent execution timed out.", req)
    except Exception as e:
        telemetry_log("manager.graph_invoke_error", {"tab": tab, "error": repr(e)})
        return _fail("Agent execution failed.", req, details=str(e))

    # 9a) Telemetry passthrough for tool events (if present)
    _telemetry_from_raw_tool_events(raw_result, tab=tab, org_id=req.org_id, user_id=req.user_id)

    # 9b) Update session state from tool events (e.g., asset created/selected)
    session_state = _maybe_update_current_asset_from_result(session_state, raw_result)
    # Clear "asked_for_asset" once the current asset is known again
    if session_state.get("once_flags", {}).get("asked_for_asset") and session_state.get("current_asset"):
        session_state["once_flags"]["asked_for_asset"] = False

    # 10) Normalize & validate UIResponse (do NOT override graph policy)
    try:
        resp = _ensure_uiresponse(
            raw_result,
            tab=tab,
            org_id=req.org_id,
            user_id=req.user_id,
            asset_id=(getattr(req, "selection", None) and getattr(req.selection, "asset_id", None)),
        )

        # 10a) Forward tool events in normalized response
        _telemetry_from_response_tool_events(resp, tab=tab, org_id=req.org_id, user_id=req.user_id)

        # 10b) Do NOT clobber policy returned by the graph.
        if not getattr(resp, "policy", None):
            resp.policy = {"emit_changes_enabled": True}  # minimal default; graphs usually set this

        # 10c) Ensure staging block exists but DO NOT alter it (respect preview/apply decisions).
        if not getattr(resp, "staging", None):
            resp.staging = {"needs_commit": False, "batch_id": None, "diff": None}

        # 11) Light cross-domain orchestration (suggest-next-step only; single-write per turn kept)
        _maybe_add_orchestration_suggestion(resp, tab=tab, language=language)

        # 12) Persist conversation window (user → assistant)
        await _persist_history(mem, req, resp, thread_id)

        # 13) Persist session state (current_asset + MRU + once_flags + language)
        await _save_session_state(deps, org_id=req.org_id, user_id=req.user_id, thread_id=thread_id, state=session_state)

        # --- Debug trace (response summary) ---
        try:
            if _trace is not None:
                from workshops_ai.telemetry_debug import summarize_ui_response
                summarize_ui_response(_trace, resp.model_dump(mode="json"))
        except Exception:
            pass

        return resp

    except ValidationError as ve:
        telemetry_log("manager.response_validation_error", {"tab": tab, "errors": ve.errors()})
        return _fail("Graph response failed UIResponse validation.", req, details=ve.errors())


# =============================================================================
# DI helpers (signature-aware; inject llm_client via factory)
# =============================================================================
def _deps(tab: str) -> GraphDeps:
    """
    Build GraphDeps via the centralized factory. We pass only the keywords supported
    by make_default_deps to avoid mismatches. The factory:
      - resolves checkpointer (Redis/SQLite/Memory),
      - wires ephemeral memory helpers,
      - resolves llm_client from settings (or app/provider),
      - exposes submit_changeset with uniform OUTBOX fallback.
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
        safe_kwargs = dict(candidate_kwargs)

    return make_default_deps(**safe_kwargs)  # type: ignore[call-arg]


def _resolve_backend_port() -> Any | None:
    """Try provider → fallback to a CSV backend instance → final fallback to module-level port."""
    try:
        from workshops_ai.backend.provider import get_backend_port  # type: ignore
        return get_backend_port()
    except Exception:
        pass
    try:
        from workshops_ai.backend.csv_backend import CsvBackend  # type: ignore
        return CsvBackend()
    except Exception:
        pass
    try:
        from workshops_ai.backend.port import backend_port  # type: ignore
        return backend_port
    except Exception:
        pass
    return None


def _resolve_ws_hub():
    """
    Return a WebSocket hub instance if available. We try common shapes:
      - get_hub() factory
      - module/global ws_hub
      - WsHub() constructor
    If none are available, return None (graphs will skip WS emits).
    """
    try:
        # Preferred: a getter that returns a singleton/instance
        from workshops_ai.app.services.ws_hub import get_hub  # type: ignore
        try:
            return get_hub()
        except Exception:
            pass
    except Exception:
        pass
    try:
        # Next: a module-level instance
        from workshops_ai.app.services.ws_hub import ws_hub  # type: ignore
        if ws_hub:
            return ws_hub
    except Exception:
        pass
    try:
        # Last resort: construct the class if available and cheap
        from workshops_ai.app.services.ws_hub import WsHub  # type: ignore
        try:
            return WsHub()
        except Exception:
            return None
    except Exception:
        return None


def _resolve_retriever_factory():
    try:
        from workshops_ai.rag.retriever import make_retriever  # type: ignore
        return make_retriever
    except Exception:
        return None


# =============================================================================
# UIResponse normalization (wrap telemetry-like dicts → canonical AgentEvent)
# =============================================================================
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
    Produces a validated UIResponse, wrapping agent_events where needed.
    """
    payload = raw

    # Unwrap common {"result": {...}} envelope
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

    # Last resort: wrap plain content
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


# =============================================================================
# Timeouts (robust against both namespaced and flat settings)
# =============================================================================
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


# =============================================================================
# Telemetry passthrough for tool.call.* events
# =============================================================================
def _telemetry_from_raw_tool_events(raw: Any, *, tab: str, org_id: str, user_id: Optional[str]) -> None:
    """Forward tool.call.* events found in raw result/envelope to telemetry."""
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
        "tool": "...",
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


# =============================================================================
# Lightweight audit and minimal memory hints
# =============================================================================
def _audit_in(req: UIRequest | dict[str, Any]) -> None:
    try:
        payload = req.model_dump() if hasattr(req, "model_dump") else cast(Any, req)
        telemetry_log("manager.route_chat_in", {"req": payload})
    except Exception:
        pass


def _fail(msg: str, req: Optional[UIRequest], details: Any = None) -> UIResponse:
    """Return a minimal failure envelope."""
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
        # History persistence must never break main flow
        pass


# =============================================================================
# Cross-domain orchestration (suggest-next-step, SK fallback)
# =============================================================================
def _maybe_add_orchestration_suggestion(resp: UIResponse, *, tab: str, language: str) -> None:
    """
    Inspect agent_events to detect key operations (e.g., asset created) and add a
    localized suggestion for the next step in another domain (impacts → threats).
    Preserves the "one step per turn" rule while guiding the user forward.

    Fallback language for the suggestion is SK.
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

        # Localized lightweight suggestion (fallback = SK)
        if (language or "").startswith("cs"):
            suggestion = "Chceš rovnou nastavit dopady (impacts) a navrhnout hrozby (threats) pro nový asset?"
        elif (language or "").startswith("en"):
            suggestion = "Do you want to set impacts and suggest threats for the new asset now?"
        else:
            suggestion = "Chceš hneď nastaviť dopady (impacts) a navrhnúť hrozby (threats) pre nový asset?"

        msgs = list(resp.messages or [])
        msgs.append(suggestion)
        resp.messages = msgs  # type: ignore[attr-defined]
    except Exception:
        # Suggestion is best-effort
        pass


# =============================================================================
# Language autodetection (very lightweight; prefers explicit req.language/locale)
# =============================================================================
_CZ_CHARS = "ěščřžýáíéůúďťňĚŠČŘŽÝÁÍÉŮÚĎŤŇ"
_SK_CHARS = "äôľĺŕÄÔĽĹŔ"


def _normalize_lang_code(lang_or_locale: Optional[str]) -> Optional[str]:
    """
    Normalize free-form 'cs|sk|en' or IETF variants like 'cs-CZ', 'sk-SK', 'en-US'
    → returns 'cs' | 'sk' | 'en' or None.
    """
    if not lang_or_locale:
        return None
    s = str(lang_or_locale).strip().lower()
    if not s:
        return None
    if s[:2] in {"cs", "sk", "en"}:
        return s[:2]
    return None


def _default_locale_for(lang: str) -> str:
    """Return a stable IETF-like locale for a given normalized language code."""
    if (lang or "").startswith("cs"):
        return "cs-CZ"
    if (lang or "").startswith("en"):
        return "en-US"
    return "sk-SK"  # default/fallback


def _autodetect_language(message: Optional[str], explicit: Optional[str], locale: Optional[str]) -> Optional[str]:
    """
    Try to choose 'cs' | 'sk' | 'en' by:
    - explicit language (normalized) if in {cs, sk, en}
    - locale prefix (normalized) if in {cs, sk, en}
    - presence of Czech/Slovak diacritics
    Returns None if no signal is available (caller applies default SK).
    """
    norm_explicit = _normalize_lang_code(explicit)
    if norm_explicit in {"cs", "sk", "en"}:
        return norm_explicit
    norm_locale = _normalize_lang_code(locale)
    if norm_locale in {"cs", "sk", "en"}:
        return norm_locale
    text = (message or "").strip()
    if any(ch in text for ch in _CZ_CHARS):
        return "cs"
    if any(ch in text for ch in _SK_CHARS):
        return "sk"
    return None


# =============================================================================
# Per-turn confirmation detector for create (SK fallback)
# =============================================================================
def _looks_like_confirm_create(message: Optional[str], language: str) -> bool:
    """
    Return True when the user's current message looks like an explicit confirmation
    to proceed with creation (used to allow a one-shot write in non-staging mode).

    Rules:
      - Czech:    "Potvrdit vytvoření" (diacritics optional), or any message starting with "potvrdit".
      - English:  "confirm creation" / "confirm create" / "confirm", or messages starting with "confirm".
      - Fallback: Slovak → "Potvrdiť vytvorenie"/"Potvrdit vytvorenie", or messages starting with "potvrdiť"/"potvrdit".
    """
    s = (message or "").strip().lower()
    if not s:
        return False

    if (language or "").startswith("cs"):
        return s in {"potvrdit vytvoření", "potvrdit vytvoreni"} or s.startswith("potvrdit")
    if (language or "").startswith("en"):
        return s in {"confirm creation", "confirm create", "confirm"} or s.startswith("confirm")
    # default/fallback → SK
    return s in {"potvrdiť vytvorenie", "potvrdit vytvorenie"} or s.startswith("potvrdiť") or s.startswith("potvrdit")


# =============================================================================
# Session state helpers (current_asset + MRU + one-time clarify flags + language)
# =============================================================================
async def _load_session_state(deps: GraphDeps, *, org_id: str, user_id: Optional[str], thread_id: str) -> dict:
    """
    Load session state dict from KV.
    - JSON is the canonical format (this function decodes JSON).
    - For backward compatibility, also accepts a legacy Python dict string and
      tries ast.literal_eval as a safe fallback.
    Returns a normalized structure (keeps language if present).
    """
    try:
        if not getattr(deps, "memory_get", None):
            return {"current_asset": None, "recent_assets": [], "once_flags": {"asked_for_asset": False}, "language": None}
        raw = await deps.memory_get(  # type: ignore[attr-defined]
            org_id=org_id, user_id=user_id, thread_id=thread_id, namespace=_SESSION_NS, key=_SESSION_STATE_KEY
        )
        if raw is None:
            return {"current_asset": None, "recent_assets": [], "once_flags": {"asked_for_asset": False}, "language": None}

        state: dict[str, Any] | None = None
        if isinstance(raw, Mapping):
            state = dict(raw)
        elif isinstance(raw, str):
            # Prefer JSON
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, Mapping):
                    state = dict(parsed)
            except Exception:
                # Back-compat: legacy str(dict) with single quotes
                try:
                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, Mapping):
                        state = dict(parsed)
                except Exception:
                    state = None

        if not isinstance(state, dict):
            return {"current_asset": None, "recent_assets": [], "once_flags": {"asked_for_asset": False}, "language": None}
    except Exception:
        state = {}

    current = state.get("current_asset") if isinstance(state.get("current_asset"), Mapping) else None
    recent = state.get("recent_assets") if isinstance(state.get("recent_assets"), list) else []
    once = state.get("once_flags") if isinstance(state.get("once_flags"), Mapping) else {"asked_for_asset": False}
    lang_norm = _normalize_lang_code(state.get("language")) if isinstance(state, Mapping) else None

    return {
        "current_asset": current,
        "recent_assets": recent[:_MRU_MAX],
        "once_flags": {"asked_for_asset": bool(once.get("asked_for_asset", False))},
        "language": lang_norm,
    }


async def _save_session_state(deps: GraphDeps, *, org_id: str, user_id: Optional[str], thread_id: str, state: dict) -> None:
    """
    Persist session state dict into KV as JSON (canonical).
    If the memory backend stores raw strings (e.g., Redis), this guarantees
    a round-trip that _load_session_state() can decode.
    """
    try:
        if not getattr(deps, "memory_set", None):
            return
        payload = json.dumps(state, separators=(",", ":"), ensure_ascii=False)
        await deps.memory_set(  # type: ignore[attr-defined]
            org_id=org_id, user_id=user_id, thread_id=thread_id, namespace=_SESSION_NS, key=_SESSION_STATE_KEY, value=payload
        )
    except Exception:
        pass


def _session_set_current_asset(state: dict, *, asset_id: str, name: Optional[str]) -> dict:
    """Set current_asset and push into MRU, trimming to MRU_MAX; clear clarify flag."""
    state = dict(state or {})
    cur = {"asset_id": asset_id, "name": name}
    state["current_asset"] = cur
    mru = [i for i in state.get("recent_assets", []) if isinstance(i, Mapping)]
    mru = [i for i in mru if str(i.get("asset_id")) != asset_id]
    mru.insert(0, cur)
    state["recent_assets"] = mru[:_MRU_MAX]
    once = dict(state.get("once_flags", {}))
    once["asked_for_asset"] = False
    state["once_flags"] = once
    return state


def _maybe_update_current_asset_from_result(state: dict, raw_result: Any) -> dict:
    """
    Examine the raw graph result for agent events that imply the "current asset"
    should be updated (and MRU refreshed). Handles:
      - Successful create tool call (assets_create/create_asset/asset_create/asset.create)
      - Explicit selection event (event_type == "asset.selected")
      - NEW: asset detail payload in messages → {"kind": "asset.get", "asset": {...}}
    """
    try:
        events: list[Any] = []

        # Unwrap common envelopes:
        if isinstance(raw_result, Mapping):
            ev = raw_result.get("agent_events")
            if isinstance(ev, list):
                events = ev
            else:
                result = raw_result.get("result")
                if isinstance(result, Mapping) and isinstance(result.get("agent_events"), list):
                    events = result["agent_events"]  # type: ignore[index]

        # --- existing handlers (create + explicit selection) ---
        for ev in events:
            data = ev
            if isinstance(ev, Mapping) and ev.get("event") == "agent.tool_event" and isinstance(ev.get("data"), Mapping):
                data = ev["data"]
            if not isinstance(data, Mapping):
                continue

            etype = str(data.get("event_type") or "")
            tool_raw = str(data.get("tool") or "")
            tool_norm = tool_raw.replace(".", "_").strip().lower()

            # Successful "create asset"
            if etype == "tool.call.success" and tool_norm in {"assets_create", "create_asset", "asset_create"}:
                res = data.get("result") if isinstance(data.get("result"), Mapping) else data.get("meta")
                if isinstance(res, Mapping):
                    asset_id = res.get("asset_id") or res.get("id")
                    name = res.get("name")
                    if asset_id:
                        return _session_set_current_asset(
                            state,
                            asset_id=str(asset_id),
                            name=(str(name) if name is not None else None),
                        )

            # Explicit selection
            if etype == "asset.selected":
                asset_id = data.get("asset_id")
                name = data.get("name")
                if asset_id:
                    return _session_set_current_asset(
                        state,
                        asset_id=str(asset_id),
                        name=(str(name) if name is not None else None),
                    )

        # derive selection from messages that carry an "asset.get" payload ---
        def _iter_messages(container: Any) -> list[Any]:
            if not isinstance(container, Mapping):
                return []
            msgs = container.get("messages")
            if isinstance(msgs, list) and msgs:
                return msgs
            res = container.get("result")
            if isinstance(res, Mapping):
                msgs2 = res.get("messages")
                if isinstance(msgs2, list):
                    return msgs2
            return []

        for msg in _iter_messages(raw_result):
            # We expect a dict payload like: {"kind": "asset.get", "asset": {"id": "...", "name": "...", ...}}
            if not isinstance(msg, Mapping):
                continue
            kind = str(msg.get("kind") or msg.get("type") or "")
            if kind != "asset.get":
                continue
            asset = msg.get("asset")
            if isinstance(asset, Mapping) and asset.get("id"):
                return _session_set_current_asset(
                    state,
                    asset_id=str(asset.get("id")),
                    name=(str(asset.get("name")) if asset.get("name") is not None else None),
                )

    except Exception:
        pass

    return state
