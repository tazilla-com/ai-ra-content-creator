# File: workshops_ai/graph/impact_graph.py
# -----------------------------------------------------------------------------
# LangGraph: IMPACTS tab — LLM-first assistant (clarify-first, one-call-per-turn)
#
# Design goals
# - Planner/Router uses deps.llm_client (no local LLM instantiation, no regex heuristics).
# - Clarify-first: ask at most one concise follow-up when required data is missing.
# - Exactly one external effect per turn (writes → single submit).
# - Unified tool naming with registry (incl. alias: `impact.levels.for_asset` → get).
# - Explain uses RAG→LLM fallback (retriever first; if no context → concise LLM).
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import time
from typing import Any, Callable, Literal, Optional, cast, List, Dict

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.graph.factory import GraphDeps
from workshops_ai.graph.agent_node import agent_decide, AgentDecision

# ChangeSet (typed source of truth)
from workshops_ai.models.changeset import (
    ChangeSet,
    build_changeset,
)

# LangGraph
try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("langgraph must be installed to use impact_graph") from e

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Optional payload guard (best-effort, tolerant)
# -----------------------------------------------------------------------------
try:
    from workshops_ai.tools.guard import guard_validate_payload as _guard_validate  # type: ignore
except Exception:  # pragma: no cover
    def _guard_validate(model_cls, payload):  # type: ignore
        return payload


# -----------------------------------------------------------------------------
# Types & State (Pydantic-validated tool args)
# -----------------------------------------------------------------------------
Intent = Literal["help", "clarify", "chitchat", "list", "get", "set", "bulk_set", "create_impact", "explain"]


class ImpactItem(BaseModel):
    impact_code: str
    level: int = Field(ge=0, le=4)
    note: Optional[str] = None


class ImpactSetPayload(BaseModel):
    asset_id: str
    items: List[ImpactItem]


class ImpactBulkPayload(BaseModel):
    entries: List[ImpactSetPayload]


class ImpactCreateItem(BaseModel):
    code: str
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    weight: Optional[float] = None


class ToolEvent(TypedDict, total=False):
    event_type: str            # "tool.call.start" | "tool.call.success" | "tool.call.fail"
    tool: str
    ok: bool
    duration_ms: int
    meta: dict


class ImpactState(BaseModel):
    # Request context
    org_id: str
    active_tab: str = Field(default="impacts")
    message: Optional[str] = None
    selection: Optional[dict] = None   # {"asset_id": "..."} | ...
    user_id: Optional[str] = None
    locale: Optional[str] = None       # "sk" | "cs" | "en"
    language: Optional[str] = None
    history: Optional[List[dict]] = None  # short rolling window

    # Structured inputs (from planner or API)
    action: Optional[Intent] = None
    set_payload: Optional[ImpactSetPayload] = None
    bulk_payload: Optional[ImpactBulkPayload] = None
    create_items: Optional[List[ImpactCreateItem]] = None
    question: Optional[str] = None

    # Working outputs
    intent: Optional[Intent] = None
    actions: Optional[List[dict]] = None
    change: Optional[ChangeSet | dict] = None
    result: Optional[dict] = None

    # Telemetry
    agent_events: List[ToolEvent] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# i18n helpers (short)
# -----------------------------------------------------------------------------
def _lang(locale: Optional[str], fallback: str = "en") -> str:
    if not locale:
        return fallback
    l = locale.lower()
    if l.startswith("sk"):
        return "sk"
    if l.startswith(("cs", "cz")):
        return "cs"
    if l.startswith("en"):
        return "en"
    return fallback


def _t(loc: str, key: str, **kw) -> str:
    M = {
        "greet": {
            "sk": "Ahoj! Môžem: „zoznam dopadov“, „dopady pre aktívum“, „nastaviť úroveň“, „hromadne nastaviť“, „pridať dopad“. 😉",
            "cs": "Ahoj! Umím: „seznam dopadů“, „dopady pro aktivum“, „nastavit úroveň“, „hromadně nastavit“, „přidat dopad“. 😉",
            "en": "Hi! I can: “list impacts”, “asset impacts”, “set level”, “bulk set”, or “create impact”. 😉",
        },
        "catalog_fail": {
            "sk": "Katalóg dopadov sa nepodarilo načítať: {err}",
            "cs": "Katalog dopadů se nepodařilo načíst: {err}",
            "en": "Failed to load impact catalog: {err}",
        },
        "catalog_found": {
            "sk": "Katalóg dopadov ({n}):",
            "cs": "Katalog dopadů ({n}):",
            "en": "Impact catalog ({n}):",
        },
        "select_prompt": {
            "sk": "Vyber prosím aktívum. Tu je pár možností:",
            "cs": "Vyber prosím aktivum. Tady je pár možností:",
            "en": "Please select an asset. Here are a few options:",
        },
        "asset_impacts_fail": {
            "sk": "Čítanie dopadov pre „{id}“ zlyhalo: {err}",
            "cs": "Čtení dopadů pro „{id}“ selhalo: {err}",
            "en": "Failed to read impacts for “{id}”: {err}",
        },
        "nothing_to_set": {
            "sk": "Nie je čo nastaviť. Potrebujem asset_id a items[{impact_code, level}].",
            "cs": "Není co nastavit. Potřebuji asset_id a items[{impact_code, level}].",
            "en": "Nothing to set. Need asset_id and items[{impact_code, level}].",
        },
        "bulk_missing": {
            "sk": "Chýbajú entries pre hromadné nastavenie.",
            "cs": "Chybí entries pro hromadné nastavení.",
            "en": "Missing entries for bulk update.",
        },
        "create_missing": {
            "sk": "Zadaj aspoň jeden dopad {code,name,...}.",
            "cs": "Zadej alespoň jeden dopad {code,name,...}.",
            "en": "Provide at least one impact {code,name,...}.",
        },
        "validation_error": {
            "sk": "Chyba validácie: {err}",
            "cs": "Chyba validace: {err}",
            "en": "Validation error: {err}",
        },
        "submitted_summary": {
            "sk": "{set} nastavení úrovní, {created} nových dopadov.",
            "cs": "{set} nastavení úrovní, {created} nových dopadů.",
            "en": "{set} level updates, {created} new impacts created.",
        },
        "submitted_generic": {
            "sk": "{count} akcií odoslaných.",
            "cs": "{count} akcí odesláno.",
            "en": "{count} actions submitted.",
        },
    }
    s = M.get(key, {}).get(loc, M.get(key, {}).get("en", key))
    return s.format(**kw)


def _ui_response(text: str, *, extra: Optional[dict] = None) -> dict:
    base = {
        "messages": [text],
        "proposals": None,
        "staging": {"needs_commit": False, "batch_id": None, "diff": None},
        "policy": {"emit_changes_enabled": True},
    }
    if extra:
        base.update(extra)
    return base


# -----------------------------------------------------------------------------
# Utilities + telemetry
# -----------------------------------------------------------------------------
def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _event(state: ImpactState, *, kind: str, tool: str, ok: bool = True, meta: dict | None = None) -> float:
    """Start a tool event and return the high-precision start time."""
    t0 = time.perf_counter()
    state.agent_events.append({"event_type": kind, "tool": tool, "ok": ok, "meta": meta or {}})
    return t0


def _finish(state: ImpactState, t0: float, *, tool: str, ok: bool, meta: dict | None = None) -> None:
    """Finish a tool event, recording duration and status."""
    dur = int((time.perf_counter() - t0) * 1000)
    state.agent_events.append(
        {"event_type": ("tool.call.success" if ok else "tool.call.fail"), "tool": tool, "ok": ok, "duration_ms": dur, "meta": meta or {}}
    )


# -----------------------------------------------------------------------------
# Backend access helpers (align with routers; prefer deps.backend)
# -----------------------------------------------------------------------------
def _backend(deps: GraphDeps):
    """
    Access the backend port. Prefer deps.backend (same as routers) and
    fall back to deps.backend_port for older wiring.
    """
    return getattr(deps, "backend", None) or getattr(deps, "backend_port", None)


# -----------------------------------------------------------------------------
# RAG→LLM explain helpers
# -----------------------------------------------------------------------------
EXPLAIN_SYSTEM = (
    "You are an Impact domain assistant. Explain impact definitions clearly and concisely. "
    "Always prioritize the organization's catalog definitions when provided in context."
)

def _get_impacts_retriever(deps: GraphDeps, org_id: Optional[str] = None):
    """
    Prefer factory.make_retriever(org_id, 'impacts') as provided by factory.py.
    Fall back to older access patterns if present.
    """
    mk = getattr(deps, "make_retriever", None)
    if callable(mk) and org_id:
        try:
            r = mk(org_id, "impacts")
            if r:
                return r
        except Exception:  # pragma: no cover
            logger.debug("make_retriever(org_id, 'impacts') failed", exc_info=True)

    factory = getattr(deps, "retriever_factory", None)
    if callable(factory):
        try:
            r = factory("impacts")
            if r:
                return r
        except Exception:  # pragma: no cover
            logger.debug("retriever_factory('impacts') failed", exc_info=True)

    getter = getattr(deps, "get_retriever", None)
    if callable(getter):
        try:
            r = getter("impacts")
            if r:
                return r
        except Exception:  # pragma: no cover
            logger.debug("get_retriever('impacts') failed", exc_info=True)

    return getattr(deps, "impacts_retriever", None)


def _llm_text(
    *,
    deps: GraphDeps,
    messages: list[dict[str, str]],
    temperature: float | None = 0.2,
    model: Optional[str] = None,
) -> str:
    """
    Minimal, robust bridge for LangChain ChatModel clients (e.g., langchain_openai.ChatOpenAI).
    """
    client = getattr(deps, "llm_client", None) or getattr(deps, "llm", None)
    if client is None:
        raise RuntimeError("LLM client is not configured (deps.llm_client is None).")

    temp = temperature if temperature is not None else getattr(deps, "llm_temperature", None) or 0.2

    # LangChain ChatModel via .invoke([...BaseMessage...]) — PRIORITY
    try:
        invoke = getattr(client, "invoke", None)
        if callable(invoke):
            try:
                from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # type: ignore
            except Exception:
                from langchain.schema import SystemMessage, HumanMessage, AIMessage  # type: ignore

            lc_messages: list[object] = []
            for m in messages:
                role = (m.get("role") or "").lower()
                content = m.get("content") or ""
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role in ("user", "human", ""):
                    lc_messages.append(HumanMessage(content=content))
                elif role in ("assistant", "ai"):
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))

            try:
                resp = client.invoke(lc_messages, temperature=temp)  # type: ignore[misc]
            except TypeError:
                resp = client.invoke(lc_messages)  # type: ignore[misc]

            txt = (getattr(resp, "content", None) or "").strip()
            if txt:
                return txt
    except Exception:
        pass

    # Generic fallback
    try:
        obj = client(messages=messages, temperature=temp, model=model)  # type: ignore[misc]
        if isinstance(obj, str):
            return obj.strip()
        txt = (getattr(obj, "content", None) or getattr(obj, "text", None) or str(obj) or "").strip()
        if txt:
            return txt
    except Exception:
        pass

    raise RuntimeError("LLM bridge failed: unsupported client shape or empty content.")

async def run_explain(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

async def run_explain_impacts(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    return await run_explain(deps=deps, ui_req=ui_req)

async def _llm_text_async(*, deps: GraphDeps, messages: List[dict], temperature: float = 0.2) -> str:
    """
    Async bridge that executes the sync _llm_text in a thread, avoiding improper awaits.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _llm_text(deps=deps, messages=messages, temperature=temperature))


# -----------------------------------------------------------------------------
# Planner (agent_decide → slot-filling; no regex heuristics)
# -----------------------------------------------------------------------------
def _history_tail(state: ImpactState, k: int = 5) -> List[dict]:
    """Return the last k turns of the conversational history (best-effort)."""
    try:
        h = state.history or []
        return h[-k:]
    except Exception:
        return []


async def _nlu(deps: GraphDeps, state: ImpactState) -> ImpactState:
    """NLU step: choose intent via agent_decide and parse tool args into typed payloads."""
    raw = (state.message or "").strip()

    # Explicit passthrough (API-driven)
    if state.action in ("help", "clarify", "chitchat", "list", "get", "set", "bulk_set", "create_impact", "explain"):
        state.intent = cast(Intent, state.action)
        return state

    if not raw:
        state.intent = "help"
        return state

    sel = state.selection or {}

    # Allowed tool names that the planner may emit for this tab (registry-aligned)
    allowed = [
        "impact.list",
        "impact.asset_get",            # internal legacy (maps to 'get')
        "impact.levels.for_asset",     # registry alias (also maps to 'get')
        "impact.set_level",
        "impact.bulk_set",
        "impact.create_impact",
        "impact.explain",
        "help",
        "clarify",
        "chitchat",
    ]

    decision: AgentDecision = await agent_decide(
        deps=deps,
        tab="impacts",
        message=raw,
        language=state.locale or "en",
        history=_history_tail(state, 5),
        selection=sel,
        allowed_actions=allowed,
        one_action_only=True,
    )

    logger.debug("impacts.decide -> action=%s args=%s", getattr(decision, "action", None), getattr(decision, "args", None))

    # Tool → intent mapping (registry names normalized to internal intents)
    tool_to_intent = {
        "impact.list": "list",
        "impact.asset_get": "get",
        "impact.levels.for_asset": "get",   # alias to 'get'
        "impact.set_level": "set",
        "impact.bulk_set": "bulk_set",
        "impact.create_impact": "create_impact",
        "impact.explain": "explain",
        "help": "help",
        "clarify": "clarify",
        "chitchat": "chitchat",
    }
    state.intent = cast(Intent, tool_to_intent.get((getattr(decision, "action", "") or "").strip(), "help"))

    # ---- slot filling (strict Pydantic validation) ----
    if getattr(decision, "args", None):
        args = decision.args or {}

        # set payload
        if state.intent == "set":
            asset_id = args.get("asset_id") or sel.get("asset_id")
            items = args.get("items")
            if asset_id and isinstance(items, list):
                try:
                    parsed = [ImpactItem(**it) for it in items]
                    state.set_payload = ImpactSetPayload(asset_id=asset_id, items=parsed)
                except Exception:
                    state.set_payload = None

        # bulk payload
        if state.intent == "bulk_set" and isinstance(args.get("entries"), list):
            try:
                entries: List[ImpactSetPayload] = []
                for e in args["entries"]:
                    if not isinstance(e, dict):
                        continue
                    aid = e.get("asset_id") or sel.get("asset_id")
                    arr = [ImpactItem(**it) for it in (e.get("items") or [])]
                    if aid and arr:
                        entries.append(ImpactSetPayload(asset_id=aid, items=arr))
                if entries:
                    state.bulk_payload = ImpactBulkPayload(entries=entries)
            except Exception:
                state.bulk_payload = None

        # create impact (catalog)
        if state.intent == "create_impact":
            items_src = []
            if isinstance(args.get("item"), dict):
                items_src.append(args["item"])
            if isinstance(args.get("items"), list):
                items_src.extend([x for x in args["items"] if isinstance(x, dict)])
            if items_src:
                try:
                    state.create_items = [ImpactCreateItem(**x) for x in items_src]
                except Exception:
                    state.create_items = None

        # explain question
        if state.intent == "explain":
            q = args.get("q") or args.get("impactCode") or args.get("impactName")
            if isinstance(q, str) and q.strip():
                state.question = q.strip()

    return state


# -----------------------------------------------------------------------------
# Clarify-first helpers (at most one concise follow-up)
# -----------------------------------------------------------------------------
def _suggest_select(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    """If no asset is selected, show a short preview list to help user pick one."""
    def _inner(state: ImpactState) -> ImpactState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="suggest_select")
        try:
            # Prefer deps.backend (router-consistent), fallback to backend_port
            backend = _backend(deps)
            items = (backend.assets_list(org_id=state.org_id) if backend else []) or []
            preview = []
            for a in items[:8]:
                try:
                    d = a.model_dump() if hasattr(a, "model_dump") else dict(a)
                except Exception:
                    d = dict(a)
                aid = d.get("id") or d.get("asset_id") or "?"
                name = d.get("name") or "Unnamed"
                a_type = d.get("assetType") or d.get("type") or "-"
                preview.append(f"- {aid}: {name} [{a_type}]")
            msg = "\n".join([_t(loc, "select_prompt")] + preview) if preview else _t(loc, "catalog_found", n=0)
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events, "assets_preview": preview})
            _finish(state, t0, tool="suggest_select", ok=True, meta={"count": len(preview)})
        except Exception as e:
            _finish(state, t0, tool="suggest_select", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "catalog_fail", err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner


def _clarify_once() -> Callable[[ImpactState], ImpactState]:
    """Produce one short follow-up question based on what’s missing (never loops)."""
    def _inner(state: ImpactState) -> ImpactState:
        loc = _lang(state.locale)
        sel_id = (state.selection or {}).get("asset_id")
        if state.intent == "set":
            if not (state.set_payload and state.set_payload.asset_id):
                if not sel_id:
                    return _suggest_select(None)(state)  # type: ignore[arg-type]
                # ask for items only
                txt = {"cs": "Uveď prosím items[{impact_code, level}].",
                       "sk": "Uveď prosím items[{impact_code, level}].",
                       "en": "Please provide items[{impact_code, level}]."}.get(loc, "Please provide items.")
                state.result = _ui_response(txt, extra={"agent_events": state.agent_events})
                return state
        if state.intent == "bulk_set" and not (state.bulk_payload and state.bulk_payload.entries):
            txt = {"cs": "Pošli prosím entries[{asset_id, items[{impact_code, level}]}].",
                   "sk": "Pošli prosím entries[{asset_id, items[{impact_code, level}]}].",
                   "en": "Please send entries[{asset_id, items[{impact_code, level}]}]."}.get(loc)
            state.result = _ui_response(txt, extra={"agent_events": state.agent_events})
            return state
        if state.intent == "create_impact" and not (state.create_items and len(state.create_items) > 0):
            txt = {"cs": "Zadej alespoň jeden dopad {code,name[,category,description,weight]}.",
                   "sk": "Zadaj aspoň jeden dopad {code,name[,category,description,weight]}.",
                   "en": "Provide at least one impact {code,name[,category,description,weight]}."}.get(loc)
            state.result = _ui_response(txt, extra={"agent_events": state.agent_events})
            return state
        # default greet/help
        state.result = _ui_response(_t(loc, "greet"), extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Read-only nodes
# -----------------------------------------------------------------------------
def _list_impacts(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    """List the impact catalog in a compact, localized form."""
    def _inner(state: ImpactState) -> ImpactState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="impact.list")
        try:
            backend = _backend(deps)
            # Align with routers: prefer impact_defs_list(org_id)
            defs = (backend.impact_defs_list(org_id=state.org_id) if backend else []) or []
            lines = [_t(loc, "catalog_found", n=len(defs))]
            for row in defs[:30]:
                d = row.model_dump() if hasattr(row, "model_dump") else dict(row)
                code = d.get("code") or d.get("impact_code") or "?"
                name = d.get("name") or ""
                lines.append(f"- {code}: {name}")
            state.result = _ui_response(
                "\n".join(lines),
                extra={
                    "impacts": [getattr(i, "model_dump", lambda: dict(i))() for i in defs[:50]],
                    "agent_events": state.agent_events,
                },
            )
            _finish(state, t0, tool="impact.list", ok=True, meta={"count": len(defs)})
        except Exception as e:
            _finish(state, t0, tool="impact.list", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "catalog_fail", err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner


def _asset_get_impacts_or_suggest(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    """Show impacts for the selected asset, or suggest an asset if none is selected."""
    def _inner(state: ImpactState) -> ImpactState:
        loc = _lang(state.locale)
        backend = _backend(deps)
        asset_id = (state.selection or {}).get("asset_id")
        if not asset_id:
            return _suggest_select(deps)(state)

        t0 = _event(state, kind="tool.call.start", tool="impact.asset_get", meta={"asset_id": asset_id})
        try:
            levels = backend.asset_get_impacts(asset_id) if backend else []  # [{code/impact_code, level}, ...]
            summary = [f"Impacts for {asset_id} ({len(levels)}):"]
            for it in levels[:30]:
                code = it.get("impact_code") or it.get("code")
                lvl = it.get("level")
                summary.append(f"- {code}: {lvl}")
            state.result = _ui_response(
                "\n".join(summary),
                extra={"asset_id": asset_id, "impacts": levels, "agent_events": state.agent_events},
            )
            _finish(state, t0, tool="impact.asset_get", ok=True, meta={"count": len(levels)})
        except Exception as e:
            _finish(state, t0, tool="impact.asset_get", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "asset_impacts_fail", id=asset_id, err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Explain node (RAG→LLM fallback)
# -----------------------------------------------------------------------------
async def _explain_node_async(deps: GraphDeps, state: ImpactState) -> ImpactState:
    """Explain impacts via RAG→LLM fallback; writes a user-facing message into state.result."""
    query = (state.question or state.message or "").strip() or "impacts overview"
    retriever = _get_impacts_retriever(deps, state.org_id)
    chunks: List[Dict[str, Any]] = []
    rag_used = False

    if retriever:
        try:
            docs = []
            if hasattr(retriever, "aget_relevant_documents"):
                docs = await retriever.aget_relevant_documents(query)
            elif hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(query)  # type: ignore
            elif callable(retriever):
                maybe = retriever(query)
                docs = await maybe if hasattr(maybe, "__await__") else maybe
            for d in docs or []:
                chunks.append({
                    "text": getattr(d, "page_content", None) or getattr(d, "text", "") or str(d),
                    "meta": getattr(d, "metadata", {}) or {},
                })
            rag_used = len(chunks) > 0
        except Exception:
            logger.debug("impacts retriever failed; falling back to LLM-only", exc_info=True)

    messages: List[dict] = [{"role": "system", "content": EXPLAIN_SYSTEM}]
    if rag_used:
        ctx = "\n\n---\n".join([c["text"] for c in chunks[:5] if c.get("text")])
        messages.append({"role": "system", "content": f"Catalog/context snippets:\n{ctx}"})
    messages.append({"role": "user", "content": f"Explain: {query}"})

    answer = await _llm_text_async(deps=deps, messages=messages, temperature=0.2)
    state.result = _ui_response(
        answer.strip(),
        extra={
            "rag_used": rag_used,
            "sources": chunks[:5] if rag_used else [],
            "agent_events": state.agent_events,
        },
    )
    return state

async def run_explain(*, deps: GraphDeps, text: str, locale: Optional[str] = None, org_id: Optional[str] = None) -> dict:
    """
    Router-facing explain for IMPACTS. Reuses internal async explain node and returns UI payload.
    """
    state = ImpactState(
        org_id=org_id or getattr(getattr(deps, "settings", object()), "org_id", "demo-org"),
        active_tab="impacts",
        message=text,
        question=text,
        locale=locale or "en",
        agent_events=[],
    )
    state = await _explain_node_async(deps, state)
    return state.result or {"ok": False, "error": "no_result"}

# -----------------------------------------------------------------------------
# Write nodes (no regex; strictly use provided codes/levels)
# -----------------------------------------------------------------------------
def _set_impact_level(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    """Set one or more impact levels for a single asset (merge mode)."""
    def _inner(state: ImpactState) -> ImpactState:
        loc = _lang(state.locale)
        payload = state.set_payload

        if not payload or not payload.items:
            if not (state.selection and state.selection.get("asset_id")):
                return _suggest_select(deps)(state)
            state.result = _ui_response(_t(loc, "nothing_to_set"), extra={"agent_events": state.agent_events})
            return state

        # Guard + build action (registry validates "ActionSetImpacts")
        action = {
            "op": "set_impacts",
            "mode": "merge",
            "asset_id": payload.asset_id,
            "items": [it.model_dump() for it in payload.items],
        }
        try:
            _guard_validate("ActionSetImpacts", action)  # type: ignore[arg-type]
        except Exception as e:
            state.result = _ui_response(_t(loc, "validation_error", err=e), extra={"agent_events": state.agent_events})
            return state

        state.actions = [action]

        # Short confirmation (echo last item)
        last = payload.items[-1]
        lvl_i = max(0, min(4, int(last.level)))
        if loc == "cs":
            txt = f"✅ Nastaven dopad {last.impact_code}={lvl_i} pro aktivum „{payload.asset_id}“."
        elif loc == "sk":
            txt = f"✅ Nastavený dopad {last.impact_code}={lvl_i} pre aktívum „{payload.asset_id}“."
        else:
            txt = f"✅ Set impact {last.impact_code}={lvl_i} for asset “{payload.asset_id}”."
        state.result = _ui_response(txt, extra={"agent_events": state.agent_events})
        return state
    return _inner


def _bulk_set_impact_levels(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    """Bulk set impact levels across multiple assets (merge mode)."""
    def _inner(state: ImpactState) -> ImpactState:
        loc = _lang(state.locale)
        bulk = state.bulk_payload
        if not bulk or not bulk.entries:
            state.result = _ui_response(_t(loc, "bulk_missing"), extra={"agent_events": state.agent_events})
            return state

        actions: List[dict] = []
        for e in bulk.entries:
            if not e.asset_id or not e.items:
                continue
            a = {
                "op": "set_impacts",
                "mode": "merge",
                "asset_id": e.asset_id,
                "items": [it.model_dump() for it in e.items],
            }
            try:
                _guard_validate("ActionSetImpacts", a)  # type: ignore[arg-type]
            except Exception as ex:
                state.result = _ui_response(_t(loc, "validation_error", err=ex), extra={"agent_events": state.agent_events})
                return state
            actions.append(a)

        if not actions:
            state.result = _ui_response(_t(loc, "bulk_missing"), extra={"agent_events": state.agent_events})
            return state

        state.actions = actions
        text = {"cs": "✅ Připraveno k aplikaci (hromadné nastavení).",
                "sk": "✅ Pripravené na aplikáciu (hromadné nastavenie).",
                "en": "✅ Ready to apply (bulk set)."}[_lang(state.locale)]
        state.result = _ui_response(text, extra={"agent_events": state.agent_events})
        return state
    return _inner


def _create_impact(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    """Create or update impact catalog entries (upsert semantics)."""
    def _inner(state: ImpactState) -> ImpactState:
        loc = _lang(state.locale)
        items = state.create_items or []
        if not items:
            # Optional tiny JSON fallback: {"items":[{code,name,...}]}
            try:
                raw = json.loads(state.message or "{}")
                src = raw.get("items") or []
                items = [ImpactCreateItem(**x) for x in src]
            except Exception:
                pass

        if not items:
            state.result = _ui_response(_t(loc, "create_missing"), extra={"agent_events": state.agent_events})
            return state

        action = {"op": "upsert_impacts", "items": [it.model_dump() for it in items]}  # keep compatible via upsert op
        try:
            _guard_validate("ActionUpsertImpacts", action)  # type: ignore[arg-type]
        except Exception as e:
            state.result = _ui_response(_t(loc, "validation_error", err=e), extra={"agent_events": state.agent_events})
            return state

        state.actions = [action]

        # Short confirmation (echo first)
        first = items[0]
        text = {"cs": f"✅ Přidán dopad „{first.name}“ ({first.code}).",
                "sk": f"✅ Pridaný dopad „{first.name}“ ({first.code}).",
                "en": f"✅ Impact “{first.name}” ({first.code}) queued."}[_lang(state.locale)]
        state.result = _ui_response(text, extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Submit (single submit per turn) — unified with routers (submit_or_outbox)
# -----------------------------------------------------------------------------
async def _submit_changeset_async(deps: GraphDeps, state: ImpactState) -> ImpactState:
    """Submit accumulated actions as a typed ChangeSet via deps.submit_or_outbox."""
    loc = _lang(state.locale)
    acts = state.actions or []
    if not acts:
        state.result = _ui_response(_t(loc, "submitted_generic", count=0), extra={"agent_events": state.agent_events})
        return state

    cs: ChangeSet | dict = build_changeset(
        org_id=state.org_id,
        user_id=state.user_id or "unknown",
        actions=acts,
        intent="apply",
    )
    state.change = cs

async def _submit_changeset_async(deps: GraphDeps, state: ImpactState) -> ImpactState:
    """Submit accumulated actions as a typed ChangeSet via deps.submit_changeset (with OUTBOX fallback handled inside)."""
    loc = _lang(state.locale)
    acts = state.actions or []
    if not acts:
        state.result = _ui_response(_t(loc, "submitted_generic", count=0), extra={"agent_events": state.agent_events})
        return state

    cs: ChangeSet | dict = build_changeset(
        org_id=state.org_id,
        user_id=state.user_id or "unknown",
        actions=acts,
        intent="apply",
    )
    state.change = cs

    submit_fn = getattr(deps, "submit_changeset", None)
    if not callable(submit_fn):
        state.result = _ui_response(_t(loc, "validation_error", err="submit_changeset not available"), extra={"agent_events": state.agent_events})
        return state

    t0 = _event(state, kind="tool.call.start", tool="submit_changeset", meta={"actions": len(acts)})
    try:
        cs_typed = cs if hasattr(cs, "model_dump") else ChangeSet.model_validate(cs)
        submit_result = await submit_fn(cs_typed, require_success=False)  # factory handles online→OUTBOX fallback

        ok = bool(getattr(submit_result, "submitted", False))
        if not ok:
            fallback_path = getattr(submit_result, "fallback_path", None)
            _finish(state, t0, tool="submit_changeset", ok=False, meta={"fallback_path": str(fallback_path) if fallback_path else None})
            state.result = _ui_response(
                {"cs": "Odeslání selhalo, změny uloženy lokálně.", "sk": "Odoslanie zlyhalo, zmeny uložené lokálne.", "en": "Submission failed; queued locally."}[_lang(state.locale)],
                extra={"agent_events": state.agent_events, "outbox_path": str(fallback_path) if fallback_path else None},
            )
            return state

        _finish(state, t0, tool="submit_changeset", ok=True, meta={"actions": len(acts)})
    except Exception as e:
        _finish(state, t0, tool="submit_changeset", ok=False, meta={"error": str(e)})
        state.result = _ui_response(_t(loc, "validation_error", err=e), extra={"agent_events": state.agent_events})
        return state

    n_set = sum(1 for a in acts if a.get("op") == "set_impacts")
    n_created = sum(1 for a in acts if a.get("op") == "upsert_impacts")
    text = _t(loc, "submitted_summary", set=n_set, created=n_created) if (n_set or n_created) else _t(loc, "submitted_generic", count=len(acts))

    try:
        cs_dump = cs.model_dump(mode="json") if hasattr(cs, "model_dump") else dict(cs)  # type: ignore[arg-type]
    except Exception:
        cs_dump = {"_error": "cannot_dump_changeset"}

    state.result = _ui_response(
        text,
        extra={
            "changeset_actions": len(acts),
            "agent_events": state.agent_events,
            "messages": [text, {"kind": "changeset", "changeset": cs_dump, "ts": _now_iso()}],
        },
    )
    state.actions = None
    state.change = None
    return state


# -----------------------------------------------------------------------------
# Graph assembly
# -----------------------------------------------------------------------------
def create_graph(deps: GraphDeps):
    """
    Flow:
      start → nlu → route → {help|clarify|list|get|set|bulk_set|create_impact|explain} → (submit?) → END

    - Clarify-first guarantees at most one short follow-up when data is missing.
    - Exactly one external effect per turn (writes → single submit).
    - Registry-aligned tool names with alias: `impact.levels.for_asset` → internal "get".
    """
    graph = StateGraph(ImpactState)

    # Async nodes
    async def _nlu_node(state: ImpactState) -> ImpactState:
        return await _nlu(deps, state)

    async def _submit_node(state: ImpactState) -> ImpactState:
        return await _submit_changeset_async(deps, state)

    async def _explain_node(state: ImpactState) -> ImpactState:
        return await _explain_node_async(deps, state)

    # Nodes
    graph.add_node("nlu", _nlu_node)
    graph.add_node("route", lambda s: s)

    # clarify/help
    graph.add_node("help", lambda s: _ui_response(_t(_lang(s.locale), "greet"), extra={"agent_events": s.agent_events}) or s)  # type: ignore[arg-type]
    graph.add_node("clarify", _clarify_once())
    graph.add_node("suggest_select", _suggest_select(deps))

    # read
    graph.add_node("impact.list", _list_impacts(deps))
    graph.add_node("impact.asset_get", _asset_get_impacts_or_suggest(deps))
    graph.add_node("impact.levels.for_asset", _asset_get_impacts_or_suggest(deps))  # registry alias to 'get'

    # write
    graph.add_node("impact.set_level", _set_impact_level(deps))
    graph.add_node("impact.bulk_set", _bulk_set_impact_levels(deps))
    graph.add_node("impact.create_impact", _create_impact(deps))

    # explain
    graph.add_node("impact.explain", _explain_node)

    # submit
    graph.add_node("submit_changeset", _submit_node)

    # Entry
    graph.set_entry_point("nlu")
    graph.add_edge("nlu", "route")

    # Router
    def _route(state: ImpactState) -> str:
        intent = state.intent or "clarify"
        if intent in ("help", "chitchat"):
            return "help"
        if intent == "clarify":
            return "clarify"
        if intent == "list":
            return "impact.list"
        if intent == "get":
            # Keep internal semantic node; registry alias is mapped separately
            return "impact.asset_get"
        if intent == "set":
            return "impact.set_level"
        if intent == "bulk_set":
            return "impact.bulk_set"
        if intent == "create_impact":
            return "impact.create_impact"
        if intent == "explain":
            return "impact.explain"
        return "help"

    graph.add_conditional_edges(
        "route",
        _route,
        {
            "help": "help",
            "clarify": "clarify",
            "impact.list": "impact.list",
            "impact.asset_get": "impact.asset_get",
            "impact.levels.for_asset": "impact.levels.for_asset",  # reachable if planner emits alias directly
            "impact.set_level": "impact.set_level",
            "impact.bulk_set": "impact.bulk_set",
            "impact.create_impact": "impact.create_impact",
            "impact.explain": "impact.explain",
        },
    )

    # Writes → single submit per turn
    graph.add_edge("impact.set_level", "submit_changeset")
    graph.add_edge("impact.bulk_set", "submit_changeset")
    graph.add_edge("impact.create_impact", "submit_changeset")

    # Terminal
    for node in (
        "help",
        "clarify",
        "impact.list",
        "impact.asset_get",
        "impact.levels.for_asset",
        "impact.explain",
        "submit_changeset",
    ):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=deps.checkpointer)


__all__ = ["router", "graph_router", "run_explain", "run_explain_impacts"]
