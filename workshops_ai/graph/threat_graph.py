# File: workshops_ai/graph/threat_graph.py
# -----------------------------------------------------------------------------
# LangGraph: THREATS tab — LLM-first assistant (clarify-first, one-call-per-turn)
#
# Goals
# - Use deps.llm_client (no local LLM instantiation here).
# - Clarify-first NLU via agent_decide (exactly one tool choice per turn).
# - Short localized replies (cs/sk/en).
# - Writes go through typed ChangeSet from models/changeset.
# - Submit via deps.submit_or_outbox (handles online + OUTBOX fallback).
# - Keep tool-call telemetry and avoid multi-step cascades in a single turn.
# - Registry-aligned tools: threat.list, threat.set_probability, threat.bulk_set, threat.explain
#   (plus an internal convenience node threat.asset_get for “GET” previews per asset).
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

from workshops_ai.graph.factory import GraphDeps
from workshops_ai.graph.agent_node import agent_decide, AgentDecision

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse

# ChangeSet (typed source of truth)
from workshops_ai.models.changeset import (
    ChangeSet,
    build_changeset,
)

# LangGraph
try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("langgraph must be installed to use threat_graph") from e

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
Intent = Literal["help", "clarify", "chitchat", "list", "get", "set", "bulk_set", "upsert", "explain"]

class RiskItem(BaseModel):
    threat_code: str
    probability: int = Field(ge=0, le=4)
    note: Optional[str] = None

class RiskSetPayload(BaseModel):
    asset_id: str
    items: List[RiskItem]

class RiskBulkPayload(BaseModel):
    entries: List[RiskSetPayload]

class UpsertThreatItem(BaseModel):
    # Catalog upsert (kept for parity with impacts/assets)
    code: Optional[str] = None
    name: str
    category: Optional[str] = None
    asset_type_codes: List[str] = Field(default_factory=list)
    intentional: bool = False
    unintentional: bool = False
    environmental: bool = False
    description: Optional[str] = None
    present: Optional[bool] = None

class ToolEvent(TypedDict, total=False):
    event_type: str            # "tool.call.start" | "tool.call.success" | "tool.call.fail"
    tool: str
    ok: bool
    duration_ms: int
    meta: dict

class ThreatState(BaseModel):
    # Request context
    org_id: str
    active_tab: str = Field(default="threats")
    message: Optional[str] = None
    selection: Optional[dict] = None   # {"asset_id": "..."} | ...
    user_id: Optional[str] = None
    locale: Optional[str] = None       # "sk" | "cs" | "en"
    language: Optional[str] = None
    history: Optional[List[dict]] = None  # short rolling window

    # Structured inputs (from planner or API)
    action: Optional[Intent] = None
    set_payload: Optional[RiskSetPayload] = None
    bulk_payload: Optional[RiskBulkPayload] = None
    upserts: Optional[List[UpsertThreatItem]] = None
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
            "sk": "Ahoj! Môžem: „zoznam hrozieb“, „riziká pre aktívum“, „nastaviť P“, „hromadne nastaviť“, „pridať/aktualizovať hrozby“. 😉",
            "cs": "Ahoj! Umím: „seznam hrozeb“, „rizika pro aktivum“, „nastavit P“, „hromadně nastavit“, „přidat/aktualizovat hrozby“. 😉",
            "en": "Hi! I can: “list threats”, “asset risks”, “set P”, “bulk set”, or “upsert threats”. 😉",
        },
        "catalog_fail": {
            "sk": "Katalóg hrozieb sa nepodarilo načítať: {err}",
            "cs": "Katalog hrozeb se nepodařilo načíst: {err}",
            "en": "Failed to load the threat catalog: {err}",
        },
        "catalog_found": {
            "sk": "Katalóg hrozieb ({n}):",
            "cs": "Katalog hrozeb ({n}):",
            "en": "Threat catalog ({n}):",
        },
        "select_prompt": {
            "sk": "Vyber prosím aktívum. Tu je pár možností:",
            "cs": "Vyber prosím aktivum. Tady je pár možností:",
            "en": "Please select an asset. Here are a few options:",
        },
        "asset_risks_fail": {
            "sk": "Čítanie rizík pre „{id}“ zlyhalo: {err}",
            "cs": "Čtení rizik pro „{id}“ selhalo: {err}",
            "en": "Failed to read risks for “{id}”: {err}",
        },
        "nothing_to_set": {
            "sk": "Nie je čo nastaviť. Potrebujem asset_id a items[{threat_code, probability}].",
            "cs": "Není co nastavit. Potřebuji asset_id a items[{threat_code, probability}].",
            "en": "Nothing to set. Need asset_id and items[{threat_code, probability}].",
        },
        "bulk_missing": {
            "sk": "Chýbajú entries pre hromadné nastavenie.",
            "cs": "Chybí entries pro hromadné nastavení.",
            "en": "Missing entries for bulk update.",
        },
        "upsert_missing": {
            "sk": "Zadaj upserts s aspoň jednou položkou (min. name; code voliteľné).",
            "cs": "Zadej upserts s alespoň jednou položkou (min. name; code volitelné).",
            "en": "Provide upserts with at least one item (name required; code optional).",
        },
        "validation_error": {
            "sk": "Chyba validácie: {err}",
            "cs": "Chyba validace: {err}",
            "en": "Validation error: {err}",
        },
        "submitted_summary": {
            "sk": "{set} nastavení pravdepodobností, {upsert} úprav hrozieb.",
            "cs": "{set} nastavení pravděpodobností, {upsert} úprav hrozeb.",
            "en": "{set} probability updates, {upsert} threat upserts.",
        },
        "submitted_generic": {
            "sk": "{count} akcií odoslaných.",
            "cs": "{count} akcí odesláno.",
            "en": "{count} actions submitted.",
        },
        "explain_need_question": {
            "sk": "Zadaj otázku (state.question alebo message) pre vysvetlenie hrozieb/rizík.",
            "cs": "Zadej otázku (state.question nebo message) pro vysvětlení hrozeb/rizik.",
            "en": "Provide a question (state.question or message) to explain threats/risks.",
        },
        "explain_no_sources": {
            "sk": "Momentálne nemám kontext a LLM fallback nevrátil odpoveď.",
            "cs": "Momentálně nemám kontext a LLM fallback nevrátil odpověď.",
            "en": "No context is available right now and the LLM fallback did not return an answer.",
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

def _event(state: ThreatState, *, kind: str, tool: str, ok: bool = True, meta: dict | None = None) -> float:
    """Start a tool event and return the high-precision start time."""
    t0 = time.perf_counter()
    state.agent_events.append({"event_type": kind, "tool": tool, "ok": ok, "meta": meta or {}})
    return t0

def _finish(state: ThreatState, t0: float, *, tool: str, ok: bool, meta: dict | None = None) -> None:
    """Finish a tool event, recording duration and status."""
    dur = int((time.perf_counter() - t0) * 1000)
    state.agent_events.append(
        {"event_type": ("tool.call.success" if ok else "tool.call.fail"), "tool": tool, "ok": ok, "duration_ms": dur, "meta": meta or {}}
    )

# Backend accessor (align with routers)
def _backend(deps: GraphDeps):
    """Prefer deps.backend (router-consistent), fallback to deps.backend_port."""
    return getattr(deps, "backend", None) or getattr(deps, "backend_port", None)


# -----------------------------------------------------------------------------
# RAG→LLM explain helpers
# -----------------------------------------------------------------------------
EXPLAIN_SYSTEM = (
    "You are a cybersecurity assistant focused on threats and risk probabilities. "
    "Explain clearly and concisely, prioritizing the organization's catalog/context when available."
)

def _get_threats_retriever(deps: GraphDeps, org_id: Optional[str] = None):
    """
    Prefer factory.make_retriever(org_id, 'threats') (viz factory.py).
    Fall back na starší přístupy, pokud nejsou deps.make_retriever k dispozici.
    """
    mk = getattr(deps, "make_retriever", None)
    if callable(mk) and org_id:
        try:
            r = mk(org_id, "threats")
            if r:
                return r
        except Exception:
            logger.debug("make_retriever(org_id, 'threats') failed", exc_info=True)

    factory = getattr(deps, "retriever_factory", None)
    if callable(factory):
        try:
            r = factory("threats")
            if r:
                return r
        except Exception:
            logger.debug("retriever_factory('threats') failed", exc_info=True)

    getter = getattr(deps, "get_retriever", None)
    if callable(getter):
        try:
            r = getter("threats")
            if r:
                return r
        except Exception:
            logger.debug("get_retriever('threats') failed", exc_info=True)

    return getattr(deps, "threats_retriever", None)

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
def _history_tail(state: ThreatState, k: int = 5) -> List[dict]:
    """Return the last k turns of the conversational history (best-effort)."""
    try:
        h = state.history or []
        return h[-k:]
    except Exception:
        return []

async def _nlu(deps: GraphDeps, state: ThreatState) -> ThreatState:
    """NLU step: choose intent via agent_decide and parse tool args into typed payloads."""
    raw = (state.message or "").strip()

    # Explicit passthrough (API-driven)
    if state.action in ("help", "clarify", "chitchat", "list", "get", "set", "bulk_set", "upsert", "explain"):
        state.intent = cast(Intent, state.action)
        return state

    if not raw:
        state.intent = "help"
        return state

    sel = state.selection or {}

    # Allowed tool names (registry-aligned + one internal read alias)
    allowed = [
        "threat.list",
        "threat.set_probability",
        "threat.bulk_set",
        "threat.explain",
        "threat.asset_get",          # internal convenience for "get" preview
        "help",
        "clarify",
        "chitchat",
    ]

    decision: AgentDecision = await agent_decide(
        deps=deps,
        tab="threats",
        message=raw,
        language=state.locale or "en",
        history=_history_tail(state, 5),
        selection=sel,
        allowed_actions=allowed,
        one_action_only=True,
    )

    logger.debug("threats.decide -> action=%s args=%s", getattr(decision, "action", None), getattr(decision, "args", None))

    # Tool → intent mapping (registry names normalized to internal intents)
    tool_to_intent = {
        "threat.list": "list",
        "threat.set_probability": "set",
        "threat.bulk_set": "bulk_set",
        "threat.explain": "explain",
        "threat.asset_get": "get",
        "help": "help",
        "clarify": "clarify",
        "chitchat": "chitchat",
    }
    state.intent = cast(Intent, tool_to_intent.get((getattr(decision, "action", "") or "").strip(), "help"))

    # ---- slot filling (strict Pydantic validation) ----
    if getattr(decision, "args", None):
        args = decision.args or {}

        # set payload (probabilities for one asset)
        if state.intent == "set":
            asset_id = args.get("asset_id") or sel.get("asset_id")
            items = args.get("items")
            if asset_id and isinstance(items, list):
                try:
                    parsed = [RiskItem(**it) for it in items]
                    state.set_payload = RiskSetPayload(asset_id=asset_id, items=parsed)
                except Exception:
                    state.set_payload = None

        # bulk payload (probabilities across multiple assets)
        if state.intent == "bulk_set" and isinstance(args.get("entries"), list):
            try:
                entries: List[RiskSetPayload] = []
                for e in args["entries"]:
                    if not isinstance(e, dict):
                        continue
                    aid = e.get("asset_id") or sel.get("asset_id")
                    arr = [RiskItem(**it) for it in (e.get("items") or [])]
                    if aid and arr:
                        entries.append(RiskSetPayload(asset_id=aid, items=arr))
                if entries:
                    state.bulk_payload = RiskBulkPayload(entries=entries)
            except Exception:
                state.bulk_payload = None

        # upsert (threat catalog)
        if state.intent == "upsert":
            items_src = []
            if isinstance(args.get("item"), dict):
                items_src.append(args["item"])
            if isinstance(args.get("items"), list):
                items_src.extend([x for x in args["items"] if isinstance(x, dict)])
            if items_src:
                try:
                    state.upserts = [UpsertThreatItem(**x) for x in items_src]
                except Exception:
                    state.upserts = None

        # explain (RAG question)
        if state.intent == "explain":
            q = args.get("q") or args.get("threatName") or args.get("question")
            if isinstance(q, str) and q.strip():
                state.question = q.strip()

    return state


# -----------------------------------------------------------------------------
# Clarify-first helpers (at most one concise follow-up)
# -----------------------------------------------------------------------------
def _suggest_select(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    """If no asset is selected, show a short preview list to help user pick one."""
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="suggest_select")
        try:
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

def _clarify_once() -> Callable[[ThreatState], ThreatState]:
    """Produce one short follow-up (never loops)."""
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)
        txt = {
            "cs": "Chceš LIST / GET / SET / BULK / UPSERT / EXPLAIN?",
            "sk": "Chceš LIST / GET / SET / BULK / UPSERT / EXPLAIN?",
            "en": "Would you like LIST / GET / SET / BULK / UPSERT / EXPLAIN?",
        }[_lang(state.locale)]
        state.result = _ui_response(txt, extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Read-only nodes
# -----------------------------------------------------------------------------
def _list_threats(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    """List the threat catalog in a compact, localized form (router-consistent)."""
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="threat.list")
        try:
            backend = _backend(deps)
            defs = (backend.threat_defs_list(org_id=state.org_id) if backend else []) or []
            lines = [_t(loc, "catalog_found", n=len(defs))]
            for r in defs[:30]:
                d = r.model_dump() if hasattr(r, "model_dump") else dict(r)
                code = d.get("code") or d.get("threat_code") or "?"
                name = d.get("name") or ""
                lines.append(f"- {code}: {name}")
            state.result = _ui_response(
                "\n".join(lines),
                extra={"threats": [getattr(i, "model_dump", lambda: dict(i))() for i in defs[:50]], "agent_events": state.agent_events},
            )
            _finish(state, t0, tool="threat.list", ok=True, meta={"count": len(defs)})
        except Exception as e:
            _finish(state, t0, tool="threat.list", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "catalog_fail", err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner

def _asset_get_risks_or_suggest(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    """Show risks for the selected asset, or suggest an asset if none is selected."""
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)
        backend = _backend(deps)
        asset_id = (state.selection or {}).get("asset_id")
        if not asset_id:
            return _suggest_select(deps)(state)

        t0 = _event(state, kind="tool.call.start", tool="threat.asset_get", meta={"asset_id": asset_id})
        try:
            # Prefer threats_for_asset(); fallback to asset_get_risks()
            rows = []
            if backend:
                getter = getattr(backend, "threats_for_asset", None) or getattr(backend, "asset_get_risks", None)
                rows = getter(asset_id) if getter else []
            rows = rows or []
            lines = [f"Risks for {asset_id} ({len(rows)}):"]
            for it in rows[:30]:
                code = it.get("threat_code") or it.get("code")
                prob = it.get("probability") or it.get("level")
                lines.append(f"- {code}: P={prob}")
            state.result = _ui_response(
                "\n".join(lines),
                extra={"asset_id": asset_id, "risks": rows, "agent_events": state.agent_events},
            )
            _finish(state, t0, tool="threat.asset_get", ok=True, meta={"count": len(rows)})
        except Exception as e:
            _finish(state, t0, tool="threat.asset_get", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "asset_risks_fail", id=asset_id, err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Explain node (RAG→LLM fallback)
# -----------------------------------------------------------------------------
async def _explain_threats_async(deps: GraphDeps, state: ThreatState) -> ThreatState:
    """Explain threats via retriever grounding (if available), else concise LLM fallback."""
    loc = _lang(state.locale)
    q = (state.question or state.message or "").strip()
    if not q:
        state.result = _ui_response(_t(loc, "explain_need_question"), extra={"agent_events": state.agent_events})
        return state

    retriever = _get_threats_retriever(deps, state.org_id)
    chunks: List[Dict[str, Any]] = []
    rag_used = False

    if retriever:
        t0 = _event(state, kind="tool.call.start", tool="retriever.search", meta={"tab": "threats"})
        try:
            docs = []
            if hasattr(retriever, "aget_relevant_documents"):
                docs = await retriever.aget_relevant_documents(q)
            elif hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(q)  # type: ignore
            elif callable(retriever):
                maybe = retriever(q)
                docs = await maybe if hasattr(maybe, "__await__") else maybe
            for d in docs or []:
                chunks.append({
                    "text": getattr(d, "page_content", None) or getattr(d, "text", "") or str(d),
                    "meta": getattr(d, "metadata", {}) or {},
                })
            rag_used = len(chunks) > 0
            _finish(state, t0, tool="retriever.search", ok=True, meta={"docs": len(chunks)})
        except Exception as e:
            _finish(state, t0, tool="retriever.search", ok=False, meta={"error": str(e)})
            logger.debug("threats retriever failed; falling back to LLM-only", exc_info=True)

    messages: List[dict] = [{"role": "system", "content": EXPLAIN_SYSTEM}]
    if rag_used:
        ctx = "\n\n---\n".join([c["text"] for c in chunks[:5] if c.get("text")])
        messages.append({"role": "system", "content": f"Catalog/context snippets:\n{ctx}"})
    messages.append({"role": "user", "content": f"Explain: {q}"})

    try:
        answer = await _llm_text_async(deps=deps, messages=messages, temperature=0.2)
        state.result = _ui_response(
            answer.strip(),
            extra={"rag_used": rag_used, "sources": chunks[:5] if rag_used else [], "agent_events": state.agent_events},
        )
        return state
    except Exception:
        # fall through to explicit second attempt without snippets (already done above if rag_used=False)
        pass

    state.result = _ui_response(_t(loc, "explain_no_sources"), extra={"snippets": (chunks or None), "agent_events": state.agent_events})
    return state

async def run_explain(*, deps: GraphDeps, text: str, locale: Optional[str] = None, org_id: Optional[str] = None) -> dict:
    """
    Router-facing explain for THREATS. Reuses internal _explain_threats_async and returns UI payload.
    """
    state = ThreatState(
        org_id=org_id or getattr(getattr(deps, "settings", object()), "org_id", "demo-org"),
        active_tab="threats",
        message=text,
        question=text,
        locale=locale or "en",
        agent_events=[],
    )
    state = await _explain_threats_async(deps, state)
    return state.result or {"ok": False, "error": "no_result"}


# -----------------------------------------------------------------------------
# Write nodes (no regex; strictly use provided codes/probabilities)
# -----------------------------------------------------------------------------
def _set_probability(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    """Set one or more threat probabilities for a single asset (merge mode)."""
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)
        payload = state.set_payload

        if not payload or not payload.items:
            if not (state.selection and state.selection.get("asset_id")):
                return _suggest_select(deps)(state)
            state.result = _ui_response(_t(loc, "nothing_to_set"), extra={"agent_events": state.agent_events})
            return state

        action = {
            "op": "set_risks",
            "mode": "merge",
            "asset_id": payload.asset_id,
            "items": [it.model_dump() for it in payload.items],
        }
        try:
            _guard_validate("ActionSetRisks", action)  # type: ignore[arg-type]
        except Exception as e:
            state.result = _ui_response(_t(loc, "validation_error", err=e), extra={"agent_events": state.agent_events})
            return state

        state.actions = [action]

        last = payload.items[-1]
        p = max(0, min(4, int(last.probability)))
        txt = {
            "cs": f"✅ Nastavena pravděpodobnost {last.threat_code}=P{p} pro aktivum „{payload.asset_id}“.",
            "sk": f"✅ Nastavená pravdepodobnosť {last.threat_code}=P{p} pre aktívum „{payload.asset_id}“.",
            "en": f"✅ Set probability {last.threat_code}=P{p} for asset “{payload.asset_id}”.",
        }[_lang(state.locale)]
        state.result = _ui_response(txt, extra={"agent_events": state.agent_events})
        return state
    return _inner

def _bulk_set_probability(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    """Bulk set threat probabilities across multiple assets (merge mode)."""
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)
        bulk = state.bulk_payload
        if not bulk or not bulk.entries:
            state.result = _ui_response(_t(loc, "bulk_missing"), extra={"agent_events": state.agent_events})
            return state

        actions: List[dict] = []
        for entry in bulk.entries:
            if not entry.asset_id or not entry.items:
                continue
            a = {
                "op": "set_risks",
                "mode": "merge",
                "asset_id": entry.asset_id,
                "items": [it.model_dump() for it in entry.items],
            }
            try:
                _guard_validate("ActionSetRisks", a)  # type: ignore[arg-type]
            except Exception as ex:
                state.result = _ui_response(_t(loc, "validation_error", err=ex), extra={"agent_events": state.agent_events})
                return state
            actions.append(a)

        if not actions:
            state.result = _ui_response(_t(loc, "bulk_missing"), extra={"agent_events": state.agent_events})
            return state

        state.actions = actions
        txt = {"cs": "✅ Připraveno k aplikaci (hromadné nastavení).",
               "sk": "✅ Pripravené na aplikáciu (hromadné nastavenie).",
               "en": "✅ Ready to apply (bulk set)."}[_lang(state.locale)]
        state.result = _ui_response(txt, extra={"agent_events": state.agent_events})
        return state
    return _inner

def _upsert_threats(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    """Create or update threat catalog entries (upsert semantics)."""
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)

        items = state.upserts or []
        if not items:
            # Optional tiny JSON fallback: {"items":[{...}]} or {"upserts":[{...}]}
            try:
                raw = json.loads(state.message or "{}")
                items = [UpsertThreatItem(**x) for x in (raw.get("items") or raw.get("upserts") or [])]
            except Exception:
                pass

        if not items:
            state.result = _ui_response(_t(loc, "upsert_missing"), extra={"agent_events": state.agent_events})
            return state

        # Normalize defaults & ensure booleans present
        normalized = []
        for it in items:
            payload = it.model_dump(exclude_none=True)
            payload.setdefault("asset_type_codes", [])
            for b in ("intentional", "unintentional", "environmental"):
                payload[b] = bool(payload.get(b, False))
            normalized.append(payload)

        action = {"op": "upsert_threats", "items": normalized}
        try:
            _guard_validate("ActionUpsertThreats", action)  # type: ignore[arg-type]
        except Exception as e:
            state.result = _ui_response(_t(loc, "validation_error", err=e), extra={"agent_events": state.agent_events})
            return state

        state.actions = [action]

        first = items[0]
        txt = {
            "cs": f"✅ Upsert hrozby „{first.name}“.",
            "sk": f"✅ Upsert hrozby „{first.name}“.",
            "en": f"✅ Upsert threat “{first.name}”.",
        }[_lang(state.locale)]
        state.result = _ui_response(txt, extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Submit (single submit per turn) — unified with routers (submit_or_outbox)
# -----------------------------------------------------------------------------
async def _submit_changeset_async(deps: GraphDeps, state: ThreatState) -> ThreatState:
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
        submit_result = await submit_fn(cs_typed, require_success=False)  # factory řeší online → OUTBOX fallback

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

    n_set = sum(1 for a in acts if a.get("op") == "set_risks")
    n_upsert = sum(1 for a in acts if a.get("op") == "upsert_threats")
    text = _t(loc, "submitted_summary", set=n_set, upsert=n_upsert) if (n_set or n_upsert) else _t(loc, "submitted_generic", count=len(acts))

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
      start → nlu → route → {help|clarify|list|get|set|bulk_set|upsert|explain} → (submit?) → END

    - Clarify-first guarantees at most one short follow-up when data is missing.
    - Exactly one external effect per turn (writes → single submit).
    - Registry-aligned tool names: threat.list / set_probability / bulk_set / explain.
    """
    graph = StateGraph(ThreatState)

    # Async nodes
    async def _nlu_node(state: ThreatState) -> ThreatState:
        return await _nlu(deps, state)

    async def _submit_node(state: ThreatState) -> ThreatState:
        return await _submit_changeset_async(deps, state)

    async def _explain_node(state: ThreatState) -> ThreatState:
        return await _explain_threats_async(deps, state)

    # Nodes
    graph.add_node("nlu", _nlu_node)
    graph.add_node("route", lambda s: s)

    # clarify/help
    graph.add_node("help", lambda s: _ui_response(_t(_lang(s.locale), "greet"), extra={"agent_events": s.agent_events}) or s)  # type: ignore[arg-type]
    graph.add_node("clarify", _clarify_once())
    graph.add_node("suggest_select", _suggest_select(deps))

    # read
    graph.add_node("threat.list", _list_threats(deps))
    graph.add_node("threat.asset_get", _asset_get_risks_or_suggest(deps))  # internal convenience

    # write (registry-aligned)
    graph.add_node("threat.set_probability", _set_probability(deps))
    graph.add_node("threat.bulk_set", _bulk_set_probability(deps))
    graph.add_node("threat.upsert", _upsert_threats(deps))  # kept for symmetry; not in registry if you don't need it

    # explain (registry-aligned)
    graph.add_node("threat.explain", _explain_node)

    # submit
    graph.add_node("submit_changeset", _submit_node)

    # Entry
    graph.set_entry_point("nlu")
    graph.add_edge("nlu", "route")

    # Router
    def _route(state: ThreatState) -> str:
        intent = state.intent or "clarify"
        if intent in ("help", "chitchat"):
            return "help"
        if intent == "clarify":
            return "clarify"
        if intent == "list":
            return "threat.list"
        if intent == "get":
            return "threat.asset_get"
        if intent == "set":
            return "threat.set_probability"
        if intent == "bulk_set":
            return "threat.bulk_set"
        if intent == "upsert":
            return "threat.upsert"
        if intent == "explain":
            return "threat.explain"
        return "help"

    graph.add_conditional_edges(
        "route",
        _route,
        {
            "help": "help",
            "clarify": "clarify",
            "threat.list": "threat.list",
            "threat.asset_get": "threat.asset_get",
            "threat.set_probability": "threat.set_probability",
            "threat.bulk_set": "threat.bulk_set",
            "threat.upsert": "threat.upsert",
            "threat.explain": "threat.explain",
        },
    )

    # Writes → single submit per turn
    graph.add_edge("threat.set_probability", "submit_changeset")
    graph.add_edge("threat.bulk_set", "submit_changeset")
    graph.add_edge("threat.upsert", "submit_changeset")

    # Terminal
    for node in ("help", "clarify", "threat.list", "threat.asset_get", "threat.explain", "submit_changeset"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=deps.checkpointer)


__all__ = ["router", "graph_router", "run_explain", "run_explain_threats"]
