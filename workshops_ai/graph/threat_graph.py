# File: workshops_ai/graph/threat_graph.py
# -----------------------------------------------------------------------------
# LangGraph: THREATS tab — LLM-first assistant (clarify-first, one-call-per-turn)
#
# This version aligns behavior and UX with Asset/Impact graphs:
#   1) Allowed actions & node names stay canonical:
#        • threat.asset_get (GET)
#        • threat.set_probability (single write)
#        • threat.bulk_set (batch write)
#        • threat.upsert (presence add/remove on asset)
#        • threat.explain (RAG→LLM)
#        (+ threat.list kept for catalog)
#   2) Read "get" node renders compact lines:  CODE: P=level — note
#      with i18n title and agent_events in UI payload.
#   3) All writes submit exactly once per turn via deps.submit_changeset(...).
#      After submit, return:
#        • a short localized LLM confirmation sentence,
#        • {"kind":"changeset", ...},
#        • {"kind":"submit_result", submitted|queued, ...}
#   4) Thin runners so the router doesn’t build confirmations itself:
#        • run_set_probability(...), run_bulk_set(...), run_upsert(...),
#          run_explain_threats(...).
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
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

# UI formatting helpers (preferred). Provide safe fallbacks if unavailable.
try:
    from workshops_ai.ui.formatting import title, bullet_list, chunk, table  # type: ignore
except Exception:  # pragma: no cover
    def title(text: str) -> str:
        return f"# {text}".strip()

    def bullet_list(items: List[str]) -> str:
        return "\n".join(f"- {x}" for x in items)

    def table(headers: List[str], rows: List[List[str]]) -> str:
        if not headers:
            return bullet_list([" | ".join(r) for r in rows])
        sep = "| " + " | ".join(["---"] * len(headers)) + " |"
        head = "| " + " | ".join(headers) + " |"
        body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
        return "\n".join([head, sep, body]) if rows else head + "\n" + sep

    def chunk(text: str, *, max_chars: int = 1800) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        parts, buf, size = [], [], 0
        for line in text.splitlines():
            add = len(line) + 1
            if size + add > max_chars and buf:
                parts.append("\n".join(buf))
                buf, size = [line], add
            else:
                buf.append(line)
                size += add
        if buf:
            parts.append("\n".join(buf))
        return parts

# LangGraph
try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("langgraph must be installed to use threat_graph") from e

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Types & State (LLM-first planner; clarify-first for missing slots)
# -----------------------------------------------------------------------------
Intent = Literal[
    "agent_reply", "clarify", "list", "get", "set", "bulk_set", "upsert", "explain"
]

class RiskItem(BaseModel):
    threat_code: str
    probability: int = Field(ge=0, le=4)  # 0..4 inclusive
    note: Optional[str] = None

class RiskSetPayload(BaseModel):
    asset_id: str
    items: List[RiskItem]

class RiskBulkPayload(BaseModel):
    entries: List[RiskSetPayload]

# Presence upsert (add/remove threat presence on an asset)
class ThreatPresenceItem(BaseModel):
    threatName: str
    present: bool
    note: Optional[str] = None

class ThreatPresencePayload(BaseModel):
    asset_id: str
    items: List[ThreatPresenceItem]

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
    locale: Optional[str] = None       # "cs" | "sk" | "en" | ...
    language: Optional[str] = None
    history: Optional[List[dict]] = None  # short rolling window

    # Structured inputs (from planner or API)
    action: Optional[Intent] = None
    set_payload: Optional[RiskSetPayload] = None
    bulk_payload: Optional[RiskBulkPayload] = None
    upsert_payload: Optional[ThreatPresencePayload] = None
    question: Optional[str] = None

    # Working outputs
    intent: Optional[Intent] = None
    actions: Optional[List[dict]] = None
    change: Optional[ChangeSet | dict] = None
    result: Optional[dict] = None

    # Telemetry
    agent_events: List[ToolEvent] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Utilities: localization (short), telemetry, UI payload
# -----------------------------------------------------------------------------
def _lang(locale: Optional[str], fallback: str = "en") -> str:
    if not locale:
        return fallback
    l = locale.lower()
    if l.startswith(("cs", "cz")):
        return "cs"
    if l.startswith("sk"):
        return "sk"
    if l.startswith("en"):
        return "en"
    return fallback

def _ui_response(content: str | dict, *, extra: Optional[dict] = None) -> dict:
    """
    Standard UIResponse payload (messages + optional structured data).
    - content: string (chat bubble) or dict (UI side panel/event)
    """
    base = {
        "messages": [content] if isinstance(content, str) else [content],
        "proposals": None,
        "staging": {"needs_commit": False, "batch_id": None, "diff": None},
        "policy": {"emit_changes_enabled": True},
    }
    if extra:
        base.update(extra)
    return base

def _now_iso() -> str:
    import datetime as dt
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
    Prefer factory.make_retriever(org_id, 'threats'); fall back to older accessors if present.
    """
    mk = getattr(deps, "make_retriever", None)
    if callable(mk) and org_id:
        try:
            return mk(org_id, "threats")
        except Exception:  # pragma: no cover
            logger.debug("make_retriever(org_id, 'threats') failed", exc_info=True)
    factory = getattr(deps, "retriever_factory", None)
    if callable(factory):
        try:
            return factory("threats")
        except Exception:  # pragma: no cover
            logger.debug("retriever_factory('threats') failed", exc_info=True)
    getter = getattr(deps, "get_retriever", None)
    if callable(getter):
        try:
            return getter("threats")
        except Exception:  # pragma: no cover
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

async def _llm_text_async(*, deps: GraphDeps, messages: List[dict], temperature: float = 0.2) -> str:
    """Async bridge that executes the sync _llm_text in a thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _llm_text(deps=deps, messages=messages, temperature=temperature))


# -----------------------------------------------------------------------------
# Planner (agent_decide → one intent per turn; no hardcoded help/greet)
# -----------------------------------------------------------------------------
def _history_tail(state: ThreatState, k: int = 5) -> List[dict]:
    """Return the last k turns from the rolling history (best-effort)."""
    try:
        return (state.history or [])[-k:]
    except Exception:
        return []

async def _nlu(deps: GraphDeps, state: ThreatState) -> ThreatState:
    """
    Choose a single intent via planner (agent_decide).
    - Honors explicit 'action' if provided.
    - LLM-first: if planner picks no actionable tool, fall back to agent_reply.
    - Parse tool args into typed payloads (Pydantic) where applicable.
    """
    raw = (state.message or "").strip()

    # Explicit passthrough (API-driven)
    if state.action in ("agent_reply", "clarify", "list", "get", "set", "bulk_set", "upsert", "explain"):
        state.intent = cast(Intent, state.action)
        return state

    sel = state.selection or {}

    allowed = [
        "threat.list",
        "threat.asset_get",
        "threat.set_probability",
        "threat.bulk_set",
        "threat.upsert",
        "threat.explain",
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

    # Tool → intent mapping
    tool_to_intent = {
        "threat.list": "list",
        "threat.asset_get": "get",
        "threat.set_probability": "set",
        "threat.bulk_set": "bulk_set",
        "threat.upsert": "upsert",
        "threat.explain": "explain",
    }
    state.intent = cast(Intent, tool_to_intent.get((decision.action or "").strip(), "agent_reply"))

    # Strict args harvesting into Pydantic payloads
    if getattr(decision, "args", None):
        args = decision.args or {}

        if state.intent == "set":
            asset_id = args.get("asset_id") or sel.get("asset_id")
            items = args.get("items")
            if asset_id and isinstance(items, list):
                try:
                    parsed = [RiskItem(**it) for it in items]
                    state.set_payload = RiskSetPayload(asset_id=asset_id, items=parsed)
                except Exception:
                    state.set_payload = None

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

        if state.intent == "upsert":
            # Accept single {threatName,present[,note]} or items:[...]; need asset_id.
            aid = args.get("asset_id") or sel.get("asset_id")
            items_src: List[dict] = []
            if "threatName" in args and "present" in args:
                items_src.append({"threatName": args.get("threatName"), "present": bool(args.get("present")), "note": args.get("note")})
            if isinstance(args.get("items"), list):
                for it in args["items"]:
                    if isinstance(it, dict) and "threatName" in it and "present" in it:
                        items_src.append({"threatName": it.get("threatName"), "present": bool(it.get("present")), "note": it.get("note")})
            if aid and items_src:
                try:
                    parsed_items = [ThreatPresenceItem(**x) for x in items_src]
                    state.upsert_payload = ThreatPresencePayload(asset_id=aid, items=parsed_items)
                except Exception:
                    state.upsert_payload = None

        if state.intent == "explain":
            q = args.get("q") or args.get("threatName") or args.get("question")
            if isinstance(q, str) and q.strip():
                state.question = q.strip()

    # If no explicit content and nothing actionable, still allow a natural reply
    if not raw and state.intent == "agent_reply":
        state.intent = "agent_reply"
    return state


# -----------------------------------------------------------------------------
# Clarify-first helpers (ask ONE concise follow-up)
# -----------------------------------------------------------------------------
def _suggest_select(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    """If no asset is selected, render a compact preview to help the user pick one."""
    def _inner(state: ThreatState) -> ThreatState:
        t0 = _event(state, kind="tool.call.start", tool="suggest_select")
        try:
            backend = _backend(deps)
            items = (backend.assets_list(org_id=state.org_id) if backend else []) or []
            rows: List[List[str]] = []
            for a in items[:10]:  # preview (not a full catalog)
                d = a.model_dump() if hasattr(a, "model_dump") else dict(a)
                rows.append([d.get("id") or "?", d.get("name") or "-", d.get("assetType") or "-"])
            heading = {"cs": "Vyber prosím aktivum", "sk": "Vyber prosím aktívum", "en": "Please select an asset"}[_lang(state.locale)]
            text = "\n\n".join([title(heading), table(["ID", "Name", "Type"], rows)]) if rows else title(heading)
            state.result = _ui_response(text, extra={"agent_events": state.agent_events})
            _finish(state, t0, tool="suggest_select", ok=True, meta={"count": len(rows)})
        except Exception as e:
            _finish(state, t0, tool="suggest_select", ok=False, meta={"error": str(e)})
            msg = {"cs": f"Nepodařilo se načíst seznam aktiv: {e}", "sk": f"Nepodarilo sa načítať zoznam aktív: {e}", "en": f"Failed to load assets: {e}"}[_lang(state.locale)]
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
        return state
    return _inner

def _clarify_once(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    """Ask at most ONE short follow-up question based on missing slot(s)."""
    def _inner(state: ThreatState) -> ThreatState:
        l = _lang(state.locale)

        def _ask(txt_cs: str, txt_sk: str, txt_en: str) -> str:
            return {"cs": txt_cs, "sk": txt_sk}.get(l, txt_en)

        if state.intent == "set":
            if not (state.set_payload and state.set_payload.asset_id):
                if not (state.selection and state.selection.get("asset_id")):
                    return _suggest_select(deps)(state)
                state.result = _ui_response(_ask("Uveď items[{threat_code, probability 0..4}].",
                                                 "Uveď items[{threat_code, probability 0..4}].",
                                                 "Please provide items[{threat_code, probability 0..4}]."),
                                            extra={"agent_events": state.agent_events})
                return state
            if not state.set_payload.items:
                state.result = _ui_response(_ask("Pošli items[{threat_code, probability 0..4}].",
                                                 "Pošli items[{threat_code, probability 0..4}].",
                                                 "Send items[{threat_code, probability 0..4}]."),
                                            extra={"agent_events": state.agent_events})
                return state

        if state.intent == "bulk_set":
            if not (state.bulk_payload and state.bulk_payload.entries):
                state.result = _ui_response(_ask(
                    "Pošli entries[{asset_id, items[{threat_code, probability 0..4}]}].",
                    "Pošli entries[{asset_id, items[{threat_code, probability 0..4}]}].",
                    "Send entries[{asset_id, items[{threat_code, probability 0..4}]}].",
                ), extra={"agent_events": state.agent_events})
                return state

        if state.intent == "upsert":
            if not (state.upsert_payload and state.upsert_payload.items and state.upsert_payload.asset_id):
                if not (state.selection and state.selection.get("asset_id")):
                    return _suggest_select(deps)(state)
                state.result = _ui_response(_ask(
                    "Uveď {threatName, present[, note]} (nebo items[...]).",
                    "Uveď {threatName, present[, note]} (alebo items[...]).",
                    "Provide {threatName, present[, note]} (or items[...]).",
                ), extra={"agent_events": state.agent_events})
                return state

        # Default: agent-style natural reply (no tools)
        return _agent_reply(deps)(state)
    return _inner


# -----------------------------------------------------------------------------
# Read-only nodes
# -----------------------------------------------------------------------------
def _list_threats(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    """
    List the entire THREAT CATALOG (no hidden truncation). Uses ui/formatting (title + bullets).
    Long outputs are chunked to multiple chat messages.
    """
    def _inner(state: ThreatState) -> ThreatState:
        t0 = _event(state, kind="tool.call.start", tool="threat.list")
        try:
            backend = _backend(deps)
            defs = (backend.threat_defs_list(org_id=state.org_id) if backend else []) or []
            lines: List[str] = []
            for r in defs:
                d = r.model_dump() if hasattr(r, "model_dump") else dict(r)
                code = str(d.get("code") or d.get("threat_code") or "").strip()
                name = str(d.get("name") or "").strip()
                cat = str(d.get("category") or "").strip()
                s = f"{code} — {name}" if name else code
                if cat:
                    s += f" [{cat}]"
                lines.append(s)

            heading = {"cs": "Katalog hrozeb", "sk": "Katalóg hrozieb", "en": "Threat catalog"}[_lang(state.locale)]
            text = "\n\n".join([title(f"{heading} ({len(lines)})"), bullet_list(lines) if lines else "_empty_"])
            parts = chunk(text, max_chars=1800)

            state.result = _ui_response(parts[0], extra={"agent_events": state.agent_events, "threats_count": len(lines)})
            if len(parts) > 1:
                state.result["messages"].extend(parts[1:])  # type: ignore[index]
            _finish(state, t0, tool="threat.list", ok=True, meta={"count": len(lines)})
        except Exception as e:
            _finish(state, t0, tool="threat.list", ok=False, meta={"error": str(e)})
            msg = {"cs": f"Katalog hrozeb se nepodařilo načíst: {e}", "sk": f"Katalóg hrozieb sa nepodarilo načítať: {e}", "en": f"Failed to load the threat catalog: {e}"}[_lang(state.locale)]
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
        return state
    return _inner

def _asset_get_risks_or_suggest(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    """
    Show risks for the selected asset, or suggest an asset if none is selected.
    Render as title + bullets; chunk long lists.
    """
    def _inner(state: ThreatState) -> ThreatState:
        asset_id = (state.selection or {}).get("asset_id")
        if not asset_id:
            return _suggest_select(deps)(state)

        t0 = _event(state, kind="tool.call.start", tool="threat.asset_get", meta={"asset_id": asset_id})
        try:
            backend = _backend(deps)
            rows = []
            if backend:
                getter = getattr(backend, "threats_for_asset", None) or getattr(backend, "asset_get_risks", None)
                rows = getter(asset_id) if getter else []
            rows = rows or []

            # Lines like "CODE: P=level — note?"
            lines: List[str] = []
            for it in rows:
                code = str(it.get("threat_code") or it.get("code") or "").strip()
                prob = it.get("probability") if it.get("probability") is not None else it.get("level")
                note = str(it.get("note") or "").strip()
                s = f"{code}: P={prob if prob is not None else '-'}"
                if note:
                    s += f" — {note}"
                lines.append(s)

            heading = {"cs": "Rizika pro aktivum", "sk": "Riziká pre aktívum", "en": "Risks for asset"}[_lang(state.locale)]
            text = "\n\n".join([title(f"{heading} {asset_id} ({len(lines)})"), bullet_list(lines) if lines else "_none_"])
            parts = chunk(text, max_chars=1800)

            state.result = _ui_response(parts[0], extra={"agent_events": state.agent_events, "asset_id": asset_id, "risks_count": len(lines)})
            if len(parts) > 1:
                state.result["messages"].extend(parts[1:])  # type: ignore[index]
            _finish(state, t0, tool="threat.asset_get", ok=True, meta={"count": len(lines)})
        except Exception as e:
            _finish(state, t0, tool="threat.asset_get", ok=False, meta={"error": str(e)})
            msg = {"cs": f"Čtení rizik pro „{asset_id}“ selhalo: {e}", "sk": f"Čítanie rizík pre „{asset_id}“ zlyhalo: {e}", "en": f"Failed to read risks for “{asset_id}”: {e}"}[_lang(state.locale)]
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Explain (RAG→LLM; wrap as title + short bullets)
# -----------------------------------------------------------------------------
async def _explain_threats_async(deps: GraphDeps, state: ThreatState) -> ThreatState:
    """Explain threats via retriever grounding (if available), else concise LLM fallback."""
    query = (state.question or state.message or "").strip() or "threats overview"
    retriever = _get_threats_retriever(deps, state.org_id)

    snippets: List[str] = []
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
                txt = getattr(d, "page_content", None) or getattr(d, "text", "") or str(d)
                if txt:
                    snippets.append(str(txt))
            rag_used = len(snippets) > 0
        except Exception:
            logger.debug("threats retriever failed; falling back to LLM-only", exc_info=True)

    messages: List[dict] = [{"role": "system", "content": EXPLAIN_SYSTEM}]
    if rag_used:
        ctx = "\n\n---\n".join(snippets[:5])
        messages.append({"role": "system", "content": f"Context snippets:\n{ctx}"})
    messages.append({
        "role": "user",
        "content": json.dumps({
            "task": "Summarize in <=6 short bullets; concrete and non-repetitive.",
            "query": query,
            "locale": _lang(state.locale),
        }, ensure_ascii=False)
    })

    answer = await _llm_text_async(deps=deps, messages=messages, temperature=0.2)
    bullets = [ln.strip("-• ").strip() for ln in answer.splitlines() if ln.strip()]
    bullets = bullets if bullets else [answer.strip()]

    heading = {"cs": "Vysvětlení hrozeb", "sk": "Vysvetlenie hrozieb", "en": "Threats — explanation"}[_lang(state.locale)]
    text = "\n\n".join([title(f"{heading}: {query}"), bullet_list(bullets)])
    parts = chunk(text, max_chars=1800)

    state.result = _ui_response(parts[0], extra={
        "agent_events": state.agent_events,
        "rag_used": rag_used,
        "sources_count": min(5, len(snippets)) if rag_used else 0,
    })
    if len(parts) > 1:
        state.result["messages"].extend(parts[1:])  # type: ignore[index]
    return state


# -----------------------------------------------------------------------------
# Write nodes (strictly use provided codes/probabilities; no guessing)
# -----------------------------------------------------------------------------
def _set_probability(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    """Set one or more threat probabilities for a single asset (merge mode)."""
    def _inner(state: ThreatState) -> ThreatState:
        l = _lang(state.locale)
        payload = state.set_payload

        if not payload or not payload.items:
            if not (state.selection and state.selection.get("asset_id")):
                return _suggest_select(deps)(state)
            msg = {"cs": "Není co nastavit. Potřebuji items[{threat_code, probability 0..4}].",
                   "sk": "Nie je čo nastaviť. Potrebujem items[{threat_code, probability 0..4}].",
                   "en": "Nothing to set. Need items[{threat_code, probability 0..4}]."}[l]
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
            return state

        action = {
            "op": "set_risks",
            "mode": "merge",
            "asset_id": payload.asset_id,
            "items": [it.model_dump() for it in payload.items],
        }
        # Optional guard
        try:
            from workshops_ai.tools.guard import guard_validate_payload as _guard_validate  # type: ignore
            _guard_validate("ActionSetRisks", action)  # type: ignore[arg-type]
        except Exception:
            pass

        state.actions = [action]

        # Short pre-submit echo
        lines = [f"{it.threat_code}=P{int(max(0, min(4, it.probability)))}" + (f" — {it.note}" if it.note else "") for it in payload.items]
        heading = {"cs": f"Připraveno nastavit pro „{payload.asset_id}“",
                   "sk": f"Pripravené nastaviť pre „{payload.asset_id}“",
                   "en": f"Ready to set for “{payload.asset_id}”"}[l]
        text = "\n\n".join([title(heading), bullet_list(lines)])
        state.result = _ui_response(text, extra={"agent_events": state.agent_events})
        return state
    return _inner

def _bulk_set_probability(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    """Bulk set threat probabilities across multiple assets (merge mode)."""
    def _inner(state: ThreatState) -> ThreatState:
        l = _lang(state.locale)
        bulk = state.bulk_payload
        if not bulk or not bulk.entries:
            msg = {"cs": "Chybí entries pro hromadné nastavení.",
                   "sk": "Chýbajú entries pre hromadné nastavenie.",
                   "en": "Missing entries for bulk update."}[l]
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
            return state

        actions: List[dict] = []
        preview_lines: List[str] = []
        for e in bulk.entries:
            if not e.asset_id or not e.items:
                continue
            actions.append({
                "op": "set_risks",
                "mode": "merge",
                "asset_id": e.asset_id,
                "items": [it.model_dump() for it in e.items],
            })
            per_asset = ", ".join(f"{it.threat_code}=P{int(max(0, min(4, it.probability)))}" for it in e.items)
            preview_lines.append(f"{e.asset_id}: {per_asset}")

        if not actions:
            msg = {"cs": "Chybí entries pro hromadné nastavení.",
                   "sk": "Chýbajú entries pre hromadné nastavenie.",
                   "en": "Missing entries for bulk update."}[l]
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
            return state

        state.actions = actions
        heading = {"cs": "Připraveno k aplikaci (hromadné nastavení)",
                   "sk": "Pripravené na aplikáciu (hromadné nastavenie)",
                   "en": "Ready to apply (bulk set)"}[l]
        text = "\n\n".join([title(heading), bullet_list(preview_lines)])
        state.result = _ui_response(text, extra={"agent_events": state.agent_events})
        return state
    return _inner

def _upsert_presence(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    """Ensure presence (present=true) or remove presence (present=false) of threat(s) on an asset."""
    def _inner(state: ThreatState) -> ThreatState:
        l = _lang(state.locale)

        payload = state.upsert_payload
        if not payload or not payload.asset_id or not payload.items:
            if not (state.selection and state.selection.get("asset_id")):
                return _suggest_select(deps)(state)
            msg = {"cs": "Uveď {threatName, present[, note]} (nebo items[...]).",
                   "sk": "Uveď {threatName, present[, note]} (alebo items[...]).",
                   "en": "Provide {threatName, present[, note]} (or items[...])."}[l]
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
            return state

        # Build one asset-scoped presence action per item
        actions: List[dict] = []
        for it in payload.items:
            code = (it.threatName or "").strip()
            if not code:
                continue
            action = {
                "op": "asset_upsert_threat_presence",
                "asset_id": payload.asset_id,
                "threat_code": code,
                "present": bool(it.present),
                "note": (it.note or None),
            }
            # Optional guard (safe best-effort)
            try:
                from workshops_ai.tools.guard import guard_validate_payload as _guard_validate  # type: ignore
                _guard_validate("ActionAssetUpsertThreatPresence", action)  # type: ignore[arg-type]
            except Exception:
                pass
            actions.append(action)

        state.actions = actions

        # Short pre-submit echo
        lines = [f"{it.threatName}: {'present' if it.present else 'absent'}" + (f" — {it.note}" if it.note else "") for it in payload.items]
        heading = {"cs": f"Připraveno upravit přítomnost pro „{payload.asset_id}“",
                   "sk": f"Pripravené upraviť prítomnosť pre „{payload.asset_id}“",
                   "en": f"Ready to upsert presence for “{payload.asset_id}”"}[l]
        text = "\n\n".join([title(heading), bullet_list(lines)])
        state.result = _ui_response(text, extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# LLM-confirmation (short, localized) + unified submit result messages
# -----------------------------------------------------------------------------
async def _llm_confirm_async(
    deps: GraphDeps,
    *,
    locale: Optional[str],
    actions: List[dict],
    submitted: bool,
    fallback_path: Optional[str],
) -> str:
    """
    Ask the LLM to produce a single short confirmation sentence (<= 20 words),
    localized to locale ("cs"/"sk"/"en"). If LLM bridge fails, return a safe fallback.
    """
    l = _lang(locale)
    n_set = sum(1 for a in actions if a.get("op") == "set_risks")
    n_presence = sum(1 for a in actions if a.get("op") == "asset_upsert_threat_presence")

    payload = {
        "locale": l,
        "submitted": bool(submitted),
        "queued": not bool(submitted),
        "counts": {"probability_updates": n_set, "presence_upserts": n_presence},
        "hint": "If queued, mention it briefly; otherwise confirm saved. One sentence; <=20 words.",
    }

    sys = (
        "You generate a single, very short confirmation sentence for end users. "
        "No extra punctuation, no emojis, no technical jargon."
    )
    user = json.dumps(payload, ensure_ascii=False)

    try:
        txt = await _llm_text_async(
            deps=deps,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.1,
        )
        if txt:
            return txt.strip()
    except Exception:
        pass

    # Deterministic localized fallback
    if submitted:
        return {"cs": f"Uloženo ({n_set} pravděpodobností, {n_presence} změn přítomnosti).",
                "sk": f"Uložené ({n_set} pravdepodobností, {n_presence} zmien prítomnosti).",
                "en": f"Saved ({n_set} probabilities, {n_presence} presence updates)."}[l]
    else:
        return {"cs": f"Zařazeno do fronty ({n_set} pravděpodobností, {n_presence} změn přítomnosti).",
                "sk": f"Zaradené do fronty ({n_set} pravdepodobností, {n_presence} zmien prítomnosti).",
                "en": f"Queued ({n_set} probabilities, {n_presence} presence updates)."}[l]


# -----------------------------------------------------------------------------
# Submit (single submit per turn) — unified with Asset/Impact (submit_changeset)
# -----------------------------------------------------------------------------
async def _submit_changeset_async(deps: GraphDeps, state: ThreatState) -> ThreatState:
    """Submit accumulated actions as a typed ChangeSet via deps.submit_changeset (with OUTBOX fallback handled inside)."""
    l = _lang(state.locale)
    acts = state.actions or []
    if not acts:
        state.result = _ui_response({"cs": "Žádné změny k uložení.", "sk": "Žiadne zmeny na uloženie.", "en": "No actions to apply."}[l], extra={"agent_events": state.agent_events})
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
        msg = {"cs": "submit_changeset není k dispozici.", "sk": "submit_changeset nie je k dispozícii.", "en": "submit_changeset not available."}[l]
        state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
        return state

    t0 = _event(state, kind="tool.call.start", tool="submit_changeset", meta={"actions": len(acts)})
    try:
        cs_typed = cs if hasattr(cs, "model_dump") else ChangeSet.model_validate(cs)
        submit_result = await submit_fn(cs_typed, require_success=False)  # factory handles online→OUTBOX fallback
        ok = bool(getattr(submit_result, "submitted", False))
        fallback_path = getattr(submit_result, "fallback_path", None)
        _finish(state, t0, tool="submit_changeset", ok=ok, meta={"actions": len(acts), "fallback_path": str(fallback_path) if fallback_path else None})

        # LLM-powered short confirmation
        confirm = await _llm_confirm_async(
            deps,
            locale=state.locale,
            actions=acts,
            submitted=ok,
            fallback_path=str(fallback_path) if fallback_path else None,
        )

        # Changeset dump (best effort)
        try:
            cs_dump = cs_typed.model_dump(mode="json") if hasattr(cs_typed, "model_dump") else dict(cs_typed)  # type: ignore[arg-type]
        except Exception:
            cs_dump = {"_error": "cannot_dump_changeset"}

        # Unified messages: confirm + structured changeset + structured submit_result
        messages: List[Any] = [
            confirm,
            {"kind": "changeset", "changeset": cs_dump, "ts": _now_iso()},
            {"kind": "submit_result", "submitted": ok, "queued": (not ok), "fallback_path": str(fallback_path) if fallback_path else None},
        ]

        state.result = _ui_response(confirm, extra={
            "changeset_actions": len(acts),
            "agent_events": state.agent_events,
            "messages": messages,  # override with the complete messages list for UI
        })
        state.actions = None
        state.change = None
        return state

    except Exception as e:
        _finish(state, t0, tool="submit_changeset", ok=False, meta={"error": str(e)})
        msg = {"cs": f"Odeslání selhalo: {e}", "sk": f"Odoslanie zlyhalo: {e}", "en": f"Submission failed: {e}"}[l]
        state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
        return state


# -----------------------------------------------------------------------------
# Agent-only reply (no tools) — concise, friendly, domain-aware
# -----------------------------------------------------------------------------
def _agent_reply(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    """
    Produce a concise assistant reply with no tool usage.
    Catches greetings/chitchat/generic questions and keeps the conversation natural.
    """
    def _inner(state: ThreatState) -> ThreatState:
        l = _lang(state.locale)
        system = (
            "You are a helpful assistant for a 'Threats' tab (risk management).\n"
            "- Be concise (<=80 words), neutral, and specific.\n"
            "- If relevant, mention you can list the threat catalog, show risks for an asset,\n"
            "  set probabilities (single or bulk), upsert presence, or explain threat concepts.\n"
            "- Do not claim actions were taken."
        )
        user = state.message or ""
        try:
            ans = _llm_text(deps=deps, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.3)
        except Exception:
            ans = {"cs": "Jak mohu s hrozbami pomoci?", "sk": "Ako môžem s hrozbami pomôcť?", "en": "How can I help with threats?"}[l]
        state.result = _ui_response(ans, extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Graph assembly
# -----------------------------------------------------------------------------
def create_graph(deps: GraphDeps):
    """
    Flow:
      start → nlu → route → {agent.reply | clarify | list | get | set | bulk_set | upsert | explain} → (submit?) → END

    - One decisive step per turn; writes always go through a single submit.
    - No hardcoded greet/help/chitchat nodes; agent.reply handles non-tool answers.
    - Catalog lists are not truncated (only chunked for UI limits).
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

    graph.add_node("agent.reply", _agent_reply(deps))
    graph.add_node("clarify", _clarify_once(deps))
    graph.add_node("suggest_select", _suggest_select(deps))

    # read
    graph.add_node("threat.list", _list_threats(deps))
    graph.add_node("threat.asset_get", _asset_get_risks_or_suggest(deps))

    # write
    graph.add_node("threat.set_probability", _set_probability(deps))
    graph.add_node("threat.bulk_set", _bulk_set_probability(deps))
    graph.add_node("threat.upsert", _upsert_presence(deps))

    # explain
    graph.add_node("threat.explain", _explain_node)

    # submit
    graph.add_node("submit_changeset", _submit_node)

    # Entry
    graph.set_entry_point("nlu")
    graph.add_edge("nlu", "route")

    # Router
    def _route(state: ThreatState) -> str:
        intent = state.intent or "agent_reply"
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
        return "agent.reply"

    graph.add_conditional_edges(
        "route",
        _route,
        {
            "agent.reply": "agent.reply",
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

    # Terminal nodes
    for node in ("agent.reply", "clarify", "threat.list", "threat.asset_get", "threat.explain", "submit_changeset"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=deps.checkpointer)


# -----------------------------------------------------------------------------
# Thin runners used by routers (compat)
# -----------------------------------------------------------------------------
async def run_explain(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

async def run_explain_threats(*, deps: GraphDeps, text: str, locale: Optional[str] = None, org_id: Optional[str] = None) -> dict:
    """Router-facing explain for THREATS that returns a UI payload dict."""
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

async def run_set_probability(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    """Thin runner so the router doesn't assemble confirmations itself."""
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

async def run_bulk_set(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    """Thin runner so the router doesn't assemble confirmations itself."""
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

async def run_upsert(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    """Thin runner so the router doesn't assemble confirmations itself."""
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)


__all__ = [
    "ThreatState",
    "create_graph",
    "run_explain",
    "run_explain_threats",
    "run_set_probability",
    "run_bulk_set",
    "run_upsert",
]
