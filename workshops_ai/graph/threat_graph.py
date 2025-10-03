# File: workshops_ai/graph/threat_graph.py
# -----------------------------------------------------------------------------
# LangGraph: THREATS tab — LLM-first conversational assistant with SEND-NOW writes
#
# Parity with Impact graph:
# - LLM-first NLU (clarify-first fallback) → deterministic routing.
# - No preview/staging/confirm flows — writes are SEND-NOW.
# - Suggest-select when asset_id is missing.
# - Optimistic WS emits (batch-preferred, per-item fallback) before submit.
# - Deterministic submit with retries/backoff + outbox fallback.
# - Short multilingual replies (SK/CZ/EN) and compact outputs.
# - Optional RAG → LLM "explain" (uses deps.make_retriever if present).
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
import time
import unicodedata
import uuid

from typing import Any, Callable, Literal, Optional, cast
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, model_validator
from workshops_ai.app.services.outbox_writer import submit_or_fallback, SubmitOrFallbackResult
from workshops_ai.graph.factory import GraphDeps

logger = logging.getLogger(__name__)

# Optional DTO + mappers for WS compatibility (kept, but not required)
try:
    from workshops_ai.api.schema.threat import AssetRisk  # type: ignore
    from workshops_ai.agents.utils.mappers import map_asset_risk_to_ws_event  # type: ignore
except Exception:  # pragma: no cover
    AssetRisk = object  # type: ignore

    def map_asset_risk_to_ws_event(_dto):  # type: ignore
        return {}

# LangGraph
try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("langgraph must be installed to use threat_graph") from e

# Optional LLM (planner + explain)
_HAS_LLM = True
try:
    from langchain_openai import ChatOpenAI  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
except Exception:  # pragma: no cover
    _HAS_LLM = False
    ChatOpenAI = object  # type: ignore
    SystemMessage = HumanMessage = object  # type: ignore

# -----------------------------------------------------------------------------
# ChangeSet & Guard (shared builder, soft guard adapter) — THREAT GRAPH
# -----------------------------------------------------------------------------
try:
    # Use the unified models + deterministic builder from models/changeset.py
    from workshops_ai.models.changeset import (  # type: ignore
        ChangeSet,
        build_changeset,
    )
except Exception:  # pragma: no cover
    # Dev fallback: return a dict that matches the ChangeSet JSON shape
    ChangeSet = Any  # type: ignore

    def build_changeset(  # type: ignore
        *,
        org_id: str,
        user_id: str,
        actions: list[dict] | list[Any],
        intent: str = "apply",
        idempotency_key: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        batch_id: str | None = None,
        source: str = "AI",
        tags: list[str] | None = None,
    ) -> dict:
        return {
            "schema_version": "1.0",
            "idempotency_key": idempotency_key
            or f"cs:fallback:{org_id}:{user_id}:{intent}:{uuid.uuid4()}",
            "org_id": org_id,
            "intent": "preview" if intent == "preview" else "apply",
            "actions": actions,
            "audit": {
                "user_id": user_id,
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source": source,
                "tags": tags or [],
                "org_id": org_id,
                "session_id": session_id,
                "request_id": request_id,
                "batch_id": batch_id,
            },
        }

# Optional guard (soft dependency; tolerant adapter)
try:
    from workshops_ai.tools.guard import guard_validate_payload as _guard_validate  # type: ignore
except Exception:  # pragma: no cover
    def _guard_validate(model_cls, payload):  # type: ignore
        return payload

def _guard(model_hint: Any, payload: dict) -> dict:
    """Adapter that tolerates different guard signatures or missing guard module."""
    try:
        return _guard_validate(model_hint, payload)
    except TypeError:
        try:
            return _guard_validate(payload)
        except Exception:
            return payload


# -----------------------------------------------------------------------------
# Types & State
# -----------------------------------------------------------------------------

Intent = Literal["help", "clarify", "chitchat", "list", "get", "set", "bulk_set", "upsert", "explain"]

class RiskItem(BaseModel):
    threat_code: str
    probability: int = Field(ge=0, le=4)
    note: Optional[str] = None

class RiskSetPayload(BaseModel):
    asset_id: str
    items: list[RiskItem]

class RiskBulkPayload(BaseModel):
    entries: list[RiskSetPayload]

class UpsertThreatItem(BaseModel):
    code: Optional[str] = None               # původní threat_code -> code
    name: str
    category: Optional[str] = None
    asset_type_codes: list[str] = Field(default_factory=list)
    intentional: bool = False
    unintentional: bool = False
    environmental: bool = False
    description: Optional[str] = None

class ToolEvent(TypedDict, total=False):
    event_type: str            # "tool.call.start" | "tool.call.end"
    tool: str                  # tool/node name
    ok: bool                   # success flag
    duration_ms: int           # only for "end"
    meta: dict                 # arbitrary metadata

class ThreatState(BaseModel):
    # Request context
    org_id: str
    active_tab: str = Field(default="threats")
    message: Optional[str] = None
    selection: Optional[dict] = None   # {"asset_id": "...", "asset_type": "..."} | ...
    user_id: Optional[str] = None
    locale: Optional[str] = None       # "sk" | "cs" | "en" (or full locales)
    language: Optional[str] = None
    history: Optional[list[dict]] = None  # short rolling window (3–5 turns), optional

    # Structured inputs (planner results / API callers may prefill)
    action: Optional[Intent] = None
    set_payload: Optional[RiskSetPayload] = None
    bulk_payload: Optional[RiskBulkPayload] = None
    upserts: Optional[list[UpsertThreatItem]] = None
    question: Optional[str] = None  # for explain

    # Working outputs
    intent: Optional[Intent] = None
    actions: Optional[list[dict]] = None
    change: Optional[ChangeSet | dict] = None
    result: Optional[dict] = None

    # Telemetry
    agent_events: list[ToolEvent] = Field(default_factory=list)

# -----------------------------------------------------------------------------
# i18n helpers (short & friendly)
# -----------------------------------------------------------------------------

def _lang(locale: Optional[str], fallback: str = "en") -> str:
    if not locale:
        return fallback
    l = locale.lower()
    if l.startswith("sk"):
        return "sk"
    if l.startswith("cs") or l.startswith("cz"):
        return "cs"
    if l.startswith("en"):
        return "en"
    return fallback

def _t(loc: str, key: str, **kw) -> str:
    M = {
        "greet": {
            "sk": "Ahoj! Môžem: „zoznam hrozieb“, „ukáž riziká pre aktívum“, „nastav P“, „hromadne nastav“, „vysvetli hrozby“. 😉",
            "cs": "Ahoj! Umím: „seznam hrozeb“, „ukaž rizika pro aktivum“, „nastav P“, „hromadně nastav“, „vysvětli hrozby“. 😉",
            "en": "Hi! I can: “list threats”, “show risks for an asset”, “set P”, “bulk set”, or “explain threats”. 😉",
        },
        "catalog_fail": {
            "sk": "Katalóg hrozieb sa nepodarilo načítať: {err}",
            "cs": "Katalog hrozeb se nepodařilo načíst: {err}",
            "en": "Failed to load the threat catalog: {err}",
        },
        "catalog_found": {
            "sk": "Katalóg hrozieb ({n}){flt}:",
            "cs": "Katalog hrozeb ({n}){flt}:",
            "en": "Threat catalog ({n}){flt}:",
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
            "sk": "Nie je čo nastaviť. Uveď asset_id a aspoň jedno P (0..4).",
            "cs": "Není co nastavit. Uveď asset_id a alespoň jedno P (0..4).",
            "en": "Nothing to set. Provide an asset_id and at least one P (0..4).",
        },
        "bulk_missing": {
            "sk": "Chýbajú položky pre hromadné nastavenie.",
            "cs": "Chybí položky pro hromadné nastavení.",
            "en": "Missing entries for bulk update.",
        },
        "upsert_missing": {
            "sk": "Zadaj upsert_payload s „threat_code“.",
            "cs": "Zadej upsert_payload s „threat_code“.",
            "en": "Provide upsert_payload with a 'threat_code'.",
        },
        "validation_error": {
            "sk": "Chyba validácie: {err}",
            "cs": "Chyba validace: {err}",
            "en": "Validation error: {err}",
        },
        "submit_failed": {
            "sk": "Odoslanie zlyhalo, zmeny boli uložené lokálne. Dôvod: {err}",
            "cs": "Odeslání selhalo, změny byly uloženy lokálně. Důvod: {err}",
            "en": "Submission failed; changes were queued locally. Reason: {err}",
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
        "upsert_missing": {
            "sk": "Zadaj upserts s aspoň jednou položkou (min. name; code je voliteľné).",
            "cs": "Zadej upserts s alespoň jednou položkou (min. name; code je volitelné).",
            "en": "Provide upserts with at least one item (name required; code optional).",
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
# Utilities
# -----------------------------------------------------------------------------

_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)
_NORM_LEVEL_RX = re.compile(r"\b(prob(ability)?|pravdepodobnost|pravdepodobnosť|p)\s*[:=]?\s*([0-4])\b", re.I)
_THREAT_CODE_RX = re.compile(r"\b([A-Z][A-Z0-9_]{1,31})\b")

def _norm(text: str) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii").lower()

def _schedule(coro_or_fn):
    try:
        loop = asyncio.get_running_loop()
        if asyncio.iscoroutine(coro_or_fn):
            loop.create_task(coro_or_fn)
        else:
            loop.create_task(coro_or_fn())
    except RuntimeError:
        if asyncio.iscoroutine(coro_or_fn):
            asyncio.run(coro_or_fn)
        else:
            asyncio.run(coro_or_fn())


async def _amemory_set(deps: GraphDeps, key: str, value: Any) -> None:
    fn = getattr(deps, "memory_set", None)
    if callable(fn):
        r = fn(key, value)
        if asyncio.iscoroutine(r):
            await r

def _now_iso() -> str:
    return dt.datetime.now().replace(tzinfo=dt.timezone.utc).isoformat()

# -----------------------------------------------------------------------------
# Telemetry helpers
# -----------------------------------------------------------------------------

def _event(state: ThreatState, *, kind: str, tool: str, ok: bool = True, meta: dict | None = None) -> float:
    t0 = time.perf_counter()
    state.agent_events.append({"event_type": kind, "tool": tool, "ok": ok, "meta": meta or {}})
    return t0

def _finish(state: ThreatState, t0: float, *, tool: str, ok: bool, meta: dict | None = None) -> None:
    dur = int((time.perf_counter() - t0) * 1000)
    state.agent_events.append({"event_type": "tool.call.end", "tool": tool, "ok": ok, "duration_ms": dur, "meta": meta or {}})

# -----------------------------------------------------------------------------
# Planner (NLU) — LLM-first with regex fallback (clarify-first)
# -----------------------------------------------------------------------------

HELP_RX     = re.compile(r"^\s*(help|pomoc|napoveda|nápověda)\s*$", re.I)
GREET_RX    = re.compile(r"^\s*(ahoj|cau|čau|hej|nazdar|dobr(y|ý)\s+de[nň]|hello|hi|hey|good\s+(morning|afternoon|evening))[\s!.]*$", re.I)
LIST_RX     = re.compile(r"\b(list|seznam|zoznam|catalog|katalog|katal[oó]g|threats)\b", re.I)
GET_RX      = re.compile(r"\b(get|show|detail|ukaz|zobraz)\b", re.I)
SET_RX      = re.compile(r"\b(set|update|nastav|zmen|uprav|prob|probability|pravdepodobnost|pravdepodobnosť|risk|riziko)\b", re.I)
BULK_RX     = re.compile(r"\b(bulk|hromadn[eě]|vsetky|všetky|all\s+assets)\b", re.I)
UPSERT_RX   = re.compile(r"(?:^|\b)(?:vytvoř(?:it)?|přidej|pridaj|create|add|new)\b.*\b(impact|dopad|threat|hrozb[auy]?)?\b", re.I)
EXPLAIN_RX  = re.compile(r"\b(explain|vysvetli|vysvětli|preco|pro[cč]|\bwhy\b|context|kontext)\b", re.I)

class NLUResult(BaseModel):
    action: Optional[Intent] = None
    # For 'set'
    asset_id: Optional[str] = None
    items: Optional[list[RiskItem]] = None
    # For 'bulk_set'
    entries: Optional[list[RiskSetPayload]] = None
    # For 'upsert'
    upserts: Optional[list[UpsertThreatItem]] = None
    # For 'list' (optional filter)
    asset_type: Optional[str] = None
    # For 'explain'
    question: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _defaults(cls, v):
        d = dict(v or {})
        ups = d.get("upserts")
        if ups is None:
            return d
        fixed = []
        for u in ups or []:
            if u is None:
                continue
            x = dict(u)
            # None→[] pro list
            if x.get("asset_type_codes") is None:
                x["asset_type_codes"] = []
            # None→False pro bool
            for b in ("intentional", "unintentional", "environmental"):
                if x.get(b) is None:
                    x[b] = False
            fixed.append(x)
        d["upserts"] = fixed
        return d

def _get_llm_from_deps(deps: GraphDeps):
    """
    Resolve an async-callable LLM client.

    - Prefer deps.llm_client (a ChatOpenAI-like client exposing .ainvoke).
    - Do NOT return deps.llm (settings object).
    - Finally, build a default ChatOpenAI client if available.
    """
    llm_client = getattr(deps, "llm_client", None)
    if llm_client is not None:
        return llm_client
    if not _HAS_LLM:
        return None
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, timeout=12, max_retries=1)  # type: ignore[call-arg]

def _parse_set_from_text(text: str, selection_id: Optional[str]) -> tuple[Optional[str], list[RiskItem]]:
    # asset_id from selection or inline UUID
    m_uuid = _UUID_RE.search(text or "")
    asset_id = selection_id or (m_uuid.group(0) if m_uuid else None)
    items: list[RiskItem] = []
    m_code = _THREAT_CODE_RX.search(text or "")
    m_prob = _NORM_LEVEL_RX.search(text or "")
    if m_code and m_prob:
        items.append(RiskItem(threat_code=m_code.group(1).upper(), probability=int(m_prob.group(3))))
    return asset_id, items

async def _nlu(deps: GraphDeps, state: ThreatState) -> ThreatState:
    raw = (state.message or "").strip()
    selection = state.selection or {}
    selection_id = selection.get("asset_id")
    if state.action == "defs_create":
        state.intent = "upsert"
        return state

    # Shortcut: explicit action provided
    if state.action in ("help", "chitchat", "list", "get", "set", "bulk_set", "upsert", "explain"):
        state.intent = cast(Intent, state.action)
        return state

    # Greeting/help
    if not raw or GREET_RX.match(raw) or HELP_RX.match(raw):
        state.intent = "help"
        return state

    llm = _get_llm_from_deps(deps)

    if llm is not None:
        system = (
            "You are a strict NLU for a 'threats' assistant.\n"
            "Primary languages: Slovak (SK) and English (EN); Czech (CZ) tolerated.\n"
            "Extract intent and slots and return ONLY compact JSON with fields:\n"
            "  action: one of [help, chitchat, list, get, set, bulk_set, upsert, explain]\n"
            "  asset_type?: string                                                   # for list (filter)\n"
            "  asset_id?: string                                                     # for get/set\n"
            "  items?: [{threat_code:string, probability:0..4, note?}]               # for set\n"
            "  entries?: [{asset_id:string, items:[...]}]                             # for bulk_set\n"
            "  upserts?: [{code?,name,category?,asset_type_codes?,intentional?,unintentional?,environmental?,description?}]  # for upsert\n"                    
            "  question?: string                                                      # for explain\n"
            "Rules:\n"
            "- Never invent IDs.\n"
            "- If a UI selection.asset_id is present, you MAY copy it to asset_id when helpful.\n"
            "- Detect greetings as 'chitchat'/'help'.\n"
            "- Probability numeric scale is 0..4.\n"
            "- Output STRICT JSON; no prose."
        )
        human = {
            "message": raw,
            "selection": {"asset_id": selection_id, "asset_type": selection.get("asset_type")} if selection else None,
            "locale": state.locale,
            "history": (state.history or [])[-5:] or None,
        }
        try:
            resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=json.dumps(human))])  # type: ignore
            txt = (getattr(resp, "content", None) or "").strip()
            start, end = txt.find("{"), txt.rfind("}")
            data = json.loads(txt[start:end + 1] if start != -1 and end != -1 else txt)
            n = NLUResult(**data)

            # Fill into state if present
            if n.action:
                state.intent = n.action
            if n.asset_type:
                state.selection = {**selection, "asset_type": n.asset_type}
            if n.asset_id and n.items and (state.set_payload is None):
                state.set_payload = RiskSetPayload(asset_id=n.asset_id, items=n.items)
            if n.entries and (state.bulk_payload is None):
                state.bulk_payload = RiskBulkPayload(entries=n.entries)
            if n.upserts and (state.upserts is None):
                state.upserts = n.upserts
            if n.question and not state.question:
                state.question = n.question

            # Best-effort parse for 'set' if slots are incomplete
            if (state.intent == "set") and (state.set_payload is None):
                aid, items = _parse_set_from_text(raw, selection_id)
                if aid and items:
                    state.set_payload = RiskSetPayload(asset_id=aid, items=items)

            if state.intent:
                return state
        except Exception as e:
            logger.debug("Threats NLU LLM failed, fallback to regex: %s", e)

    # Regex fallback (clarify-first)
    t = _norm(raw)
    if EXPLAIN_RX.search(t):
        state.intent = "explain"
    elif UPSERT_RX.search(t):
        state.intent = "upsert"
    elif BULK_RX.search(t):
        state.intent = "bulk_set"
    elif SET_RX.search(t):
        state.intent = "set"
    elif GET_RX.search(t):
        state.intent = "get"
    elif LIST_RX.search(t):
        state.intent = "list"
    else:
        state.intent = "clarify"

    if state.intent == "set" and state.set_payload is None:
        aid, items = _parse_set_from_text(raw, selection_id)
        if aid and items:
            state.set_payload = RiskSetPayload(asset_id=aid, items=items)

    return state

# -----------------------------------------------------------------------------
# Suggest-select (if missing asset selection) & Clarify
# -----------------------------------------------------------------------------

def _suggest_select(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="suggest_select")
        try:
            items = deps.backend_port.assets_list(org_id=state.org_id) or []
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
            msg = "\n".join([_t(loc, "select_prompt")] + preview) if preview else _t(loc, "catalog_found", n=0, flt="")
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events, "assets_preview": preview})
            _finish(state, t0, tool="suggest_select", ok=True, meta={"count": len(preview)})
        except Exception as e:
            _finish(state, t0, tool="suggest_select", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "catalog_fail", err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner

def _clarify() -> Callable[[ThreatState], ThreatState]:
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)
        # Lightweight binary clarify (keep short)
        msg = state.message or ""
        t = _norm(msg)
        if EXPLAIN_RX.search(t):
            a, b = "EXPLAIN", "GET"
        elif _NORM_LEVEL_RX.search(msg) or _THREAT_CODE_RX.search(msg):
            a, b = "SET", "GET"
        else:
            a, b = "GET", "LIST"
        txt = _t(loc, "greet") + f"\n→ {a}/{b}?"
        state.result = _ui_response(txt, extra={"agent_events": state.agent_events})
        return state
    return _inner

# -----------------------------------------------------------------------------
# Read-only nodes (catalog/risk reads)
# -----------------------------------------------------------------------------

def _list_threats(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="list_threats")
        try:
            sel = state.selection or {}
            asset_type = sel.get("asset_type")
            rows = deps.backend_port.list_threats(asset_type=asset_type) or []
            lines = [_t(loc, "catalog_found", n=len(rows), flt=(f" for type={asset_type}" if asset_type else ""))]
            for r in rows[:30]:
                if isinstance(r, dict):
                    code = r.get("threat_code") or r.get("code") or r.get("EN Code") or "?"
                    name = r.get("name") or r.get("EN Name") or r.get("Threat") or ""
                else:
                    code = getattr(r, "threat_code", None) or getattr(r, "code", None) or "?"
                    name = getattr(r, "name", None) or ""
                lines.append(f"- {code}: {name}")
            state.result = _ui_response("\n".join(lines), extra={"threats": rows, "agent_events": state.agent_events})
            _finish(state, t0, tool="list_threats", ok=True, meta={"count": len(rows), "asset_type": asset_type})
        except Exception as e:
            _finish(state, t0, tool="list_threats", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "catalog_fail", err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner

def _asset_get_risks_or_suggest(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)
        asset_id = (state.selection or {}).get("asset_id")
        if not asset_id:
            return _suggest_select(deps)(state)

        t0 = _event(state, kind="tool.call.start", tool="asset_get_risks", meta={"asset_id": asset_id})
        try:
            rows = deps.backend_port.asset_get_risks(asset_id) or []
            lines = [f"Risks for {asset_id} ({len(rows)}):"]
            for it in rows[:30]:
                code = it.get("threat_code") or it.get("code")
                prob = it.get("probability")
                lines.append(f"- {code}: P={prob}")
            state.result = _ui_response("\n".join(lines), extra={"asset_id": asset_id, "risks": rows, "agent_events": state.agent_events})
            _finish(state, t0, tool="asset_get_risks", ok=True, meta={"count": len(rows)})
        except Exception as e:
            _finish(state, t0, tool="asset_get_risks", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "asset_risks_fail", id=asset_id, err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner

# -----------------------------------------------------------------------------
# WS emit helpers (batch preferred, per-item fallback)
# -----------------------------------------------------------------------------

async def _emit_ws_from_dto(dto: AssetRisk, deps: GraphDeps, org_id: str, actor: Optional[str]) -> None:
    """
    Prefer a single list-shaped event via aemit_risk_set_event(dto-based payload).
    If not available, fall back to per-item aemit_risk_set_probability with the
    schema-correct envelope: {"event":"risk.set_probability","payload":{...}}.
    """
    hub = getattr(deps, "ws_hub", None)
    if hub is None:
        return

    # 1) Try batch/list-shaped emitter
    try:
        ws_batch = map_asset_risk_to_ws_event(dto)
    except Exception:
        ws_batch = None

    if ws_batch and hasattr(hub, "aemit_risk_set_event"):
        try:
            await hub.aemit_risk_set_event(ws_batch)  # type: ignore[attr-defined]
            return
        except Exception as e:  # pragma: no cover
            logger.debug("WS batch emit failed, falling back to per-item: %s", e)

    # 2) Fallback: per-item
    if hasattr(hub, "aemit_risk_set_probability"):
        try:
            asset_id = str(getattr(dto, "assetId", None) or "")
            for r in (getattr(dto, "risks", None) or []):
                evt = {
                    "event": "risk.set_probability",
                    "payload": {
                        "org_id": org_id,
                        "asset_id": asset_id,
                        "threat_code": getattr(r, "threatName", None),
                        "probability": int(getattr(r, "probability", 0)),
                        **({"note": getattr(r, "note")} if getattr(r, "note", None) else {}),
                        "actor": actor,
                        "ts": _now_iso(),
                    },
                }
                try:
                    _guard("WSRiskSetProbability", evt)
                except Exception:
                    pass
                await hub.aemit_risk_set_probability(evt)  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover
            logger.debug("WS per-item emit failed: %s", e)

# -----------------------------------------------------------------------------
# Write nodes (emit-first) → actions → submit
# -----------------------------------------------------------------------------

def _adapt_set_payload_to_dto(payload: RiskSetPayload, *, asset_name: Optional[str] = None) -> AssetRisk:
    risks = [{
        "threatName": it.threat_code.upper(),
        "probability": max(0, min(4, int(it.probability)))
    } for it in payload.items]
    dto_raw = {"assetId": payload.asset_id, "assetName": asset_name, "risks": risks}
    try:
        return AssetRisk.model_validate(dto_raw)  # type: ignore[attr-defined]
    except Exception:
        return cast(AssetRisk, dto_raw)  # type: ignore[return-value]

def _set_risk_probability(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)
        payload = state.set_payload

        # Fill from inline text if needed
        if payload is None:
            aid, items = _parse_set_from_text(state.message or "", (state.selection or {}).get("asset_id"))
            if aid and items:
                payload = RiskSetPayload(asset_id=aid, items=items)

        # Presence checks
        if not payload or not payload.items:
            if not (state.selection and state.selection.get("asset_id")):
                return _suggest_select(deps)(state)
            state.result = _ui_response(_t(loc, "nothing_to_set"), extra={"agent_events": state.agent_events})
            return state

        # Normalize codes; if any missing code, try to guess from message
        msg = state.message or ""
        if any(not (it.threat_code or "").strip() for it in payload.items):
            m = _THREAT_CODE_RX.search(msg)
            guessed = m.group(1).upper() if m else None
            new_items: list[RiskItem] = []
            for it in payload.items:
                code = (it.threat_code or "").strip().upper() or (guessed or "")
                if not code:
                    state.result = _ui_response(_t(loc, "nothing_to_set"), extra={"agent_events": state.agent_events})
                    return state
                new_items.append(RiskItem(threat_code=code, probability=int(it.probability), note=it.note))
            payload = RiskSetPayload(asset_id=payload.asset_id, items=new_items)

        # Guard + build actions
        # Normalize items (clamp probability to 0..4 and uppercase codes)
        items_dump = [{
            "threat_code": it.threat_code.upper(),
            "probability": max(0, min(4, int(it.probability))),
            **({"note": it.note} if it.note else {})
        } for it in payload.items]

        # Build the deterministic action for a single asset
        action = {
            "op": "set_risks",
            "mode": "merge",
            "asset_id": payload.asset_id,
            "items": [{
                "threat_code": it.threat_code.upper(),
                "probability": max(0, min(4, int(it.probability))),
                **({"note": it.note} if it.note else {})
            } for it in payload.items],
        }

        # Validate before submit
        try:
            _guard("ActionSetRisks", action)
        except Exception as e:
            state.result = _ui_response(_t(loc, "validation_error", err=e), extra={"agent_events": state.agent_events})
            return state

        # Queue for submission node
        state.actions = [action]


        # Optimistic WS emit (batch-preferred, per-item fallback)
        try:
            dto_for_ws = _adapt_set_payload_to_dto(payload)
            _schedule(_emit_ws_from_dto(dto_for_ws, deps, org_id=state.org_id, actor=state.user_id))
        except Exception:
            pass

        # Short inline confirmation
        last = payload.items[-1]
        p = max(0, min(4, int(last.probability)))
        if loc == "cs":
            confirm = f"✅ Nastavena pravděpodobnost {last.threat_code}=P{p} pro aktivum „{payload.asset_id}“."
        elif loc == "sk":
            confirm = f"✅ Nastavená pravdepodobnosť {last.threat_code}=P{p} pre aktívum „{payload.asset_id}“."
        else:
            confirm = f"✅ Set probability {last.threat_code}=P{p} for asset “{payload.asset_id}”."
        state.result = _ui_response(confirm, extra={"agent_events": state.agent_events})

        return state
    return _inner

def _bulk_set_risk_probabilities(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)
        bulk = state.bulk_payload
        if not bulk or not bulk.entries:
            state.result = _ui_response(_t(loc, "bulk_missing"), extra={"agent_events": state.agent_events})
            return state

        actions: list[dict] = []
        for entry in bulk.entries:
            if not entry.asset_id or not entry.items:
                continue
            action = {
                "op": "set_risks",
                "mode": "merge",
                "asset_id": entry.asset_id,
                "items": [it.model_dump() for it in entry.items],
            }
            try:
                _guard("ActionSetRisks", action)
            except Exception as e:
                state.result = _ui_response(_t(loc, "validation_error", err=e), extra={"agent_events": state.agent_events})
                return state
            actions.append(action)

        if not actions:
            state.result = _ui_response(_t(loc, "bulk_missing"), extra={"agent_events": state.agent_events})
            return state

        state.actions = actions

        # Optimistic WS emits (per-item; batch-per-asset if available)
        hub = getattr(deps, "ws_hub", None)
        if hub and hasattr(hub, "aemit_risk_set_probability"):
            async def _emit_now(entries: list[RiskSetPayload]):
                for e in entries:
                    for it in e.items:
                        evt = {
                            "event": "risk.set_probability",
                            "payload": {
                                "org_id": state.org_id,
                                "asset_id": e.asset_id,
                                "threat_code": it.threat_code,
                                "probability": int(it.probability),
                                **({"note": it.note} if it.note else {}),
                                "actor": state.user_id,
                                "ts": _now_iso(),
                            },
                        }
                        try:
                            _guard("WSRiskSetProbability", evt)
                        except Exception:
                            pass
                        try:
                            await hub.aemit_risk_set_probability(evt)  # type: ignore[attr-defined]
                        except Exception:
                            pass
            _schedule(_emit_now(list(bulk.entries)))

        return state
    return _inner

def _upsert_threat(deps: GraphDeps) -> Callable[[ThreatState], ThreatState]:
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)

        items = state.upserts if hasattr(state, "upserts") and state.upserts else None
        if items is None:
            try:
                raw = json.loads(state.message or "{}")
                def _sanitize_upsert_dict(x: dict) -> dict:
                    y = dict(x or {})
                    if y.get("asset_type_codes") is None:
                        y["asset_type_codes"] = []
                    for b in ("intentional", "unintentional", "environmental"):
                        if y.get(b) is None:
                            y[b] = False
                    return y
                items = [UpsertThreatItem(**_sanitize_upsert_dict(x)) for x in (raw.get("items") or raw.get("upserts") or [])]
            except Exception:
                items = []

        if not items:
            state.result = _ui_response(_t(loc, "upsert_missing"), extra={"agent_events": state.agent_events})
            return state

        normalized = []
        for it in items:
            category = (it.category or it.name or "").strip()
            if not category:
                state.result = _ui_response(_t(loc, "validation_error", err="category required"), extra={"agent_events": state.agent_events})
                return state
            payload = it.model_dump(exclude_none=True)
            payload["category"] = category
            payload.setdefault("asset_type_codes", [])
            payload["intentional"] = bool(payload.get("intentional", False))
            payload["unintentional"] = bool(payload.get("unintentional", False))
            payload["environmental"] = bool(payload.get("environmental", False))
            normalized.append(payload)
        action = {"op": "upsert_threats", "items": normalized}

        try:
            _guard("ActionUpsertThreats", action)
        except Exception as e:
            state.result = _ui_response(_t(loc, "validation_error", err=e), extra={"agent_events": state.agent_events})
            return state

        # WS emit (batch je fajn, per-item fallback může zůstat)
        hub = getattr(deps, "ws_hub", None)
        async def _emit():
            if hub and hasattr(hub, "aemit_threat_upsert_batch"):
                evt = {"event": "threat.upsert_batch", "org_id": state.org_id, "items": [it.model_dump() for it in items], "actor": state.user_id}
                try:
                    _guard("WSThreatUpsertBatch", evt)
                except Exception:
                    pass
                try:
                    await hub.aemit_threat_upsert_batch(evt)  # type: ignore[attr-defined]
                    return
                except Exception:
                    pass
            if hub and hasattr(hub, "aemit_threat_upsert"):
                for it in items:
                    evt = {"event": "threat.upsert", "org_id": state.org_id, "item": it.model_dump(), "actor": state.user_id}
                    try:
                        _guard("WSThreatUpsert", evt)
                    except Exception:
                        pass
                    try:
                        await hub.aemit_threat_upsert(evt)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        _schedule(_emit())

        state.actions = [action]
        return state
    return _inner

# -----------------------------------------------------------------------------
# Submit with backoff + jitter and outbox fallback
# -----------------------------------------------------------------------------



async def _submit_changeset_async(deps: GraphDeps, state: ThreatState) -> ThreatState:
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

    submitter = getattr(deps, "tazila_client", None) or getattr(deps, "submitter", None)

    t0 = _event(state, kind="tool.call.start", tool="submit_changeset", meta={"actions": len(acts)})
    try:
        cs_typed = cs if hasattr(cs, "model_dump") else ChangeSet.model_validate(cs)
        result: SubmitOrFallbackResult = await submit_or_fallback(
            changeset=cs_typed,
            submitter=submitter,
        )
        if result.submitted:
            transport = (
                result.submit_result.transport
                if (result.submit_result and getattr(result.submit_result, "transport", None))
                else "unknown"
            )
            _finish(state, t0, tool="submit_changeset", ok=True, meta={"actions": len(acts), "transport": transport})
        else:
            _finish(state, t0, tool="submit_changeset", ok=False, meta={"fallback_path": str(result.fallback_path)})
            state.result = _ui_response(
                _t(loc, "submit_failed", err="queued to outbox"),
                extra={"agent_events": state.agent_events, "outbox_path": str(result.fallback_path)}
            )
            return state
    except Exception as e:
        _finish(state, t0, tool="submit_changeset", ok=False, meta={"error": str(e)})
        state.result = _ui_response(_t(loc, "submit_failed", err=e), extra={"agent_events": state.agent_events})
        return state

    n_set = sum(1 for a in acts if a.get("op") == "set_risks")
    n_upsert = sum(1 for a in acts if a.get("op") == "upsert_threats")
    text = _t(loc, "submitted_summary", set=n_set, upsert=n_upsert) if (n_set or n_upsert) \
           else _t(loc, "submitted_generic", count=len(acts))

    try:
        touched = [a.get("asset_id") for a in acts if a.get("op") == "set_risks" and a.get("asset_id")]
        if touched:
            _schedule(_amemory_set(deps, f"{state.org_id}:threats.last_touched_asset_ids", touched))
    except Exception:
        pass

    state.result = _ui_response(text, extra={"changeset_actions": len(acts), "agent_events": state.agent_events})
    return state


# -----------------------------------------------------------------------------
# Explain (RAG → LLM synthesis)
# -----------------------------------------------------------------------------

async def _explain_threats_async(deps: GraphDeps, state: ThreatState) -> ThreatState:
    loc = _lang(state.locale)
    q = (state.question or state.message or "").strip()
    if not q:
        state.result = _ui_response(_t(loc, "explain_need_question"), extra={"agent_events": state.agent_events})
        return state

    retriever = None
    try:
        if hasattr(deps, "make_retriever") and callable(deps.make_retriever):
            retriever = deps.make_retriever(state.org_id, "threats")
    except Exception:
        retriever = None

    snippets: list[dict] = []
    if retriever:
        t0 = _event(state, kind="tool.call.start", tool="retriever.search", meta={"tab": "threats"})
        try:
            docs = await retriever.asearch(q, k=5) if hasattr(retriever, "asearch") else retriever.search(q, k=5)  # type: ignore
            for d in (docs or [])[:5]:
                if isinstance(d, dict):
                    text = (d.get("text") or d.get("page_content") or "").strip()
                    src = d.get("source") or (d.get("metadata") or {}).get("source")
                    if text:
                        snippets.append({"text": text[:800], "source": src})
                else:
                    snippets.append({"text": str(d)})
            _finish(state, t0, tool="retriever.search", ok=True, meta={"docs": len(snippets)})
        except Exception as e:
            _finish(state, t0, tool="retriever.search", ok=False, meta={"error": str(e)})

    llm = _get_llm_from_deps(deps)
    if snippets and llm is not None:
        system = (
            "You are a cybersecurity assistant specializing in threats and risk probabilities. "
            "Synthesize a concise answer (≤120 words) grounded ONLY in the provided snippets. "
            "If a detail is not present in snippets, do not invent it. Primary languages: SK/EN (CZ tolerated)."
        )
        human = {"question": q, "snippets": snippets, "locale": loc}
        t0 = _event(state, kind="tool.call.start", tool="llm.synthesize", meta={"snippets": len(snippets)})
        try:
            resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=json.dumps(human))])  # type: ignore
            txt = (getattr(resp, "content", None) or "").strip()
            if txt:
                _finish(state, t0, tool="llm.synthesize", ok=True, meta={})
                state.result = _ui_response(txt, extra={"snippets": snippets, "agent_events": state.agent_events})
                return state
        except Exception as e:
            _finish(state, t0, tool="llm.synthesize", ok=False, meta={"error": str(e)})

    # Fallback: plain LLM on question
    if llm is not None:
        system = (
            "You are a cybersecurity assistant specializing in threats and risk probabilities. "
            "Provide a concise (≤120 words) answer without hallucinations."
        )
        human = {"question": q, "locale": loc}
        t0 = _event(state, kind="tool.call.start", tool="llm.explain")
        try:
            resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=json.dumps(human))])  # type: ignore
            txt = (getattr(resp, "content", None) or "").strip()
            if txt:
                _finish(state, t0, tool="llm.explain", ok=True, meta={})
                state.result = _ui_response(txt, extra={"agent_events": state.agent_events})
                return state
        except Exception as e:
            _finish(state, t0, tool="llm.explain", ok=False, meta={"error": str(e)})

    state.result = _ui_response(_t(loc, "explain_no_sources"), extra={"snippets": snippets or None, "agent_events": state.agent_events})
    return state

# -----------------------------------------------------------------------------
# Help / chitchat
# -----------------------------------------------------------------------------

def _help_or_greet() -> Callable[[ThreatState], ThreatState]:
    def _inner(state: ThreatState) -> ThreatState:
        loc = _lang(state.locale)
        state.result = _ui_response(_t(loc, "greet"), extra={"agent_events": state.agent_events})
        return state
    return _inner

# -----------------------------------------------------------------------------
# Graph assembly
# -----------------------------------------------------------------------------

def create_graph(deps: GraphDeps):
    """
    Build & compile the THREATS graph.
    Flow:
      start → nlu → route → {help|clarify|list|get|set|bulk_set|upsert|explain} → (submit?) → END
    Missing selection for get/set triggers suggest-select (clarify-first).
    """
    graph = StateGraph(ThreatState)

    # Nodes
    async def _nlu_node(state: ThreatState) -> ThreatState:
        return await _nlu(deps, state)

    async def _submit_node(state: ThreatState) -> ThreatState:
        return await _submit_changeset_async(deps, state)

    async def _explain_node(state: ThreatState) -> ThreatState:
        return await _explain_threats_async(deps, state)

    graph.add_node("nlu", _nlu_node)
    graph.add_node("route", lambda s: s)

    graph.add_node("help", _help_or_greet())
    graph.add_node("chitchat", _help_or_greet())
    graph.add_node("clarify", _clarify())
    graph.add_node("suggest_select", _suggest_select(deps))

    graph.add_node("list_threats", _list_threats(deps))
    graph.add_node("asset_get_risks", _asset_get_risks_or_suggest(deps))

    graph.add_node("set_risk_probability", _set_risk_probability(deps))
    graph.add_node("bulk_set_risk_probabilities", _bulk_set_risk_probabilities(deps))
    graph.add_node("upsert_threat", _upsert_threat(deps))

    graph.add_node("submit_changeset", _submit_node)
    graph.add_node("explain_threats", _explain_node)

    # Entry
    graph.set_entry_point("nlu")
    graph.add_edge("nlu", "route")

    # Routing
    def _route(state: ThreatState) -> str:
        intent = state.intent or "clarify"  # clarify-first default
        if intent in ("help", "chitchat"):
            return "help"
        if intent == "clarify":
            return "clarify"
        if intent == "list":
            return "list_threats"
        if intent == "get":
            return "asset_get_risks"
        if intent == "set":
            return "set_risk_probability"
        if intent == "bulk_set":
            return "bulk_set_risk_probabilities"
        if intent == "upsert":
            return "upsert_threat"
        if intent == "explain":
            return "explain_threats"
        return "clarify"

    graph.add_conditional_edges(
        "route",
        _route,
        {
            "help": "help",
            "clarify": "clarify",
            "list_threats": "list_threats",
            "asset_get_risks": "asset_get_risks",
            "set_risk_probability": "set_risk_probability",
            "bulk_set_risk_probabilities": "bulk_set_risk_probabilities",
            "upsert_threat": "upsert_threat",
            "explain_threats": "explain_threats",
        },
    )

    # Write → submit (send-now semantics)
    graph.add_edge("set_risk_probability", "submit_changeset")
    graph.add_edge("bulk_set_risk_probabilities", "submit_changeset")
    graph.add_edge("upsert_threat", "submit_changeset")

    # Terminal
    for node in ("help", "clarify", "list_threats", "asset_get_risks", "explain_threats", "submit_changeset"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=deps.checkpointer)

__all__ = ["ThreatState", "create_graph"]
