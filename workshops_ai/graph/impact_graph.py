# File: workshops_ai/graph/impact_graph.py
# -----------------------------------------------------------------------------
# LangGraph: IMPACTS tab — LLM-first conversational assistant with SEND-NOW writes
#
# What’s new vs. the legacy version
# - Planner (LLM-first) extracts intent & slots (SK/CZ/EN) → deterministic tools.
# - Suggest-select: if an asset_id is missing for get/set/explain, we propose a
#   short list of assets instead of returning a dry error.
# - Write semantics preserved (emit-first → submit with backoff). Guards optional.
# - Multilingual, short replies (SK/CZ/EN). Manager should pass locale + short
#   history (last 3–5 turns) but this graph works even without it.
#
# Supported intents
#   - help/chitchat   (friendly guidance; no tools)
#   - list            (read impact catalog)
#   - get             (read current impact levels for selected asset)
#   - set             (set impact levels for one asset)
#   - bulk_set        (set impact levels for multiple assets)
#   - upsert          (create/update impact catalog entries)
#   - explain         (RAG → LLM synthesis, if retriever/LLM available)
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import random
import re
import time
import unicodedata
import uuid
from typing import Any, Callable, Literal, Optional, Sequence, cast
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from workshops_ai.graph.factory import GraphDeps

# LangGraph
try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("langgraph must be installed to use impact_graph") from e

# Optional LLM (planner + explain)
_HAS_LLM = True
try:
    from langchain_openai import ChatOpenAI  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
except Exception:  # pragma: no cover
    _HAS_LLM = False
    ChatOpenAI = object  # type: ignore
    SystemMessage = HumanMessage = object  # type: ignore

# ChangeSet (shared schema + builder)
try:
    from workshops_ai.models.changeset import ChangeSet  # type: ignore
    from workshops_ai.agents.utils.changeset_builder import build_changeset  # type: ignore
except Exception:  # pragma: no cover
    ChangeSet = Any  # type: ignore

    def build_changeset(**kwargs):  # type: ignore
        return {
            "id": kwargs.get("id") or str(uuid.uuid4()),
            "org_id": kwargs.get("org_id"),
            "user_id": kwargs.get("user_id"),
            "intent": kwargs.get("intent", "apply"),
            "actions": kwargs.get("actions", []),
            "audit": {"source": "AI", "timestamp": dt.datetime.utcnow().isoformat()},
        }

# Optional guard (soft dependency; flexible adapter)
try:
    from workshops_ai.tools.guard import guard_validate_payload as _guard_validate  # type: ignore
except Exception:  # pragma: no cover
    def _guard_validate(*args, **kwargs):  # type: ignore
        return args[-1] if args else kwargs  # passthrough

def _guard(model_hint: Any, payload: dict) -> dict:
    """
    Adapter: supports guard_validate_payload(model_cls, payload) or guard_validate_payload(payload)
    without breaking when the guard package is absent.
    """
    try:
        return _guard_validate(model_hint, payload)  # type: ignore[misc]
    except TypeError:
        try:
            return _guard_validate(payload)  # type: ignore[misc]
        except Exception:
            return payload

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Types & State
# -----------------------------------------------------------------------------

Intent = Literal["help", "chitchat", "list", "get", "set", "bulk_set", "upsert", "explain"]


class ImpactItem(BaseModel):
    impact_code: str  # "C" | "I" | "A" | custom code
    level: int = Field(ge=0, le=4)
    note: Optional[str] = None


class ImpactSetPayload(BaseModel):
    asset_id: str
    items: list[ImpactItem]


class ImpactBulkPayload(BaseModel):
    entries: list[ImpactSetPayload]


class ImpactUpsertItem(BaseModel):
    code: str
    name: str
    category: Optional[str] = None
    description: Optional[str] = None


class ToolEvent(TypedDict, total=False):
    event_type: str            # "tool.call.start" | "tool.call.end"
    tool: str                  # tool/node name
    ok: bool                   # success flag
    duration_ms: int           # only for "end"
    meta: dict                 # arbitrary metadata


class ImpactState(BaseModel):
    # Request context
    org_id: str
    active_tab: str = Field(default="impacts")
    message: Optional[str] = None
    selection: Optional[dict] = None   # {"asset_id": "..."} | {"asset_ids": [...]} | {"assets": [...]}
    user_id: Optional[str] = None
    locale: Optional[str] = None       # "sk" | "cs" | "en" (or full locales)

    # Structured inputs (planner results, wizard/API callers may prefill)
    action: Optional[Intent] = None
    set_payload: Optional[ImpactSetPayload] = None
    bulk_payload: Optional[ImpactBulkPayload] = None
    upsert_items: Optional[list[ImpactUpsertItem]] = None
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
            "sk": "Ahoj! Môžem: „zoznam dopadov“, „ukáž dopady pre aktívum“, „nastav C/I/A“, „hromadne nastav“, „vysvetli dopady“. 😉",
            "cs": "Ahoj! Umím: „seznam dopadů“, „ukaž dopady pro aktivum“, „nastav C/I/A“, „hromadně nastav“, „vysvětli dopady“. 😉",
            "en": "Hi! I can: “list impacts”, “show impacts for an asset”, “set C/I/A”, “bulk set”, or “explain impacts”. 😉",
        },
        "catalog_fail": {
            "sk": "Katalóg dopadov sa nepodarilo načítať: {err}",
            "cs": "Katalog dopadů se nepodařilo načíst: {err}",
            "en": "Failed to load the impact catalog: {err}",
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
            "sk": "Nie je čo nastaviť. Uveď asset_id a aspoň jeden z C/I/A.",
            "cs": "Není co nastavit. Uveď asset_id a alespoň jednu z hodnot C/I/A.",
            "en": "Nothing to set. Provide an asset_id and at least one of C/I/A.",
        },
        "bulk_missing": {
            "sk": "Chýbajú položky pre hromadné nastavenie.",
            "cs": "Chybí položky pro hromadné nastavení.",
            "en": "Missing entries for bulk update.",
        },
        "upsert_missing": {
            "sk": "Zadaj upsert_items s aspoň jednou položkou {code,name,...}.",
            "cs": "Zadej upsert_items s alespoň jednou položkou {code,name,...}.",
            "en": "Provide upsert_items with at least one {code,name,...}.",
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
            "sk": "{set} nastavení úrovní, {upsert} úprav katalógu.",
            "cs": "{set} nastavení úrovní, {upsert} úprav katalogu.",
            "en": "{set} level updates, {upsert} catalog upserts.",
        },
        "submitted_generic": {
            "sk": "{count} akcií odoslaných.",
            "cs": "{count} akcí odesláno.",
            "en": "{count} actions submitted.",
        },
        "explain_need_question": {
            "sk": "Zadaj otázku (state.question alebo message) pre vysvetlenie dopadov.",
            "cs": "Zadej otázku (state.question nebo message) pro vysvětlení dopadů.",
            "en": "Provide a question (state.question or message) to explain impacts.",
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
# Utilities
# -----------------------------------------------------------------------------

_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)

def _norm(text: str) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii").lower()


def _safe_emit(ws_hub: Any, event: str, payload: dict) -> None:
    try:
        if hasattr(ws_hub, "emit"):
            ws_hub.emit(event, payload)
        elif hasattr(ws_hub, "broadcast"):
            ws_hub.broadcast(event, payload)
    except Exception as e:  # pragma: no cover
        logging.getLogger(__name__).debug("WS emit failed (%s): %s", event, e)


def _schedule(coro):
    try:
        asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        pass


async def _amemory_set(deps: GraphDeps, key: str, value: Any) -> None:
    fn = getattr(deps, "memory_set", None)
    if callable(fn):
        r = fn(key, value)
        if asyncio.iscoroutine(r):
            await r


# -----------------------------------------------------------------------------
# Telemetry helpers
# -----------------------------------------------------------------------------

def _event(state: ImpactState, *, kind: str, tool: str, ok: bool = True, meta: dict | None = None) -> float:
    t0 = time.perf_counter()
    state.agent_events.append({"event_type": kind, "tool": tool, "ok": ok, "meta": meta or {}})
    return t0


def _finish(state: ImpactState, t0: float, *, tool: str, ok: bool, meta: dict | None = None) -> None:
    dur = int((time.perf_counter() - t0) * 1000)
    state.agent_events.append({"event_type": "tool.call.end", "tool": tool, "ok": ok, "duration_ms": dur, "meta": meta or {}})


# -----------------------------------------------------------------------------
# Planner (NLU) — LLM-first with regex fallback
# -----------------------------------------------------------------------------

# Intent cues (SK/EN; tolerant to CZ)
HELP_RX     = re.compile(r"^\s*(help|pomoc|napoveda|nápověda)\s*$", re.I)
GREET_RX    = re.compile(r"^\s*(ahoj|cau|čau|hej|nazdar|dobr(y|ý)\s+de[nň]|hello|hi|hey|good\s+(morning|afternoon|evening))[\s!.]*$", re.I)
LIST_RX     = re.compile(r"\b(list|seznam|zoznam|catalog|katalog|katal[oó]g)\b", re.I)
GET_RX      = re.compile(r"\b(get|show|detail|ukaz|zobraz)\b", re.I)
SET_RX      = re.compile(r"\b(set|update|nastav|zmen|uprav|impact|dopad|level|uroven|úroveň)\b", re.I)
BULK_RX     = re.compile(r"\b(bulk|hromadn[eě]|vsetky|všetky|all\s+assets)\b", re.I)
UPSERT_RX   = re.compile(r"\b(upsert|add\s+impact|new\s+impact|pridaj\s+dopad|přidej\s+dopad)\b", re.I)
EXPLAIN_RX  = re.compile(r"\b(explain|vysvetli|vysvětli|preco|pro[cč]|\bwhy\b|context|kontext)\b", re.I)

# CIA dimension parsing
DIM_RX = {
    "C": re.compile(r"\b(c|confidentiality|d[ôo]vernost|d[uů]věrnost)\b[^0-9-]*([0-4])\b", re.I),
    "I": re.compile(r"\b(i|integrity|integrita)\b[^0-9-]*([0-4])\b", re.I),
    "A": re.compile(r"\b(a|availability|dostupnost)\b[^0-9-]*([0-4])\b", re.I),
}

class NLUResult(BaseModel):
    action: Optional[Intent] = None
    # For 'set'
    asset_id: Optional[str] = None
    items: Optional[list[ImpactItem]] = None
    # For 'bulk_set'
    entries: Optional[list[ImpactSetPayload]] = None
    # For 'upsert'
    upserts: Optional[list[ImpactUpsertItem]] = None
    # For 'explain'
    question: Optional[str] = None

def _get_llm_from_deps(deps: GraphDeps):
    llm = getattr(deps, "llm", None)
    if llm is not None:
        return llm
    if not _HAS_LLM:
        return None
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, timeout=12, max_retries=1)  # type: ignore[call-arg]


def _parse_set_from_text(text: str, selection_id: Optional[str]) -> tuple[Optional[str], list[ImpactItem]]:
    t = _norm(text)
    # asset_id from selection or inline UUID
    m_uuid = _UUID_RE.search(text)
    asset_id = selection_id or (m_uuid.group(0) if m_uuid else None)
    items: list[ImpactItem] = []
    for code, rx in DIM_RX.items():
        m = rx.search(t)
        if m:
            items.append(ImpactItem(impact_code=code, level=int(m.group(2))))
    return asset_id, items


async def _nlu(deps: GraphDeps, state: ImpactState) -> ImpactState:
    raw = (state.message or "").strip()
    selection_id = (state.selection or {}).get("asset_id")

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
            "You are a strict NLU for an 'impacts' assistant.\n"
            "Primary languages: Slovak (SK) and English (EN); Czech (CZ) tolerated.\n"
            "Extract intent and slots and return ONLY compact JSON with fields:\n"
            "  action: one of [help, chitchat, list, get, set, bulk_set, upsert, explain]\n"
            "  asset_id?: string\n"
            "  items?: [{impact_code:'C'|'I'|'A'|custom, level:0..4, note?}]           # for set\n"
            "  entries?: [{asset_id:string, items:[...]}]                              # for bulk_set\n"
            "  upserts?: [{code,name,category?,description?}]                          # for upsert\n"
            "  question?: string                                                       # for explain\n"
            "Rules:\n"
            "- Never invent IDs.\n"
            "- If a UI selection.asset_id is present, you MAY copy it to asset_id when helpful.\n"
            "- Detect greetings as 'chitchat'/'help'.\n"
            "- CIA numeric scale is 0..4.\n"
            "- Output STRICT JSON; no prose."
        )
        human = {
            "message": raw,
            "selection": {"asset_id": selection_id} if selection_id else None,
            "locale": state.locale,
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
            if n.asset_id and (state.set_payload is None):
                if n.items:
                    state.set_payload = ImpactSetPayload(asset_id=n.asset_id, items=n.items)
            if n.entries and (state.bulk_payload is None):
                state.bulk_payload = ImpactBulkPayload(entries=n.entries)
            if n.upserts and (state.upsert_items is None):
                state.upsert_items = n.upserts
            if n.question and not state.question:
                state.question = n.question

            # If still nothing parsed for 'set', try a best-effort parse
            if (state.intent == "set") and (state.set_payload is None):
                aid, items = _parse_set_from_text(raw, selection_id)
                if aid and items:
                    state.set_payload = ImpactSetPayload(asset_id=aid, items=items)

            if state.intent:
                return state
        except Exception as e:
            logger.debug("Impacts NLU LLM failed, fallback to regex: %s", e)

    # Regex fallback
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
        state.intent = "help"

    if state.intent == "set" and state.set_payload is None:
        aid, items = _parse_set_from_text(raw, selection_id)
        if aid and items:
            state.set_payload = ImpactSetPayload(asset_id=aid, items=items)

    return state


# -----------------------------------------------------------------------------
# Suggest-select (if missing asset selection)
# -----------------------------------------------------------------------------

def _suggest_select(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    def _inner(state: ImpactState) -> ImpactState:
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
            msg = "\n".join([_t(loc, "select_prompt")] + preview) if preview else _t(loc, "catalog_found", n=0)
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events, "assets_preview": preview})
            _finish(state, t0, tool="suggest_select", ok=True, meta={"count": len(preview)})
        except Exception as e:
            _finish(state, t0, tool="suggest_select", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "catalog_fail", err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Read-only nodes
# -----------------------------------------------------------------------------

def _list_impacts(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    def _inner(state: ImpactState) -> ImpactState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="list_impacts")
        try:
            impacts = deps.backend_port.list_impacts() or []
            lines = [_t(loc, "catalog_found", n=len(impacts))]
            for row in impacts[:30]:
                code = row.get("impact_code") or row.get("code") or row.get("EN Code") or "?"
                name = row.get("EN Name") or row.get("name") or row.get("Impact") or ""
                lines.append(f"- {code}: {name}")
            state.result = _ui_response("\n".join(lines), extra={"impacts": impacts, "agent_events": state.agent_events})
            _finish(state, t0, tool="list_impacts", ok=True, meta={"count": len(impacts)})
        except Exception as e:
            _finish(state, t0, tool="list_impacts", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "catalog_fail", err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner


def _asset_get_impacts_or_suggest(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    def _inner(state: ImpactState) -> ImpactState:
        loc = _lang(state.locale)
        asset_id = (state.selection or {}).get("asset_id")
        if not asset_id:
            return _suggest_select(deps)(state)

        t0 = _event(state, kind="tool.call.start", tool="asset_get_impacts", meta={"asset_id": asset_id})
        try:
            levels = deps.backend_port.asset_get_impacts(asset_id) or []
            summary = [f"Impacts for {asset_id} ({len(levels)}):"]
            for it in levels[:30]:
                code = it.get("impact_code") or it.get("code")
                lvl = it.get("level")
                summary.append(f"- {code}: {lvl}")
            state.result = _ui_response("\n".join(summary), extra={"asset_id": asset_id, "impacts": levels, "agent_events": state.agent_events})
            _finish(state, t0, tool="asset_get_impacts", ok=True, meta={"count": len(levels)})
        except Exception as e:
            _finish(state, t0, tool="asset_get_impacts", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "asset_impacts_fail", id=asset_id, err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Write nodes (emit-first) → actions → submit
# -----------------------------------------------------------------------------

_LEVEL_VALUE = [5, 20, 50, 70, 100]  # example scale for WS preview

def _set_impact_level(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    def _inner(state: ImpactState) -> ImpactState:
        loc = _lang(state.locale)
        payload = state.set_payload
        if not payload or not payload.items:
            # If missing because selection absent → suggest-select
            if not (state.selection and state.selection.get("asset_id")):
                return _suggest_select(deps)(state)
            state.result = _ui_response(_t(loc, "nothing_to_set"), extra={"agent_events": state.agent_events})
            return state

        # Guard before WS emit
        try:
            _guard("ActionSetImpacts", {
                "op": "set_impacts",
                "mode": "merge",
                "asset_id": payload.asset_id,
                "items": [it.model_dump() for it in payload.items],
            })
        except Exception as e:
            state.result = _ui_response(_t(loc, "validation_error", err=e), extra={"agent_events": state.agent_events})
            return state

        # Optimistic per-item WS emit (if hub supports async helpers)
        hub = getattr(deps, "ws_hub", None)
        if hub and hasattr(hub, "aemit_impact_set_level"):
            async def _emit_all():
                for it in payload.items:
                    level = max(0, min(4, int(it.level)))
                    evt = {
                        "event": "impact.set_level",
                        "org_id": state.org_id,
                        "asset_id": payload.asset_id,
                        "impact_code": it.impact_code,
                        "level_index": level,
                        "level_value": _LEVEL_VALUE[level],
                        "note": it.note,
                        "actor": state.user_id,
                    }
                    try:
                        _guard("WSImpactSetLevel", evt)
                    except Exception:
                        pass
                    try:
                        await hub.aemit_impact_set_level(evt)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            _schedule(_emit_all())

        # Build actions
        state.actions = [{
            "op": "set_impacts",
            "mode": "merge",
            "asset_id": payload.asset_id,
            "items": [it.model_dump() for it in payload.items],
        }]
        return state
    return _inner


def _bulk_set_impact_levels(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    def _inner(state: ImpactState) -> ImpactState:
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
                "op": "set_impacts",
                "mode": "merge",
                "asset_id": entry.asset_id,
                "items": [it.model_dump() for it in entry.items],
            }
            try:
                _guard("ActionSetImpacts", action)
            except Exception as e:
                state.result = _ui_response(_t(loc, "validation_error", err=e), extra={"agent_events": state.agent_events})
                return state
            actions.append(action)

            # Optimistic WS per entry
            hub = getattr(deps, "ws_hub", None)
            if hub and hasattr(hub, "aemit_impact_set_level"):
                async def _emit_entry(aid: str, items: list[ImpactItem]):
                    for it in items:
                        level = max(0, min(4, int(it.level)))
                        evt = {
                            "event": "impact.set_level",
                            "org_id": state.org_id,
                            "asset_id": aid,
                            "impact_code": it.impact_code,
                            "level_index": level,
                            "level_value": _LEVEL_VALUE[level],
                            "note": it.note,
                            "actor": state.user_id,
                        }
                        try:
                            _guard("WSImpactSetLevel", evt)
                        except Exception:
                            pass
                        try:
                            await hub.aemit_impact_set_level(evt)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                _schedule(_emit_entry(entry.asset_id, list(entry.items)))

        if not actions:
            state.result = _ui_response(_t(loc, "bulk_missing"), extra={"agent_events": state.agent_events})
            return state

        state.actions = actions
        return state
    return _inner


def _upsert_impact(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    def _inner(state: ImpactState) -> ImpactState:
        loc = _lang(state.locale)
        items = state.upsert_items or []
        if not items:
            # Tiny inline JSON fallback: {"items":[{code,name,category?,description?},...]}
            try:
                raw = json.loads(state.message or "{}")
                items = [ImpactUpsertItem(**x) for x in (raw.get("items") or [])]
            except Exception:
                pass
        if not items:
            state.result = _ui_response(_t(loc, "upsert_missing"), extra={"agent_events": state.agent_events})
            return state

        action = {"op": "upsert_impacts", "items": [it.model_dump() for it in items]}
        try:
            _guard("ActionUpsertImpacts", action)
        except Exception as e:
            state.result = _ui_response(_t(loc, "validation_error", err=e), extra={"agent_events": state.agent_events})
            return state

        # Optimistic WS batch emit if present
        hub = getattr(deps, "ws_hub", None)
        async def _emit():
            if hub and hasattr(hub, "aemit_impact_upsert_batch"):
                evt = {"event": "impact.upsert_batch", "org_id": state.org_id, "items": [it.model_dump() for it in items], "actor": state.user_id}
                try:
                    _guard("WSImpactUpsertBatch", evt)
                except Exception:
                    pass
                try:
                    await hub.aemit_impact_upsert_batch(evt)  # type: ignore[attr-defined]
                    return
                except Exception:
                    pass
            if hub and hasattr(hub, "aemit_impact_upsert"):
                for it in items:
                    evt = {"event": "impact.upsert", "org_id": state.org_id, "item": it.model_dump(), "actor": state.user_id}
                    try:
                        _guard("WSImpactUpsert", evt)
                    except Exception:
                        pass
                    try:
                        await hub.aemit_impact_upsert(evt)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        _schedule(_emit())

        state.actions = [action]
        return state
    return _inner


# -----------------------------------------------------------------------------
# Submit with backoff + jitter and outbox fallback (same pattern as Asset)
# -----------------------------------------------------------------------------

def _outbox_path_from_deps(deps: GraphDeps) -> str:
    settings = getattr(deps, "settings", None)
    try:
        return getattr(getattr(settings, "tazila", None), "outbox_path", None) or os.getenv("WORKSHOPS_AI_TAZILA__OUTBOX_PATH") or "/app/workshops_ai/data/outbox"
    except Exception:
        return "/app/workshops_ai/data/outbox"


async def _write_outbox(payload: dict, deps: GraphDeps) -> None:
    path_dir = _outbox_path_from_deps(deps)
    os.makedirs(path_dir, exist_ok=True)
    cs_id = payload.get("id") or str(uuid.uuid4())
    path = os.path.join(path_dir, f"changeset_{cs_id}.json")
    def _write():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    await asyncio.to_thread(_write)
    logger.info("ChangeSet written to outbox: %s", path)


async def _submit_with_backoff(deps: GraphDeps, payload: dict, *, attempts: int = 4, base_s: float = 0.25, jitter_s: float = 0.25) -> None:
    last_err: Optional[Exception] = None

    async def _try_once():
        nonlocal last_err
        submit = getattr(deps, "submit_changeset", None)
        if callable(submit):
            try:
                r = submit(payload)
                if asyncio.iscoroutine(r):
                    await r
                return
            except Exception as e:
                last_err = e

        try:
            from workshops_ai.app.services import tazila_client  # type: ignore
        except Exception:
            tazila_client = None  # type: ignore

        if tazila_client is not None:
            try:
                ikey = payload.get("idempotency_key") or payload.get("id")
                if hasattr(tazila_client, "asubmit_changeset"):
                    await tazila_client.asubmit_changeset(payload, idempotency_key=ikey)  # type: ignore
                    return
                if hasattr(tazila_client, "submit_changeset"):
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, tazila_client.submit_changeset, payload)  # type: ignore
                    return
            except Exception as e:
                last_err = e
        else:
            raise RuntimeError("No submitter available")

        raise last_err or RuntimeError("Unknown submit failure")

    for i in range(1, attempts + 1):
        try:
            await _try_once()
            return
        except Exception as e:
            last_err = e
            if i >= attempts:
                break
            delay = base_s * (2 ** (i - 1)) + random.random() * jitter_s
            await asyncio.sleep(delay)

    await _write_outbox(payload, deps)
    if last_err:
        raise last_err


async def _submit_changeset_async(deps: GraphDeps, state: ImpactState) -> ImpactState:
    loc = _lang(state.locale)
    acts = state.actions or []
    if not acts:
        state.result = _ui_response(_t(loc, "submitted_generic", count=0), extra={"agent_events": state.agent_events})
        return state

    cs: ChangeSet | dict = build_changeset(org_id=state.org_id, user_id=state.user_id, actions=acts, intent="apply")
    if isinstance(cs, dict) and "idempotency_key" not in cs:
        cs["idempotency_key"] = cs.get("id")
    state.change = cs

    t0 = _event(state, kind="tool.call.start", tool="submit_changeset", meta={"actions": len(acts)})
    try:
        payload = cs.model_dump() if hasattr(cs, "model_dump") else cast(dict, cs)
        await _submit_with_backoff(deps, payload, attempts=4, base_s=0.25, jitter_s=0.25)
        _finish(state, t0, tool="submit_changeset", ok=True, meta={"actions": len(acts)})
    except Exception as e:
        _finish(state, t0, tool="submit_changeset", ok=False, meta={"error": str(e)})
        state.result = _ui_response(_t(loc, "submit_failed", err=e), extra={"agent_events": state.agent_events})
        return state

    n_set = sum(1 for a in acts if a.get("op") == "set_impacts")
    n_upsert = sum(1 for a in acts if a.get("op") == "upsert_impacts")
    if n_set or n_upsert:
        text = _t(loc, "submitted_summary", set=n_set, upsert=n_upsert)
    else:
        text = _t(loc, "submitted_generic", count=len(acts))

    # Memory hint: last touched asset ids
    try:
        touched = [a.get("asset_id") for a in acts if a.get("op") == "set_impacts" and a.get("asset_id")]
        if touched:
            _schedule(_amemory_set(deps, f"{state.org_id}:impacts.last_touched_asset_ids", touched))
    except Exception:
        pass

    state.result = _ui_response(text, extra={"changeset_actions": len(acts), "agent_events": state.agent_events})
    return state


# -----------------------------------------------------------------------------
# Explain (RAG → LLM synthesis)
# -----------------------------------------------------------------------------

async def _explain_impacts_async(deps: GraphDeps, state: ImpactState) -> ImpactState:
    loc = _lang(state.locale)
    q = (state.question or state.message or "").strip()
    if not q:
        state.result = _ui_response(_t(loc, "explain_need_question"), extra={"agent_events": state.agent_events})
        return state

    retriever = None
    try:
        if hasattr(deps, "make_retriever") and callable(deps.make_retriever):
            retriever = deps.make_retriever(state.org_id, "impacts")
    except Exception:
        retriever = None

    snippets: list[dict] = []
    if retriever:
        t0 = _event(state, kind="tool.call.start", tool="retriever.search", meta={"tab": "impacts"})
        try:
            docs = await retriever.asearch(q, k=5) if hasattr(retriever, "asearch") else retriever.search(q, k=5)  # type: ignore
            for d in (docs or [])[:5]:
                if isinstance(d, dict):
                    txt = (d.get("text") or d.get("page_content") or "").strip()
                    src = d.get("source") or (d.get("metadata") or {}).get("source")
                    if txt:
                        snippets.append({"text": txt[:800], "source": src})
                else:
                    snippets.append({"text": str(d)})
            _finish(state, t0, tool="retriever.search", ok=True, meta={"docs": len(snippets)})
        except Exception as e:
            _finish(state, t0, tool="retriever.search", ok=False, meta={"error": str(e)})

    llm = _get_llm_from_deps(deps)
    if snippets and llm is not None:
        system = (
            "You are a cybersecurity assistant specializing in business impact analysis (CIA). "
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
            "You are a cybersecurity assistant specializing in business impact analysis (CIA). "
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

    # Nothing worked
    state.result = _ui_response(_t(loc, "explain_no_sources"), extra={"snippets": snippets or None, "agent_events": state.agent_events})
    return state


# -----------------------------------------------------------------------------
# Help / chitchat
# -----------------------------------------------------------------------------

def _help_or_greet() -> Callable[[ImpactState], ImpactState]:
    def _inner(state: ImpactState) -> ImpactState:
        loc = _lang(state.locale)
        state.result = _ui_response(_t(loc, "greet"), extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Graph assembly
# -----------------------------------------------------------------------------

def create_graph(deps: GraphDeps):
    """
    Build & compile the IMPACTS graph.
    Flow:
      start → nlu → route → {help|list|get|set|bulk_set|upsert|explain} → (submit?) → END
    Missing selection for get/set triggers suggest-select.
    """
    graph = StateGraph(ImpactState)

    # Nodes
    async def _nlu_node(state: ImpactState) -> ImpactState:
        return await _nlu(deps, state)

    async def _submit_node(state: ImpactState) -> ImpactState:
        return await _submit_changeset_async(deps, state)

    async def _explain_node(state: ImpactState) -> ImpactState:
        return await _explain_impacts_async(deps, state)

    graph.add_node("nlu", _nlu_node)
    graph.add_node("route", lambda s: s)

    graph.add_node("help", _help_or_greet())
    graph.add_node("chitchat", _help_or_greet())
    graph.add_node("suggest_select", _suggest_select(deps))

    graph.add_node("list_impacts", _list_impacts(deps))
    graph.add_node("asset_get_impacts", _asset_get_impacts_or_suggest(deps))

    graph.add_node("set_impact_level", _set_impact_level(deps))
    graph.add_node("bulk_set_impact_levels", _bulk_set_impact_levels(deps))
    graph.add_node("upsert_impact", _upsert_impact(deps))

    graph.add_node("submit_changeset", _submit_node)
    graph.add_node("explain_impacts", _explain_node)

    # Entry
    graph.set_entry_point("nlu")
    graph.add_edge("nlu", "route")

    # Routing
    def _route(state: ImpactState) -> str:
        intent = state.intent or "list"
        if intent in ("help", "chitchat"):
            return "help"
        if intent == "list":
            return "list_impacts"
        if intent == "get":
            return "asset_get_impacts"
        if intent == "set":
            return "set_impact_level"
        if intent == "bulk_set":
            return "bulk_set_impact_levels"
        if intent == "upsert":
            return "upsert_impact"
        if intent == "explain":
            return "explain_impacts"
        return "list_impacts"

    graph.add_conditional_edges(
        "route",
        _route,
        {
            "help": "help",
            "list_impacts": "list_impacts",
            "asset_get_impacts": "asset_get_impacts",
            "set_impact_level": "set_impact_level",
            "bulk_set_impact_levels": "bulk_set_impact_levels",
            "upsert_impact": "upsert_impact",
            "explain_impacts": "explain_impacts",
        },
    )

    # Write → submit
    graph.add_edge("set_impact_level", "submit_changeset")
    graph.add_edge("bulk_set_impact_levels", "submit_changeset")
    graph.add_edge("upsert_impact", "submit_changeset")

    # Terminal
    for node in ("help", "list_impacts", "asset_get_impacts", "explain_impacts", "submit_changeset"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=deps.checkpointer)


__all__ = ["ImpactState", "create_graph"]
