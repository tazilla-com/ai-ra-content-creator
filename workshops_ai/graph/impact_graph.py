# File: workshops_ai/graph/impact_graph.py
# -----------------------------------------------------------------------------
# LangGraph: IMPACTS tab — LLM-first conversational assistant with SEND-NOW writes
#
# What’s new vs. legacy:
# - Clarify-first NLU (no hard default to "list"): asks for missing slots or suggests a selection.
# - Multilingual replies (SK/CZ/EN) with unified short messages.
# - Uses a short rolling history (last turns) to stay on topic; still safe if absent.
# - Deterministic writes preserved (set / bulk_set / upsert) → submit_changeset with backoff & outbox.
# - WS emits now follow schema shape: {"event":"impact.set_level|impact.set_levels", "payload": {...}}.
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
from workshops_ai.app.services.outbox_writer import submit_or_fallback, SubmitOrFallbackResult
from workshops_ai.agents.utils.mappers import resolve_impact_category
from workshops_ai.graph.factory import GraphDeps

from pydantic import BaseModel, Field



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

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# ChangeSet & Guard (shared builder, soft guard adapter)
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

# -----------------------------------------------------------------------------
# Optional guard (soft dependency; flexible adapter)
# -----------------------------------------------------------------------------
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
    weight: Optional[float] = None

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
    language: Optional[str] = None
    history: Optional[list[dict]] = None  # short rolling window (3–5 turns), optional

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

def _event(state: ImpactState, *, kind: str, tool: str, ok: bool = True, meta: dict | None = None) -> float:
    t0 = time.perf_counter()
    state.agent_events.append({"event_type": kind, "tool": tool, "ok": ok, "meta": meta or {}})
    return t0

def _finish(state: ImpactState, t0: float, *, tool: str, ok: bool, meta: dict | None = None) -> None:
    dur = int((time.perf_counter() - t0) * 1000)
    state.agent_events.append({"event_type": "tool.call.end", "tool": tool, "ok": ok, "duration_ms": dur, "meta": meta or {}})

# -----------------------------------------------------------------------------
# Planner (NLU) — LLM-first with regex fallback (clarify-first)
# -----------------------------------------------------------------------------

# Intent cues (SK/EN; tolerant to CZ)
HELP_RX     = re.compile(r"^\s*(help|pomoc|napoveda|nápověda)\s*$", re.I)
GREET_RX    = re.compile(r"^\s*(ahoj|cau|čau|hej|nazdar|dobr(y|ý)\s+de[nň]|hello|hi|hey|good\s+(morning|afternoon|evening))[\s!.]*$", re.I)
LIST_RX     = re.compile(r"\b(list|seznam|zoznam|catalog|katalog|katal[oó]g)\b", re.I)
GET_RX      = re.compile(r"\b(get|show|detail|ukaz|zobraz)\b", re.I)
SET_RX      = re.compile(r"\b(set|update|nastav|zmen|uprav|impact|dopad|level|uroven|úroveň)\b", re.I)
BULK_RX     = re.compile(r"\b(bulk|hromadn[eě]|vsetky|všetky|all\s+assets)\b", re.I)
UPSERT_RX   = re.compile(
    r"(?:^|\b)(?:vytvoř(?:it)?|přidej|pridaj|create|add|new)\b.*\b(impact|dopad)\b",
    re.I
)
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
    """
    Resolve an async-callable LLM client.

    - Prefer deps.llm_client (a ChatOpenAI-like client exposing .ainvoke).
    - Do NOT return deps.llm (that often holds settings/config, not a client).
    - Finally, build a default ChatOpenAI client if available.
    """
    llm_client = getattr(deps, "llm_client", None)
    if llm_client is not None:
        return llm_client

    if not _HAS_LLM:
        return None

    # Conservative defaults for responsiveness.
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

def _history_tail(state: ImpactState, k: int = 5) -> list[dict]:
    """Return last k turns of the short history (if provided)."""
    try:
        h = state.history or []
        return h[-k:]
    except Exception:
        return []

async def _nlu(deps: GraphDeps, state: ImpactState) -> ImpactState:
    raw = (state.message or "").strip()
    selection_id = (state.selection or {}).get("asset_id")

    # Shortcut: explicit action provided
    if state.action in ("help", "chitchat", "list", "get", "set", "bulk_set", "upsert", "explain"):
        state.intent = cast(Intent, state.action)
        return state

    if state.action == "defs_create":   # alias from tool vocabulary
        state.intent = "upsert"
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
            "history": _history_tail(state, 5),  # short rolling context
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
        # Clarify-first default
        state.intent = "clarify"

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
                if isinstance(row, dict):
                    code = row.get("impact_code") or row.get("code") or row.get("EN Code") or "?"
                    name = row.get("EN Name") or row.get("name") or row.get("Impact") or ""
                else:
                    code = getattr(row, "impact_code", None) or getattr(row, "code", None) or "?"
                    name = getattr(row, "name", None) or ""
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

_LEVEL_VALUE = [5, 20, 50, 70, 100]  # example scale for WS preview/telemetry

def _set_impact_level(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    def _inner(state: ImpactState) -> ImpactState:
        loc = _lang(state.locale)
        payload = state.set_payload

        # Basic presence checks
        if not payload or not payload.items:
            if not (state.selection and state.selection.get("asset_id")):
                return _suggest_select(deps)(state)
            state.result = _ui_response(_t(loc, "nothing_to_set"), extra={"agent_events": state.agent_events})
            return state

        # If any item is missing impact_code, try to infer or use catalog once
        need_code_fill = any((not (it.impact_code or "").strip()) for it in payload.items)
        catalog: list[dict] | None = None
        if need_code_fill:
            try:
                catalog = deps.backend_port.list_impacts() or []
            except Exception as e:
                logger.debug("impacts.defs.list failed: %s", e)

        def _choose_code_from_context(default: str = "C") -> str:
            msg = (state.message or "").lower()
            if DIM_RX["C"].search(msg):
                return "C"
            if DIM_RX["I"].search(msg):
                return "I"
            if DIM_RX["A"].search(msg):
                return "A"
            if isinstance(catalog, list) and catalog:
                codes = []
                for row in catalog:
                    if isinstance(row, dict):
                        c = row.get("impact_code") or row.get("code") or row.get("EN Code")
                        if isinstance(c, str) and c.strip():
                            codes.append(c.strip())
                for prefer in ("C", "I", "A"):
                    if prefer in codes:
                        return prefer
                for c in codes:
                    if c:
                        return c
            return default

        if need_code_fill:
            guessed = _choose_code_from_context("C")
            new_items = []
            for it in payload.items:
                code = (it.impact_code or "").strip() or guessed
                new_items.append(type(it)(impact_code=code, level=it.level, note=it.note))
            payload = ImpactSetPayload(asset_id=payload.asset_id, items=new_items)
            state.set_payload = payload  # persist normalized payload in state

        # --- Guard before WS emit / submit ---
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

        # Optimistic WS emit per item (non-blocking) — envelope with payload
        hub = getattr(deps, "ws_hub", None)
        if hub and hasattr(hub, "aemit_impact_set_level"):
            async def _emit_all():
                for it in payload.items:
                    # clamp into 0..4 (defensive)
                    try:
                        lvl = int(it.level)
                    except Exception:
                        lvl = 0
                    level = max(0, min(4, lvl))
                    evt = {
                        "event": "impact.set_level",
                        "payload": {
                            "org_id": state.org_id,
                            "asset_id": payload.asset_id,
                            "impact_code": it.impact_code,
                            "level_index": level,
                            "level_value": _LEVEL_VALUE[level],
                            **({"note": it.note} if it.note else {}),
                            "actor": state.user_id,
                            "ts": _now_iso(),
                        },
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

        # Build actions (tool-first)
        items_dump = [it.model_dump() for it in payload.items]
        state.actions = [{
            "op": "set_impacts",
            "mode": "merge",
            "asset_id": payload.asset_id,
            "items": items_dump,
        }]

        # Short inline confirmation
        last = payload.items[-1]
        try:
            lvl_i = int(last.level)
        except Exception:
            lvl_i = 0
        lvl_i = max(0, min(4, lvl_i))

        if loc == "cs":
            confirm = f"✅ Nastaven dopad {last.impact_code}={lvl_i} pro aktivum „{payload.asset_id}“."
        elif loc == "sk":
            confirm = f"✅ Nastavený dopad {last.impact_code}={lvl_i} pre aktívum „{payload.asset_id}“."
        else:
            confirm = f"✅ Set impact {last.impact_code}={lvl_i} for asset “{payload.asset_id}”."

        state.result = _ui_response(confirm, extra={"agent_events": state.agent_events})
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

        if not actions:
            state.result = _ui_response(_t(loc, "bulk_missing"), extra={"agent_events": state.agent_events})
            return state

        state.actions = actions

        # Optimistic WS emit — prefer batch/list event; fallback to per-item
        hub = getattr(deps, "ws_hub", None)
        if hub:
            async def _emit_bulk(entries: list[ImpactSetPayload]):
                for e in entries:
                    # 1) Try batch/list event: impact.set_levels with payload.impacts[]
                    if hasattr(hub, "aemit_impact_set_event"):
                        try:
                            impacts = []
                            for it in e.items:
                                level = max(0, min(4, int(it.level)))
                                item = {
                                    "impact_code": it.impact_code,
                                    "level_index": level,
                                    "level_value": _LEVEL_VALUE[level],
                                    **({"note": it.note} if it.note else {}),
                                }
                                impacts.append(item)
                            evt = {
                                "event": "impact.set_levels",
                                "payload": {
                                    "org_id": state.org_id,
                                    "asset_id": e.asset_id,
                                    "impacts": impacts,
                                    "actor": state.user_id,
                                    "ts": _now_iso(),
                                },
                            }
                            try:
                                _guard("WSImpactSetListEvent", evt)
                            except Exception:
                                pass
                            await hub.aemit_impact_set_event(evt)  # type: ignore[attr-defined]
                            continue  # next entry
                        except Exception:
                            pass
                    # 2) Fallback: per-item impact.set_level
                    if hasattr(hub, "aemit_impact_set_level"):
                        for it in e.items:
                            try:
                                level = max(0, min(4, int(it.level)))
                            except Exception:
                                level = 0
                            evt = {
                                "event": "impact.set_level",
                                "payload": {
                                    "org_id": state.org_id,
                                    "asset_id": e.asset_id,
                                    "impact_code": it.impact_code,
                                    "level_index": level,
                                    "level_value": _LEVEL_VALUE[level],
                                    **({"note": it.note} if it.note else {}),
                                    "actor": state.user_id,
                                    "ts": _now_iso(),
                                },
                            }
                            try:
                                _guard("WSImpactSetLevel", evt)
                            except Exception:
                                pass
                            try:
                                await hub.aemit_impact_set_level(evt)  # type: ignore[attr-defined]
                            except Exception:
                                pass
            _schedule(_emit_bulk(list(bulk.entries)))

        return state
    return _inner

def _upsert_impact(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    def _inner(state: ImpactState) -> ImpactState:
        # Localized language code (sk/cs/en)
        loc = _lang(state.locale)

        # Prefer structured items from state; fall back to inline JSON in the message:
        # {"items":[{code,name,category?,description?}, ...]}
        items = state.upsert_items or []
        if not items:
            try:
                raw = json.loads(state.message or "{}")
                src = (raw.get("items") or raw.get("upserts") or [])
                # lehká sanitace None → vynechat klíč
                clean = [{k: v for k, v in (x or {}).items() if v is not None} for x in src]
                items = [ImpactUpsertItem(**x) for x in clean]
            except Exception:
                items = []

        # If still empty, return a localized hint
        if not items:
            state.result = _ui_response(_t(loc, "upsert_missing"), extra={"agent_events": state.agent_events})
            return state

       # Normalize categories (resolve → fallback na name), ať neblokujeme tok
        normalized: list[ImpactUpsertItem] = []
        for it in items:
            category = (it.category or "").strip() or resolve_impact_category(code=it.code, name=it.name) or it.name
            normalized.append(
                ImpactUpsertItem(
                    code=it.code,
                    name=it.name,
                    category=category,
                    description=it.description,
                    weight=it.weight,
                )
            )
        items = normalized

        # Build the deterministic action
        action = {"op": "upsert_impacts", "items": [it.model_dump() for it in items]}
        try:
            _guard("ActionUpsertImpacts", action)
        except Exception as e:
            state.result = _ui_response(_t(loc, "validation_error", err=e), extra={"agent_events": state.agent_events})
            return state

        # Optimistic WS emit (batch preferred; per-item fallback)
        hub = getattr(deps, "ws_hub", None)

        async def _emit():
            if hub and hasattr(hub, "aemit_impact_upsert_batch"):
                evt = {
                    "event": "impact.upsert_batch",
                    "org_id": state.org_id,
                    "items": [it.model_dump() for it in items],
                    "actor": state.user_id,
                }
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
                    evt = {
                        "event": "impact.upsert",
                        "org_id": state.org_id,
                        "item": it.model_dump(),
                        "actor": state.user_id,
                    }
                    try:
                        _guard("WSImpactUpsert", evt)
                    except Exception:
                        pass
                    try:
                        await hub.aemit_impact_upsert(evt)  # type: ignore[attr-defined]
                    except Exception:
                        pass

        _schedule(_emit())

        # Queue the write for submit node
        state.actions = [action]
        return state

    return _inner


# -----------------------------------------------------------------------------
# Submit with backoff + jitter and outbox fallback (deterministic submit)
# -----------------------------------------------------------------------------

async def _submit_changeset_async(deps: GraphDeps, state: ImpactState) -> ImpactState:
    loc = _lang(state.locale)
    acts = state.actions or []
    if not acts:
        state.result = _ui_response(_t(loc, "submitted_generic", count=0), extra={"agent_events": state.agent_events})
        return state

    # Build typed ChangeSet
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

    n_set    = sum(1 for a in acts if a.get("op") == "set_impacts")
    n_upsert = sum(1 for a in acts if a.get("op") == "upsert_impacts")
    text = _t(loc, "submitted_summary", set=n_set, upsert=n_upsert) if (n_set or n_upsert) \
           else _t(loc, "submitted_generic", count=len(acts))
    state.result = _ui_response(text, extra={"changeset_actions": len(acts), "agent_events": state.agent_events})
    state.actions = None
    state.change = None
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
# Help / clarify
# -----------------------------------------------------------------------------

def _clarify() -> Callable[[ImpactState], ImpactState]:
    def _inner(state: ImpactState) -> ImpactState:
        loc = _lang(state.locale)
        msg = state.message or ""
        t = _norm(msg)
        # jednoduchý hint podle vzoru threats
        if EXPLAIN_RX.search(t):
            a, b = "EXPLAIN", "GET"
        elif any(rx.search(t) for rx in DIM_RX.values()):
            a, b = "SET", "GET"
        else:
            a, b = "GET", "LIST"
        txt = _t(loc, "greet") + f"\n→ {a}/{b}?"
        state.result = _ui_response(txt, extra={"agent_events": state.agent_events})
        return state
    return _inner

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
    Missing selection for get/set triggers suggest-select (clarify-first).
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
    graph.add_node("clarify", _clarify())
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
        intent = state.intent or "clarify"  # clarify-first default
        if intent in ("help", "chitchat"):
            return "help"
        if intent == "clarify":
            return "clarify"
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
        return "help"

    graph.add_conditional_edges(
        "route",
        _route,
        {
            "help": "help",
            "clarify": "clarify",
            "list_impacts": "list_impacts",
            "asset_get_impacts": "asset_get_impacts",
            "set_impact_level": "set_impact_level",
            "bulk_set_impact_levels": "bulk_set_impact_levels",
            "upsert_impact": "upsert_impact",
            "explain_impacts": "explain_impacts",
        },
    )

    # Write → submit (send-now semantics)
    graph.add_edge("set_impact_level", "submit_changeset")
    graph.add_edge("bulk_set_impact_levels", "submit_changeset")
    graph.add_edge("upsert_impact", "submit_changeset")

    # Terminal
    for node in ("help", "clarify", "list_impacts", "asset_get_impacts", "explain_impacts", "submit_changeset"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=deps.checkpointer)

__all__ = ["ImpactState", "create_graph"]
