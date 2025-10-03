# File: workshops_ai/graph/asset_graph.py
# -----------------------------------------------------------------------------
# LangGraph: ASSETS tab — LLM-first conversational assistant with SEND-NOW writes
#
# 
# - Clarify-first NLU (no “list” hard default). If selection is missing → suggest-select.
# - Multilingual short replies (SK/CZ/EN) with deterministic tool calls.
# - SEND-NOW write semantics (create_asset / set_asset_meta) → submit_changeset with backoff & outbox.
# - Optimistic WS emits (asset.created / asset.updated) — best effort, schema-tolerant.
# -----------------------------------------------------------------------------

from __future__ import annotations

from workshops_ai.graph.nlu import (
    parse_nlu_for_tab,
    ensure_action_is_string,
    Tab,
)

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

from pydantic import BaseModel, Field

from workshops_ai.graph.factory import GraphDeps

from workshops_ai.app.services.outbox_writer import submit_or_fallback, SubmitOrFallbackResult

# LangGraph
try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("langgraph must be installed to use asset_graph") from e

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
    # Everything (models + deterministic builder) now lives in models/changeset.py
    from workshops_ai.models.changeset import (  # type: ignore
        ChangeSet,
        ActionCreateAsset,
        ActionSetAssetMeta,
        build_changeset,
    )
except Exception:  # pragma: no cover
    # Lightweight fallback for dev: returns a dict matching the ChangeSet JSON shape
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

try:
    from workshops_ai.tools.guard import guard_validate_payload as _guard_validate  # type: ignore
except Exception:  # pragma: no cover
    def _guard_validate(model_cls, payload):  # type: ignore
        # Best-effort fallback; accept payload as-is
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

Intent = Literal["help", "clarify", "chitchat", "list", "get", "create", "set_meta", "explain", "list_types"]


class ToolEvent(TypedDict, total=False):
    event_type: str            # "tool.call.start" | "tool.call.end"
    tool: str                  # tool/node name
    ok: bool                   # success flag
    duration_ms: int           # only for "end"
    meta: dict                 # arbitrary metadata

class AssetCreateItem(BaseModel):
    id: Optional[str] = None
    name: str
    assetType: Optional[str] = None
    isPrimary: Optional[bool] = None
    confidentiality: Optional[int] = Field(default=None, ge=0, le=4)
    integrity: Optional[int] = Field(default=None, ge=0, le=4)
    availability: Optional[int] = Field(default=None, ge=0, le=4)
    description: Optional[str] = None

class AssetUpdateItem(BaseModel):
    asset_id: str
    assetType: Optional[str] = None
    isPrimary: Optional[bool] = None
    confidentiality: Optional[int] = Field(default=None, ge=0, le=4)
    integrity: Optional[int] = Field(default=None, ge=0, le=4)
    availability: Optional[int] = Field(default=None, ge=0, le=4)
    description: Optional[str] = None

class AssetState(BaseModel):
    # Request context
    org_id: str
    active_tab: str = Field(default="assets")
    message: Optional[str] = None
    selection: Optional[dict] = None   # {"asset_id": "..."} | {"asset_ids": [...]} | {"assets": [...]}
    user_id: Optional[str] = None
    locale: Optional[str] = None       # "sk" | "cs" | "en"
    language: Optional[str] = None
    history: Optional[list[dict]] = None  # short rolling window (3–5 turns), optional

    # Structured inputs
    action: Optional[Intent] = None
    create_items: Optional[list[AssetCreateItem]] = None
    update_items: Optional[list[AssetUpdateItem]] = None
    question: Optional[str] = None  # for explain

    # Working outputs
    intent: Optional[Intent] = None
    actions: Optional[list[dict]] = None
    change: Optional[ChangeSet | dict] = None
    result: Optional[dict] = None

    # Telemetry
    agent_events: list[ToolEvent] = Field(default_factory=list)

# -----------------------------------------------------------------------------
# i18n helpers
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
            "sk": "Ahoj! Môžem: „zoznam aktív“, „ukáž detail“, „vytvor aktívum“, „zmeň metadata“, „vysvetli“. 😉",
            "cs": "Ahoj! Umím: „seznam aktiv“, „ukaž detail“, „vytvoř aktivum“, „změň metadata“, „vysvětli“. 😉",
            "en": "Hi! I can: “list assets”, “show details”, “create asset”, “update metadata”, or “explain”. 😉",
        },
        "list_fail": {
            "sk": "Zoznam aktív sa nepodarilo načítať: {err}",
            "cs": "Seznam aktiv se nepodařilo načíst: {err}",
            "en": "Failed to list assets: {err}",
        },
        "list_found": {
            "sk": "Našiel som {n} aktív:",
            "cs": "Našel jsem {n} aktiv:",
            "en": "Found {n} assets:",
        },
        "select_prompt": {
            "sk": "Vyber prosím aktívum. Tu je pár možností:",
            "cs": "Vyber prosím aktivum. Tady je pár možností:",
            "en": "Please select an asset. Here are a few options:",
        },
        "asset_not_found": {
            "sk": "Aktívum „{id}“ sa nenašlo.",
            "cs": "Aktivum „{id}“ se nenašlo.",
            "en": "Asset “{id}” not found.",
        },
        "asset_load_fail": {
            "sk": "Čítanie aktíva „{id}“ zlyhalo: {err}",
            "cs": "Načtení aktiva „{id}“ selhalo: {err}",
            "en": "Failed to load asset “{id}”: {err}",
        },
        "create_missing": {
            "sk": "Nie je čo vytvoriť. Zadaj aspoň name.",
            "cs": "Není co vytvořit. Uveď alespoň name.",
            "en": "Nothing to create. Provide at least name.",
        },
        "update_missing": {
            "sk": "Nie je čo zmeniť. Uveď asset_id a aspoň jedno pole.",
            "cs": "Není co změnit. Uveď asset_id a alespoň jedno pole.",
            "en": "Nothing to update. Provide asset_id and at least one field.",
        },
        "validation_error": {
            "sk": "Chyba validácie: {err}",
            "cs": "Chyba validace: {err}",
            "en": "Validation error: {err}",
        },
        "submitted_summary": {
            "sk": "{created} vytvorených, {updated} upravených.",
            "cs": "{created} vytvořených, {updated} upravených.",
            "en": "{created} created, {updated} updated.",
        },
        "submitted_generic": {
            "sk": "{count} akcií odoslaných.",
            "cs": "{count} akcí odesláno.",
            "en": "{count} actions submitted.",
        },
        "submit_failed": {
            "sk": "Odoslanie zlyhalo, zmeny boli uložené lokálne. Dôvod: {err}",
            "cs": "Odeslání selhalo, změny byly uloženy lokálně. Důvod: {err}",
            "en": "Submission failed; changes were queued locally. Reason: {err}",
        },
        "explain_need_question": {
            "sk": "Zadaj otázku (state.question alebo message) pre vysvetlenie aktív.",
            "cs": "Zadej otázku (state.question nebo message) pro vysvětlení aktiv.",
            "en": "Provide a question (state.question or message) to explain assets.",
        },
        "explain_no_sources": {
            "sk": "Momentálne nemám kontext a LLM fallback nevrátil odpoveď.",
            "cs": "Momentálně nemám kontext a LLM fallback nevrátil odpověď.",
            "en": "No context is available and the LLM fallback did not return an answer.",
        },
         "types_fail": {
            "sk": "Katalóg typov aktív sa nepodarilo načítať: {err}",
            "cs": "Katalog typů aktiv se nepodařilo načíst: {err}",
            "en": "Failed to load asset type catalog: {err}",
        },
        "types_found": {
            "sk": "Môžeš vytvoriť tieto typy aktív: {types}",
            "cs": "Můžeš vytvořit tato aktiva (typy): {types}",
            "en": "You can create these asset types: {types}",
        },
        "types_empty": {
            "sk": "V katalógu nie sú žiadne typy aktív.",
            "cs": "V katalogu nejsou žádné typy aktiv.",
            "en": "No asset types found in the catalog.",
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
        # nejsme v async loopu → spusť to hned (bez warningu)
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

def _event(state: AssetState, *, kind: str, tool: str, ok: bool = True, meta: dict | None = None) -> float:
    t0 = time.perf_counter()
    state.agent_events.append({"event_type": kind, "tool": tool, "ok": ok, "meta": meta or {}})
    return t0

def _finish(state: AssetState, t0: float, *, tool: str, ok: bool, meta: dict | None = None) -> None:
    dur = int((time.perf_counter() - t0) * 1000)
    state.agent_events.append({"event_type": "tool.call.end", "tool": tool, "ok": ok, "duration_ms": dur, "meta": meta or {}})

# -----------------------------------------------------------------------------
# Planner (NLU) — LLM-first with regex fallback (clarify-first)
# -----------------------------------------------------------------------------

GREET_RX    = re.compile(r"^\s*(ahoj|cau|čau|hej|nazdar|dobr(y|ý)\s+de[nň]|hello|hi|hey|good\s+(morning|afternoon|evening))[\s!.]*$", re.I)
LIST_RX     = re.compile(r"\b(list|seznam|zoznam|all\s+assets)\b", re.I)
GET_RX      = re.compile(r"\b(get|show|detail|ukaz|zobraz)\b", re.I)
CREATE_RX   = re.compile(r"\b(create|new|add|vytvor|zalo[zž]|pridaj|přidej)\b", re.I)
UPDATE_RX   = re.compile(r"\b(update|set|change|zmen|nastav|uprav)\b", re.I)
EXPLAIN_RX  = re.compile(r"\b(explain|vysvetli|vysvětli|popis|why|context|kontext)\b", re.I)
LIST_TYPES_RX = re.compile(
    r"(co\s+se\s+d[aá]\s+vytvo[rř]it|jak[eé]\s+typy\s+akt(i|í)v|list\s+types|what\s+can\s+i\s+create|available\s+asset\s+types)",
    re.I
)


class NLUResult(BaseModel):
    action: Optional[Intent] = None
    # For create
    create_items: Optional[list[AssetCreateItem]] = None
    # For set_meta
    update_items: Optional[list[AssetUpdateItem]] = None
    # For explain
    question: Optional[str] = None

def _get_llm_from_deps(deps: GraphDeps):
    """
    Resolve an async-callable LLM client.

    - Prefer deps.llm_client (a ChatOpenAI-like client exposing .ainvoke).
    - Do NOT return deps.llm (often a settings/config object).
    - Finally, build a default ChatOpenAI client if available.
    """
    llm_client = getattr(deps, "llm_client", None)
    if llm_client is not None:
        return llm_client
    if not _HAS_LLM:
        return None
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, timeout=12, max_retries=1)  # type: ignore[call-arg]

def _history_tail(state: AssetState, k: int = 5) -> list[dict]:
    try:
        h = state.history or []
        return h[-k:]
    except Exception:
        return []

async def _nlu(deps: GraphDeps, state: AssetState) -> AssetState:
    raw = (state.message or "").strip()
    selection_id = (state.selection or {}).get("asset_id")

    # Explicit action passthrough
    if state.action in ("help", "chitchat", "list", "get", "create", "set_meta", "explain"):
        state.intent = cast(Intent, state.action)
        return state

    # Greeting/help
    if not raw or GREET_RX.match(raw):
        state.intent = "help"
        return state

    llm = _get_llm_from_deps(deps)
    if llm is not None:
        system = (
            "You are a strict NLU for an 'assets' assistant.\n"
            "Primary languages: Slovak (SK) and English (EN); Czech (CZ) tolerated.\n"
            "Extract intent and slots and return ONLY compact JSON with fields:\n"
            "  action: one of [help, chitchat, list, get, create, set_meta, explain]\n"
            "  create_items?: [{id?, name, assetType?, isPrimary?, confidentiality?, integrity?, availability?, description?}]  # for create\n"
            "  update_items?: [{asset_id, assetType?, isPrimary?, confidentiality?, integrity?, availability?, description?}]   # for set_meta\n"
            "  question?: string                                                                                               # for explain\n"
            "Rules:\n"
            "- Never invent IDs.\n"
            "- If a UI selection.asset_id is present, you MAY copy it into update_items[].asset_id when helpful.\n"
            "- Detect greetings as 'chitchat'/'help'.\n"
            "- CIA numeric scale is 0..4.\n"
            "- Output STRICT JSON; no prose."
        )
        human = {
            "message": raw,
            "selection": {"asset_id": selection_id} if selection_id else None,
            "locale": state.locale,
            "history": _history_tail(state, 5),
        }
        try:
            resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=json.dumps(human))])  # type: ignore
            txt = (getattr(resp, "content", None) or "").strip()
            start, end = txt.find("{"), txt.rfind("}")
            data = json.loads(txt[start:end + 1] if start != -1 and end != -1 else txt)
            n = NLUResult(**data)

            if n.action:
                state.intent = n.action
            if n.create_items and (state.create_items is None):
                state.create_items = n.create_items
            if n.update_items and (state.update_items is None):
                # fill missing asset_id from selection if useful
                if selection_id:
                    for it in n.update_items:
                        if not it.asset_id:
                            it.asset_id = selection_id
                state.update_items = n.update_items
            if n.question and not state.question:
                state.question = n.question

            if state.intent:
                return state
        except Exception as e:
            logger.debug("Assets NLU LLM failed, fallback to regex: %s", e)

    # Regex fallback (clarify-first)
    t = _norm(raw)
    if EXPLAIN_RX.search(t):
        state.intent = "explain"
    elif LIST_TYPES_RX.search(t):
        state.intent = "list_types"
    elif CREATE_RX.search(t):
        state.intent = "create"
    elif UPDATE_RX.search(t):
        state.intent = "set_meta"
    elif GET_RX.search(t):
        state.intent = "get"
    elif LIST_RX.search(t):
        state.intent = "list"
    else:
        state.intent = "clarify"

    # Minor extraction for fallback
    if state.intent == "set_meta" and state.update_items is None and selection_id:
        state.update_items = [AssetUpdateItem(asset_id=selection_id)]
    return state

# -----------------------------------------------------------------------------
# Suggest-select (if missing asset selection)
# -----------------------------------------------------------------------------

def _suggest_select(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
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
            msg = "\n".join([_t(loc, "select_prompt")] + preview) if preview else _t(loc, "list_found", n=0)
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events, "assets_preview": preview})
            _finish(state, t0, tool="suggest_select", ok=True, meta={"count": len(preview)})
        except Exception as e:
            _finish(state, t0, tool="suggest_select", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "list_fail", err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner

# -----------------------------------------------------------------------------
# Read-only nodes
# -----------------------------------------------------------------------------
def _list_asset_types(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="list_asset_types")
        try:
            # prefer unified port; pokud máš jen deps.backend_port, použij ten
            types = deps.backend_port.list_asset_types()  # type: ignore[attr-defined]
            types = types or []
            if types:
                msg = _t(loc, "types_found", types=", ".join(types))
            else:
                msg = _t(loc, "types_empty")
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events, "asset_types": types})
            _finish(state, t0, tool="list_asset_types", ok=True, meta={"count": len(types)})
        except Exception as e:
            _finish(state, t0, tool="list_asset_types", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "types_fail", err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner


def _assets_list(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="assets_list")
        try:
            items = deps.backend_port.assets_list(org_id=state.org_id) or []
            lines = [_t(loc, "list_found", n=len(items))]
            for a in items[:30]:
                d = a.model_dump() if hasattr(a, "model_dump") else dict(a)
                lines.append(f"- {d.get('id') or d.get('asset_id')}: {d.get('name','Unnamed')} [{d.get('assetType','-')}]")
            state.result = _ui_response("\n".join(lines), extra={"assets": [getattr(a, "model_dump", lambda: dict(a))() for a in items[:50]], "agent_events": state.agent_events})
            _finish(state, t0, tool="assets_list", ok=True, meta={"count": len(items)})
        except Exception as e:
            _finish(state, t0, tool="assets_list", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "list_fail", err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner

def _asset_get_or_suggest(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        asset_id = (state.selection or {}).get("asset_id")
        if not asset_id:
            return _suggest_select(deps)(state)

        t0 = _event(state, kind="tool.call.start", tool="asset_get", meta={"asset_id": asset_id})
        try:
            asset = deps.backend_port.asset_get(asset_id)
            if not asset:
                _finish(state, t0, tool="asset_get", ok=False, meta={"error": "not_found"})
                state.result = _ui_response(_t(loc, "asset_not_found", id=asset_id), extra={"agent_events": state.agent_events})
                return state

            d = asset.model_dump() if hasattr(asset, "model_dump") else dict(asset)
            cia = {"C": d.get("confidentiality"), "I": d.get("integrity"), "A": d.get("availability")}
            msg = f"{d.get('name') or asset_id} [{d.get('assetType','-')}], CIA={cia}, primary={bool(d.get('isPrimary'))}"
            state.result = _ui_response(msg, extra={"asset": d, "agent_events": state.agent_events})
            _finish(state, t0, tool="asset_get", ok=True, meta={"asset_id": asset_id})
        except Exception as e:
            _finish(state, t0, tool="asset_get", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "asset_load_fail", id=asset_id, err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner

# -----------------------------------------------------------------------------
# Write nodes (emit-first) → actions → submit
# -----------------------------------------------------------------------------

def _validate_write_actions(actions: list[dict]) -> tuple[list[dict], Optional[str]]:
    validated: list[dict] = []
    for a in actions:
        op = (a.get("op") or "").strip()
        try:
            if op == "create_asset":
                payload = {"op": "create_asset", "client_token": a.get("client_token"), "fields": a.get("fields") or {}}
                if not payload["fields"].get("name"):
                    return [], "create_asset requires fields.name."
                validated.append(_guard_validate(ActionCreateAsset, payload))  # type: ignore[arg-type]
            elif op == "set_asset_meta":
                payload = {"op": "set_asset_meta", "asset_id": a.get("asset_id"), "fields": a.get("fields") or {}}
                if not payload["asset_id"]:
                    return [], "set_asset_meta requires asset_id."
                if not payload["fields"]:
                    return [], "set_asset_meta requires at least one field."
                validated.append(_guard_validate(ActionSetAssetMeta, payload))  # type: ignore[arg-type]
            else:
                return [], f"Unsupported operation '{op}'."
        except Exception as e:
            return [], f"Write action validation failed for op='{op}': {e}"
    return validated, None

def _create_asset(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        items = state.create_items or []
        if not items:
            # tiny inline JSON fallback: [{"name": "...", "assetType":"..."}]
            try:
                arr = json.loads(state.message or "[]")
                if isinstance(arr, list):
                    items = [AssetCreateItem(**x) for x in arr]
            except Exception:
                pass
        if not items:
            state.result = _ui_response(_t(loc, "create_missing"), extra={"agent_events": state.agent_events})
            return state

        actions: list[dict] = []
        last_name = None
        for it in items:
            name = (it.name or "").strip()
            if not name:
                continue
            fields = {
                "id": it.id,
                "name": name,
                "assetType": it.assetType,
                "isPrimary": it.isPrimary,
                "confidentiality": it.confidentiality,
                "integrity": it.integrity,
                "availability": it.availability,
                "description": it.description,
            }
            fields = {k: v for k, v in fields.items() if v is not None}
            actions.append({"op": "create_asset", "client_token": f"tmp-{uuid.uuid4().hex[:10]}", "fields": fields})
            last_name = name

        validated, err = _validate_write_actions(actions)
        if err:
            state.result = _ui_response(_t(loc, "validation_error", err=err), extra={"agent_events": state.agent_events})
            return state

        # Optimistic WS emits (best effort): asset.created
        hub = getattr(deps, "ws_hub", None)
        if hub:
            async def _emit_all():
                for a in validated:
                    f = (a.get("fields") or {})
                    evt = {"event": "asset.created", "payload": { "org_id": state.org_id, **f, "actor": state.user_id, "ts": _now_iso() }}
                    try:
                        _guard("WSAssetCreated", evt)
                    except Exception:
                        pass
                    try:
                        if hasattr(hub, "aemit_asset_event"):
                            await hub.aemit_asset_event(evt)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            _schedule(_emit_all)

        state.actions = validated
        # Short inline confirmation
        confirm = f"✅ Done: created asset “{last_name or '-'}”."
        state.result = _ui_response(confirm, extra={"agent_events": state.agent_events})
        return state
    return _inner

def _set_asset_meta(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        items = state.update_items or []
        if not items:
            # tiny inline JSON fallback: [{"asset_id":"...", "assetType":"..."}]
            try:
                arr = json.loads(state.message or "[]")
                if isinstance(arr, list):
                    items = [AssetUpdateItem(**x) for x in arr]
            except Exception:
                pass
        if not items:
            if not (state.selection and state.selection.get("asset_id")):
                return _suggest_select(deps)(state)
            state.result = _ui_response(_t(loc, "update_missing"), extra={"agent_events": state.agent_events})
            return state

        actions: list[dict] = []
        last_id = None
        for it in items:
            fields = {
                "assetType": it.assetType,
                "isPrimary": it.isPrimary,
                "confidentiality": it.confidentiality,
                "integrity": it.integrity,
                "availability": it.availability,
                "description": it.description,
            }
            fields = {k: v for k, v in fields.items() if v is not None}
            if fields:
                actions.append({"op": "set_asset_meta", "asset_id": it.asset_id, "fields": fields})
                last_id = it.asset_id

        if not actions:
            state.result = _ui_response(_t(loc, "update_missing"), extra={"agent_events": state.agent_events})
            return state

        validated, err = _validate_write_actions(actions)
        if err:
            state.result = _ui_response(_t(loc, "validation_error", err=err), extra={"agent_events": state.agent_events})
            return state

        # Optimistic WS emits (best effort): asset.updated
        hub = getattr(deps, "ws_hub", None)
        if hub:
            async def _emit_all():
                for a in validated:
                    f = (a.get("fields") or {})
                    evt = {"event": "asset.updated", "payload": {"org_id": state.org_id, "asset_id": a.get("asset_id"), **f, "actor": state.user_id, "ts": _now_iso()}}
                    try:
                        _guard("WSAssetUpdated", evt)
                    except Exception:
                        pass
                    try:
                        if hasattr(hub, "aemit_asset_event"):
                            await hub.aemit_asset_event(evt)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            _schedule(_emit_all())

        state.actions = validated
        confirm = f"✅ Done: updated asset {last_id or '-'}."
        state.result = _ui_response(confirm, extra={"agent_events": state.agent_events})
        return state
    return _inner

# -----------------------------------------------------------------------------
# Submit changeset
# -----------------------------------------------------------------------------


async def _submit_changeset_async(deps: GraphDeps, state: AssetState) -> AssetState:
    loc = _lang(state.locale)
    acts = state.actions or []
    if not acts:
        state.result = _ui_response(_t(loc, "submitted_generic", count=0), extra={"agent_events": state.agent_events})
        return state

    # Build a typed ChangeSet (deterministický idempotency_key je uvnitř builderu)
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
        # Ensure we pass a typed ChangeSet (in case a dev fallback returned a dict)
        cs_typed = cs if hasattr(cs, "model_dump") else ChangeSet.model_validate(cs)

        # Unified submit with built-in backoff + atomic outbox fallback
        result: SubmitOrFallbackResult = await submit_or_fallback(
            changeset=cs_typed,
            submitter=submitter,
            # outbox_dir=None → uses settings.tazila.outbox_path and fallbacks
        )

        if result.submitted:
            transport = (
                result.submit_result.transport
                if (result.submit_result and getattr(result.submit_result, "transport", None))
                else "unknown"
            )
            _finish(
                state, t0, tool="submit_changeset", ok=True,
                meta={"actions": len(acts), "transport": transport}
            )
        else:
            # Fell back to outbox
            _finish(
                state, t0, tool="submit_changeset", ok=False,
                meta={"fallback_path": str(result.fallback_path)}
            )
            state.result = _ui_response(
                _t(loc, "submit_failed", err="queued to outbox"),
                extra={"agent_events": state.agent_events, "outbox_path": str(result.fallback_path)}
            )
            return state

    except Exception as e:
        _finish(state, t0, tool="submit_changeset", ok=False, meta={"error": str(e)})
        state.result = _ui_response(_t(loc, "submit_failed", err=e), extra={"agent_events": state.agent_events})
        return state

    n_create = sum(1 for a in acts if a.get("op") == "create_asset")
    n_update = sum(1 for a in acts if a.get("op") == "set_asset_meta")
    if n_create or n_update:
        text = _t(loc, "submitted_summary", created=n_create, updated=n_update)
    else:
        text = _t(loc, "submitted_generic", count=len(acts))

    # Memory hints 
    try:
        touched_ids = [a.get("asset_id") for a in acts if a.get("op") == "set_asset_meta" and a.get("asset_id")]
        created_names = [a.get("fields", {}).get("name") for a in acts if a.get("op") == "create_asset"]
        if touched_ids:
            _schedule(_amemory_set(deps, f"{state.org_id}:assets.last_touched_ids", touched_ids))
        if created_names:
            _schedule(_amemory_set(deps, f"{state.org_id}:assets.last_created_names", [n for n in created_names if n]))
    except Exception:
        pass

    state.result = _ui_response(text, extra={"changeset_actions": len(acts), "agent_events": state.agent_events})
    return state

# -----------------------------------------------------------------------------
# Explain (RAG → LLM synthesis)
# -----------------------------------------------------------------------------

async def _explain_assets_async(deps: GraphDeps, state: AssetState) -> AssetState:
    loc = _lang(state.locale)
    q = (state.question or state.message or "").strip()
    if not q:
        state.result = _ui_response(_t(loc, "explain_need_question"), extra={"agent_events": state.agent_events})
        return state

    retriever = None
    try:
        if hasattr(deps, "make_retriever") and callable(deps.make_retriever):
            retriever = deps.make_retriever(state.org_id, "assets")
    except Exception:
        retriever = None

    snippets: list[dict] = []
    if retriever:
        t0 = _event(state, kind="tool.call.start", tool="retriever.search", meta={"tab": "assets"})
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
            "You are a cybersecurity assistant. Synthesize a concise answer (≤120 words) "
            "grounded ONLY in the provided snippets. Primary languages: SK/EN (CZ tolerated)."
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
        system = "You are a cybersecurity assistant. Provide a concise (≤120 words) answer without hallucinations."
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

def _help_or_greet() -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        state.result = _ui_response(_t(loc, "greet"), extra={"agent_events": state.agent_events})
        return state
    return _inner

# -----------------------------------------------------------------------------
# Graph assembly
# -----------------------------------------------------------------------------

def create_graph(deps: GraphDeps):
    """
    Build & compile the ASSETS graph.
    Flow:
      start → nlu → route → {help|list|get|create|set_meta|explain} → (submit?) → END
    Missing selection for get/set_meta triggers suggest-select (clarify-first).
    """
    graph = StateGraph(AssetState)

    # Nodes
    async def _nlu_node(state: AssetState) -> AssetState:
        return await _nlu(deps, state)

    async def _submit_node(state: AssetState) -> AssetState:
        return await _submit_changeset_async(deps, state)

    async def _explain_node(state: AssetState) -> AssetState:
        return await _explain_assets_async(deps, state)

    graph.add_node("nlu", _nlu_node)
    graph.add_node("route", lambda s: s)

    graph.add_node("help", _help_or_greet())
    graph.add_node("chitchat", _help_or_greet())
    graph.add_node("suggest_select", _suggest_select(deps))

    graph.add_node("assets_list", _assets_list(deps))
    graph.add_node("list_asset_types", _list_asset_types(deps))
    graph.add_node("asset_get", _asset_get_or_suggest(deps))

    graph.add_node("create_asset", _create_asset(deps))
    graph.add_node("set_asset_meta", _set_asset_meta(deps))

    graph.add_node("submit_changeset", _submit_node)
    graph.add_node("explain", _explain_node)

    # Entry
    graph.set_entry_point("nlu")
    graph.add_edge("nlu", "route")

    # Routing
    def _route(state: AssetState) -> str:
        intent = state.intent or "clarify"
        if intent in ("help", "chitchat"):
            return "help"
        if intent == "list":
            return "assets_list"
        if intent == "list_types":
            return "list_asset_types"
        if intent == "get":
            return "asset_get"
        if intent == "create":
            return "create_asset"
        if intent == "set_meta":
            return "set_asset_meta"
        if intent == "explain":
            return "explain"
        return "help"

    graph.add_conditional_edges(
        "route",
        _route,
        {
            "help": "help",
            "assets_list": "assets_list",
            "list_asset_types": "list_asset_types",
            "asset_get": "asset_get",
            "create_asset": "create_asset",
            "set_asset_meta": "set_asset_meta",
            "explain": "explain",
        },
    )

    # Write → submit (send-now semantics)
    graph.add_edge("create_asset", "submit_changeset")
    graph.add_edge("set_asset_meta", "submit_changeset")

    # Terminal
    for node in ("help", "assets_list", "asset_get", "explain", "submit_changeset", "list_asset_types"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=deps.checkpointer)

__all__ = ["AssetState", "create_graph"]
