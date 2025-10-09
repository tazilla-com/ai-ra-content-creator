# File: workshops_ai/graph/asset_graph.py
# -----------------------------------------------------------------------------
# LangGraph: ASSETS tab — LLM-first assistant (short, localized, one-call-per-turn)
#
# Goals
# - Use deps.llm_client (no local LLM instantiation, no regex planners).
# - Clarify-first NLU via agent_decide (exactly one tool choice per turn).
# - Short localized replies (cs/sk/en).
# - Writes go through typed ChangeSet from models/changeset (no local fallback builder).
# - Submit via deps.submit_changeset (which itself handles online + OUTBOX fallback).
# - Keep tool-call telemetry and avoid multi-step cascades in a single turn.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import csv
import re
import difflib
import asyncio
import datetime as dt
import json
import logging
import time
import uuid
from typing import Any, Callable, Literal, Optional, cast, TYPE_CHECKING
if TYPE_CHECKING:
    from workshops_ai.graph.factory import GraphDeps
from workshops_ai.schemas.ui_contract import UIRequest, UIResponse

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from workshops_ai.graph.agent_node import agent_decide, AgentDecision
from workshops_ai.rag.port import Scope
# ChangeSet (typed actions; source of truth)
from workshops_ai.models.changeset import (
    ChangeSet,
    ActionCreateAsset,
    ActionSetAssetMeta,
    build_changeset,
)

# LangGraph
try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("langgraph must be installed to use asset_graph") from e

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Optional payload guards (best-effort)
# -----------------------------------------------------------------------------
try:
    from workshops_ai.tools.guard import guard_validate_payload as _guard_validate  # type: ignore
except Exception:  # pragma: no cover
    def _guard_validate(model_cls, payload):  # type: ignore
        return payload


# -----------------------------------------------------------------------------
# State & minimal models
# -----------------------------------------------------------------------------
Intent = Literal["help", "clarify", "chitchat", "list", "get", "create", "set_meta", "explain", "list_types"]


class ToolEvent(TypedDict, total=False):
    event_type: str            # "tool.call.start" | "tool.call.end" | "tool.call.success" | "tool.call.fail"
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

    # Conversational create 
    create_draft: Optional[dict] = None
    pending_fields: Optional[list[str]] = None

    # Telemetry
    agent_events: list[ToolEvent] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# i18n helpers (short, localized)
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
            "sk": "Ahoj! Viem ti pomôcť so zoznamom aktív, detailmi, vytváraním nových položiek, úpravou metaúdajov a zrozumiteľným vysvetlením.",
            "cs": "Ahoj! Umím ti pomoct se seznamem aktiv, detaily, vytvořením nových položek, úpravou metadat a srozumitelným vysvětlením.",
            "en": "Hi! I can help you list assets, view details, create new items, update metadata, and explain things clearly.",
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
            "sk": "Nemám kontext a LLM fallback nevrátil odpoveď.",
            "cs": "Nemám kontext a LLM fallback nevrátil odpověď.",
            "en": "No context is available and the LLM fallback did not return an answer.",
        },
        "types_fail": {
            "sk": "Katalóg typov aktív sa nepodarilo načítať: {err}",
            "cs": "Katalog typů aktiv se nepodařilo načíst: {err}",
            "en": "Failed to load asset type catalog: {err}",
        },
        "types_found": {
            "sk": "Môžeš vytvoriť tieto typy: {types}",
            "cs": "Můžeš vytvořit tyto typy: {types}",
            "en": "You can create these types: {types}",
        },
        "types_empty": {
            "sk": "V katalógu nie sú žiadne typy aktív.",
            "cs": "V katalogu nejsou žádné typy aktiv.",
            "en": "No asset types found.",
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

# --- Catalog helpers (CSV: data/catalogs/asset_types.csv) --------------------
def _catalog_asset_types(deps: GraphDeps) -> list[str]:
    """
    Load asset types from CSV if available; otherwise fallback to backend_port.list_asset_types().
    Returns a flat list of strings.
    """
    try:
        paths = getattr(getattr(deps, "settings", object()), "paths", None)
        cat_dir = getattr(paths, "catalogs_dir", None)
    except Exception:
        cat_dir = None

    if not cat_dir:
        # Fallback when settings.paths.catalogs_dir is not set
        try:
            types = deps.backend_port.list_asset_types()  # type: ignore[attr-defined]
            return [t for t in (types or []) if isinstance(t, str)]
        except Exception:
            return []

    path = os.path.join(cat_dir, "asset_types.csv")
    if not os.path.exists(path):
        # CSV missing → fallback to backend
        try:
            types = deps.backend_port.list_asset_types()  # type: ignore[attr-defined]
            return [t for t in (types or []) if isinstance(t, str)]
        except Exception:
            return []

    out: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            cell = (row[0] or "").strip()
            # Ignore commented or empty lines
            if cell and not cell.lower().startswith("#"):
                out.append(cell)
    return out


def _match_asset_type(user_text: str, types: list[str]) -> Optional[str]:
    """
    Try exact, startswith/substring, then fuzzy (difflib) match of user_text to a known asset type.
    Returns the best match or None.
    """
    if not user_text or not types:
        return None
    low = user_text.strip().lower()
    for t in types:
        if t.strip().lower() == low:
            return t
    for t in types:
        if t.strip().lower().startswith(low) or low in t.strip().lower():
            return t
    cand = difflib.get_close_matches(user_text, types, n=1, cutoff=0.6)
    return cand[0] if cand else None

def _parse_level(text: Optional[str | int]) -> Optional[int]:
    """
    Normalize level to an int 0..4.
    Accepts:
      - int in range 0..4
      - first digit 0..4 found in the text
    Returns None if no digit 0..4 is present.
    """
    if text is None:
        return None
    if isinstance(text, int):
        return max(0, min(4, text))
    m = re.search(r"\b([0-4])\b", str(text))
    return int(m.group(1)) if m else None

def _friendly_catalog_bullets(types: list[str]) -> str:
    """Render list of types as bullet points for the chat message."""
    if not types:
        return "-"
    return "\n".join([f"- {t}" for t in types])

def _ask(loc: str, prompt_cs: str, prompt_sk: str, prompt_en: str) -> str:
    """Pick a localized prompt string; default to EN."""
    if loc == "cs":
        return prompt_cs
    if loc == "sk":
        return prompt_sk
    return prompt_en

def _extract_create_draft(*, deps: GraphDeps, text: str, loc: str, asset_types: list[str]) -> dict:
    """
    Ask the LLM to extract a STRICT JSON draft from free-form text.
    The model is constrained to a small whitelist of keys and must output JSON ONLY.
    Any uncertain field must be null.
    """
    system = (
        "You convert a short user request about creating an asset into STRICT JSON.\n"
        "Keys allowed: name (str), assetType (str|null), isPrimary (bool|null), "
        "confidentiality (int 0-4|null), integrity (int 0-4|null), availability (int 0-4|null), "
        "description (str|null).\n"
        "Rules:\n"
        "- If you are NOT SURE about a value, output null.\n"
        "- assetType MUST be one from the provided list or null.\n"
        "- confidentiality/integrity/availability MUST be integers 0..4 or null.\n"
        "- Output JSON ONLY, no markdown."
    )
    human = {
        "locale": loc,
        "user_text": text,
        "asset_types": asset_types,
        "hints": {
            "levels": "0..4 where 0=none/public and 4=critical/secret"
        },
    }
    try:
        raw = _llm_text(
            deps=deps,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": json.dumps(human)}],
            temperature=getattr(deps, "llm_temperature", None) or 0.0,
        )
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return {}
        # Normalize levels
        for k in ("confidentiality", "integrity", "availability"):
            obj[k] = _parse_level(obj.get(k))
        # Validate/match assetType to known catalog
        at = obj.get("assetType")
        if isinstance(at, str) and at:
            match = _match_asset_type(at, asset_types)
            obj["assetType"] = match
        return obj
    except Exception:
        return {}

def _list_asset_types(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="asset.list_types")
        try:
            # Prefer CSV catalog, fallback to backend if not available
            types = _catalog_asset_types(deps)
            msg = _t(loc, "types_found", types=", ".join(types)) if types else _t(loc, "types_empty")
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events, "asset_types": types})
            _finish(state, t0, tool="asset.list_types", ok=True, meta={"count": len(types)})
        except Exception as e:
            _finish(state, t0, tool="asset.list_types", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "types_fail", err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner



# -----------------------------------------------------------------------------
# Utilities + telemetry
# -----------------------------------------------------------------------------
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
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _event(state: AssetState, *, kind: str, tool: str, ok: bool = True, meta: dict | None = None) -> float:
    t0 = time.perf_counter()
    state.agent_events.append({"event_type": kind, "tool": tool, "ok": ok, "meta": meta or {}})
    return t0


def _finish(state: AssetState, t0: float, *, tool: str, ok: bool, meta: dict | None = None) -> None:
    dur = int((time.perf_counter() - t0) * 1000)
    state.agent_events.append(
        {"event_type": ("tool.call.success" if ok else "tool.call.fail"), "tool": tool, "ok": ok, "duration_ms": dur, "meta": meta or {}}
    )


# -----------------------------------------------------------------------------
# LLM bridge (deps.llm_client only; no local ChatOpenAI)
# -----------------------------------------------------------------------------
def _llm_text(
    *,
    deps: GraphDeps,
    messages: list[dict[str, str]],
    temperature: float | None = 0.2,
    model: Optional[str] = None,
) -> str:
    """
    Minimal, robust bridge for LangChain ChatModel clients (e.g., langchain_openai.ChatOpenAI).

    Rules:
    - Prefer LangChain `.invoke()` FIRST.
    - Convert dict messages -> LangChain BaseMessage (System/Human/AI).
    - Do NOT mix in OpenAI SDK branches here.
    - Provide a very conservative fallback and fail loudly if empty.
    """
    client = getattr(deps, "llm_client", None) or getattr(deps, "llm", None)
    if client is None:
        raise RuntimeError("LLM client is not configured (deps.llm_client is None).")

    temp = temperature if temperature is not None else getattr(deps, "llm_temperature", None) or 0.2

    # 1) LangChain ChatModel via .invoke([...BaseMessage...]) — PRIORITY
    try:
        invoke = getattr(client, "invoke", None)
        if callable(invoke):
            try:
                # LangChain >= 0.1 / 0.3 import path
                from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # type: ignore
            except Exception:
                # Older alias (kept for compatibility)
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
                    # Unknown roles: degrade to HumanMessage to stay safe.
                    lc_messages.append(HumanMessage(content=content))

            # Some LC models accept kwargs; if not, call without them.
            try:
                resp = client.invoke(lc_messages, temperature=temp)  # type: ignore[misc]
            except TypeError:
                resp = client.invoke(lc_messages)  # type: ignore[misc]

            txt = (getattr(resp, "content", None) or "").strip()
            if txt:
                return txt
    except Exception:
        # Move on to fallback; do NOT try direct OpenAI SDK here.
        pass

    # 2) Conservative generic fallback: call and extract .content/.text/str
    try:
        obj = client(messages=messages, temperature=temp, model=model)  # type: ignore[misc]
        if isinstance(obj, str):
            return obj.strip()
        txt = (getattr(obj, "content", None) or getattr(obj, "text", None) or str(obj) or "").strip()
        if txt:
            return txt
    except Exception:
        pass

    # 3) If everything failed, make it explicit.
    raise RuntimeError("LLM bridge failed: unsupported client shape or empty content.")

def _ask_llm(*, deps: GraphDeps, loc: str, draft: dict, types: list[str]) -> str:
    """
    Ask EXACTLY ONE next question to complete asset creation.
    Output: a single, friendly question in the user's locale (cs/sk/en), no preamble, no markdown fences.
    The model must:
    - If assetType is missing, ask the user to choose from the provided list and INCLUDE the list as short bullet points.
    - If name is missing, ask for the asset name.
    - If both type+name are missing, combine into one question (still concise) and include the list.
    - If CIA fields (confidentiality/integrity/availability) are missing, ask for them with NUMERIC scale 0–4
      (explain briefly: 0 = nejméně/lowest, 4 = nejvíce/highest). Ask for all missing CIA in one line.
    - If isPrimary is missing, ask yes/no in the same tone.
    - Keep it short (≤2 lines if possible), friendly, no over-explaining unless user asked.
    - Do not fabricate values; only ask.
    """
    # Locale hint
    lang = "cs" if loc == "cs" else ("sk" if loc == "sk" else "en")

    system = (
        "You are a concise, friendly assistant helping a user create an asset.\n"
        "Your job is to ask exactly ONE next question needed to complete the data.\n"
        "Language: " + lang + ".\n"
        "Tone: brief, helpful, not verbose. Output ONLY the question text.\n"
        "If asking about CIA, require integers 0–4 where 0=lowest and 4=highest.\n"
        "If asking about assetType, show the provided catalog as bullet points.\n"
    )

    human_payload = {
        "known_draft": {
            "name": draft.get("name"),
            "assetType": draft.get("assetType"),
            "isPrimary": draft.get("isPrimary"),
            "confidentiality": draft.get("confidentiality"),
            "integrity": draft.get("integrity"),
            "availability": draft.get("availability"),
            "description": draft.get("description"),
        },
        "catalog_asset_types": types,
        "rules": {
            "cia_numeric": "Use integers 0..4 only (0=lowest, 4=highest).",
            "yes_no_primary": True,
            "combine_when_possible": True,
            "one_question_only": True,
        },
    }

    question = _llm_text(
        deps=deps,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(human_payload)},
        ],
        temperature=getattr(deps, "llm_temperature", None) or 0.2,
    )
    return question.strip()

# -----------------------------------------------------------------------------
# Planner (agent_decide → one tool per turn)
# -----------------------------------------------------------------------------
def _history_tail(state: AssetState, k: int = 5) -> list[dict]:
    try:
        h = state.history or []
        return h[-k:]
    except Exception:
        return []


async def _nlu(deps: GraphDeps, state: AssetState) -> AssetState:
    raw = (state.message or "").strip()

    # Explicit passthrough
    if state.action in ("help", "chitchat", "list", "get", "create", "set_meta", "explain", "list_types"):
        state.intent = cast(Intent, state.action)
        return state

    if not raw:
        state.intent = "help"
        return state

    sel = state.selection or {}

    # Allowed (aliases are normalized inside agent_decide)
    allowed = [
        "llm.explain",
        "asset.list",
        "asset.list_types",
        "asset.create",
        "asset.set_meta",
        "asset.update_meta",
        "asset.get", 
    ]

    decision: AgentDecision = await agent_decide(
        deps=deps,
        tab="assets",
        message=raw,
        language=state.locale or "en",
        history=_history_tail(state, 5),
        selection=sel,
        allowed_actions=allowed,
        one_action_only=True,
    )

    logger.debug("assets.decide -> action=%s args=%s", getattr(decision, "action", None), getattr(decision, "args", None))

    # Tool → intent
    tool_to_intent = {
        "asset.list": "list",
        "asset.list_types": "list_types",
        "asset.create": "create",
        "asset.set_meta": "set_meta",
        "asset.update_meta": "set_meta",
        "llm.explain": "explain",
    }

    state.intent = cast(Intent, tool_to_intent.get((decision.action or "").strip(), "help"))

    # Args → structured
    if getattr(decision, "args", None):
        args = decision.args or {}
        if "items" in args and state.intent == "create" and isinstance(args["items"], list):
            try:
                state.create_items = [AssetCreateItem(**it) for it in args["items"]]
            except Exception:
                state.create_items = None

        if state.intent == "set_meta":
            # unified shape: either single 'update' or array 'items'
            upd_items: list[dict] = []
            if isinstance(args.get("update"), dict):
                upd_items.append(args["update"])
            if isinstance(args.get("items"), list):
                upd_items.extend([it for it in args["items"] if isinstance(it, dict)])
            if upd_items:
                try:
                    state.update_items = [AssetUpdateItem(**it) for it in upd_items]
                except Exception:
                    state.update_items = None
                # auto-fill from selection
                sel_id = sel.get("asset_id")
                if sel_id and state.update_items:
                    for it in state.update_items:
                        if not it.asset_id:
                            it.asset_id = str(sel_id)

        if "question" in args and isinstance(args["question"], str):
            state.question = args["question"]

    return state


# -----------------------------------------------------------------------------
# Read-only nodes
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


def _assets_list(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="asset.list")
        try:
            items = deps.backend_port.assets_list(org_id=state.org_id) or []
            lines = [_t(loc, "list_found", n=len(items))]
            out_items = []
            for a in items[:30]:
                d = a.model_dump() if hasattr(a, "model_dump") else dict(a)
                lines.append(f"- {d.get('id') or d.get('asset_id')}: {d.get('name','Unnamed')} [{d.get('assetType','-')}]")
                out_items.append(d)
            state.result = _ui_response("\n".join(lines), extra={"assets": out_items, "agent_events": state.agent_events})
            _finish(state, t0, tool="asset.list", ok=True, meta={"count": len(items)})
        except Exception as e:
            _finish(state, t0, tool="asset.list", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "list_fail", err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner


def _asset_get_or_suggest(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        asset_id = (state.selection or {}).get("asset_id")
        if not asset_id:
            return _suggest_select(deps)(state)

        t0 = _event(state, kind="tool.call.start", tool="asset.get", meta={"asset_id": asset_id})
        try:
            asset = deps.backend_port.asset_get(asset_id)
            if not asset:
                _finish(state, t0, tool="asset.get", ok=False, meta={"error": "not_found"})
                state.result = _ui_response(_t(loc, "asset_not_found", id=asset_id), extra={"agent_events": state.agent_events})
                return state

            d = asset.model_dump() if hasattr(asset, "model_dump") else dict(asset)
            cia = {"C": d.get("confidentiality"), "I": d.get("integrity"), "A": d.get("availability")}
            msg = f"{d.get('name') or asset_id} [{d.get('assetType','-')}], CIA={cia}, primary={bool(d.get('isPrimary'))}"
            state.result = _ui_response(msg, extra={"asset": d, "agent_events": state.agent_events})
            _finish(state, t0, tool="asset.get", ok=True, meta={"asset_id": asset_id})
        except Exception as e:
            _finish(state, t0, tool="asset.get", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "asset_load_fail", id=asset_id, err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Write nodes (build typed actions → (later) submit via deps.submit_changeset)
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

        # Conversational creation: prefer building a draft over legacy `items`
        types = _catalog_asset_types(deps)

        # Start/continue a draft persisted in the graph state
        draft = dict(state.create_draft or {})

        # Try to extract/merge user-provided info from the current message via LLM
        if state.message:
            try:
                inc = _extract_create_draft(deps=deps, text=state.message or "", loc=loc, asset_types=types)
                # Merge only non-null keys to avoid overwriting known values with nulls
                for k, v in (inc or {}).items():
                    if v is not None:
                        draft[k] = v
            except Exception:
                # If extraction fails, we continue with the draft as-is
                pass

        # Minimum required for creation: name + assetType
        missing: list[str] = []
        if not (draft.get("name") and isinstance(draft.get("name"), str) and draft["name"].strip()):
            missing.append("name")
        if not (draft.get("assetType") and isinstance(draft.get("assetType"), str)):
            missing.append("assetType")

        # If anything is missing, ask exactly ONE question and finish the turn without actions
        if missing:
            question = _ask_llm(deps=deps, loc=loc, draft=draft, types=types)
            state.create_draft = draft
            state.pending_fields = missing
            state.actions = None  # critical: prevents the submit node from running
            state.result = _ui_response(
                question,
                extra={
                    "agent_events": state.agent_events,
                    "asset_types": types or None,
                    "draft": draft,
                },
            )
            return state

        # Draft is minimally complete; optional CIA/isPrimary may remain unset (that’s fine)
        fields = {
            "id": draft.get("id"),
            "name": draft.get("name"),
            "assetType": draft.get("assetType"),
            "isPrimary": draft.get("isPrimary"),
            "confidentiality": _parse_level(draft.get("confidentiality")),
            "integrity": _parse_level(draft.get("integrity")),
            "availability": _parse_level(draft.get("availability")),
            "description": draft.get("description"),
        }
        # Remove nulls so we only send provided values
        fields = {k: v for k, v in fields.items() if v is not None}

        actions = [{"op": "create_asset", "client_token": f"tmp-{uuid.uuid4().hex[:10]}", "fields": fields}]
        validated, err = _validate_write_actions(actions)
        if err:
            state.result = _ui_response(_t(loc, "validation_error", err=err), extra={"agent_events": state.agent_events})
            return state

        # Best-effort optimistic WS event
        hub = getattr(deps, "ws_hub", None)
        if hub:
            async def _emit_all():
                for a in validated:
                    f = (a.get("fields") or {})
                    evt = {"event": "asset.created", "payload": {"org_id": state.org_id, **f, "actor": state.user_id, "ts": _now_iso()}}
                    try:
                        _guard_validate("WSAssetCreated", evt)  # type: ignore[arg-type]
                    except Exception:
                        pass
                    try:
                        if hasattr(hub, "aemit_asset_event"):
                            await hub.aemit_asset_event(evt)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            _schedule(_emit_all)

        # Ready to submit
        state.actions = validated
        state.create_draft = None
        state.pending_fields = None
        confirm = f"✅ Created “{fields.get('name') or '-'}”."
        changes_json = {"kind": "changeset.actions", "tab": "assets", "actions": validated, "ts": _now_iso()}
        state.result = _ui_response(confirm, extra={"agent_events": state.agent_events, "messages": [confirm, changes_json]})
        return state

    return _inner




def _set_asset_meta(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        items = state.update_items or []
        if not items:
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

        # Optimistic WS emits (best effort)
        hub = getattr(deps, "ws_hub", None)
        if hub:
            async def _emit_all():
                for a in validated:
                    f = (a.get("fields") or {})
                    evt = {
                        "event": "asset.updated",
                        "payload": {"org_id": state.org_id, "asset_id": a.get("asset_id"), **f, "actor": state.user_id, "ts": _now_iso()},
                    }
                    try:
                        _guard_validate("WSAssetUpdated", evt)  # type: ignore[arg-type]
                    except Exception:
                        pass
                    try:
                        if hasattr(hub, "aemit_asset_event"):
                            await hub.aemit_asset_event(evt)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            _schedule(_emit_all)

        state.actions = validated
        confirm = f"✅ Updated {last_id or '-'}."
        changes_json = {"kind": "changeset.actions", "tab": "assets", "actions": validated, "ts": _now_iso()}
        state.result = _ui_response(confirm, extra={"agent_events": state.agent_events, "messages": [confirm, changes_json]})
        return state
    return _inner
    
async def run_create(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    """
    Thin runner for `asset.create` (kept for routers).
    Builds and executes the LangGraph for assets; produces a valid UIResponse.
    """
    # Implementace 1: pokud už máš společný manager-entry pro assets, deleguj:
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

async def run_update_meta(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    """
    Thin runner for `asset.update_meta`.
    """
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

# -----------------------------------------------------------------------------
# Submit changeset (via deps.submit_changeset)
# -----------------------------------------------------------------------------
async def _submit_changeset_async(deps: GraphDeps, state: AssetState) -> AssetState:
    """
    Submit the accumulated write actions as a typed ChangeSet.
    - On success: publish a 'changeset.submitted' WS event and include compact details in response.messages.
    - On failure (queued to outbox): publish a 'changeset.queued' WS event and attach a JSON message so the demo UI
      shows it in the right-side Events panel.
    """
    loc = _lang(state.locale)

    # 1) Nothing to submit → short "0 actions" response
    acts = state.actions or []
    if not acts:
        state.result = _ui_response(
            _t(loc, "submitted_generic", count=0),
            extra={"agent_events": state.agent_events},
        )
        return state

    # 2) Build the typed ChangeSet and pin it onto state for visibility
    cs: ChangeSet | dict = build_changeset(
        org_id=state.org_id,
        user_id=state.user_id or "unknown",
        actions=acts,
        intent="apply",
    )
    state.change = cs

    # 3) Ensure we have a submit function
    submit_fn = getattr(deps, "submit_changeset", None)
    if not callable(submit_fn):
        state.result = _ui_response(
            _t(loc, "submit_failed", err="submit_changeset not available"),
            extra={"agent_events": state.agent_events},
        )
        return state

    # 4) Emit telemetry: starting the submit tool call
    t0 = _event(state, kind="tool.call.start", tool="submit_changeset", meta={"actions": len(acts)})

    try:
        # 5) Validate/normalize to typed model for the submit layer
        cs_typed = cs if hasattr(cs, "model_dump") else ChangeSet.model_validate(cs)

        # 6) Submit with soft-failure (will return queued/outbox info on network errors)
        submit_result = await submit_fn(cs_typed, require_success=False)  # type: ignore[misc]

        ok = bool(getattr(submit_result, "submitted", False))
        if ok:
            # -------- SUCCESS PATH --------
            transport = getattr(getattr(submit_result, "submit_result", None), "transport", None) or "unknown"
            _finish(state, t0, tool="submit_changeset", ok=True, meta={"actions": len(acts), "transport": transport})

            # Try to dump the changeset for echo & WS payloads
            try:
                cs_dump = cs_typed.model_dump(mode="json") if hasattr(cs_typed, "model_dump") else dict(cs_typed)  # type: ignore[arg-type]
            except Exception:
                cs_dump = {"_error": "cannot_dump_changeset"}

            # Optional WS broadcast so the UI can reflect the success in real time
            hub = getattr(deps, "ws_hub", None)
            if hub:
                payload = {
                    "tab": "assets",
                    "submitted": True,
                    "transport": transport,
                    "actions": len(acts),
                    "changeset": cs_dump,
                    "ts": _now_iso(),
                    "org_id": state.org_id,
                    "actor": state.user_id,
                }
                try:
                    if hasattr(hub, "publish_agent_event"):
                        # publish_agent_event(org_id, event, payload, mode)
                        await hub.publish_agent_event(state.org_id, "changeset.submitted", payload, "raw_json")  # type: ignore[attr-defined]
                    elif hasattr(hub, "aemit_asset_event"):
                        await hub.aemit_asset_event({"event": "changeset.submitted", "payload": payload})  # type: ignore[attr-defined]
                except Exception:
                    # WS is best-effort; never fail the turn on a WS error
                    pass

            # Human-friendly summary text
            n_create = sum(1 for a in acts if a.get("op") == "create_asset")
            n_update = sum(1 for a in acts if a.get("op") == "set_asset_meta")
            text = (
                _t(loc, "submitted_summary", created=n_create, updated=n_update)
                if (n_create or n_update) else
                _t(loc, "submitted_generic", count=len(acts))
            )

            # Echo compact details into the chat response (right panel will also log JSON items)
            submit_info = {"submitted": True, "transport": transport}
            state.result = _ui_response(
                text,
                extra={
                    "changeset_actions": len(acts),
                    "agent_events": state.agent_events,
                    "messages": [
                        text,  # string → visible as a chat bubble
                        {"kind": "changeset", "changeset": cs_dump, "ts": _now_iso()},
                        {"kind": "submit_result", "result": submit_info, "ts": _now_iso()},
                    ],
                },
            )
            return state

        else:
            # -------- QUEUED / OUTBOX PATH --------
            fallback_path = getattr(submit_result, "fallback_path", None)
            meta = {"fallback_path": str(fallback_path) if fallback_path else None}
            _finish(state, t0, tool="submit_changeset", ok=False, meta=meta)

            # Prepare a JSON message for the right panel (non-string entries in `messages` are logged there)
            try:
                cs_dump = cs_typed.model_dump(mode="json") if hasattr(cs_typed, "model_dump") else dict(cs_typed)  # type: ignore[arg-type]
            except Exception:
                cs_dump = {"_error": "cannot_dump_changeset"}

            msg_json = {
                "kind": "changeset.queued",
                "tab": "assets",
                "outbox_path": meta.get("fallback_path"),
                "changeset": cs_dump,
                "ts": _now_iso(),
                "org_id": state.org_id,
                "actor": state.user_id,
            }

            # Also broadcast over WS so an open websocket shows it live in the Events panel
            hub = getattr(deps, "ws_hub", None)
            if hub:
                try:
                    if hasattr(hub, "publish_agent_event"):
                        await hub.publish_agent_event(state.org_id, "changeset.queued", msg_json, "raw_json")  # type: ignore[attr-defined]
                    elif hasattr(hub, "aemit_asset_event"):
                        await hub.aemit_asset_event({"event": "changeset.queued", "payload": msg_json})  # type: ignore[attr-defined]
                except Exception:
                    pass  # best-effort

            # Build the response: a concise text plus the JSON message (for the right panel)
            state.result = _ui_response(
                _t(loc, "submit_failed", err="queued to outbox"),
                extra={
                    "agent_events": state.agent_events,
                    "messages": [msg_json],  # non-string → demo UI logs this JSON in the right-hand Events panel
                },
            )
            return state

    except Exception as e:
        # -------- HARD FAILURE PATH (unexpected exception) --------
        _finish(state, t0, tool="submit_changeset", ok=False, meta={"error": str(e)})
        state.result = _ui_response(
            _t(loc, "submit_failed", err=e),
            extra={"agent_events": state.agent_events},
        )
        return state

# -----------------------------------------------------------------------------
# Explain (RAG → deps.llm_client synthesis; short localized)
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
            retriever = deps.make_retriever(Scope(org_id=state.org_id, tab="assets"))
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

    client = getattr(deps, "llm_client", None)
    if snippets and client is not None:
        tagged = [{"id": f"S{i+1}", "text": s["text"], "source": s.get("source")} for i, s in enumerate(snippets[:5])]
        system = (
            "You are a precise cybersecurity assistant.\n"
            "- Answer briefly (≤120 words) in the user's locale (cs/sk/en).\n"
            "- GROUND EVERY CLAIM in provided snippets; include inline [S#] citations.\n"
            "- If unsupported, state you don't have evidence. Use bullet points when listing."
        )
        human_payload = {"question": q, "locale": loc, "snippets": tagged}
        t0 = _event(state, kind="tool.call.start", tool="llm.synthesize", meta={"snippets": len(tagged)})
        try:
            txt = _llm_text(
                deps=deps,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(human_payload)}],
                temperature=getattr(deps, "llm_temperature", None) or 0.2,
            )
            if txt:
                _finish(state, t0, tool="llm.synthesize", ok=True, meta={})
                state.result = _ui_response(txt, extra={"snippets": tagged, "agent_events": state.agent_events})
                return state
        except Exception as e:
            _finish(state, t0, tool="llm.synthesize", ok=False, meta={"error": str(e)})

    # Fallback: plain concise answer (if client exists)
    if client is not None:
        system = "You are a cybersecurity assistant. Provide a concise (≤120 words) answer without hallucinations."
        human = {"question": q, "locale": loc}
        t0 = _event(state, kind="tool.call.start", tool="llm.explain")
        try:
            txt = _llm_text(
                deps=deps,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(human)}],
                temperature=getattr(deps, "llm_temperature", None) or 0.2,
            )
            if txt:
                _finish(state, t0, tool="llm.explain", ok=True, meta={})
                state.result = _ui_response(txt, extra={"agent_events": state.agent_events})
                return state
        except Exception as e:
            _finish(state, t0, tool="llm.explain", ok=False, meta={"error": str(e)})

    state.result = _ui_response(_t(loc, "explain_no_sources"), extra={"snippets": (snippets or None), "agent_events": state.agent_events})
    return state


# -----------------------------------------------------------------------------
# Help / chitchat (short)
# -----------------------------------------------------------------------------
def _help_or_greet() -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        state.result = _ui_response(_t(loc, "greet"), extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Graph assembly (one tool call per turn; writes → submit)
# -----------------------------------------------------------------------------
def create_graph(deps: GraphDeps):
    """
    Build & compile the ASSETS graph.
    Flow:
      start → nlu → route → {help|list|list_types|get|create|set_meta|explain} → (submit?) → END

    Notes
    - Each turn performs at most one external action from the user's perspective.
    - Writes: node builds actions → submit node (single submit per turn).
    - Read-only routes return short, localized summaries.
    """
    graph = StateGraph(AssetState)

    # Async node wrappers
    async def _nlu_node(state: AssetState) -> AssetState:
        return await _nlu(deps, state)

    async def _submit_node(state: AssetState) -> AssetState:
        return await _submit_changeset_async(deps, state)

    async def _explain_node(state: AssetState) -> AssetState:
        return await _explain_assets_async(deps, state)

    # Nodes
    graph.add_node("nlu", _nlu_node)
    graph.add_node("route", lambda s: s)

    graph.add_node("help", _help_or_greet())
    graph.add_node("chitchat", _help_or_greet())
    graph.add_node("suggest_select", _suggest_select(deps))

    graph.add_node("asset.list", _assets_list(deps))
    graph.add_node("asset.list_types", _list_asset_types(deps))
    graph.add_node("asset.get", _asset_get_or_suggest(deps))

    graph.add_node("asset.create", _create_asset(deps))
    graph.add_node("asset.update_meta", _set_asset_meta(deps))

    graph.add_node("submit_changeset", _submit_node)
    graph.add_node("llm.explain", _explain_node)


    # Entry
    graph.set_entry_point("nlu")
    graph.add_edge("nlu", "route")

    # Routing
    def _route(state: AssetState) -> str:
        intent = state.intent or "clarify"
        if intent in ("help", "chitchat"):
            return "help"
        if intent == "list":
            return "asset.list"
        if intent == "list_types":
            return "asset.list_types"
        if intent == "get":
            return "asset.get"
        if intent == "create":
            return "asset.create"
        if intent == "set_meta":
            return "asset.update_meta"
        if intent == "explain":
            return "llm.explain"
        return "help"

    graph.add_conditional_edges(
        "route",
        _route,
            {
                "help": "help",
                "clarify": "help",                        # nebo vlastní "clarify" node, pokud jej máš
                "asset.list": "asset.list",
                "asset.list_types": "asset.list_types",
                "asset.get": "asset.get",
                "asset.create": "asset.create",
                "asset.update_meta": "asset.update_meta",
                "llm.explain": "llm.explain",

                # --- ALIASY (legacy / raw) ---
                "asset.set_meta": "asset.update_meta",    # ← hlavní fix pro tvou chybu
                "set_meta": "asset.update_meta",          # ← když by NLU vrátilo „surový“ intent
                "get": "asset.get",
                "create": "asset.create",
                "explain": "llm.explain",
                "list": "asset.list",
                "list_types": "asset.list_types",
            },
    )

    def _after_create(state: AssetState) -> str:
        # If we have actions, proceed to submission; otherwise end the turn (we just asked a question).
        return "submit_changeset" if state.actions else "END"

    graph.add_conditional_edges(
    "asset.create",
    _after_create,
    {"submit_changeset": "submit_changeset", "END": END},
    )

    graph.add_edge("asset.update_meta", "submit_changeset")
    
    # Terminal nodes
    for node in ("help", "chitchat", "asset.list", "asset.list_types", "asset.get",
             "llm.explain", "submit_changeset"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=deps.checkpointer)


async def run_explain(*, deps: GraphDeps, text: str, locale: Optional[str] = None) -> dict:
    """
    Router-facing explain for ASSETS. Thin shim that reuses our internal async node.
    Returns the same UI payload the graph would place into state.result.
    """
    # Build a minimal state and reuse our internal node
    state = AssetState(
        org_id=getattr(getattr(deps, "settings", object()), "org_id", "demo-org"),  # safe default
        active_tab="assets",
        message=text,
        question=text,
        locale=locale or "en",
        agent_events=[],
    )
    state = await _explain_assets_async(deps, state)
    return state.result or {"ok": False, "error": "no_result"}

__all__ = ["AssetState", "create_graph", "run_create", "run_update_meta", "run_explain"]
