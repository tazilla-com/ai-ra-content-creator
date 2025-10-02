# File: workshops_ai/graph/asset_graph.py
# -----------------------------------------------------------------------------
# LangGraph: ASSETS tab — tool-choice agent + LLM-first conversational assistant
#
# Flow:
#   start → agent_decide → agent_route → {assets_list | asset_get | create_asset
#         | set_meta | explain} → (submit_changeset?) → END
# Missing selection for get/set/explain triggers suggest-select.
#
# External contracts unchanged:
# - Deterministic use of backend_port + ws_hub
# - ChangeSet submission with backoff; on failure → JSON outbox

from __future__ import annotations

import asyncio
import datetime as dt
import time
import json
import logging
import os
import random
import re
import unicodedata
import uuid
from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Callable, Literal, Optional, Sequence, cast
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, TypeAdapter

from workshops_ai.graph.factory import GraphDeps
from workshops_ai.api.schema.asset import Asset
from workshops_ai.graph.agent_node import agent_decide

# LangGraph
try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("langgraph must be installed to use asset_graph") from e

# Optional LLM (NLU + explain fallback)
_HAS_LLM = True
try:
    from langchain_openai import ChatOpenAI  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
except Exception:  # pragma: no cover
    _HAS_LLM = False
    ChatOpenAI = object  # type: ignore
    SystemMessage = HumanMessage = object  # type: ignore

# ChangeSet & write actions (+ guard)
try:
    from workshops_ai.models.changeset import (  # type: ignore
        ChangeSet,
        ActionCreateAsset,
        ActionSetAssetMeta,
    )
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

try:
    from workshops_ai.tools.guard import guard_validate_payload as _guard_validate  # type: ignore
except Exception:  # pragma: no cover
    def _guard_validate(model_cls, payload):  # type: ignore
        if hasattr(model_cls, "model_validate"):
            return model_cls.model_validate(payload).model_dump()
        return payload

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Types & State
# -----------------------------------------------------------------------------

Intent = Literal["help", "chitchat", "list", "get", "create", "set_meta", "explain"]


class ToolEvent(TypedDict, total=False):
    event_type: str            # "tool.call.start" | "tool.call.end"
    tool: str                  # tool/node name
    ok: bool                   # success flag
    duration_ms: int           # only for "end"
    meta: dict                 # arbitrary metadata


class AssetState(BaseModel):
    # Request context
    org_id: str
    active_tab: str = Field(default="assets")
    message: Optional[str] = None
    selection: Optional[dict] = None  # {"asset_id": "..."} | {"asset_ids": [...]} | {"assets": [...]}
    user_id: Optional[str] = None
    locale: Optional[str] = None  # "sk", "cs", "en" (or full locales like "sk-SK")

    # Structured inputs (legacy planner; still supported)
    action: Optional[Intent] = None
    new_asset: Optional[dict] = None   # when action == "create"
    update: Optional[dict] = None      # when action == "set_meta"

    # Working outputs
    intent: Optional[Intent] = None
    actions: Optional[list[dict]] = None
    change: Optional[Any] = None
    result: Optional[dict] = None

    # Telemetry
    agent_events: list[ToolEvent] = Field(default_factory=list)

    # Scratchpad for agent
    tmp: dict = Field(default_factory=dict)

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
            "sk": "Ahoj! Povedz, čo chceš s aktívami: „vytvor“, „zmeň“, „ukáž detail“ alebo „vysvetli“. 😉",
            "cs": "Ahoj! Řekni, co chceš s aktivy: „vytvoř“, „změň“, „ukaž detail“ nebo „vysvětli“. 😉",
            "en": "Hi! Tell me what you want to do with assets: “create”, “update”, “show details”, or “explain”. 😉",
        },
        "list_fail": {
            "sk": "Zoznam aktív sa nepodarilo načítať: {err}",
            "cs": "Nepodařilo se načíst seznam aktiv: {err}",
            "en": "Failed to list assets: {err}",
        },
        "list_found": {
            "sk": "Našiel som {n} aktív.",
            "cs": "Našel jsem {n} aktiv.",
            "en": "Found {n} assets.",
        },
        "select_prompt": {
            "sk": "Vyber prosím aktívum. Tu je pár možností:",
            "cs": "Vyber prosím aktivum. Tady je pár možností:",
            "en": "Please select an asset. Here are a few options:",
        },
        "asset_not_found": {
            "sk": "Aktívum „{id}“ som nenašiel.",
            "cs": "Aktivum „{id}“ jsem nenašel.",
            "en": "Asset “{id}” not found.",
        },
        "asset_load_fail": {
            "sk": "Načítanie aktíva „{id}“ zlyhalo: {err}",
            "cs": "Načtení aktiva „{id}“ selhalo: {err}",
            "en": "Failed to load asset “{id}”: {err}",
        },
        "nothing_to_create": {
            "sk": "Nie je čo vytvoriť. Uveď aspoň názov (name).",
            "cs": "Není co vytvořit. Uveď alespoň název (name).",
            "en": "Nothing to create. Provide at least a name.",
        },
        "nothing_to_update": {
            "sk": "Nie je čo zmeniť. Uveď cieľové aktíva a aspoň jedno pole.",
            "cs": "Není co změnit. Uveď cílová aktiva a alespoň jedno pole.",
            "en": "Nothing to update. Provide target asset ids and at least one field.",
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
        "submit_failed_queued": {
            "sk": "Odoslanie zlyhalo, zmeny som uložil do outboxu. Dôvod: {err}",
            "cs": "Odeslání selhalo, změny jsem uložil do outboxu. Důvod: {err}",
            "en": "Submission failed; changes were queued locally. Reason: {err}",
        },
        "explain_select_first": {
            "sk": "Vyber najprv konkrétne aktívum (selection.asset_id).",
            "cs": "Nejprve vyber konkrétní aktivum (selection.asset_id).",
            "en": "Please select a specific asset (selection.asset_id) first.",
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
# Low-level utils
# -----------------------------------------------------------------------------

def _norm(text: str) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii").lower()


def _safe_emit(ws_hub: Any, event: str, payload: dict) -> None:
    try:
        if hasattr(ws_hub, "emit"):
            ws_hub.emit(event, payload)
        elif hasattr(ws_hub, "broadcast"):
            ws_hub.broadcast(event, payload)
    except Exception as e:  # pragma: no cover
        logger.debug("WS emit failed (%s): %s", event, e)


def _schedule(coro):
    try:
        asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        pass


async def _amemory_set(deps: GraphDeps, key: str, value: Any) -> None:
    fn = getattr(deps, "memory_set", None)
    if callable(fn):
        res = fn(key, value)
        if asyncio.iscoroutine(res):
            await res


async def _amemory_get(deps: GraphDeps, key: str) -> Any:
    fn = getattr(deps, "memory_get", None)
    if callable(fn):
        res = fn(key)
        return await res if asyncio.iscoroutine(res) else res
    return None


def _to_asset_dict(a: Any) -> dict:
    if a is None:
        return {}
    if isinstance(a, dict):
        return a
    try:
        from pydantic import BaseModel as _BM
        if isinstance(a, _BM):  # type: ignore
            return a.model_dump()
    except Exception:
        pass
    if is_dataclass(a):
        try:
            return asdict(a)
        except Exception:
            pass
    try:
        return {"id": getattr(a, "id", None) or getattr(a, "asset_id", None), "name": getattr(a, "name", None)}
    except Exception:
        return {}

# CIA normalization for preview
_CONF_INDEX_TO_LABEL = {1: "PUBLIC", 2: "INTERNAL", 3: "PROTECTED", 4: "STRICTLY_PROTECTED"}
_IA_INDEX_TO_LABEL = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}
TA_ASSET = TypeAdapter(Asset)

def _first_nonempty(*vals: Any) -> Any | None:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None


def _as_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _num_conf_to_label(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return _CONF_INDEX_TO_LABEL.get(int(v))
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"none", "null"}:
            return None
        return s.upper().replace("-", "_")
    return None


def _num_ia_to_label(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return _IA_INDEX_TO_LABEL.get(int(v))
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"none", "null"}:
            return None
        return s.upper()
    return None


def _to_asset_dto(item: Any, *, org_id: str | None = None) -> Asset:
    data = item.model_dump() if hasattr(item, "model_dump") else dict(item)
    payload: dict[str, Any] = {}
    _org = _first_nonempty(org_id, data.get("org_id"))
    if _org:
        payload["org_id"] = str(_org)
    _id = _first_nonempty(data.get("id"), data.get("asset_id"))
    if _id:
        payload["id"] = str(_id)
    _name = _first_nonempty(data.get("name"), data.get("title"))
    if _name:
        payload["name"] = str(_name)
    _desc = _first_nonempty(data.get("description"), data.get("desc"))
    if _desc is not None:
        payload["description"] = _desc
    _prim = _as_bool(_first_nonempty(data.get("isPrimary"), data.get("primary"), data.get("is_primary")))
    if _prim is not None:
        payload["isPrimary"] = _prim
    _atype = _first_nonempty(data.get("assetType"), data.get("type"), data.get("asset_type"))
    if _atype:
        payload["assetType"] = str(_atype)
    conf = _num_conf_to_label(_first_nonempty(data.get("confidentiality"), data.get("c")))
    if conf is not None:
        payload["confidentiality"] = conf
    integ = _num_ia_to_label(_first_nonempty(data.get("integrity"), data.get("i")))
    if integ is not None:
        payload["integrity"] = integ
    avail = _num_ia_to_label(_first_nonempty(data.get("availability"), data.get("a")))
    if avail is not None:
        payload["availability"] = avail
    return TA_ASSET.validate_python(payload)

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
# Submit with backoff + outbox
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
            if i == attempts:
                break
            delay = base_s * (2 ** (i - 1)) + random.random() * jitter_s
            await asyncio.sleep(delay)

    await _write_outbox(payload, deps)
    if last_err:
        raise last_err

# -----------------------------------------------------------------------------
# HARD GUARD: validate write actions
# -----------------------------------------------------------------------------

def _validate_write_actions(actions: list[dict]) -> tuple[list[dict], Optional[str]]:
    validated: list[dict] = []
    for a in actions:
        op = (a.get("op") or "").strip()
        try:
            if op == "create_asset":
                payload = {
                    "op": "create_asset",
                    "client_token": a.get("client_token"),
                    "fields": a.get("fields") or {},
                }
                if not payload["fields"].get("name"):
                    return [], "create_asset requires fields.name."
                validated.append(_guard_validate(ActionCreateAsset, payload))
            elif op == "set_asset_meta":
                payload = {
                    "op": "set_asset_meta",
                    "asset_id": a.get("asset_id"),
                    "asset_ref": a.get("asset_ref"),
                    "fields": a.get("fields") or {},
                }
                if not payload["asset_id"]:
                    return [], "set_asset_meta requires asset_id."
                if not payload["fields"]:
                    return [], "set_asset_meta requires at least one field."
                validated.append(_guard_validate(ActionSetAssetMeta, payload))
            else:
                return [], f"Unsupported operation '{op}'."
        except Exception as e:
            return [], f"Write action validation failed for op='{op}': {e}"
    return validated, None

# -----------------------------------------------------------------------------
# (Legacy) Planner (NLU) — kept for resilience & tests
# -----------------------------------------------------------------------------

GREET_RX = re.compile(r"^\s*(ahoj|cau|čau|hej|nazdar|dobr(y|ý)\s+de[nň]|hello|hi|hey|good\s+(morning|afternoon|evening))[\s!.]*$", re.I)
CREATE_RX = re.compile(r"\b(create|new|add|vytvor|zaloz|pridaj|pridej)\b", re.I)
UPDATE_RX = re.compile(r"\b(update|set|change|zmen|nastav|uprav)\b", re.I)
GET_RX    = re.compile(r"\b(get|show|detail|ukaz|zobraz|tell me|čo mi povieš|co mi reknes)\b", re.I)
LIST_RX   = re.compile(r"\b(list|seznam|vypis|zoznam|all assets)\b", re.I)
EXPLAIN_RX= re.compile(r"\b(explain|vysvetli|popis|povedz viac|tell me more)\b", re.I)

CIA_RX = {
    "confidentiality": re.compile(r"\b(c|confidentiality)\s*([0-4])\b", re.I),
    "integrity":       re.compile(r"\b(i|integrity)\s*([0-4])\b", re.I),
    "availability":    re.compile(r"\b(a|availability)\s*([0-4])\b", re.I),
}
PRIMARY_TRUE_RX  = re.compile(r"\b(primarn(y|í)|hlavn(y|í)|primary|make primary|ozna[cč]\s+ako\s+hlavn[ýe])\b", re.I)
PRIMARY_FALSE_RX = re.compile(r"\b(neprimarn(y|í)|zru[sš]i[ťt]\s+hlavn[ýe]|unprimary|odobra[tť]\s+hlavn[ýe])\b", re.I)
TYPE_RX = re.compile(r"\b(typ|type)\s*[:=]?\s*([a-z0-9 _\-]+)", re.I)
NAME_QUOTED_RX = re.compile(r"[\"“”']\s*([^\"“”']+?)\s*[\"“”']")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)


class NLUResult(BaseModel):
    action: Optional[Intent] = None
    language: Optional[str] = None
    new_asset: Optional[dict] = None
    update: Optional[dict] = None

    def sanitized(self) -> "NLUResult":
        def _only(d: Optional[dict], allowed: Sequence[str]) -> Optional[dict]:
            if not d:
                return None
            return {k: d.get(k) for k in allowed if k in d}

        na = _only(self.new_asset, ("id", "name", "assetType", "isPrimary", "confidentiality", "integrity", "availability"))
        up = _only(self.update, ("asset_id", "assetType", "isPrimary", "confidentiality", "integrity", "availability"))
        return NLUResult(action=self.action, language=self.language, new_asset=na, update=up)


def _get_llm_from_deps(deps: GraphDeps):
    llm = getattr(deps, "llm", None)
    if llm is not None:
        return llm
    if not _HAS_LLM:
        return None
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, timeout=12, max_retries=1)  # type: ignore[call-arg]


async def _nlu(deps: GraphDeps, state: AssetState) -> AssetState:
    text = (state.message or "").strip()
    if not text:
        state.action = "help"
        return state

    if GREET_RX.match(text):
        state.action = "chitchat"
        return state

    llm = _get_llm_from_deps(deps)
    selection_id = (state.selection or {}).get("asset_id")

    if llm is not None:
        system = (
            "You are a strict NLU for an 'assets' assistant.\n"
            "Primary languages: Slovak (SK), English (EN); Czech (CZ) tolerated.\n"
            "Return ONLY compact JSON with:\n"
            "  action: one of [help, chitchat, list, get, create, set_meta, explain]\n"
            "  new_asset?: {id?, name?, assetType?, isPrimary?, confidentiality?, integrity?, availability?}\n"
            "  update?: {asset_id?, assetType?, isPrimary?, confidentiality?, integrity?, availability?}\n"
            "Rules:\n"
            "- Never invent IDs.\n"
            "- If a UI selection asset_id exists, you MAY copy it into update.asset_id.\n"
            "- Detect greetings as chitchat.\n"
            "- CIA numeric scale is 0..4 if present.\n"
            "- Output strict JSON only.\n"
        )
        human = {
            "message": text,
            "selection": {"asset_id": selection_id} if selection_id else None,
            "hints": {"allowed_actions": ["help", "chitchat", "list", "get", "create", "set_meta", "explain"]},
            "locale": state.locale,
        }
        try:
            resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=json.dumps(human))])  # type: ignore
            raw = (getattr(resp, "content", None) or "").strip()
            start, end = raw.find("{"), raw.rfind("}")
            data = json.loads(raw[start:end + 1] if start != -1 and end != -1 else raw)
            n = NLUResult(**data).sanitized()
            if n.action:
                state.action = n.action
            if n.new_asset and not state.new_asset:
                state.new_asset = n.new_asset
            if n.update:
                if selection_id and not n.update.get("asset_id"):
                    n.update["asset_id"] = selection_id
                if not state.update:
                    state.update = n.update
        except Exception as e:
            logger.debug("NLU LLM failed, fallback to regex: %s", e)

    if not state.action:
        t = _norm(text)
        if EXPLAIN_RX.search(t):
            state.action = "explain"
        elif CREATE_RX.search(t):
            state.action = "create"
        elif UPDATE_RX.search(t) or any(rx.search(t) for rx in CIA_RX.values()) or TYPE_RX.search(t):
            state.action = "set_meta"
        elif GET_RX.search(t):
            state.action = "get"
        elif LIST_RX.search(t):
            state.action = "list"
        else:
            state.action = "help"

        if state.action == "create" and not state.new_asset:
            na = {}
            m_name = NAME_QUOTED_RX.search(text)
            if m_name:
                na["name"] = m_name.group(1).strip()
            m_type = TYPE_RX.search(t)
            if m_type:
                na["assetType"] = m_type.group(2).strip()
            if na:
                state.new_asset = na

        if state.action == "set_meta" and not state.update:
            fields: dict[str, Any] = {}
            for k, rx in CIA_RX.items():
                m = rx.search(t)
                if m:
                    fields[k] = int(m.group(2))
            m_type = TYPE_RX.search(t)
            if m_type:
                fields["assetType"] = m_type.group(2).strip()
            if PRIMARY_TRUE_RX.search(t):
                fields["isPrimary"] = True
            elif PRIMARY_FALSE_RX.search(t):
                fields["isPrimary"] = False
            if fields:
                if selection_id:
                    fields["asset_id"] = selection_id
                state.update = fields

    if state.action in ("help", "chitchat", "list", "get", "create", "set_meta", "explain"):
        state.intent = cast(Intent, state.action)
    else:
        state.intent = "list"
    return state

# -----------------------------------------------------------------------------
# Suggest-select (if missing selection)
# -----------------------------------------------------------------------------

def _suggest_select(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="suggest_select")
        try:
            items = deps.backend_port.assets_list(org_id=state.org_id) or []
            dtos: list[Asset] = []
            for a in items[:50]:
                try:
                    dtos.append(_to_asset_dto(a, org_id=state.org_id))
                except Exception:
                    continue

            lines = [_t(loc, "select_prompt")]
            preview = []
            for dto in dtos[:8]:
                d = dto.model_dump()
                preview.append(f"- {d.get('id')}: {d.get('name', 'Unnamed')} [{d.get('assetType','-')}]")
            msg = "\n".join(lines + preview) if preview else _t(loc, "list_found", n=0)

            state.result = _ui_response(
                msg,
                extra={"assets": [a.model_dump() for a in dtos[:20]], "agent_events": state.agent_events},
            )
            _finish(state, t0, tool="suggest_select", ok=True, meta={"count": len(dtos)})
        except Exception as e:
            _finish(state, t0, tool="suggest_select", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "list_fail", err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner

# -----------------------------------------------------------------------------
# Read-only nodes
# -----------------------------------------------------------------------------

def _assets_list(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="assets_list")
        try:
            items = deps.backend_port.assets_list(org_id=state.org_id) or []
            dtos: list[Asset] = []
            for a in items:
                try:
                    dtos.append(_to_asset_dto(a, org_id=state.org_id))
                except Exception:
                    continue

            header = _t(loc, "list_found", n=len(dtos))
            preview = []
            for dto in dtos[:10]:
                d = dto.model_dump()
                preview.append(
                    f"- {d.get('id')}: {d.get('name','Unnamed')} [{d.get('assetType','-')}]"
                    f"{' (primary)' if d.get('isPrimary') else ''}"
                )
            msg = "\n".join([header] + preview) if preview else header

            state.result = _ui_response(
                msg,
                extra={"assets": [a.model_dump() for a in dtos[:50]], "agent_events": state.agent_events},
            )
            _finish(state, t0, tool="assets_list", ok=True, meta={"count": len(dtos)})
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

            dto = _to_asset_dto(asset, org_id=state.org_id)
            d = dto.model_dump()
            preview_cia = {"C": d.get("confidentiality"), "I": d.get("integrity"), "A": d.get("availability")}
            msg = f"{d.get('name') or asset_id} [{d.get('assetType','-')}], CIA={preview_cia}, primary={bool(d.get('isPrimary'))}"
            state.result = _ui_response(msg, extra={"asset": d, "agent_events": state.agent_events})
            _finish(state, t0, tool="asset_get", ok=True, meta={"asset_id": asset_id})
        except Exception as e:
            _finish(state, t0, tool="asset_get", ok=False, meta={"error": str(e)})
            state.result = _ui_response(_t(loc, "asset_load_fail", id=asset_id, err=e), extra={"agent_events": state.agent_events})
        return state
    return _inner

# -----------------------------------------------------------------------------
# Chitchat/help
# -----------------------------------------------------------------------------

def _help_or_greet() -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        state.result = _ui_response(_t(loc, "greet"), extra={"agent_events": state.agent_events})
        return state
    return _inner

# -----------------------------------------------------------------------------
# Explain (RAG → LLM)
# -----------------------------------------------------------------------------

async def _asset_explain(deps: GraphDeps, state: AssetState) -> AssetState:
    loc = _lang(state.locale)
    asset_id = (state.selection or {}).get("asset_id") \
        or (state.update or {}).get("asset_id") \
        or (await _amemory_get(deps, "assets.last_hint") or {}).get("asset_id")

    if not asset_id:
        return _suggest_select(deps)(state)

    asset_dto_dump: dict | None = None
    try:
        a = deps.backend_port.asset_get(asset_id)
        if a:
            asset_dto_dump = _to_asset_dto(a).model_dump()
    except Exception:
        asset_dto_dump = None

    retriever = None
    try:
        if hasattr(deps, "make_retriever"):
            retriever = deps.make_retriever(state.org_id, "assets")  # type: ignore[arg-type]
    except Exception:
        retriever = None

    context_chunks: list[dict] = []
    if retriever is not None:
        t0r = _event(state, kind="tool.call.start", tool="retriever.search", meta={"tab": "assets"})
        try:
            query = "Explain the asset " + (asset_dto_dump.get("name") if asset_dto_dump else "")
            docs = await retriever.asearch(query, k=4) if hasattr(retriever, "asearch") else retriever.search(query, k=4)  # type: ignore
            for d in (docs or [])[:5]:
                context_chunks.append(d if isinstance(d, dict) else {"text": str(d)})
            _finish(state, t0r, tool="retriever.search", ok=True, meta={"docs": len(context_chunks)})
        except Exception as e:
            _finish(state, t0r, tool="retriever.search", ok=False, meta={"error": str(e)})

    llm = _get_llm_from_deps(deps) if _HAS_LLM else None
    if llm is not None:
        sys = (
            "You are a helpful security workshop assistant. Explain the selected asset to a non-expert. "
            "Use provided context if relevant; else reason from the asset metadata. Keep it concise (5–8 short sentences). "
            "Avoid hallucinating unknown fields. Output plain text."
        )
        human_payload = {
            "asset": asset_dto_dump or {},
            "selection": (state.selection or {}),
            "org_id": state.org_id,
            "context": context_chunks or None,
            "question": state.message or "Explain this asset",
            "locale": loc,
        }
        t0l = _event(state, kind="tool.call.start", tool="llm.explain")
        try:
            resp = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=json.dumps(human_payload))])  # type: ignore
            text = (getattr(resp, "content", None) or "").strip()
            if text:
                _finish(state, t0l, tool="llm.explain", ok=True, meta={"used_sources": bool(context_chunks)})
                state.result = _ui_response(text, extra={"asset": asset_dto_dump, "sources": context_chunks or None, "agent_events": state.agent_events})
                return state
        except Exception as e:
            _finish(state, t0l, tool="llm.explain", ok=False, meta={"error": str(e)})

    if asset_dto_dump:
        name = asset_dto_dump.get("name") or "Unnamed asset"
        a_type = asset_dto_dump.get("assetType") or "-"
        c = asset_dto_dump.get("confidentiality"); i = asset_dto_dump.get("integrity"); a = asset_dto_dump.get("availability")
        text = f"{name} [{a_type}]. From a security perspective: C={c}, I={i}, A={a}. More detail will appear as the knowledge base grows."
        state.result = _ui_response(text, extra={"asset": asset_dto_dump, "agent_events": state.agent_events})
        return state

    state.result = _ui_response(_t(loc, "explain_select_first"), extra={"agent_events": state.agent_events})
    return state

# -----------------------------------------------------------------------------
# Write helpers (derive multi)
# -----------------------------------------------------------------------------

_ALLOWED_CREATE = {"id", "name", "assetType", "isPrimary", "confidentiality", "integrity", "availability"}
_ALLOWED_UPDATE = {"asset_id", "assetType", "isPrimary", "confidentiality", "integrity", "availability"}

def _json_array_from_text(s: str) -> list[Any]:
    s = (s or "").strip()
    if not s:
        return []
    try:
        data = json.loads(s)
        return data if isinstance(data, list) else []
    except Exception:
        pass
    try:
        start, end = s.find("["), s.rfind("]")
        if start != -1 and end != -1 and end > start:
            data = json.loads(s[start:end + 1])
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def _collect_selected_ids(selection: Optional[dict], message: Optional[str]) -> list[str]:
    ids: list[str] = []
    sel = selection or {}
    if sel.get("asset_id"):
        ids.append(str(sel["asset_id"]))
    if isinstance(sel.get("asset_ids"), (list, tuple)):
        ids += [str(x) for x in sel["asset_ids"] if x]
    if isinstance(sel.get("assets"), (list, tuple)):
        for a in sel["assets"]:
            ad = _to_asset_dict(a)
            aid = ad.get("id") or ad.get("asset_id")
            if aid:
                ids.append(str(aid))
    msg = message or ""
    ids += _UUID_RE.findall(msg)
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def _derive_create_items(state: AssetState) -> list[dict]:
    # 1) Prefer items proposed by the agent (bulk create)
    agent_items = (state.tmp.get("agent_create_items") or [])
    out_from_agent = []
    for it in agent_items:
        if isinstance(it, dict):
            obj = {k: it.get(k) for k in _ALLOWED_CREATE if k in it}
            if (obj.get("name") or "").strip():
                out_from_agent.append(obj)
    if out_from_agent:
        return out_from_agent

    # 2) Parse strict JSON array from message
    items: list[dict] = []
    raw = state.message or ""
    arr = _json_array_from_text(raw)
    for it in arr:
        if isinstance(it, dict):
            obj = {k: it.get(k) for k in _ALLOWED_CREATE if k in it}
            if (obj.get("name") or "").strip():
                items.append(obj)
    if items:
        return items

    # 3) Names in quotes + shared slots
    names = [m.strip() for m in NAME_QUOTED_RX.findall(raw) if m and m.strip()]
    if names:
        shared = {k: v for k, v in (state.new_asset or {}).items() if k in _ALLOWED_CREATE and k != "name"}
        for nm in names:
            obj = {"name": nm}
            obj.update(shared)
            items.append(obj)
        return items

    # 4) Single object from state
    if state.new_asset:
        obj = {k: state.new_asset.get(k) for k in _ALLOWED_CREATE if k in state.new_asset}
        if (obj.get("name") or "").strip():
            items.append(obj)
    return items


def _derive_update_items(state: AssetState) -> list[dict]:
    # 1) Prefer items proposed by the agent (bulk update)
    agent_items = (state.tmp.get("agent_update_items") or [])
    out_from_agent = []
    for it in agent_items:
        if isinstance(it, dict):
            aid = it.get("asset_id") or it.get("id")
            if not aid:
                continue
            fields = {k: it.get(k) for k in _ALLOWED_UPDATE if k in it and k != "asset_id"}
            fields = {k: v for k, v in fields.items() if v is not None}
            if fields:
                out_from_agent.append({"asset_id": str(aid), **fields})
    if out_from_agent:
        return out_from_agent

    # 2) Parse strict JSON array from message
    items: list[dict] = []
    raw = state.message or ""
    arr = _json_array_from_text(raw)
    for it in arr:
        if isinstance(it, dict):
            aid = it.get("asset_id") or it.get("id")
            if not aid:
                continue
            fields = {k: it.get(k) for k in _ALLOWED_UPDATE if k in it and k != "asset_id"}
            fields = {k: v for k, v in fields.items() if v is not None}
            if fields:
                items.append({"asset_id": str(aid), **fields})
    if items:
        return items

    # 3) Use selection + shared fields
    ids = _collect_selected_ids(state.selection, state.message)
    shared = {k: v for k, v in (state.update or {}).items() if k in _ALLOWED_UPDATE and k != "asset_id" and v is not None}
    for aid in ids:
        if shared:
            items.append({"asset_id": aid, **shared})

    # 4) Single object from state
    if not items and state.update and state.update.get("asset_id"):
        fields = {k: v for k, v in state.update.items() if k in _ALLOWED_UPDATE and k != "asset_id" and v is not None}
        if fields:
            items.append({"asset_id": str(state.update["asset_id"]), **fields})
    return items

# -----------------------------------------------------------------------------
# Write nodes (create / set_meta) — emit-only then submit
# -----------------------------------------------------------------------------

def _create_asset(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="create_asset")
        draft: list[dict] = []
        create_items = _derive_create_items(state)
        if not create_items:
            _finish(state, t0, tool="create_asset", ok=False, meta={"error": "no_items"})
            state.result = _ui_response(_t(loc, "nothing_to_create"), extra={"agent_events": state.agent_events})
            return state

        for it in create_items:
            name = (it.get("name") or "").strip()
            if not name:
                continue
            fields = {
                "id": it.get("id") or None,
                "name": name,
                "assetType": it.get("assetType"),
                "isPrimary": bool(it.get("isPrimary")) if it.get("isPrimary") is not None else None,
                "confidentiality": it.get("confidentiality"),
                "integrity": it.get("integrity"),
                "availability": it.get("availability"),
            }
            fields = {k: v for k, v in fields.items() if v is not None}
            draft.append({"op": "create_asset", "client_token": f"tmp-{uuid.uuid4().hex[:10]}", "fields": fields})

        validated, err = _validate_write_actions(draft)
        if err:
            _finish(state, t0, tool="create_asset", ok=False, meta={"error": err})
            state.result = _ui_response(_t(loc, "validation_error", err=err), extra={"agent_events": state.agent_events})
            return state

        for a in validated:
            _safe_emit(getattr(deps, "ws_hub", None), "asset.created", {"org_id": state.org_id, **(a.get("fields") or {})})

        state.actions = validated
        last_fields = (validated[-1] or {}).get("fields", {})
        _schedule(_amemory_set(deps, "assets.last_hint", {"asset_name": last_fields.get("name"), "asset_id": last_fields.get("id")}))

        _finish(state, t0, tool="create_asset", ok=True, meta={"count": len(validated)})
        return state
    return _inner


def _set_meta(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        loc = _lang(state.locale)
        t0 = _event(state, kind="tool.call.start", tool="set_asset_meta")
        update_items = _derive_update_items(state)
        if not update_items:
            if not (state.update and state.update.get("asset_id")) and not (state.selection and state.selection.get("asset_id")):
                _finish(state, t0, tool="set_asset_meta", ok=False, meta={"error": "missing_selection"})
                return _suggest_select(deps)(state)
            _finish(state, t0, tool="set_asset_meta", ok=False, meta={"error": "no_items"})
            state.result = _ui_response(_t(loc, "nothing_to_update"), extra={"agent_events": state.agent_events})
            return state

        draft: list[dict] = []
        for it in update_items:
            aid = it.get("asset_id")
            fields = {k: v for k, v in it.items() if k in _ALLOWED_UPDATE and k != "asset_id" and v is not None}
            if aid and fields:
                draft.append({"op": "set_asset_meta", "asset_id": aid, "fields": fields})

        validated, err = _validate_write_actions(draft)
        if err:
            _finish(state, t0, tool="set_asset_meta", ok=False, meta={"error": err})
            state.result = _ui_response(_t(loc, "validation_error", err=err), extra={"agent_events": state.agent_events})
            return state

        for a in validated:
            _safe_emit(getattr(deps, "ws_hub", None), "asset.updated", {"org_id": state.org_id, "asset_id": a.get("asset_id"), **(a.get("fields") or {})})

        state.actions = validated
        _schedule(_amemory_set(deps, "assets.last_hint", {"asset_id": validated[-1].get("asset_id")}))

        _finish(state, t0, tool="set_asset_meta", ok=True, meta={"count": len(validated)})
        return state
    return _inner


async def _submit_changeset_async(deps: GraphDeps, state: AssetState) -> AssetState:
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
        state.result = _ui_response(_t(loc, "submit_failed_queued", err=e), extra={"agent_events": state.agent_events})
        return state

    n_create = sum(1 for a in acts if a.get("op") == "create_asset")
    n_update = sum(1 for a in acts if a.get("op") == "set_asset_meta")
    if n_create or n_update:
        state.result = _ui_response(
            _t(loc, "submitted_summary", created=n_create, updated=n_update),
            extra={"changeset_actions": len(acts), "agent_events": state.agent_events},
        )
    else:
        state.result = _ui_response(_t(loc, "submitted_generic", count=len(acts)), extra={"agent_events": state.agent_events})

    created_names = [a.get("fields", {}).get("name") for a in acts if a.get("op") == "create_asset"]
    updated_ids = [a.get("asset_id") for a in acts if a.get("op") == "set_asset_meta"]
    _schedule(_amemory_set(deps, "assets.last_batch", {
        "created_names": [n for n in created_names if n],
        "updated_ids": [i for i in updated_ids if i],
        "count": len(acts),
        "ts": dt.datetime.utcnow().isoformat()
    }))
    if updated_ids and updated_ids[-1]:
        _schedule(_amemory_set(deps, "assets.last_hint", {"asset_id": updated_ids[-1]}))
    elif created_names and created_names[-1]:
        _schedule(_amemory_set(deps, "assets.last_hint", {"asset_name": created_names[-1]}))

    return state

# -----------------------------------------------------------------------------
# Graph assembly
# -----------------------------------------------------------------------------

def create_graph(deps: GraphDeps):
    """
    Build & compile the Asset graph.
    Flow:
      start → agent_decide → agent_route → {assets_list|asset_get|create_asset|set_meta|explain} → (submit?) → END
    """
    graph = StateGraph(AssetState)

    # --- Agent nodes ---
    def _agent_decide_node(state: AssetState) -> AssetState:
        t0 = _event(state, kind="tool.call.start", tool="agent_decide")
        try:
            decision = agent_decide(
                tab="assets",
                user_text=(state.message or ""),
                context=state.model_dump(),
                deps=deps,
            )
            d = decision.model_dump()
            state.tmp["decision"] = d

            # Hydrate state for downstream nodes
            act = (d.get("action") or "").strip()
            asset_id = (d.get("args") or {}).get("asset_id") or (d.get("selection") or {}).get("asset_id")

            # asset_get
            if act == "asset_get" and asset_id:
                state.selection = {"asset_id": asset_id}

            # create_asset
            items = (d.get("items") or [])
            if act == "create_asset":
                if items:
                    first = {k: v for k, v in items[0].items() if k in {"id","name","assetType","isPrimary","confidentiality","integrity","availability"}}
                    if first:
                        state.new_asset = first
                state.tmp["agent_create_items"] = items

            # set_meta
            if act == "set_meta":
                upd = (d.get("update") or {})
                if not upd.get("asset_id") and asset_id:
                    upd["asset_id"] = asset_id
                state.update = {k: v for k, v in upd.items() if k in {"asset_id","assetType","isPrimary","confidentiality","integrity","availability"}}
                state.tmp["agent_update_items"] = (d.get("items") or [])

            _finish(state, t0, tool="agent_decide", ok=True, meta={"action": d.get("action")})
            return state
        except Exception as e:
            _finish(state, t0, tool="agent_decide", ok=False, meta={"error": str(e)})
            state.tmp["next_node"] = "explain"
            return state

    def _agent_route_node(state: AssetState) -> AssetState:
        action = (state.tmp.get("decision") or {}).get("action")
        allowed = {"assets_list", "asset_get", "create_asset", "set_meta", "explain"}
        state.tmp["next_node"] = action if action in allowed else "explain"
        return state

    graph.add_node("agent_decide", _agent_decide_node)
    graph.add_node("agent_route", _agent_route_node)

    # --- Business nodes ---
    async def _explain_node(state: AssetState) -> AssetState:
        return await _asset_explain(deps, state)

    async def _submit_node(state: AssetState) -> AssetState:
        return await _submit_changeset_async(deps, state)

    graph.add_node("assets_list", _assets_list(deps))
    graph.add_node("asset_get", _asset_get_or_suggest(deps))
    graph.add_node("explain", _explain_node)
    graph.add_node("create_asset", _create_asset(deps))
    graph.add_node("set_meta", _set_meta(deps))
    graph.add_node("submit_changeset", _submit_node)
    graph.add_node("help", _help_or_greet())  # only used if you route to it explicitly

    # Entry & routing via agent
    graph.set_entry_point("agent_decide")
    graph.add_edge("agent_decide", "agent_route")
    graph.add_conditional_edges(
        "agent_route",
        lambda s: s.tmp.get("next_node", "explain"),
        {
            "assets_list": "assets_list",
            "asset_get": "asset_get",
            "create_asset": "create_asset",
            "set_meta": "set_meta",
            "explain": "explain",
        },
        default="explain",
    )

    # Write → submit
    graph.add_edge("create_asset", "submit_changeset")
    graph.add_edge("set_meta", "submit_changeset")

    # Terminal nodes
    for node in ("help", "assets_list", "asset_get", "explain", "submit_changeset"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=deps.checkpointer)

# -----------------------------------------------------------------------------
# Local preview struct (unchanged)
# -----------------------------------------------------------------------------

@dataclass
class AssetPreview:
    id: str
    name: str
    assetType: str
    isPrimary: bool | None = None
    confidentiality: int | None = None
    integrity: int | None = None
    availability: int | None = None
