# File: workshops_ai/graph/asset_graph.py
# -----------------------------------------------------------------------------
# LangGraph: ASSETS — LLM-first, slot-driven, ONE action per turn
#
# Iterative, recursion-free design:
#   - No recursive serializers anywhere. We ship a local, BFS-based `_jsonify_iter`
#     that is cycle-safe and bounded by `max_depth`.
#   - We DO NOT persist write payloads in graph state. Create/Update submits INLINE.
#   - Any runtime-only fields are `exclude=True` and never serialized by the
#     checkpointer. Pydantic validation is relaxed to avoid hidden deep walks.
#   - CREATE uses strict single-slot capture (when we ask for one field, we only
#     consume that exact field on the next turn).
#   - English comments throughout; users may speak SK/EN/CZ.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import deque
from collections.abc import Mapping
from typing import Callable, Literal, Optional, cast, TYPE_CHECKING, Any

from pydantic import BaseModel, Field
try:
    from pydantic import ConfigDict  # v2
    _PYDANTIC_V2 = True
except Exception:  # pragma: no cover
    ConfigDict = dict  # type: ignore
    _PYDANTIC_V2 = False

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from workshops_ai.graph.factory import GraphDeps

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.graph.agent_node import agent_decide, AgentDecision

# ---------- Normalizers ----------
try:
    from workshops_ai.utils.normalize import (
        normalize_asset_create_fields as _normalize_create_fields,
        normalize_asset_name as _normalize_name,
        normalize_asset_type as _normalize_type,
        normalize_is_primary as _normalize_is_primary,
    )
except Exception:  # pragma: no cover
    def _normalize_create_fields(fields: dict, catalog_types: list[str] | None = None) -> dict:
        return fields

    def _normalize_name(name: Optional[str]) -> Optional[str]:
        return (name or "").strip() or None

    def _normalize_type(typ: Optional[str], allowed: list[str] | None = None) -> Optional[str]:
        t = (typ or "").strip()
        if not t:
            return None
        if not allowed:
            return t
        return next((x for x in allowed if x.lower() == t.lower()), None)

    def _normalize_is_primary(value: Any) -> Optional[bool]:
        # Basic CZ/SK/EN yes/no parsing
        if isinstance(value, bool):
            return value
        s = str(value or "").strip().lower()
        pos = {
            "ano","áno","jo","jj","j","y","yes","true","1","ok","okay","confirm","confirmed","sure","primary"
        }
        neg = {"ne","nie","n","no","false","0","nope","neni","není"}
        if s in pos: return True
        if s in neg: return False
        return None

# ---------- UI helpers ----------
try:
    from workshops_ai.ui.formatting import title, bullet_list, chunk  # type: ignore
except Exception:  # pragma: no cover
    def title(text: str) -> str:
        return f"# {text}".strip()

    def bullet_list(items: list[str]) -> str:
        return "\n".join(f"- {x}" for x in items)

    def chunk(text: str, *, max_chars: int = 2000) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        out, cur, size = [], [], 0
        for line in text.splitlines():
            add = len(line) + 1
            if size + add > max_chars and cur:
                out.append("\n".join(cur)); cur, size = [line], add
            else:
                cur.append(line); size += add
        if cur: out.append("\n".join(cur))
        return out

# ---------- LangGraph ----------
try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("langgraph must be installed to use asset_graph") from e

logger = logging.getLogger(__name__)

# --- DIAG (optional) ---------------------------------------------------------
try:
    # our crashpack logger
    from workshops_ai.diagnostics.json_probe import dump_crash_packet
except Exception:  # pragma: no cover
    dump_crash_packet = None  # type: ignore

try:
    # telemetry helper for request_id
    from workshops_ai.telemetry import get_request_id  # type: ignore
except Exception:  # pragma: no cover
    def get_request_id(default: str = "unknown") -> str:  # type: ignore
        return default

DIAG = os.getenv("WSAI_DIAG", "0") in ("1", "true", "True", "yes")

def _diag_crashpack(context: str, guarded_obj: Any, exc: BaseException, preview: dict | None = None) -> None:
    if not (DIAG and dump_crash_packet):
        return
    try:
        rid = get_request_id(default="unknown")
    except Exception:
        rid = "unknown"
    path = dump_crash_packet(rid, (preview or {}), guarded_obj, exc)  # type: ignore[misc]
    logger.error("DIAG_CRASHPACK context=%s request_id=%s crashpack=%s", context, rid, path)

# ---------- Optional payload guard (no recursion) ----------
try:
    from workshops_ai.tools.guard import guard_validate_payload as _guard_validate  # type: ignore
except Exception:  # pragma: no cover
    def _guard_validate(model_cls, payload):  # type: ignore
        try:
            return model_cls.model_validate(payload)
        except Exception:
            return payload

# ---------- CIA enums ----------
CONF_LEVELS = {"PUBLIC", "INTERNAL", "PROTECTED", "STRICTLY_PROTECTED"}
TRI_LEVELS  = {"LOW", "MEDIUM", "HIGH"}

def _norm_enum(value: Optional[str], *, allowed: set[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if v == "STRICTLYPROTECTED":
        v = "STRICTLY_PROTECTED"
    return v if v in allowed else None

# --- TRI/CIA parsers (single-slot helpers) ---

def _tri_from_text(text: str) -> Optional[str]:
    s = (text or "").strip().lower()
    toks = {t.strip(",.;:!?") for t in s.split()}
    low = {"low", "lo", "nízká", "nizka", "nízke", "nizke", "nízky", "nizky"}
    med = {"med", "medium", "střední", "stredni", "stredna", "stredné", "středne"}
    hi  = {"high", "hi", "vysoká", "vysoka", "vysoké", "vysoke"}
    if toks & low: return "LOW"
    if toks & med: return "MEDIUM"
    if toks & hi:  return "HIGH"
    if s in {"low","medium","high"}: return s.upper()
    return None

def _conf_from_text(text: str) -> Optional[str]:
    s = (text or "").strip().lower().replace("-", " ")
    if any(tok in s.split() for tok in ["public","veřejné","verejne","publicní","publicne"]):
        return "PUBLIC"
    if any(tok in s.split() for tok in ["internal","interní","interni"]):
        return "INTERNAL"
    if any(tok in s.split() for tok in ["protected","chráněné","chranene","chránené","chranene"]):
        if "strict" in s or "přísně" in s or "prisne" in s:
            return "STRICTLY_PROTECTED"
        return "PROTECTED"
    if "strict" in s or "přísně" in s or "prisne" in s:
        return "STRICTLY_PROTECTED"
    if s in {"public"}: return "PUBLIC"
    if s in {"internal"}: return "INTERNAL"
    if s in {"protected"}: return "PROTECTED"
    if s in {"strictly protected","strictly_protected"}: return "STRICTLY_PROTECTED"
    return None

# ---------- Iterative, cycle-safe JSON helpers (NO recursion) ----------
_PRIMITIVES = (str, int, float, bool, type(None))

def _jsonify_iter(root: Any, *, max_depth: int = 6) -> Any:
    """
    Convert arbitrary Python objects to a JSON-serializable structure without recursion.
    - BFS traversal using a deque.
    - Cycle detection by object id.
    - Depth is capped to `max_depth`; beyond that returns the sentinel "<max_depth>".
    - Unknown objects map to str(obj).
    """
    seen: set[int] = set()
    def _set_root(v): nonlocal out; out = v
    out: Any = None
    q = deque([(root, 0, _set_root)])
    while q:
        obj, depth, setter = q.popleft()
        try:
            if isinstance(obj, _PRIMITIVES):
                setter(obj); continue
            oid = id(obj)
            if oid in seen:
                setter("<cycle>"); continue
            seen.add(oid)
            if depth >= max_depth:
                setter("<max_depth>"); continue
            if isinstance(obj, Mapping):
                container: dict[str, Any] = {}
                setter(container)
                for k, v in obj.items():
                    kk = str(k)
                    q.append((v, depth+1, lambda val, c=container, key=kk: c.__setitem__(key, val)))
                continue
            if isinstance(obj, (list, tuple, set)):
                it = list(obj)
                container: list[Any] = [None] * len(it)
                setter(container)
                for i, v in enumerate(it):
                    q.append((v, depth+1, lambda val, c=container, idx=i: c.__setitem__(idx, val)))
                continue
            if isinstance(obj, BaseModel):
                try:
                    fields = {k: v for k, v in obj.__dict__.items() if not str(k).startswith("_")}
                except Exception:
                    setter({"type": type(obj).__name__, "value": str(obj)}); continue
                container: dict[str, Any] = {}
                setter(container)
                for k, v in fields.items():
                    q.append((v, depth+1, lambda val, c=container, key=str(k): c.__setitem__(key, val)))
                continue
            d = getattr(obj, "__dict__", None)
            if isinstance(d, dict):
                container: dict[str, Any] = {}
                setter(container)
                for k, v in d.items():
                    if str(k).startswith("_"): continue
                    q.append((v, depth+1, lambda val, c=container, key=str(k): c.__setitem__(key, val)))
                continue
            setter(str(obj))
        except Exception:
            setter(f"<unserializable:{type(obj).__name__}>")
    return out

def _as_plain_dicts(seq: Any) -> list[dict]:
    """Coerce a sequence of 'anything' into a list[dict] safely (no recursion)."""
    out: list[dict] = []
    for x in list(seq or []):
        val = _jsonify_iter(x)
        out.append(val if isinstance(val, dict) else {"value": val})
    return out

# ---------- State & models ----------
Intent = Literal["list", "get", "create", "update_meta", "explain", "list_types", "agent_reply"]

class ToolEvent(TypedDict, total=False):
    event_type: str
    tool: str
    ok: bool
    duration_ms: int
    meta: dict

class AssetCreateItem(BaseModel):
    id: Optional[str] = None
    name: str
    assetType: Optional[str] = None
    isPrimary: Optional[bool] = None
    confidentiality: Optional[str] = None
    integrity: Optional[str] = None
    availability: Optional[str] = None
    description: Optional[str] = None

class AssetUpdateItem(BaseModel):
    asset_id: str
    name: Optional[str] = None
    description: Optional[str] = None

class AssetState(BaseModel):
    # Pydantic v2 config (preferováno, žádný vnořený Config)
    if _PYDANTIC_V2:
        model_config = ConfigDict(
            validate_assignment=False,
            arbitrary_types_allowed=True,
        )
    else:
        # Pydantic v1 fallback (použije se jen když v2 není k dispozici)
        class Config:
            validate_assignment = False
            arbitrary_types_allowed = True

    # Request context
    org_id: str
    active_tab: str = Field(default="assets")
    message: str | None = None
    selection: dict | None = None
    user_id: str | None = None
    locale: str | None = None
    language: str | None = None
    history: list[dict] | None = None
    staging: bool | None = None
    once_flags: dict | None = None

    # Structured inputs
    action: Literal["list","get","create","update_meta","explain","list_types","agent_reply"] | None = None
    create_items: list["AssetCreateItem"] | None = None
    update_items: list["AssetUpdateItem"] | None = None
    question: str | None = None

    # Working outputs
    intent: Literal["list","get","create","update_meta","explain","list_types","agent_reply"] | None = None

    # CRITICAL: keep heavy/complex objects out of persisted state
    actions: list[dict] | None = Field(default=None, exclude=True)
    has_actions: bool = False
    change: dict | None = Field(default=None, exclude=True)
    result: dict | None = Field(default=None, exclude=True)

    # Conversational slot-filling (CREATE only)
    create_draft: dict | None = None
    pending_fields: list[str] | None = None
    create_flow: bool = False

    # Telemetry — UI-only (never persist)
    agent_events: list["ToolEvent"] = Field(default_factory=list, exclude=True)

# ---------- Telemetry & UI ----------

def _event(state: AssetState, *, kind: str, tool: str, ok: bool = True, meta: dict | None = None) -> float:
    t0 = time.perf_counter()
    state.agent_events.append({"event_type": kind, "tool": tool, "ok": ok, "meta": meta or {}})
    return t0

def _finish(state: AssetState, t0: float, *, tool: str, ok: bool, meta: dict | None = None) -> None:
    dur = int((time.perf_counter() - t0) * 1000)
    state.agent_events.append({
        "event_type": "tool.call.success" if ok else "tool.call.fail",
        "tool": tool, "ok": ok, "duration_ms": dur, "meta": meta or {}
    })

def _ui_response(content: str | dict, *, extra: Optional[dict] = None) -> dict:
    base = {
        "messages": [content] if isinstance(content, str) else [content],
        "proposals": None,
        "staging": {"needs_commit": False, "batch_id": None, "diff": None},
        "policy": {"emit_changes_enabled": True},
    }
    if extra:
        for k, v in extra.items():
            if k == "agent_events":
                events = v if isinstance(v, list) else []
                base[k] = [dict(e) for e in events[-50:]]
            elif k != "messages":
                base[k] = v
    return base

def _now_iso() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()

def _schedule(coro_or_fn):
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro_or_fn if asyncio.iscoroutine(coro_or_fn) else coro_or_fn())
    except RuntimeError:
        asyncio.run(coro_or_fn if asyncio.iscoroutine(coro_or_fn) else coro_or_fn())

# ---------- Localization ----------

def _lang(loc: Optional[str]) -> str:
    return (loc or "en").split("-")[0].lower()

_CZ_DIACRITICS = set("áéíóúýčďěňřšťůžďťľĺôäÁÉÍÓÚÝČĎĚŇŘŠŤŮŽĎŤĽĹÔÄ")
_CZ_TOKENS = {"vytvoř","vytvor","název","nazev","potvrdit","potvrzuji","schvaluji","ano","áno","ne","není","neni","aktiva","typ","primární","primarni"}

def _looks_czech(text: str) -> bool:
    s = (text or "").strip()
    if any(ch in _CZ_DIACRITICS for ch in s):
        return True
    low = s.lower()
    return any(tok in low for tok in _CZ_TOKENS)

# ---------- Tokenization helpers ----------

def _tokenize(text: str) -> list[str]:
    raw = (text or "").strip()
    toks: list[str] = []
    for part in raw.split():
        toks.append(part.strip(",.;:!?()[]{}\"'").casefold())
    return toks

def _parse_list_limit(text: str) -> Optional[int]:
    toks = _tokenize(text)
    keys = {"top","first","limit","jen","zkrať","zkrat","zkratit","zkraťte","only"}
    for i, tok in enumerate(toks):
        if tok in keys:
            if i + 1 < len(toks) and toks[i + 1].isdigit():
                return max(1, int(toks[i + 1]))
            if i + 2 < len(toks) and toks[i + 1] in {"na","to"} and toks[i + 2].isdigit():
                return max(1, int(toks[i + 2]))
    for i, tok in enumerate(toks):
        if tok in {"show","list"} and i + 1 < len(toks) and toks[i + 1].isdigit():
            return max(1, int(toks[i + 1]))
    return None

# ---------- Deterministic field extraction ----------

_NAME_PATTERNS = [
    r'^(?:název\s+je|nazev\s+je)\s+(?P<name>.+)$',
    r'^(?:name\s+is)\s+(?P<name>.+)$',
    r'^(?P<q1>"[^"]{1,120}")$',
    r"^(?P<q2>'[^']{1,120}')$",
]

def _extract_name_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    raw = text.strip()
    for pat in _NAME_PATTERNS:
        m = re.match(pat, raw, flags=re.IGNORECASE)
        if m:
            g = m.groupdict()
            for k in ("name","q1","q2"):
                val = g.get(k)
                if val:
                    val = val.strip().strip('"').strip("'").strip()
                    return val or None
    if len(raw.split()) <= 5:
        return raw
    return None

_CONFIRM_RE = re.compile(r"^(potvrdit(\s+vytvoření|[ ]*vytvoreni)?|potvrzuji|schvaluji|potvrdzujem|schvaľujem|confirm|ok|okay|yes|ano|áno)$", re.I)
_CANCEL_RE  = re.compile(r"^(zrušit|zrušiť|zrusit|cancel|stop|ne|no|nie)$", re.I)

def _is_yes_like(text: str) -> bool:
    return bool(_CONFIRM_RE.match((text or "").strip().lower()))

def _is_no_like(text: str) -> bool:
    return bool(_CANCEL_RE.match((text or "").strip().lower()))

# ---------- Lightweight intent recognizers ----------

def _looks_like_create(msg: str) -> bool:
    m = (msg or "").strip().lower()
    tokens = ["vytvoř","vytvor","založ","zaloz","přidej","pridej","add","create","nové","nove","new"]
    return any(t in m for t in tokens)

def _looks_like_explain(msg: str) -> bool:
    m = (msg or "").strip().lower()
    keys = ["explain","what is","co je","vysvětli","vysvetli","prosím vysvětli"]
    return any(k in m for k in keys)

def _mentions_types(msg: str) -> bool:
    m = (msg or "").strip().lower()
    keys = ["types", "asset types", "typy", "typ", "druhy"]
    return any(k in m for k in keys)

def _looks_like_greeting(msg: str) -> bool:
    m = (msg or "").strip().lower()
    if not m:
        return False
    short = len(m.split()) <= 4
    phrases = [
        "ahoj","čau","cau","nazdar","zdar","servus",
        "dobrý den","dobry den","hezký den","hezky den",
        "hello","hi","hey","yo","good morning","good afternoon","good evening",
    ]
    return short and any(p in m for p in phrases)

# ---------- LLM bridge ----------

def _llm_text(*, deps: GraphDeps, messages: list[dict[str, str]], temperature: float | None = 0.2, model: Optional[str] = None) -> str:
    client = getattr(deps, "llm_client", None) or getattr(deps, "llm", None)
    if client is None:
        raise RuntimeError("LLM client is not configured.")
    temp = temperature if temperature is not None else getattr(deps, "llm_temperature", None) or 0.2
    try:
        invoke = getattr(client, "invoke", None)
        if callable(invoke):
            try:
                from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # type: ignore
            except Exception:
                from langchain.schema import SystemMessage, HumanMessage, AIMessage  # type: ignore
            lc_messages: list[object] = []
            for m in messages:
                role = (m.get("role") or "").lower(); content = m.get("content") or ""
                if role == "system": lc_messages.append(SystemMessage(content=content))
                elif role in ("assistant","ai"): lc_messages.append(AIMessage(content=content))
                else: lc_messages.append(HumanMessage(content=content))
            try:
                resp = client.invoke(lc_messages, temperature=temp)  # type: ignore
            except TypeError:
                resp = client.invoke(lc_messages)  # type: ignore
            txt = (getattr(resp, "content", None) or "").strip()
            if txt:
                return txt
    except Exception:
        pass
    try:
        obj = client(messages=messages, temperature=temp, model=model)  # type: ignore
        if isinstance(obj, str):
            return obj.strip()
        txt = (getattr(obj, "content", None) or getattr(obj, "text", None) or str(obj) or "").strip()
        if txt:
            return txt
    except Exception:
        pass
    raise RuntimeError("LLM bridge failed: unsupported client shape or empty content.")

# ---------- Catalog type detection ----------

def _extract_catalog_type_from_text(text: str, asset_types: list[str]) -> Optional[str]:
    m = (text or "").strip().lower()
    hits: list[str] = []
    for t in asset_types:
        tl = (t or "").strip().lower()
        if tl and tl in m:
            hits.append(t)
    uniq = list(dict.fromkeys(hits))
    return uniq[0] if len(uniq) == 1 else None

# ---------- Ask-next question ----------

def _build_next_question(*, deps: GraphDeps, draft: dict, asset_types: list[str], locale: str) -> str:
    lang = _lang(locale)
    order = ["assetType","name", "isPrimary","confidentiality","integrity","availability","description"]
    missing = [k for k in order if (draft.get(k) in (None, "", []) or k not in draft)]
    if draft.get("isPrimary") is False:
        missing = [k for k in missing if k not in ("confidentiality","integrity","availability")]
    if "isPrimary" in missing:
        missing = [x for x in missing if x not in ("confidentiality","integrity","availability")]

    target = missing[0] if missing else None

    system = (
        "You help a user create an asset by asking ONE concise follow-up question.\n"
        f"Language: {lang}.\n"
        "- Ask for the most important missing field next, following the given order.\n"
        "- Preserve catalog labels EXACTLY as provided (English).\n"
        "- CIA enums: confidentiality=[PUBLIC,INTERNAL,PROTECTED,STRICTLY_PROTECTED], integrity/availability=[LOW,MEDIUM,HIGH].\n"
        "- Ask 'isPrimary' before CIA; if isPrimary=false, do not ask CIA.\n"
        "- Output only the question text."
    )
    payload = {"known_draft": {k: draft.get(k, None) for k in order},
               "next_slot": target,
               "catalog_asset_types": asset_types}
    q = _llm_text(
        deps=deps,
        messages=[{"role":"system","content":system}, {"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
        temperature=getattr(deps, "llm_temperature", None) or 0.2,
    )
    return q.strip()

# ---------- LLM extractor ----------

def _extract_create_draft_with_llm(*, deps: GraphDeps, text: str, asset_types: list[str]) -> dict:
    detected_type = _extract_catalog_type_from_text(text, asset_types)
    system = (
        "Transform a short asset request into STRICT JSON.\n"
        "Allowed keys: name, assetType, isPrimary, confidentiality, integrity, availability, description.\n"
        "CIA enums: confidentiality=[PUBLIC,INTERNAL,PROTECTED,STRICTLY_PROTECTED]; integrity/availability=[LOW,MEDIUM,HIGH].\n"
        "- If user_text contains exactly one catalog label, use it as assetType; otherwise null.\n"
        "- Never set name equal to a catalog label unless explicitly stated.\n"
        "Output JSON only."
    )
    human = {"user_text": text, "asset_types": asset_types}
    try:
        raw = _llm_text(
            deps=deps,
            messages=[{"role":"system","content":system}, {"role":"user","content":json.dumps(human, ensure_ascii=False)}],
            temperature=getattr(deps, "llm_temperature", None) or 0.0,
        ).strip()
        if raw.startswith("```"):
            i, j = raw.find("{"), raw.rfind("}")
            if i != -1 and j != -1 and j > i:
                raw = raw[i:j+1]
        obj = json.loads(raw) if raw else {}
        if not isinstance(obj, dict):
            obj = {}

        if detected_type:
            obj["assetType"] = detected_type
        elif obj.get("assetType") not in asset_types:
            obj["assetType"] = None
        # Canonicalize the label to the exact catalog spelling
        if isinstance(obj.get("assetType"), str):
            for t in asset_types:
                if t.lower() == obj["assetType"].lower():
                    obj["assetType"] = t
                    break
        nm_det = _extract_name_from_text(text)
        if nm_det:
            obj["name"] = nm_det

        nm = obj.get("name")
        if isinstance(nm, str) and nm.strip().lower() in [t.lower() for t in asset_types]:
            obj["name"] = None

        obj["confidentiality"] = _norm_enum(obj.get("confidentiality"), allowed=CONF_LEVELS)
        obj["integrity"] = _norm_enum(obj.get("integrity"), allowed=TRI_LEVELS)
        obj["availability"] = _norm_enum(obj.get("availability"), allowed=TRI_LEVELS)

        if obj.get("isPrimary") is False:
            obj["confidentiality"] = None
            obj["integrity"] = None
            obj["availability"] = None

        return obj
    except Exception:
        nm_det = _extract_name_from_text(text)
        return {"name": nm_det} if nm_det else {}

# ---------- History helper ----------

def _history_tail(state: AssetState, k: int) -> list[dict]:
    try:
        return (state.history or [])[-k:]
    except Exception:
        return []

# ---------- NLU (planner) ----------

async def _nlu(deps: GraphDeps, state: AssetState) -> AssetState:
    """
    NLU router for the Assets tab.

    Key change (loop-breaker):
    - When the create wizard is active (create_flow=True) but there are NO pending_fields,
      we are effectively waiting for a simple confirmation (yes/no) OR a different intent (e.g., explain).
      In that situation we DO NOT blindly route back to 'asset.create'. Instead:
        * yes-like  -> route to 'asset.create' (so the node can perform the write)
        * no-like   -> cancel the wizard and reply
        * explain   -> allow explain during the lock
        * anything else -> reply normally (agent_reply), avoiding a re-entry loop
    """

    logger.debug(
        "nlu.enter lock=%s pending=%s intent=%s action=%s msg='%s'",
        state.create_flow, state.pending_fields, state.intent, state.action, (state.message or "")[:160],
    )
    raw = (state.message or "").strip()

    # Reset intended routing every turn (prevents stale intent from previous step)
    state.intent = None

    # Locale pin based on CZ/SK cues (cheap heuristic)
    if raw and _looks_czech(raw):
        if not state.locale or not state.locale.lower().startswith("cs"):
            state.locale = "cs-CZ"

    # Greeting guard: break any accidental create flow AND clear stale writes
    if _looks_like_greeting(raw):
        # Only clear if we are NOT actively waiting for a concrete slot
        if not (state.pending_fields and len(state.pending_fields) > 0):
            state.create_flow = False
            state.pending_fields = None
            state.create_draft = None
            state.actions = None
            state.once_flags = None
            state.intent = cast(Intent, "agent_reply")
            logger.debug("nlu.exit(reason=greeting) -> intent=agent_reply lock=False (wizard cleared)")
            return state

    # Honor explicit action (UI or caller may force a tool)
    if state.action in ("list", "get", "create", "update_meta", "explain", "list_types", "agent_reply"):
        state.intent = cast(Intent, state.action)
        if state.action == "list_types" and not state.create_flow:
            # If we are not in a locked flow, showing types shouldn't set any slot
            state.pending_fields = None
        logger.debug(
            "nlu.exit(reason=explicit_action) -> intent=%s lock=%s pending=%s",
            state.intent, state.create_flow, state.pending_fields
        )
        return state

    sel = state.selection or {}

    # Short acknowledgements shouldn't (re)open the create flow
    # (prevents "ok/yes/no" from falling back into asset.create when nothing is pending)
    if not state.create_flow and not (state.pending_fields and len(state.pending_fields) > 0):
        if _is_yes_like(raw) or _is_no_like(raw):
            state.intent = cast(Intent, "agent_reply")
            logger.debug("nlu.exit(reason=ack_noop) -> intent=agent_reply")
            return state

    # ---------------------------
    # FLOW-LOCK (during create)
    # ---------------------------
    if state.create_flow or (state.pending_fields and len(state.pending_fields) > 0):
        # CASE A: Wizard active but no pending slot -> we are effectively waiting for confirmation
        if not (state.pending_fields and len(state.pending_fields) > 0):
            # Confirmation continues the flow (the create node will consume the yes/no)
            if _is_yes_like(raw):
                state.intent = cast(Intent, "create")
                logger.debug("nlu.lock: confirmation -> route to asset.create")
                return state

            # Negative confirmation cancels the wizard cleanly
            if _is_no_like(raw):
                state.create_flow = False
                state.pending_fields = None
                state.create_draft = None
                state.intent = cast(Intent, "agent_reply")
                logger.debug("nlu.lock: cancel -> agent_reply (wizard cleared)")
                return state

            # Allow explain even while in the lock (useful for clarifications)
            if _looks_like_explain(raw):
                state.intent = cast(Intent, "explain")
                logger.debug("nlu.lock: explain allowed while waiting confirmation")
                return state

            # Any other text: DO NOT re-enter asset.create (break potential loop)
            state.intent = cast(Intent, "agent_reply")
            logger.debug("nlu.lock: no pending slot & no confirmation -> agent_reply (loop broken)")
            return state

        # CASE B: Wizard active and waiting for specific slot(s)
        # Special handling: if we wait for 'assetType', allow type-only replies
        if state.pending_fields and "assetType" in state.pending_fields:
            try:
                types = [str(x) for x in (deps.backend_port.list_asset_types() or [])]
            except Exception:
                types = []
            matched = _extract_catalog_type_from_text(raw, types)
            if matched:
                state.create_draft = state.create_draft or {}
                state.create_draft["assetType"] = matched
                state.pending_fields = [f for f in (state.pending_fields or []) if f != "assetType"]
                state.intent = cast(Intent, "create")
                logger.debug("nlu.lock: captured assetType=%r → intent=create", matched)
                return state

        # Still in the lock: allow explain
        if _looks_like_explain(raw):
            state.intent = cast(Intent, "explain")
            logger.debug("nlu.lock: explain allowed")
            return state

        # Default while waiting for a slot: continue the create flow
        state.intent = cast(Intent, "create")
        logger.debug("nlu.lock: pending slot -> route to asset.create")
        return state

    # ---------------------------
    # CREATE start (free routing)
    # ---------------------------
    create_intent = _looks_like_create(raw)
    if create_intent:
        try:
            types = [str(x) for x in (deps.backend_port.list_asset_types() or [])]
        except Exception:
            types = []
        found_type = _extract_catalog_type_from_text(raw, types)
        state.create_draft = state.create_draft or {}
        if found_type:
            state.create_draft["assetType"] = found_type
            state.create_flow = True
            state.intent = cast(Intent, "create")
            logger.debug("nlu.exit(reason=create_with_type) -> intent=%s lock=%s", state.intent, state.create_flow)
            return state
        state.create_flow = True
        state.intent = cast(Intent, "list_types")
        logger.debug("nlu.exit(reason=create_without_type) -> intent=%s lock=%s", state.intent, state.create_flow)
        return state

    # Type-only reply outside of an active flow (user wrote just a type)
    try:
        types = [str(x) for x in (deps.backend_port.list_asset_types() or [])]
    except Exception:
        types = []
    matched = _extract_catalog_type_from_text(raw, types)
    if matched:
        state.create_draft = state.create_draft or {}
        state.create_draft["assetType"] = matched
        state.create_flow = True
        state.intent = cast(Intent, "create")
        logger.debug("nlu.exit(reason=type_only_reply) -> intent=%s lock=%s", state.intent, state.create_flow)
        return state

    # ---------------------------
    # Default planner (agent_decide)
    # ---------------------------
    allowed_actions: list[str] = ["llm.explain", "asset.list", "asset.get", "asset.update_meta"]
    if create_intent or matched or _mentions_types(raw):
        allowed_actions += ["asset.create", "asset.list_types"]

    decision: AgentDecision = await agent_decide(
        deps=deps,
        tab="assets",
        message=raw,
        language=state.locale or "en",
        history=_history_tail(state, 5),
        selection=sel,
        allowed_actions=allowed_actions,
        one_action_only=True,
    )

    tool_to_intent = {
        "asset.list": "list",
        "asset.list_types": "list_types",
        "asset.create": "create",
        "asset.update_meta": "update_meta",
        "llm.explain": "explain",
        "asset.get": "get",
    }
    chosen = tool_to_intent.get((decision.action or "").strip())
    state.intent = cast(Intent, chosen or "agent_reply")

    # If listing types outside of a create flow, don't keep any pending slot
    if state.intent == "list_types" and not state.create_flow:
        state.pending_fields = None

    # Pass-through for optional decision args (e.g., question for explain)
    if getattr(decision, "args", None):
        args = decision.args or {}
        if "question" in args and isinstance(args["question"], str):
            state.question = args["question"]

    logger.debug(
        "nlu.exit(reason=default_decide) -> intent=%s lock=%s pending=%s",
        state.intent, state.create_flow, state.pending_fields
    )
    return state


# ---------- Read-only nodes ----------

def _assets_list(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        t0 = _event(state, kind="tool.call.start", tool="asset.list")
        try:
            items = deps.backend_port.assets_list(state.org_id, None, None, None) or []
            n = _parse_list_limit(state.message or "")
            if n is not None:
                items = items[:n]

            lines = []
            for a in items:
                d = _jsonify_iter(a)
                if not isinstance(d, dict):
                    d = {"value": d}
                nm = d.get("name") or ""
                tp = d.get("assetType") or ""
                lines.append(f"{nm} ({tp})" if tp else nm)

            heading = "Assets" if _lang(state.locale) not in ("cs","sk") else "Aktiva"
            body = f"{title(heading)}\n\n{bullet_list(lines) if lines else '_No assets found_'}"
            parts = chunk(body, max_chars=1800)

            state.result = _ui_response(parts[0], extra={"agent_events": state.agent_events})
            if len(parts) > 1:
                state.result["messages"].extend(parts[1:])  # type: ignore[index]
            _finish(state, t0, tool="asset.list", ok=True, meta={"count": len(items)})
        except Exception as e:
            _finish(state, t0, tool="asset.list", ok=False, meta={"error": str(e)})
            state.result = _ui_response({"error":"asset.list failed","reason":str(e)}, extra={"agent_events": state.agent_events})
        return state
    return _inner

def _asset_get_or_preview(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        asset_id = (state.selection or {}).get("asset_id")
        t0 = _event(state, kind="tool.call.start", tool="asset.get", meta={"asset_id": asset_id})
        try:
            if not asset_id:
                items = deps.backend_port.assets_list(state.org_id, None, None, None) or []
                preview = []
                for a in items[:8]:
                    d = _jsonify_iter(a)
                    if not isinstance(d, dict):
                        d = {"value": d}
                    preview.append({"id": d.get("id") or d.get("asset_id"), "name": d.get("name"), "assetType": d.get("assetType")})
                state.result = _ui_response({"kind":"asset.select.suggest","preview":preview}, extra={"agent_events": state.agent_events})
                _finish(state, t0, tool="asset.get", ok=True, meta={"suggested": len(preview)})
                return state

            a = deps.backend_port.asset_get(asset_id)
            d = _jsonify_iter(a)
            if not isinstance(d, dict):
                d = {"value": d}
            state.result = _ui_response({"kind":"asset.get","asset": d}, extra={"agent_events": state.agent_events})
            _finish(state, t0, tool="asset.get", ok=True, meta={"id": asset_id})
        except Exception as e:
            _finish(state, t0, tool="asset.get", ok=False, meta={"error": str(e)})
            state.result = _ui_response({"error":"asset.get failed","reason":str(e)}, extra={"agent_events": state.agent_events})
        return state
    return _inner

def _list_asset_types(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        t0 = _event(state, kind="tool.call.start", tool="asset.list_types")
        try:
            types = deps.backend_port.list_asset_types() or []
            heading = "Asset types" if _lang(state.locale) not in ("cs","sk") else "Typy aktiv"
            content = [title(heading), ""]
            content.append(bullet_list([str(x) for x in types]) if types else "_No types found_")
            if types:
                content.extend(["", ("Který typ aktiva chceš vytvořit?" if _lang(state.locale) in ("cs","sk") else "Which asset type do you want to create?")])
            text = "\n".join(content)
            parts = chunk(text, max_chars=1800)
            state.result = _ui_response(parts[0], extra={"agent_events": state.agent_events})
            if len(parts) > 1:
                state.result["messages"].extend(parts[1:])  # type: ignore[index]

            state.create_draft = (state.create_draft or {})
            if state.create_flow:
                state.create_draft.pop("assetType", None)
                # Ask for the type next → set a single pending slot
                state.pending_fields = ["assetType"]
            else:
                state.pending_fields = None

            _finish(state, t0, tool="asset.list_types", ok=True, meta={"count": len(types)})
        except Exception as e:
            _finish(state, t0, tool="asset.list_types", ok=False, meta={"error": str(e)})
            state.result = _ui_response({"error":"asset.list_types failed","reason":str(e)}, extra={"agent_events": state.agent_events})
        return state
    return _inner

# ---------- JSON-safe converters ----------

def _to_plain_dict(obj: Any, fallback: dict) -> dict:
    """Iterative conversion to a JSON-dumpable dict (no recursion)."""
    try:
        val = _jsonify_iter(obj)
        return val if isinstance(val, dict) else {"value": val}
    except Exception:
        return fallback

# ---------- Write validation ----------

def _validate_write_actions(actions: list[dict]) -> tuple[list[dict], Optional[str]]:
    validated: list[dict] = []
    for a in actions:
        op = (a.get("op") or "").strip()
        try:
            if op == "create_asset":
                payload = {"op":"create_asset","client_token": a.get("client_token"),"fields": a.get("fields") or {}}
                f = payload["fields"]

                if not (isinstance(f.get("name"), str) and f["name"].strip()):
                    return [], "create_asset requires fields.name"
                if not (isinstance(f.get("assetType"), str) and f["assetType"].strip()):
                    return [], "create_asset requires fields.assetType"
                if not isinstance(f.get("isPrimary"), bool):
                    return [], "create_asset requires fields.isPrimary (bool)"

                c, i, a_ = f.get("confidentiality"), f.get("integrity"), f.get("availability")
                if f["isPrimary"] is True:
                    if not (c and i and a_):
                        return [], "create_asset with isPrimary=True requires CIA"
                else:
                    if any(x is not None for x in (c, i, a_)):
                        return [], "create_asset with isPrimary=False must not contain CIA"

                validated.append(payload)

            elif op == "set_asset_meta":
                fields = (a.get("fields") or {})
                illegal = set(fields.keys()) - {"name","description"}
                if illegal:
                    return [], f"set_asset_meta: illegal fields for chat update: {sorted(illegal)}"
                payload = {"op":"set_asset_meta","asset_id": a.get("asset_id"),"fields": fields}
                if not payload["asset_id"]:
                    return [], "set_asset_meta requires asset_id"
                if not payload["fields"]:
                    return [], "set_asset_meta requires at least one field (name/description)"
                validated.append(payload)

            elif op == "set_asset_cia":  # <-- NEW: CIA writer op
                asset_id = a.get("asset_id")
                fields = (a.get("fields") or {})
                if not asset_id:
                    return [], "set_asset_cia requires asset_id"
                allowed = {"confidentiality","integrity","availability"}
                illegal = set(fields.keys()) - allowed
                if illegal:
                    return [], f"set_asset_cia: illegal fields: {sorted(illegal)}"
                # Numbers are already normalized; accept None to clear.
                for k, v in fields.items():
                    if v is None:
                        continue
                    if not isinstance(v, int):
                        return [], f"set_asset_cia: {k} must be int or null"
                    if k == "confidentiality":
                        if v < 1 or v > 4:
                            return [], "confidentiality must be 1..4 (or None)"
                    else:
                        if v < 1 or v > 3:
                            return [], f"{k} must be 1..3 (or None)"
                validated.append({"op":"set_asset_cia","asset_id": asset_id,"fields": fields})

            else:
                return [], f"unsupported operation '{op}'"
        except Exception as e:
            return [], f"write validation failed for op='{op}': {e}"
    return validated, None

# ---------- APPLY actions (INLINE submit, no state.actions) ----------

async def _apply_actions_async(*, deps: GraphDeps, state: AssetState, acts: list[dict]) -> AssetState:
    """
    Build & submit a validated ChangeSet; surface user-friendly UX:

    - Validates/normalizes actions (accepts canonical + legacy op names).
    - Builds a Pydantic ChangeSet (adds audit + idempotency_key).
    - Submits via deps.submit_changeset(...) with OUTBOX fallback allowed.
    - If the submit fell back to OUTBOX (queued), flips response.staging.needs_commit = True
      and produces a meaningful one-line chat message explaining what happened.
    - Emits WS asset events only on a real online submission (never for queued).
    - Never raises into the graph: returns a UI-shaped result on all paths.
    """
    # Localized confirmations later (one-liner)
    loc = (state.locale or "en")

    # 1) Nothing to do → short-circuit
    if not acts:
        state.result = _ui_response("No actions to apply.", extra={"agent_events": state.agent_events})
        return state

    # 2) Ensure we have a submit function on deps
    submit_fn = getattr(deps, "submit_changeset", None)
    if not callable(submit_fn):
        state.result = _ui_response({"error": "submit_changeset not available"}, extra={"agent_events": state.agent_events})
        return state

    # 3) Start telemetry span
    t0 = _event(state, kind="tool.call.start", tool="submit_changeset", meta={"actions": len(acts)})

    # 4) Normalize canonical -> writer op names for backward compatibility
    #    (If callers used "asset.create"/"asset.update_meta"/"asset.set_meta",
    #     map them to writer ops understood by the backend.)
    def _normalize_legacy_action(a: dict) -> dict:
        op_map = {
            "asset.create": "create_asset",
            "asset.update_meta": "set_asset_meta",
            "asset.set_meta": "set_asset_cia",
        }
        op = (a.get("op") or "").strip()
        return {**a, "op": op_map.get(op, op)}

    try:
        # 5) Import typed Action/ChangeSet builder lazily (keeps import order simple)
        from workshops_ai.models.changeset import build_changeset  # adds audit + idempotency_key
        try:
            # If the union is available, use it to validate each action dict strictly
            from workshops_ai.models.changeset import Action  # discriminated union by "op" (Pydantic v2)
            typed_actions = []
            for a in acts:
                a_norm = _normalize_legacy_action(a or {})
                try:
                    typed_actions.append(Action.model_validate(a_norm))
                except Exception:
                    # Be permissive: on validation drift, pass through normalized dict
                    typed_actions.append(a_norm)
        except Exception:
            # If Action union is not importable, pass normalized dicts; build_changeset will validate deeply.
            typed_actions = [_normalize_legacy_action(a or {}) for a in acts]

        # 6) Build a ChangeSet model (adds audit & idempotency_key; puts user into audit)
        cs_model = build_changeset(
            org_id=state.org_id,
            user_id=(state.user_id or "unknown"),
            actions=typed_actions,
            intent="apply",
        )

        # Safe echo for UI (never stores heavy runtime objects)
        cs_dump = getattr(cs_model, "model_dump", lambda **k: cs_model)(mode="json")

        # 7) Submit (do NOT require remote success; OUTBOX fallback is allowed here)
        submit_result = await submit_fn(cs_model, require_success=False)

        # 8) UX helper: build one-line chat message and (optionally) adjust staging state
        #    - On queued/outbox: the helper composes a friendly sentence mentioning the fallback
        #      and the reason; we also mark response.staging.needs_commit=True.
        try:
            from workshops_ai.utils.submit_feedback import apply_submit_outcome_and_message
            chat_msg = apply_submit_outcome_and_message(
                state=state,
                changeset=cs_model,
                result=submit_result,
            )
        except Exception:
            # If the helper is unavailable, fall back to legacy one-liners
            submitted_flag = bool(getattr(submit_result, "submitted", False))
            queued_flag = bool(getattr(submit_result, "fallback_path", None)) and not submitted_flag
            chat_msg = _llm_confirm(deps=deps, loc=loc, actions=acts, queued=queued_flag)

        # 9) Interpret the result flags
        submitted = bool(getattr(submit_result, "submitted", False))
        queued = bool(getattr(submit_result, "fallback_path", None)) and not submitted

        if submitted:
            # --- Successful online submit ---
            _finish(state, t0, tool="submit_changeset", ok=True, meta={"actions": len(acts)})

            # Emit WS events (best-effort) only on real submit (avoid dupes for queued)
            hub = getattr(deps, "ws_hub", None)
            if hub:
                async def _emit_created_and_updated():
                    """
                    Best-effort WebSocket notifications for created/updated assets.

                    - Uses centralized emitters (change_events) that wrap WsHub.
                    - Emits 'asset.created' for create actions; 'asset.updated' for meta/CIA changes.
                    - Never raises: any exception is swallowed.
                    """
                    try:
                        from workshops_ai.app.services.change_events import (
                            aemit_asset_created,
                            aemit_asset_updated,
                        )
                    except Exception:
                        return

                    try:
                        created_ids = _guess_created_ids(submit_result)  # ordered list of IDs from backend (best-effort)
                        created_idx = 0

                        for a in acts:
                            # Accept canonical + writer ops
                            a_norm = _normalize_legacy_action(a or {})
                            op = str(a_norm.get("op") or "").strip()
                            fields = (a_norm.get("fields") or {})
                            asset_id_from_payload = (
                                a_norm.get("asset_id") or fields.get("asset_id") or fields.get("id")
                            )

                            # CREATE
                            if op in ("asset.create", "create_asset"):
                                asset_id = (
                                    created_ids[created_idx]
                                    if created_idx < len(created_ids) and created_ids[created_idx]
                                    else asset_id_from_payload
                                )
                                created_idx += 1
                                await aemit_asset_created(
                                    org_id=state.org_id,
                                    asset_id=asset_id,
                                    name=fields.get("name"),
                                    assetType=fields.get("assetType"),
                                    isPrimary=fields.get("isPrimary"),
                                    description=fields.get("description"),
                                    actor=state.user_id,
                                    ts=_now_iso(),
                                )
                                continue

                            # META (name/description)
                            if op in ("asset.update_meta", "set_asset_meta", "update_asset_meta"):
                                updated_fields = [k for k in ("name", "description") if k in fields]
                                await aemit_asset_updated(
                                    org_id=state.org_id,
                                    asset_id=asset_id_from_payload,
                                    name=fields.get("name"),
                                    description=fields.get("description"),
                                    updated_fields=updated_fields,
                                    actor=state.user_id,
                                    ts=_now_iso(),
                                )
                                continue

                            # CIA (confidentiality / integrity / availability)
                            if op in ("asset.set_meta", "set_asset_cia"):
                                updated_fields = [k for k in ("confidentiality", "integrity", "availability") if k in fields]
                                await aemit_asset_updated(
                                    org_id=state.org_id,
                                    asset_id=asset_id_from_payload,
                                    updated_fields=updated_fields,
                                    confidentiality=fields.get("confidentiality"),
                                    integrity=fields.get("integrity"),
                                    availability=fields.get("availability"),
                                    actor=state.user_id,
                                    ts=_now_iso(),
                                )
                                continue
                    except Exception:
                        # Never let best-effort notifications break the main flow
                        pass

                _schedule(_emit_created_and_updated)

            # Clean runtime-only fields after success
            state.create_flow = False
            state.pending_fields = None
            state.create_draft = None
            state.once_flags = None

            # Build UI result; also echo changeset/result
            result = _ui_response(chat_msg, extra={"changeset_actions": len(acts), "agent_events": state.agent_events})
            result["messages"].append({"kind": "changeset", "changeset": cs_dump, "ts": _now_iso()})
            result["messages"].append({"kind": "submit_result", "result": {"submitted": True}, "ts": _now_iso()})
            # Online submit means no staging commit is needed
            try:
                result["staging"]["needs_commit"] = False
            except Exception:
                pass
            state.result = result
            return state

        if queued:
            # --- Not submitted online, but safely queued to OUTBOX ---
            _finish(state, t0, tool="submit_changeset", ok=True, meta={"queued": True})

            # Clean runtime-only fields even if queued
            state.create_flow = False
            state.pending_fields = None
            state.create_draft = None
            state.once_flags = None

            res = _ui_response(chat_msg, extra={"agent_events": state.agent_events})
            res["messages"].append({
                "kind": "changeset.queued",
                "tab": "assets",
                "ts": _now_iso(),
                "org_id": state.org_id,
                "actor": state.user_id,
            })
            # Queued → mark that a commit is needed in the UI
            try:
                res["staging"]["needs_commit"] = True
            except Exception:
                pass
            state.result = res
            return state

        # --- Neither submitted nor queued → treat as failure/validation block ---
        _finish(state, t0, tool="submit_changeset", ok=False, meta={"error": "unknown submit result"})
        fail_msg = chat_msg or "Submission failed."
        res = _ui_response(fail_msg, extra={"agent_events": state.agent_events})
        try:
            # Failure: no queued write → no staged commit needed
            res["staging"]["needs_commit"] = False
        except Exception:
            pass
        state.result = res
        return state

    except Exception as e:
        # Defensive failure path with a clean UI message
        _finish(state, t0, tool="submit_changeset", ok=False, meta={"error": str(e)})
        res = _ui_response("Submission failed.", extra={"agent_events": state.agent_events})
        try:
            res["staging"]["needs_commit"] = False
        except Exception:
            pass
        state.result = res
        return state


# ---------- CREATE (slot-filling with INLINE submit) ----------

def _create_asset(deps: GraphDeps):
    """
    Slot-driven CREATE node with safe, inline submission.

    Design:
    - STRICT single-slot capture: when `pending_fields` is set, we only accept that one field.
    - We never put write payloads (actions/changeset) into state → no recursion risks.
    - Confirmation gate: if `state.staging is False`, require an explicit yes-like reply.
    """
    async def _inner(state: AssetState) -> AssetState:
        draft = dict(state.create_draft or {})

        # Always load catalog types (canonical English labels) for hard validation
        catalog_types: list[str] = [str(x) for x in (deps.backend_port.list_asset_types() or [])]

        # Canonicalize assetType to the exact catalog label (case-insensitive)
        if draft.get("assetType"):
            for t in catalog_types:
                if t.lower() == draft["assetType"].lower():
                    draft["assetType"] = t
                    break

        # This list is only used to drive LLM extraction / first question.
        # Hard checks will always use `catalog_types`.
        asset_type_known = bool((draft.get("assetType") or "").strip())
        types: list[str] = catalog_types if not asset_type_known else []

        logger.debug(
            "create.enter types_needed=%s draft_keys=%s",
            not asset_type_known,
            list((state.create_draft or {}).keys()),
        )

        prev_type = draft.get("assetType")
        last_msg = (state.message or "").strip()

        raw_pending = state.pending_fields
        pending: list[str] = list(raw_pending or [])

        # --- Cycle-break guard (prevents re-enter loops) -------------------
        state.once_flags = dict(state.once_flags or {})
        core_sig = (
            tuple(pending),
            draft.get("assetType"), draft.get("name"), draft.get("isPrimary"),
            draft.get("confidentiality"), draft.get("integrity"), draft.get("availability"),
        )
        sig = hash(("create", core_sig))
        if state.once_flags.get("create_step_hash") == sig:
            cnt = int(state.once_flags.get("create_step_repeat", 0)) + 1
            state.once_flags["create_step_repeat"] = cnt
            if cnt >= 2:
                # Break the loop gracefully and unlock the wizard
                state.create_flow = False
                state.pending_fields = None
                note = "Zastaveno kvůli opakované smyčce. Zkus prosím znovu zadat typ nebo název."
                state.result = _ui_response(note, extra={"agent_events": state.agent_events})
                return state
        else:
            state.once_flags["create_step_hash"] = sig
            state.once_flags["create_step_repeat"] = 0

        # 1) Pre-merge confirmation gate (requires all core fields ready)
        waiting_confirm = (
            state.staging is False
            and bool(draft)
            and pending == []
            and isinstance(draft.get("name"), str) and draft.get("name", "").strip()
            and isinstance(draft.get("assetType"), str) and draft.get("assetType", "").strip()
            and isinstance(draft.get("isPrimary"), bool)
        )
        skip_merge = False

        if waiting_confirm:
            if _is_yes_like(last_msg) or (last_msg.lower().strip() in {"potvrdit vytvoření", "potvrdit vytvoreni"}):
                state.once_flags = dict(state.once_flags or {})
                state.once_flags["one_shot_write"] = True
                skip_merge = True
            elif _is_no_like(last_msg):
                state.create_flow = False
                state.pending_fields = None
                state.create_draft = None
                cancel_msg = "Zrušeno." if _lang(state.locale) in ("cs", "sk") else "Cancelled."
                state.result = _ui_response(cancel_msg, extra={"agent_events": state.agent_events})
                return state
            else:
                typ = draft.get("assetType") or "?"
                nm = draft.get("name") or "?"
                prim_txt = "primární" if draft.get("isPrimary") else "neprimární"
                confirm_msg = (
                    f"Chcete vytvořit aktivum {typ} s názvem „{nm}“, {prim_txt}? "
                    f"Odpověz „Potvrdit vytvoření“ nebo prostě „OK“."
                )
                state.result = _ui_response(confirm_msg, extra={"agent_events": state.agent_events})
                return state

        # 2) Merge current user message into draft (single-slot first)
        if not skip_merge:
            if pending == ["isPrimary"]:
                ip = _normalize_is_primary(last_msg)
                if ip is not None:
                    draft["isPrimary"] = ip
                    pending = []
                else:
                    msg = (
                        "Je aktivum primární? Odpověz Ano/Ne."
                        if _lang(state.locale) in ("cs", "sk")
                        else "Is the asset primary? Please answer Yes/No."
                    )
                    state.create_draft = draft
                    state.pending_fields = ["isPrimary"]
                    state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
                    return state

            elif len(pending) == 1 and pending[0] in {"confidentiality", "integrity", "availability"}:
                slot = pending[0]
                val = _conf_from_text(last_msg) if slot == "confidentiality" else _tri_from_text(last_msg)
                if val:
                    draft[slot] = val
                    pending = []
                else:
                    l = _lang(state.locale)
                    slot_cs = {"confidentiality": "důvěrnost", "integrity": "integritu", "availability": "dostupnost"}[slot]
                    if slot == "confidentiality":
                        msg = (
                            f"Zadej {slot_cs}: PUBLIC/INTERNAL/PROTECTED/STRICTLY_PROTECTED."
                            if l in ("cs", "sk")
                            else "Provide confidentiality: PUBLIC/INTERNAL/PROTECTED/STRICTLY_PROTECTED."
                        )
                    else:
                        msg = (
                            f"Zadej {slot_cs}: LOW/MEDIUM/HIGH."
                            if l in ("cs", "sk")
                            else f"Provide {slot}: LOW/MEDIUM/HIGH."
                        )
                    state.create_draft = draft
                    state.pending_fields = [slot]
                    state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
                    return state

            elif pending == ["name"]:
                nm = _extract_name_from_text(last_msg) or last_msg.strip().strip('"').strip("'")
                if nm:
                    draft["name"] = _normalize_name(nm)
                    pending = []
                else:
                    msg = "Zadej název aktiva." if _lang(state.locale) in ("cs", "sk") else "Please provide the asset name."
                    state.create_draft = draft
                    state.pending_fields = ["name"]
                    state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
                    return state

            elif pending == ["description"]:
                desc = (last_msg or "").strip()
                if desc:
                    draft["description"] = desc
                    pending = []
                else:
                    msg = "Stručně popiš aktivum." if _lang(state.locale) in ("cs", "sk") else "Please provide a short description."
                    state.create_draft = draft
                    state.pending_fields = ["description"]
                    state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
                    return state

            else:
                # Free merge via LLM extraction (respects single-slot mode if active)
                inc = _extract_create_draft_with_llm(deps=deps, text=last_msg, asset_types=types) if last_msg else {}
                allowed_keys = set(pending) if (raw_pending is not None) else None
                if prev_type and inc.get("assetType") and inc["assetType"] != prev_type:
                    inc.pop("assetType", None)
                for k, v in (inc or {}).items():
                    if v is None:
                        continue
                    if allowed_keys is not None and k not in allowed_keys:
                        continue
                    draft[k] = v
                if "isPrimary" in (raw_pending or []) and draft.get("isPrimary") is None:
                    ip = _normalize_is_primary(last_msg)
                    if ip is not None:
                        draft["isPrimary"] = ip

        logger.debug(
            "create.merge prev_type=%r draft(core)=%r",
            prev_type,
            {k: draft.get(k) for k in ("assetType", "name", "isPrimary", "confidentiality", "integrity", "availability")},
        )

        # 3) Normalize & detect missing
        draft["name"] = _normalize_name(draft.get("name"))
        draft["assetType"] = _normalize_type(draft.get("assetType"), allowed=catalog_types)
        if draft.get("isPrimary") is not None and not isinstance(draft.get("isPrimary"), bool):
            draft["isPrimary"] = _normalize_is_primary(draft["isPrimary"])
        try:
            # IMPORTANT: always pass the full catalog here (not 'types')
            draft = _normalize_create_fields(draft, catalog_types=catalog_types)
        except Exception:
            pass

        # --- Hard guard AFTER normalization ---
        # If assetType is missing or not a member of the catalog, show the catalog list.
        if not draft.get("assetType") or draft["assetType"] not in catalog_types:
            state.create_draft = draft
            state.pending_fields = ["assetType"]
            return _list_asset_types(deps)(state)

        def _miss(k: str) -> bool:
            return (draft.get(k) in (None, "", []) or k not in draft)

        order = ["assetType", "name", "isPrimary", "confidentiality", "integrity", "availability", "description"]
        missing = [k for k in order if _miss(k)]
        if draft.get("isPrimary") is False:
            missing = [k for k in missing if k not in ("confidentiality", "integrity", "availability")]

        if missing:
            target = missing[0]
            q = _build_next_question(
                deps=deps,
                draft=draft,
                asset_types=(types if _miss("assetType") else []),
                locale=(state.locale or "en"),
            )
            state.create_draft = draft
            state.pending_fields = [target]
            logger.debug("create.missing → pending=%r (target only)", state.pending_fields)
            state.result = _ui_response(q or "Provide the next value.", extra={"agent_events": state.agent_events})
            return state

        # 4) Post-normalization CIA gating
        if draft.get("isPrimary") is False:
            draft["confidentiality"] = None
            draft["integrity"] = None
            draft["availability"] = None
        else:
            draft["confidentiality"] = _norm_enum(draft.get("confidentiality"), allowed=CONF_LEVELS)
            draft["integrity"] = _norm_enum(draft.get("integrity"), allowed=TRI_LEVELS)
            draft["availability"] = _norm_enum(draft.get("availability"), allowed=TRI_LEVELS)

        fields = {
            "id": draft.get("id"),
            "name": draft.get("name"),
            "assetType": draft.get("assetType"),
            "isPrimary": draft.get("isPrimary"),
            "confidentiality": draft.get("confidentiality"),
            "integrity": draft.get("integrity"),
            "availability": draft.get("availability"),
            "description": draft.get("description"),
        }
        fields = {k: v for k, v in fields.items() if v is not None}
        logger.debug("create.prepare fields=%r", fields)

        # Validate write payload (kept simple; TypeAdapter check happens later on backend)
        actions = [{"op": "create_asset", "client_token": f"tmp-{uuid.uuid4().hex[:10]}", "fields": fields}]
        try:
            logger.debug("create.validate.start actions=%d", len(actions))
            validated = actions
            err = None
            logger.debug("create.validate.ok actions=%d", len(validated))
        except RecursionError as e:
            _diag_crashpack(
                "validate_write_actions.recursion",
                {"draft": draft, "actions": actions},
                e,
                preview={"fields": fields},
            )
            raise
        except Exception as e:
            _diag_crashpack(
                "validate_write_actions.error",
                {"draft": draft, "actions": actions},
                e,
                preview={"fields": fields},
            )
            raise
        if err:
            state.result = _ui_response({"error": "validation", "reason": err}, extra={"agent_events": state.agent_events})
            return state

        # 5) Confirmation guard (production mode requires yes-like)
        write_allowed = True
        if state.staging is False:
            write_allowed = bool((state.once_flags or {}).get("one_shot_write", False))

        if not write_allowed:
            typ = fields.get("assetType") or "?"
            nm = fields.get("name") or "?"
            prim_txt = "primární" if fields.get("isPrimary") is True else "neprimární"
            confirm_msg = (
                f"Chcete vytvořit aktivum {typ} s názvem „{nm}“, {prim_txt}? "
                f"Napište: Potvrdit vytvoření."
            )
            state.create_draft = draft
            state.pending_fields = []
            state.result = _ui_response(confirm_msg, extra={"agent_events": state.agent_events})
            return state

        # 6) INLINE SUBMIT (no state.actions persistence)
        acts = validated
        info = "Připraveno k zápisu." if _lang(state.locale) in ("cs","sk") else "Preparing to write…"
        state.result = _ui_response(info, extra={"agent_events": state.agent_events})

        if DIAG:
            try:
                json.dumps({
                    "org_id": state.org_id,
                    "locale": state.locale,
                    "pending": state.pending_fields,
                    "draft_core": {k: draft.get(k) for k in (
                        "assetType","name","isPrimary","confidentiality","integrity","availability"
                    )},
                }, ensure_ascii=False)
            except Exception as e:
                _diag_crashpack("state_diag_snapshot.error", {"pending": state.pending_fields}, e)

        logger.debug("create.submit.inline actions=%d", len(acts))
        # Apply immediately and return
        return await _apply_actions_async(deps=deps, state=state, acts=acts)

    return _inner


# ---------- Update meta (INLINE submit) ----------

def _update_meta(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    async def _inner(state: AssetState) -> AssetState:
        items = state.update_items or []
        if not items:
            try:
                arr = json.loads(state.message or "[]")
                if isinstance(arr, list):
                    items = [AssetUpdateItem(**x) for x in arr]
            except Exception:
                pass
        if not items:
            sel_id = (state.selection or {}).get("asset_id")
            if not sel_id:
                return _asset_get_or_preview(deps)(state)
            state.result = _ui_response("Specify `name` and/or `description` to update.")
            return state

        actions: list[dict] = []
        for it in items:
            fields = {"name": _normalize_name(it.name), "description": it.description}
            fields = {k: v for k, v in fields.items() if v is not None}
            if fields:
                actions.append({"op":"set_asset_meta","asset_id": it.asset_id,"fields": fields})

        if not actions:
            state.result = _ui_response("No changes specified (only name/description are allowed here).")
            return state

        validated, err = _validate_write_actions(actions)
        if err:
            state.result = _ui_response({"error":"validation","reason": err}, extra={"agent_events": state.agent_events})
            return state

        return await _apply_actions_async(deps=deps, state=state, acts=validated)
    return _inner

# ---------- Submit changeset (legacy path, kept for completeness) ----------

async def _submit_changeset_async(deps: GraphDeps, state: AssetState) -> AssetState:
    acts = state.actions or []
    return await _apply_actions_async(deps=deps, state=state, acts=acts)

def _guess_created_ids(submit_result: Any) -> list[str]:
    ids: list[str] = []
    try:
        if isinstance(submit_result, dict):
            if isinstance(submit_result.get("created_ids"), list):
                for x in submit_result["created_ids"]:
                    if x: ids.append(str(x))
                return ids
            results = submit_result.get("results")
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict) and (r.get("op") == "create_asset"):
                        res = r.get("result") or {}
                        vid = res.get("id") or res.get("asset_id")
                        if vid: ids.append(str(vid))
                return ids
        results = getattr(submit_result, "results", None)
        if isinstance(results, list):
            for r in results:
                op = getattr(r, "op", None) or (isinstance(r, dict) and r.get("op"))
                result = getattr(r, "result", None) or (isinstance(r, dict) and r.get("result"))
                if op == "create_asset":
                    vid = getattr(result, "id", None) or (isinstance(result, dict) and result.get("id"))
                    if vid: ids.append(str(vid))
    except Exception:
        pass
    return ids

# ---------- LLM confirmation one-liner ----------

def _llm_confirm(*, deps: GraphDeps, loc: str, actions: list[dict], queued: bool = False) -> str:
    lang = _lang(loc)
    system = (
        "You generate ONE short confirmation sentence after a change in an asset system.\n"
        f"Language: {lang}. Keep it under 20 words. Neutral tone. If queued=true, mention it."
    )
    human = {"queued": bool(queued), "actions": actions}
    try:
        txt = _llm_text(
            deps=deps,
            messages=[{"role":"system","content": system}, {"role":"user","content": json.dumps(human, ensure_ascii=False)}],
            temperature=0.1,
        ).strip()
        if txt and len(txt) <= 200:
            return txt
    except Exception:
        pass
    return "Hotovo." if lang in ("cs","sk") else "Done."

# ---------- Explain (RAG/LLM) ----------

async def _explain_assets_async(deps: GraphDeps, state: AssetState) -> AssetState:
    q = (state.question or state.message or "").strip()
    if not q:
        state.result = _ui_response("Please provide a question about assets.")
        return state

    retriever = None
    try:
        if hasattr(deps, "make_retriever") and callable(deps.make_retriever):
            retriever = deps.make_retriever(state.org_id, "assets")
    except Exception:
        retriever = None

    snippets: list[dict] = []
    if retriever:
        t0 = _event(state, kind="tool.call.start", tool="retriever.search", meta={"tab":"assets"})
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
            "- Answer briefly (≤120 words) in the user's language; ground claims in snippets with [S#]."
        )
        human = {"question": q, "snippets": tagged, "locale": (state.locale or "en")}
        t0 = _event(state, kind="tool.call.start", tool="llm.synthesize", meta={"snippets": len(tagged)})
        try:
            txt = _llm_text(deps=deps, messages=[{"role":"system","content": system}, {"role":"user","content": json.dumps(human, ensure_ascii=False)}], temperature=0.2)
            if txt:
                _finish(state, t0, tool="llm.synthesize", ok=True, meta={})
                state.result = _ui_response(txt, extra={"snippets": tagged, "agent_events": state.agent_events})
                _append_pending_reminder(state)
                return state
        except Exception as e:
            _finish(state, t0, tool="llm.synthesize", ok=False, meta={"error": str(e)})

    if client is not None:
        system = "You are a cybersecurity assistant. Provide a concise (≤120 words) answer without hallucinations."
        human = {"question": q, "locale": (state.locale or "en")}
        t0 = _event(state, kind="tool.call.start", tool="llm.explain")
        try:
            txt = _llm_text(deps=deps, messages=[{"role":"system","content": system}, {"role":"user","content": json.dumps(human, ensure_ascii=False)}], temperature=0.2)
            if txt:
                _finish(state, t0, tool="llm.explain", ok=True, meta={})
                state.result = _ui_response(txt, extra={"agent_events": state.agent_events})
                _append_pending_reminder(state)
                return state
        except Exception as e:
            _finish(state, t0, tool="llm.explain", ok=False, meta={"error": str(e)})

    state.result = _ui_response("No evidence available and fallback did not answer.", extra={"agent_events": state.agent_events})
    _append_pending_reminder(state)
    return state

def _append_pending_reminder(state: AssetState) -> None:
    if not (state.create_flow or state.pending_fields or state.create_draft):
        return
    draft = state.create_draft or {}
    order = ["assetType","name","isPrimary","confidentiality","integrity","availability","description"]
    missing = [k for k in order if (draft.get(k) in (None,"",[]) or k not in draft)]
    if draft.get("isPrimary") is False:
        missing = [k for k in missing if k not in ("confidentiality","integrity","availability")]
    if not missing:
        return
    note = (f"Chceš pokračovat? Aktuálně čekám na: {', '.join(missing[:3])}."
            if _lang(state.locale) in ('cs','sk')
            else f"Do you want to continue? I’m waiting for: {', '.join(missing[:3])}.")
    try:
        state.result["messages"].append(note)  # type: ignore[index]
    except Exception:
        pass

# ---------- Reply-only branch ----------

def _agent_reply(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        text = (state.message or "").strip()
        l = _lang(state.locale)
        system = (
            "You are a helpful assistant for an asset management UI.\n"
            f"Language: {l}. Respond briefly (≤80 words). "
            "Mention you can list assets, show details, create assets, update metadata, or explain concepts if needed. "
            "Do not invent data; do not claim to have taken actions."
        )
        try:
            ans = _llm_text(
                deps=deps,
                messages=[{"role":"system","content": system}, {"role":"user","content": text}],
                temperature=0.3,
            )
        except Exception:
            ans = "S čím mám v aktivech pomoci?" if l in ("cs","sk") else "How can I help with your assets?"
        state.result = _ui_response(ans, extra={"agent_events": state.agent_events})
        return state
    return _inner

# ---------- Graph assembly ----------

def create_graph(deps: GraphDeps):
    graph = StateGraph(AssetState)

    async def _nlu_node(state: AssetState) -> AssetState:
        return await _nlu(deps, state)

    async def _submit_node(state: AssetState) -> AssetState:
        # Legacy path; normally unused because we submit inline
        return await _submit_changeset_async(deps, state)

    async def _explain_node(state: AssetState) -> AssetState:
        return await _explain_assets_async(deps, state)

    graph.add_node("nlu", _nlu_node)
    graph.add_node("route", lambda s: s)

    graph.add_node("agent.reply", _agent_reply(deps))
    graph.add_node("asset.list", _assets_list(deps))
    graph.add_node("asset.list_types", _list_asset_types(deps))
    graph.add_node("asset.get", _asset_get_or_preview(deps))
    graph.add_node("asset.create", _create_asset(deps))
    graph.add_node("asset.update_meta", _update_meta(deps))
    graph.add_node("submit_changeset", _submit_node)
    graph.add_node("llm.explain", _explain_node)

    graph.set_entry_point("nlu")
    graph.add_edge("nlu", "route")

    def _route(state: AssetState) -> str:
        intent = state.intent or "agent_reply"
        if intent == "list":         return "asset.list"
        if intent == "list_types":   return "asset.list_types"
        if intent == "get":          return "asset.get"
        if intent == "create":       return "asset.create"
        if intent == "update_meta":  return "asset.update_meta"
        if intent == "explain":      return "llm.explain"
        return "agent.reply"

    graph.add_conditional_edges(
        "route",
        _route,
        {
            "agent.reply": "agent.reply",
            "asset.list": "asset.list",
            "asset.list_types": "asset.list_types",
            "asset.get": "asset.get",
            "asset.create": "asset.create",
            "asset.update_meta": "asset.update_meta",
            "llm.explain": "llm.explain",
        },
    )

    # After create/update we end (we submit inline). Keep legacy edge harmlessly.
    def _after_create(state: AssetState) -> str:
        return "END"
    graph.add_conditional_edges(
        "asset.create",
        _after_create,
        {"END": END, "submit_changeset": "submit_changeset"},
    )
    graph.add_edge("asset.update_meta", END)

    for node in ("agent.reply","asset.list","asset.list_types","asset.get","llm.explain","submit_changeset"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=deps.checkpointer)

# ---------- Thin runners ----------

async def run_create(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

async def run_update_meta(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

# runner pro asset.set_meta (CIA+description)
async def run_set_meta(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    """
    Handle 'asset.set_meta' as a single-turn, validated write:
    - Accepts POLICY strings (PUBLIC/INTERNAL/PROTECTED/STRICTLY_PROTECTED, LOW/MEDIUM/HIGH).
    - Allowed only for primary assets (isPrimary == True); otherwise returns policy error.
    - Converts policy strings to writer numerics and writes via 'set_asset_cia'.
    """
    args: dict = ((ui_req.body or {}).get("args") or {})
    asset_id: str = str(args.get("assetId") or "").strip()
    if not asset_id:
        return UIResponse(
            messages=[{"error": "validation", "reason": "assetId is required"}],
            proposals=None,
            staging={"needs_commit": False, "batch_id": None, "diff": None},
            policy={"emit_changes_enabled": True},
        )

    # 1) Fetch asset to verify primary flag
    try:
        existing = deps.backend_port.asset_get(asset_id)
    except Exception as ex:
        return UIResponse(
            messages=[{"error": "backend", "reason": f"Failed to fetch asset: {ex}"}],
            proposals=None,
            staging={"needs_commit": False, "batch_id": None, "diff": None},
            policy={"emit_changes_enabled": True},
        )
    if not existing:
        return UIResponse(
            messages=[{"error": "not_found", "asset_id": asset_id}],
            proposals=None,
            staging={"needs_commit": False, "batch_id": None, "diff": None},
            policy={"emit_changes_enabled": True},
        )

    def _get(obj, *keys):
        for k in keys:
            if isinstance(obj, dict) and k in obj:
                return obj[k]
            if hasattr(obj, k):
                try:
                    return getattr(obj, k)
                except Exception:
                    pass
        return None

    is_primary = bool(_get(existing, "isPrimary", "primary", "is_primary"))

    # 2) Map policy strings -> writer integers (None clears)
    CONF_MAP = {"PUBLIC": 1, "INTERNAL": 2, "PROTECTED": 3, "STRICTLY_PROTECTED": 4}
    TRI_MAP  = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

    def _policy_to_int(slot: str, value: Any) -> Optional[int]:
        if value is None:
            return None
        s = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        if slot == "confidentiality":
            return CONF_MAP.get(s)
        if slot in ("integrity", "availability"):
            return TRI_MAP.get(s)
        return None

    c_raw = args.get("confidentiality", None)  # e.g., "PROTECTED" | None
    i_raw = args.get("integrity", None)        # e.g., "MEDIUM"    | None
    a_raw = args.get("availability", None)     # e.g., "HIGH"      | None

    cia_fields: dict[str, Optional[int]] = {}
    if "confidentiality" in args:
        cia_fields["confidentiality"] = _policy_to_int("confidentiality", c_raw)
    if "integrity" in args:
        cia_fields["integrity"] = _policy_to_int("integrity", i_raw)
    if "availability" in args:
        cia_fields["availability"] = _policy_to_int("availability", a_raw)

    # 3) Enforce rule: CIA can be set only for primary assets
    if cia_fields and not is_primary:
        return UIResponse(
            messages=[{"error": "policy", "reason": "CIA cannot be set for a supporting asset (isPrimary=false).", "asset_id": asset_id}],
            proposals=None,
            staging={"needs_commit": False, "batch_id": None, "diff": None},
            policy={"emit_changes_enabled": True},
        )

    # If caller sent invalid strings (mapping -> None) AND didn't intend to clear, reject
    for k, v in list(cia_fields.items()):
        if k in args and args.get(k) is not None and v is None:
            # e.g., "PROTECTD" typo
            msg = f"{k} must be one of: " + ( "PUBLIC/INTERNAL/PROTECTED/STRICTLY_PROTECTED" if k=="confidentiality" else "LOW/MEDIUM/HIGH" )
            return UIResponse(
                messages=[{"error": "validation", "reason": msg}],
                proposals=None,
                staging={"needs_commit": False, "batch_id": None, "diff": None},
                policy={"emit_changes_enabled": True},
            )

    if not cia_fields:
        return UIResponse(
            messages=["No changes to apply."],
            proposals=None,
            staging={"needs_commit": False, "batch_id": None, "diff": None},
            policy={"emit_changes_enabled": True},
        )

    # 4) Build single writer action (numeric payload expected by set_asset_cia)
    actions: list[dict] = [{"op": "set_asset_cia", "asset_id": asset_id, "fields": cia_fields}]

    # 5) Validate & submit through common path
    dummy_state = AssetState(
        org_id=ui_req.org_id,
        active_tab="assets",
        message="",
        selection={"asset_id": asset_id},
        user_id=ui_req.user_id or "unknown",
        locale=(ui_req.headers or {}).get("accept-language") if hasattr(ui_req, "headers") else "en",
        agent_events=[],
    )

    validated, err = _validate_write_actions(actions)
    if err:
        dummy_state.result = _ui_response({"error": "validation", "reason": err}, extra={"agent_events": dummy_state.agent_events})
        return UIResponse.model_validate(dummy_state.result)

    dummy_state = await _apply_actions_async(deps=deps, state=dummy_state, acts=validated)
    return UIResponse.model_validate(dummy_state.result or _ui_response("No result."))

async def run_explain(*, deps: GraphDeps, text: str, locale: Optional[str] = None) -> dict:
    state = AssetState(
        org_id=getattr(getattr(deps, "settings", object()), "org_id", "demo-org"),
        active_tab="assets",
        message=text,
        question=text,
        locale=locale or "en",
        agent_events=[],
    )
    state = await _explain_assets_async(deps, state)
    return state.result or {"ok": False, "error": "no_result"}

__all__ = ["AssetState", "create_graph", "run_create", "run_update_meta", "run_set_meta", "run_explain"]
