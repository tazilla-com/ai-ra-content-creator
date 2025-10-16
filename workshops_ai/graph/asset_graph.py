# File: workshops_ai/graph/asset_graph.py
# -----------------------------------------------------------------------------
# LangGraph: ASSETS — LLM-first, slot-driven, ONE action per turn
#
# Hardenings in this version:
# - Confirmation as a control intent:
#   * On the confirm turn (e.g., "Potvrdit vytvoření"/"Confirm"), we DO NOT parse or
#     merge any fields. We only enable a one-shot write and proceed.
# - Pending-field lock:
#   * During CREATE, merge user input ONLY into fields listed in state.pending_fields.
#   * Special case: when pending == ['isPrimary'], map yes/no and ignore everything else.
# - Yes/No detection:
#   * Yes/No is interpreted as isPrimary ONLY when the flow is explicitly waiting for it.
# - Recursion prevention / serializable state:
#   * state.actions is always a list of PLAIN DICTS (not Pydantic objects),
#     so LangGraph checkpointer (e.g., Redis) can serialize safely. This avoids
#     RecursionError during checkpointing.
# - No re-entry loops:
#   * After create → submit, we clear actions and wizard state; we do not bounce back
#     into asset.create within the same turn.
# - Language handling:
#   * Locale pinning remains defensive here, but the Manager should prefer to pass
#     a stable language in LangGraph config. This module will stay robust if it isn’t.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Callable, Literal, Optional, cast, TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from workshops_ai.graph.factory import GraphDeps

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.graph.agent_node import agent_decide, AgentDecision
from workshops_ai.models.changeset import (
    ChangeSet,
    ActionCreateAsset,
    ActionSetAssetMeta,  # used for name/description
    build_changeset,
)

# ---------- Normalizers (prefer centralized utils; safe fallbacks if missing) ----------
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
        if isinstance(value, bool):
            return value
        s = str(value or "").strip().lower()
        pos = {"ano","jo","jj","j","y","yes","true","1","ok","okay","confirm","confirmed","sure","primary"}
        neg = {"ne","n","no","false","0","nope","neni","není"}
        if s in pos: return True
        if s in neg: return False
        return None

# ---------- UI helpers (fail-safe fallbacks) ----------
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

# ---------- Optional payload guard ----------
try:
    from workshops_ai.tools.guard import guard_validate_payload as _guard_validate  # type: ignore
except Exception:  # pragma: no cover
    def _guard_validate(model_cls, payload):  # type: ignore
        return payload


# ---------- CIA enums / local normalizer ----------
CONF_LEVELS = {"PUBLIC", "INTERNAL", "PROTECTED", "STRICTLY_PROTECTED"}
TRI_LEVELS = {"LOW", "MEDIUM", "HIGH"}

def _norm_enum(value: Optional[str], *, allowed: set[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if v == "STRICTLYPROTECTED":
        v = "STRICTLY_PROTECTED"
    return v if v in allowed else None


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
    # Request context
    org_id: str
    active_tab: str = Field(default="assets")
    message: Optional[str] = None
    selection: Optional[dict] = None
    user_id: Optional[str] = None
    locale: Optional[str] = None
    language: Optional[str] = None
    history: Optional[list[dict]] = None
    staging: Optional[bool] = None            # write gate (False → confirm)
    once_flags: Optional[dict] = None         # one-shot write toggles

    # Structured inputs
    action: Optional[Intent] = None
    create_items: Optional[list[AssetCreateItem]] = None
    update_items: Optional[list[AssetUpdateItem]] = None
    question: Optional[str] = None

    # Working outputs
    intent: Optional[Intent] = None
    actions: Optional[list[dict]] = None
    change: Optional[ChangeSet | dict] = None
    result: Optional[dict] = None

    # Conversational slot-filling (CREATE only)
    create_draft: Optional[dict] = None
    pending_fields: Optional[list[str]] = None
    create_flow: bool = False

    # Telemetry
    agent_events: list[ToolEvent] = Field(default_factory=list)


# ---------- Telemetry & UI ----------
def _event(state: AssetState, *, kind: str, tool: str, ok: bool = True, meta: dict | None = None) -> float:
    t0 = time.perf_counter()
    state.agent_events.append({"event_type": kind, "tool": tool, "ok": ok, "meta": meta or {}})
    return t0

def _finish(state: AssetState, t0: float, *, tool: str, ok: bool, meta: dict | None = None) -> None:
    dur = int((time.perf_counter() - t0) * 1000)
    state.agent_events.append({"event_type": "tool.call.success" if ok else "tool.call.fail",
                               "tool": tool, "ok": ok, "duration_ms": dur, "meta": meta or {}})

def _ui_response(content: str | dict, *, extra: Optional[dict] = None) -> dict:
    """
    Builds an append-only UI payload to avoid accidental self-references.
    """
    base = {
        "messages": [content] if isinstance(content, str) else [content],
        "proposals": None,
        "staging": {"needs_commit": False, "batch_id": None, "diff": None},
        "policy": {"emit_changes_enabled": True},
    }
    if extra:
        for k, v in extra.items():
            if k != "messages":
                base[k] = v
    return base

def _now_iso() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()

def _schedule(coro_or_fn):
    """Fire-and-forget WS emits; never await inside the turn."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro_or_fn if asyncio.iscoroutine(coro_or_fn) else coro_or_fn())
    except RuntimeError:
        asyncio.run(coro_or_fn if asyncio.iscoroutine(coro_or_fn) else coro_or_fn())


# ---------- Localization ----------
def _lang(loc: Optional[str]) -> str:
    return (loc or "en").split("-")[0].lower()

_CZ_DIACRITICS = set("áéíóúýčďěňřšťůžďťľĺôäÁÉÍÓÚÝČĎĚŇŘŠŤŮŽĎŤĽĹÔÄ")
_CZ_TOKENS = {"vytvoř","vytvor","název","nazev","potvrdit","potvrzuji","schvaluji","ano","ne","není","neni","aktiva","typ","primární","primarni"}

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
    """Deterministic name capture for CZ/EN; intentionally conservative."""
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
    # Fallback: if the raw is very short, you MIGHT treat it as a name,
    # but this is dangerous during pending=['isPrimary'] or confirm.
    # We rely on pending-lock to prevent misuse in those phases.
    if len(raw.split()) <= 5:
        return raw
    return None

# Confirmation / yes-no recognizers
_CONFIRM_RE = re.compile(r"^(potvrdit(\s+vytvoření|[ ]*vytvoreni)?|potvrzuji|schvaluji|confirm|ok|okay|yes)$", re.I)
_CANCEL_RE  = re.compile(r"^(zrušit|zrusit|cancel|stop|ne|no)$", re.I)

def _is_yes_like(text: str) -> bool:
    return bool(_CONFIRM_RE.match((text or "").strip().lower()))

def _is_no_like(text: str) -> bool:
    return bool(_CANCEL_RE.match((text or "").strip().lower()))


# ---------- LLM bridge (deps.llm_client only; tool-agnostic) ----------
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
                role = (m.get("role") or "").lower()
                content = m.get("content") or ""
                if role == "system":      lc_messages.append(SystemMessage(content=content))
                elif role in ("assistant","ai"): lc_messages.append(AIMessage(content=content))
                else:                      lc_messages.append(HumanMessage(content=content))
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


# ---------- Ask-next question (LLM-generated) ----------
def _build_next_question(
    *, deps: GraphDeps, draft: dict, asset_types: list[str], locale: str
) -> str:
    lang = _lang(locale)
    order = ["assetType","name","isPrimary","confidentiality","integrity","availability","description"]
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
        "- If asking about asset type, list 3–6 catalog types (bulleted) as options.\n"
        "- Preserve catalog labels EXACTLY as provided (English). Do NOT translate catalog labels.\n"
        "- For CIA, REQUIRE these exact enums:\n"
        "  • confidentiality: PUBLIC, INTERNAL, PROTECTED, STRICTLY_PROTECTED\n"
        "  • integrity: LOW, MEDIUM, HIGH\n"
        "  • availability: LOW, MEDIUM, HIGH\n"
        "- Ask 'isPrimary' before any CIA question. If isPrimary is false, do not ask CIA.\n"
        "- Output only the question text, no preamble, no markdown fences."
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


# ---------- One-liner LLM extractor (guarded by deterministic rules) ----------
def _extract_create_draft_with_llm(*, deps: GraphDeps, text: str, asset_types: list[str]) -> dict:
    detected_type = _extract_catalog_type_from_text(text, asset_types)
    system = (
        "Transform a short asset request into STRICT JSON.\n"
        "Allowed keys: name (str), assetType (str|null), isPrimary (bool|null), "
        "confidentiality (PUBLIC|INTERNAL|PROTECTED|STRICTLY_PROTECTED|null), "
        "integrity (LOW|MEDIUM|HIGH|null), availability (LOW|MEDIUM|HIGH|null), "
        "description (str|null).\n"
        "- If user_text contains exactly one catalog label, use that exact label as assetType.\n"
        "- Never guess assetType; if none or multiple match, set assetType=null.\n"
        "- Never set name equal to any catalog label unless the user explicitly states it is the name.\n"
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

        # Deterministic override for type
        if detected_type:
            obj["assetType"] = detected_type
        elif obj.get("assetType") not in asset_types:
            obj["assetType"] = None

        # Deterministic name capture beats LLM (prevents 'Operating system' → name confusion)
        nm_det = _extract_name_from_text(text)
        if nm_det:
            obj["name"] = nm_det

        # Prevent "type-as-name" unless explicit
        nm = obj.get("name")
        if isinstance(nm, str) and nm.strip().lower() in [t.lower() for t in asset_types]:
            obj["name"] = None

        # CIA normalize via local fallback
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


# ---------- NLU helpers ----------
def _looks_like_create(msg: str) -> bool:
    m = (msg or "").strip().lower()
    tokens = ["vytvoř","vytvor","založ","zaloz","přidej","pridej","add","create","nové","nove","new"]
    return any(t in m for t in tokens)

def _looks_like_explain(msg: str) -> bool:
    m = (msg or "").strip().lower()
    keys = ["explain","what is","co je","vysvětli","vysvetli","prosím vysvětli"]
    return any(k in m for k in keys)


# ---------- NLU (planner) ----------
def _history_tail(state: AssetState, k: int = 5) -> list[dict]:
    try:
        return (state.history or [])[-k:]
    except Exception:
        return []

async def _nlu(deps: GraphDeps, state: AssetState) -> AssetState:
    """
    Routes the user input to a single intent. While in CREATE wizard,
    we force route to 'asset.create' (with a one-off 'explain' allowed).
    Also pins Czech locale after the first CZ-looking input.
    """
    logger.debug(
        "nlu.enter lock=%s pending=%s draft_keys=%s intent=%s action=%s msg=%r",
        state.create_flow, state.pending_fields, list((state.create_draft or {}).keys()),
        state.intent, state.action, (state.message or "")[:160],
    )
    raw = (state.message or "").strip()

    # --- Locale pinning: once CZ is seen, stick to cs-CZ for the session ---
    if raw and _looks_czech(raw):
        if not state.locale or not state.locale.lower().startswith("cs"):
            state.locale = "cs-CZ"

    # Honor explicit action if provided
    if state.action in ("list","get","create","update_meta","explain","list_types","agent_reply"):
        state.intent = cast(Intent, state.action)
        if state.action == "list_types" and not state.create_flow:
            state.pending_fields = None
        logger.debug("nlu.exit(reason=explicit_action) -> intent=%s lock=%s pending=%s", state.intent, state.create_flow, state.pending_fields)
        return state

    sel = state.selection or {}

    # ---------- FLOW-LOCK (during create) ----------
    if state.create_flow or (state.pending_fields and len(state.pending_fields) > 0):
        # Type capture without planner if waiting for assetType:
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

        # Allow one-off explain
        if _looks_like_explain(raw):
            state.intent = cast(Intent, "explain")
            logger.debug("nlu.lock: explain allowed")
            return state

        state.intent = cast(Intent, "create")
        logger.debug("nlu.lock: force route to asset.create")
        return state

    # ---------- CREATE start ----------
    create_intent = _looks_like_create(raw)
    if create_intent:
        types = [str(x) for x in (deps.backend_port.list_asset_types() or [])]
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

    # ---------- Type-only reply (after listing) ----------
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

    # ---------- Default planner ----------
    allowed_default = ["llm.explain","asset.list","asset.list_types","asset.create","asset.update_meta","asset.get"]
    decision: AgentDecision = await agent_decide(
        deps=deps,
        tab="assets",
        message=raw,
        language=state.locale or "en",
        history=_history_tail(state, 5),
        selection=sel,
        allowed_actions=allowed_default,
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

    if state.intent == "list_types" and not state.create_flow:
        state.pending_fields = None

    if getattr(decision, "args", None):
        args = decision.args or {}
        if "question" in args and isinstance(args["question"], str):
            state.question = args["question"]

    logger.debug("nlu.exit(reason=default_decide) -> intent=%s lock=%s pending=%s", state.intent, state.create_flow, state.pending_fields)
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
                d = a.model_dump() if hasattr(a, "model_dump") else dict(a)
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
                    d = a.model_dump() if hasattr(a, "model_dump") else dict(a)
                    preview.append({"id": d.get("id") or d.get("asset_id"), "name": d.get("name"), "assetType": d.get("assetType")})
                state.result = _ui_response({"kind":"asset.select.suggest","preview":preview}, extra={"agent_events": state.agent_events})
                _finish(state, t0, tool="asset.get", ok=True, meta={"suggested": len(preview)})
                return state

            a = deps.backend_port.asset_get(asset_id)
            d = a.model_dump() if hasattr(a, "model_dump") else dict(a)
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

            # Start slot-filling ONLY IF inside CREATE flow:
            state.create_draft = (state.create_draft or {})
            if state.create_flow:
                state.create_draft.pop("assetType", None)  # force explicit user choice after the list
                state.pending_fields = ["assetType"]       # answer expected next
            else:
                state.pending_fields = None

            _finish(state, t0, tool="asset.list_types", ok=True, meta={"count": len(types)})
        except Exception as e:
            _finish(state, t0, tool="asset.list_types", ok=False, meta={"error": str(e)})
            state.result = _ui_response({"error":"asset.list_types failed","reason":str(e)}, extra={"agent_events": state.agent_events})
        return state
    return _inner


# ---------- Write validation ----------
def _to_plain_dict(obj: Any, fallback: dict) -> dict:
    """
    Ensure a plain-JSON serializable dict (avoid Pydantic instances in state.actions).
    """
    try:
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")  # type: ignore[attr-defined]
        if isinstance(obj, dict):
            return obj
        # Last resort: JSON round-trip with __dict__ fallback
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return fallback

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
                        return [], "create_asset with isPrimary=True requires CIA (confidentiality, integrity, availability)"
                else:
                    if any(x is not None for x in (c, i, a_)):
                        return [], "create_asset with isPrimary=False must not contain CIA"

                guarded = _guard_validate(ActionCreateAsset, payload)  # may return model or dict
                validated.append(_to_plain_dict(guarded, payload))

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
                guarded = _guard_validate(ActionSetAssetMeta, payload)
                validated.append(_to_plain_dict(guarded, payload))
            else:
                return [], f"unsupported operation '{op}'"
        except Exception as e:
            return [], f"write validation failed for op='{op}': {e}"
    return validated, None


# ---------- CREATE (slot-filling) ----------
def _create_asset(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    """
    Slot-driven CREATE with strict pending-field lock and confirm-gate.
    """
    def _inner(state: AssetState) -> AssetState:
        raw_types = deps.backend_port.list_asset_types() or []
        types = [str(x) for x in raw_types]

        logger.debug("create.enter types_count=%s draft_keys=%s", len(types), list((state.create_draft or {}).keys()))

        draft = dict(state.create_draft or {})
        prev_type = draft.get("assetType")
        last_msg = (state.message or "").strip()
        pending: list[str] = list(state.pending_fields or [])

        # ---------- CONFIRM-GATE handled BEFORE any field merge ----------
        waiting_confirm = (
            state.staging is False and
            bool(draft) and
            pending == [] and
            isinstance(draft.get("name"), str) and draft.get("name").strip() and
            isinstance(draft.get("assetType"), str) and draft.get("assetType").strip() and
            isinstance(draft.get("isPrimary"), bool)
        )

        if waiting_confirm:
            # Control intent: confirm / cancel — NO field parsing here.
            if _is_yes_like(last_msg) or (last_msg.lower().strip() in {"potvrdit vytvoření","potvrdit vytvoreni"}):
                logger.debug("create.confirm: yes-like → enable one-shot write")
                state.once_flags = dict(state.once_flags or {})
                state.once_flags["one_shot_write"] = True
                # fall-through to build actions from EXISTING draft (no merge)
            elif _is_no_like(last_msg):
                # Cancel the wizard, no merge.
                state.create_flow = False
                state.pending_fields = None
                state.create_draft = None
                state.actions = None
                cancel_msg = "Zrušeno." if _lang(state.locale) in ("cs","sk") else "Cancelled."
                state.result = _ui_response(cancel_msg)
                return state
            else:
                # Repeat the confirm prompt using the current draft (no merge).
                typ = draft.get("assetType") or "?"
                nm = draft.get("name") or "?"
                prim_txt = "primární" if draft.get("isPrimary") else "neprimární"
                confirm_msg = f"Chcete vytvořit aktivum {typ} s názvem „{nm}“, {prim_txt}? Napište: Potvrdit vytvoření."
                logger.debug("create.blocked.confirm msg=%r", confirm_msg)
                state.actions = None
                state.result = _ui_response(confirm_msg)
                return state

        # ---------- PENDING-LOCKED MERGE ----------
        if pending == ["isPrimary"]:
            # Map yes/no only; ignore anything else.
            ip = _normalize_is_primary(last_msg)
            if ip is not None:
                draft["isPrimary"] = ip
                pending = []  # resolved
            else:
                msg = "Je aktivum primární? Odpověz Ano/Ne." if _lang(state.locale) in ("cs","sk") else "Is the asset primary? Please answer Yes/No."
                state.create_draft = draft
                state.pending_fields = ["isPrimary"]
                state.result = _ui_response(msg)
                state.actions = None
                logger.debug("create.pending.isPrimary-only → ask again")
                return state
        else:
            # General case: allow LLM/deterministic extraction, but ONLY merge into pending slots
            inc = _extract_create_draft_with_llm(deps=deps, text=last_msg, asset_types=types) if last_msg else {}
            allowed_keys = set(pending) if pending else None  # None → allow all (initial capture)
            if prev_type and inc.get("assetType") and inc["assetType"] != prev_type:
                inc.pop("assetType", None)  # type lock
            for k, v in (inc or {}).items():
                if v is None:
                    continue
                if allowed_keys is not None and k not in allowed_keys:
                    continue  # lock: do not overwrite non-pending fields
                draft[k] = v

            # Additionally, if isPrimary wasn't captured and it's in pending set (among others),
            # try lightweight normalization but only when 'isPrimary' is pending:
            if "isPrimary" in (state.pending_fields or []) and draft.get("isPrimary") is None:
                ip = _normalize_is_primary(last_msg)
                if ip is not None:
                    draft["isPrimary"] = ip

        logger.debug("create.merge prev_type=%r draft=%r", prev_type, {k: draft.get(k) for k in ("assetType","name","isPrimary","confidentiality","integrity","availability")})

        # ---------- NORMALIZATION (just before ask/submit) ----------
        draft["name"] = _normalize_name(draft.get("name"))
        draft["assetType"] = _normalize_type(draft.get("assetType"), allowed=types)
        if draft.get("isPrimary") is not None and not isinstance(draft.get("isPrimary"), bool):
            draft["isPrimary"] = _normalize_is_primary(draft.get("isPrimary"))
        try:
            draft = _normalize_create_fields(draft, catalog_types=types)
        except Exception:
            pass

        # Minimal requirements
        need_name = not (isinstance(draft.get("name"), str) and draft["name"].strip())
        need_type = not (isinstance(draft.get("assetType"), str) and draft["assetType"].strip())
        need_primary = draft.get("isPrimary") is None
        need_cia = bool(draft.get("isPrimary") is True and any(
            draft.get(k) in (None, "") for k in ("confidentiality","integrity","availability")
        ))

        # Ask-next if something missing
        if need_name or need_type or need_primary or need_cia:
            q = _build_next_question(deps=deps, draft=draft, asset_types=types, locale=(state.locale or "en"))
            state.create_draft = draft
            pending_list: list[str] = []
            if need_type: pending_list.append("assetType")
            if need_name: pending_list.append("name")
            if need_primary: pending_list.append("isPrimary")
            if need_cia:
                for c in ("confidentiality","integrity","availability"):
                    if draft.get(c) in (None, ""):
                        pending_list.append(c)
            state.pending_fields = pending_list
            logger.debug("create.missing need_type=%s need_name=%s need_primary=%s need_cia=%s → pending=%r", need_type, need_name, need_primary, need_cia, pending_list)
            logger.debug("create.ask_next q=%r", q)
            state.actions = None
            state.result = _ui_response(q or "Provide the next value.")
            return state

        # Enforce CIA stripping based on isPrimary
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

        actions = [{"op":"create_asset","client_token": f"tmp-{uuid.uuid4().hex[:10]}","fields": fields}]
        validated, err = _validate_write_actions(actions)
        if err:
            state.result = _ui_response({"error":"validation","reason": err}, extra={"agent_events": state.agent_events})
            return state

        # ---------- Write gate & confirm loop ----------
        write_allowed = True
        if state.staging is False:
            one_shot = bool((state.once_flags or {}).get("one_shot_write", False))
            write_allowed = one_shot

        if not write_allowed:
            typ = fields.get("assetType") or "?"
            nm = fields.get("name") or "?"
            prim_txt = "primární" if fields.get("isPrimary") is True else "neprimární"
            confirm_msg = f"Chcete vytvořit aktivum {typ} s názvem „{nm}“, {prim_txt}? Napište: Potvrdit vytvoření."
            logger.debug("create.blocked.confirm msg=%r", confirm_msg)
            state.create_draft = draft
            state.pending_fields = []    # mark that we are waiting for confirm only
            state.actions = None
            state.result = _ui_response(confirm_msg)
            return state

        # Ready to submit; clear wizard slots and pass actions to submit node
        state.actions = validated                    # already plain dicts
        state.create_draft = None
        state.pending_fields = None
        logger.debug("create.ready actions=%s", len(validated))
        state.result = _ui_response({"kind":"changeset.prepared","actions": len(validated)})
        return state
    return _inner


# ---------- Update meta (safe: name/description only) ----------
def _update_meta(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
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

        state.actions = validated                    # plain dicts
        state.result = _ui_response({"kind":"changeset.prepared","actions": len(validated)})
        return state
    return _inner


# ---------- Submit changeset (single tool-call; then END) ----------
async def _submit_changeset_async(deps: GraphDeps, state: AssetState) -> AssetState:
    loc = (state.locale or "en")
    acts = state.actions or []
    if not acts:
        state.result = _ui_response("No actions to apply.", extra={"agent_events": state.agent_events})
        return state

    cs: ChangeSet | dict = build_changeset(
        org_id=state.org_id,
        user_id=state.user_id or "unknown",
        actions=acts,                         # acts are plain dicts already
        intent="apply",
    )
    state.change = cs

    submit_fn = getattr(deps, "submit_changeset", None)
    if not callable(submit_fn):
        state.result = _ui_response({"error":"submit_changeset not available"}, extra={"agent_events": state.agent_events})
        return state

    t0 = _event(state, kind="tool.call.start", tool="submit_changeset", meta={"actions": len(acts)})
    try:
        cs_typed = cs if hasattr(cs, "model_dump") else ChangeSet.model_validate(cs)
        submit_result = await submit_fn(cs_typed, require_success=False)  # type: ignore
        ok = bool(getattr(submit_result, "submitted", False))

        try:
            cs_dump = cs_typed.model_dump(mode="json") if hasattr(cs_typed, "model_dump") else dict(cs_typed)  # type: ignore[arg-type]
        except Exception:
            cs_dump = {"_error":"cannot_dump_changeset"}

        if ok:
            _finish(state, t0, tool="submit_changeset", ok=True, meta={"actions": len(acts)})

            # WS emits AFTER success
            hub = getattr(deps, "ws_hub", None)
            if hub:
                async def _emit_created_and_updated():
                    try:
                        created_ids = _guess_created_ids(submit_result)
                        idx = 0
                        for a in acts:
                            if (a.get("op") == "create_asset"):
                                f = a.get("fields") or {}
                                item = {
                                    "org_id": state.org_id,
                                    "asset_id": created_ids[idx] if idx < len(created_ids) else (f.get("id") or f.get("asset_id")),
                                    "name": f.get("name"),
                                    "assetType": f.get("assetType"),
                                    "isPrimary": f.get("isPrimary"),
                                    "description": f.get("description"),
                                    "ts": _now_iso(),
                                    "actor": state.user_id,
                                }
                                idx += 1
                                await hub.aemit_asset_created(item)
                            elif (a.get("op") == "set_asset_meta"):
                                f = a.get("fields") or {}
                                item = {
                                    "org_id": state.org_id,
                                    "asset_id": a.get("asset_id"),
                                    "name": f.get("name"),
                                    "description": f.get("description"),
                                    "ts": _now_iso(),
                                    "actor": state.user_id,
                                }
                                await hub.aemit_asset_updated(item)
                    except Exception:
                        pass
                _schedule(_emit_created_and_updated)

            # Clean wizard state AFTER success
            state.create_flow = False
            state.pending_fields = None
            state.create_draft = None
            state.actions = None  # guard against re-entry / re-submit

            confirm = _llm_confirm(deps=deps, loc=loc, actions=acts, queued=False)
            result = _ui_response(confirm, extra={"changeset_actions": len(acts), "agent_events": state.agent_events})
            result["messages"].append({"kind":"changeset","changeset": cs_dump,"ts": _now_iso()})
            result["messages"].append({"kind":"submit_result","result":{"submitted": True},"ts": _now_iso()})
            state.result = result
            return state

        # queued/outbox path
        _finish(state, t0, tool="submit_changeset", ok=True, meta={"queued": True})
        confirm = _llm_confirm(deps=deps, loc=loc, actions=acts, queued=True)
        res = _ui_response(confirm, extra={"agent_events": state.agent_events})
        res["messages"].append({"kind":"changeset.queued","tab":"assets","changeset": cs_dump,"ts": _now_iso(),"org_id": state.org_id,"actor": state.user_id})

        state.create_flow = False
        state.pending_fields = None
        state.create_draft = None
        state.actions = None

        state.result = res
        return state

    except Exception as e:
        _finish(state, t0, tool="submit_changeset", ok=False, meta={"error": str(e)})
        # Keep wizard state so the user can correct and retry.
        state.result = _ui_response("Submission failed.", extra={"agent_events": state.agent_events})
        state.actions = None
        return state


def _guess_created_ids(submit_result: Any) -> list[str]:
    """
    Best-effort extraction of created IDs from submit_result:
      - {'results': [{'op':'create_asset','result':{'id':...}}, ...]}
      - {'created_ids': [ ... ]}
      - object with .results[i].result.id
    """
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
        f"Language: {lang}. Keep it under 20 words. Be specific when possible.\n"
        "Neutral tone. No markdown. No emojis. If queued=true, mention it."
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


# ---------- Explain (RAG/LLM) with mid-creation reminder ----------
async def _explain_assets_async(deps: GraphDeps, state: AssetState) -> AssetState:
    q = (state.question or state.message or "").strip()
    if not q:
        state.result = _ui_response("Please provide a question about assets.")
        return state

    retriever = None
    try:
        if hasattr(deps, "make_retriever") and callable(deps.make_retriever):
            # Factory signature: MakeRetriever(org_id: str, tab: str) -> Optional[Retriever]
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
            "- Answer briefly (≤120 words) in the user's language if provided; otherwise English.\n"
            "- Ground every claim in provided snippets; include inline [S#] citations.\n"
            "- If unsupported, say you don't have evidence."
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
            f"Language: {l}. Respond briefly (≤80 words). If the user seems lost, "
            "mention you can list assets, show details, create assets, update metadata, "
            "or explain concepts. Do not invent data; do not claim to have taken actions."
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


# ---------- Graph assembly (single action per turn) ----------
def create_graph(deps: GraphDeps):
    """
    Flow:
      start → nlu → route → {agent.reply | list | list_types | get | create | update_meta | explain} → (submit?) → END
    Writes always pass through submit once. After submit, we clear actions to prevent re-entry recursion.
    """
    graph = StateGraph(AssetState)

    async def _nlu_node(state: AssetState) -> AssetState:
        return await _nlu(deps, state)

    async def _submit_node(state: AssetState) -> AssetState:
        return await _submit_changeset_async(deps, state)

    async def _explain_node(state: AssetState) -> AssetState:
        return await _explain_assets_async(deps, state)

    # Nodes
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

    # Entry
    graph.set_entry_point("nlu")
    graph.add_edge("nlu", "route")

    # Routing
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

    # After create → submit or END (no recursion)
    def _after_create(state: AssetState) -> str:
        return "submit_changeset" if state.actions else "END"

    graph.add_conditional_edges("asset.create", _after_create, {"submit_changeset": "submit_changeset", "END": END})
    graph.add_edge("asset.update_meta", "submit_changeset")

    # Terminal nodes
    for node in ("agent.reply","asset.list","asset.list_types","asset.get","llm.explain","submit_changeset"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=deps.checkpointer)


# ---------- Thin runners (compat) ----------
async def run_create(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

async def run_update_meta(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

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


__all__ = ["AssetState", "create_graph", "run_create", "run_update_meta", "run_explain"]
