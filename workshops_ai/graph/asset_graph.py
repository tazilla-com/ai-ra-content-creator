# File: workshops_ai/graph/asset_graph.py
# -----------------------------------------------------------------------------
# LangGraph: ASSETS — LLM-first, slot-driven, ONE action per turn
#
# Key changes implemented here (as requested):
# - Flow-lock for CREATE: when we are mid slot-filling (`create_draft`/`pending_fields`),
#   the planner whitelists ONLY `asset.create` (plus a one-off `explain`) so the model
#   cannot drift to `asset.list_types`.
# - First-step guard for CREATE: if the user asks to create but no known catalog type
#   is present in the utterance, we route to `asset.list_types` FIRST (and ask type).
# - Stricter create gating (FINAL): require `assetType`, `name`, and `isPrimary` before submit.
#   If `isPrimary=True`, also require CIA (confidentiality/integrity/availability).
#   If `isPrimary=False`, CIA must be None and we never ask for it.
# - Type-lock inside CREATE: once `assetType` is captured in the draft, later turns
#   cannot overwrite it (prevents model drift).
# - “Type-as-name” confirmation: if only NAME is missing and the user’s reply equals
#   a catalog TYPE label, ask to confirm using it as the NAME (do not change type).
# - Mid-creation explain: allow a one-off `explain` while slot-filling; after answering,
#   append a short reminder about fields still missing.
# - After submit: clear `state.actions` to prevent duplicate re-submits.
#
# CIA levels (STRING ENUMS):
#   confidentiality: PUBLIC|INTERNAL|PROTECTED|STRICTLY_PROTECTED
#   integrity: LOW|MEDIUM|HIGH
#   availability: LOW|MEDIUM|HIGH
#
# Non-localization rules:
#   - assetType labels come from a fixed EN code list → never translate them.
#   - CIA enum tokens are fixed EN values → never translate them.
#
# Chat update path is SAFE and NARROW: only `name` and `description`.
# WS emits call hub.aemit_asset_created / hub.aemit_asset_updated directly
# (best-effort; never fail the turn).
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Callable, Literal, Optional, cast, TYPE_CHECKING

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

# UI helpers (graceful fallbacks if not available)
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
        out, cur = [], []
        size = 0
        for line in text.splitlines():
            add = len(line) + 1
            if size + add > max_chars and cur:
                out.append("\n".join(cur))
                cur, size = [line], add
            else:
                cur.append(line)
                size += add
        if cur:
            out.append("\n".join(cur))
        return out

# LangGraph
try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("langgraph must be installed to use asset_graph") from e

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Optional payload guard (best-effort)
# -----------------------------------------------------------------------------
try:
    from workshops_ai.tools.guard import guard_validate_payload as _guard_validate  # type: ignore
except Exception:  # pragma: no cover
    def _guard_validate(model_cls, payload):  # type: ignore
        return payload


# -----------------------------------------------------------------------------
# Enums & normalizers (client-aligned)
# -----------------------------------------------------------------------------
CONF_LEVELS = {"PUBLIC", "INTERNAL", "PROTECTED", "STRICTLY_PROTECTED"}
TRI_LEVELS = {"LOW", "MEDIUM", "HIGH"}  # integrity & availability

def _norm_enum(value: Optional[str], *, allowed: set[str]) -> Optional[str]:
    """Normalize to UPPER_SNAKE and return None if unknown. No regex involved."""
    if value is None:
        return None
    v = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if v == "STRICTLYPROTECTED":
        v = "STRICTLY_PROTECTED"
    if v == "MEDIUM_":
        v = "MEDIUM"
    return v if v in allowed else None


# -----------------------------------------------------------------------------
# State & typed items
# -----------------------------------------------------------------------------
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

    # Conversational slot-filling — ONLY for create
    create_draft: Optional[dict] = None
    pending_fields: Optional[list[str]] = None

    # Telemetry
    agent_events: list[ToolEvent] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Telemetry & UI helpers
# -----------------------------------------------------------------------------
def _event(state: AssetState, *, kind: str, tool: str, ok: bool = True, meta: dict | None = None) -> float:
    t0 = time.perf_counter()
    state.agent_events.append({"event_type": kind, "tool": tool, "ok": ok, "meta": meta or {}})
    return t0


def _finish(state: AssetState, t0: float, *, tool: str, ok: bool, meta: dict | None = None) -> None:
    dur = int((time.perf_counter() - t0) * 1000)
    state.agent_events.append(
        {"event_type": ("tool.call.success" if ok else "tool.call.fail"), "tool": tool, "ok": ok, "duration_ms": dur, "meta": meta or {}}
    )


def _ui_response(content: str | dict, *, extra: Optional[dict] = None) -> dict:
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


def _schedule(coro_or_fn):
    """Fire-and-forget best-effort (WS emits). Never awaits in-turn."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro_or_fn if asyncio.iscoroutine(coro_or_fn) else coro_or_fn())
    except RuntimeError:
        asyncio.run(coro_or_fn if asyncio.iscoroutine(coro_or_fn) else coro_or_fn())


# -----------------------------------------------------------------------------
# Localization
# -----------------------------------------------------------------------------
def _lang(loc: Optional[str]) -> str:
    return (loc or "en").split("-")[0].lower()


def _hint_create(locale: Optional[str], example_type: str) -> str:
    # English-only catalog labels — ask a neutral follow-up, no bias to any example type.
    l = _lang(locale)
    if l in ("cs", "sk"):
        return f"Který typ aktiva chceš vytvořit?"
    return f"Which asset type do you want to create?"


def _tokenize(text: str) -> list[str]:
    # Extremely simple tokenization: split on whitespace and strip punctuation
    raw = (text or "").strip()
    toks: list[str] = []
    for part in raw.split():
        toks.append(part.strip(",.;:!?()[]{}\"'").casefold())
    return toks


def _parse_list_limit(text: str) -> Optional[int]:
    """
    Parse simple "limit/top/first/jen/zkrať" patterns WITHOUT regex:
      - "top 10", "first 5", "limit 20", "jen 7", "zkrať na 12"
    Returns None if nothing found.
    """
    toks = _tokenize(text)
    # look for a control token and its immediate numeric neighbor
    keys = {"top", "first", "limit", "jen", "zkrať", "zkrat", "zkratit", "zkraťte", "only"}
    for i, tok in enumerate(toks):
        if tok in keys:
            # prefer next token if numeric
            if i + 1 < len(toks) and toks[i + 1].isdigit():
                return max(1, int(toks[i + 1]))
            # sometimes "na 12" appears after the key
            if i + 2 < len(toks) and toks[i + 1] in {"na", "to"} and toks[i + 2].isdigit():
                return max(1, int(toks[i + 2]))
    # fallback: "show 10" / "list 5"
    for i, tok in enumerate(toks):
        if tok in {"show", "list"} and i + 1 < len(toks) and toks[i + 1].isdigit():
            return max(1, int(toks[i + 1]))
    return None


# -----------------------------------------------------------------------------
# LLM bridge (deps.llm_client only; tool-agnostic)
# -----------------------------------------------------------------------------
def _llm_text(*, deps: GraphDeps, messages: list[dict[str, str]], temperature: float | None = 0.2, model: Optional[str] = None) -> str:
    """
    Minimal, robust bridge for LangChain ChatModel clients (or callable-like).
    Returns the assistant text content or raises if unavailable.
    """
    client = getattr(deps, "llm_client", None) or getattr(deps, "llm", None)
    if client is None:
        raise RuntimeError("LLM client is not configured.")

    temp = temperature if temperature is not None else getattr(deps, "llm_temperature", None) or 0.2

    # Prefer LangChain ChatModel .invoke
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

    # Generic callable fallback
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


# -----------------------------------------------------------------------------
# Catalog type detection helpers (deterministic, no regex)
# -----------------------------------------------------------------------------
def _extract_catalog_type_from_text(text: str, asset_types: list[str]) -> Optional[str]:
    """
    Return exactly one matching catalog label (case-insensitive substring).
    If none or multiple match, return None (do not guess).
    NOTE: Labels are EN and must not be translated.
    """
    m = (text or "").strip().lower()
    hits: list[str] = []
    for t in asset_types:
        tl = (t or "").strip().lower()
        if tl and tl in m:
            hits.append(t)
    uniq = list(dict.fromkeys(hits))
    return uniq[0] if len(uniq) == 1 else None


# -----------------------------------------------------------------------------
# Ask-next question (generated) — ONLY used inside create
# -----------------------------------------------------------------------------
def _build_next_question(
    *, deps: GraphDeps, draft: dict, asset_types: list[str], locale: str
) -> str:
    """
    Ask EXACTLY ONE short follow-up question in the user's language to complete the draft.
    Ask order (deterministic):
      [assetType, name, isPrimary, confidentiality, integrity, availability, description]
      If isPrimary == False → never ask CIA; if isPrimary is unknown → ask isPrimary before CIA.
      IMPORTANT: When listing asset types, preserve EXACT English labels from catalog (no translation).
    """
    lang = _lang(locale)
    order = ["assetType", "name", "isPrimary", "confidentiality", "integrity", "availability", "description"]
    missing = [k for k in order if (draft.get(k) in (None, "", []) or k not in draft)]
    if draft.get("isPrimary") is False:
        missing = [k for k in missing if k not in ("confidentiality", "integrity", "availability")]
    # Ensure we never ask CIA before isPrimary is known
    if "isPrimary" in missing:
        missing = [x for x in missing if x not in ("confidentiality", "integrity", "availability")]
    target = missing[0] if missing else None

    system = (
        "You help a user create an asset by asking ONE concise follow-up question.\n"
        f"Language: {lang}.\n"
        "- Ask for the most important missing field next, following the given order.\n"
        "- If asking about asset type, list 3–6 catalog types (bulleted) as options.\n"
        "- Preserve catalog labels EXACTLY as provided (English). Do NOT translate catalog labels.\n"
        "- For CIA, REQUIRE these exact enums (do not translate):\n"
        "  • confidentiality: PUBLIC, INTERNAL, PROTECTED, STRICTLY_PROTECTED\n"
        "  • integrity: LOW, MEDIUM, HIGH\n"
        "  • availability: LOW, MEDIUM, HIGH\n"
        "- Ask 'isPrimary' before any CIA question. If isPrimary is false, do not ask CIA.\n"
        "- Output only the question text, no preamble, no markdown fences."
    )
    payload = {
        "known_draft": {k: draft.get(k, None) for k in order},
        "next_slot": target,
        "catalog_asset_types": asset_types,
    }
    q = _llm_text(
        deps=deps,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        temperature=getattr(deps, "llm_temperature", None) or 0.2,
    )
    return q.strip()


# -----------------------------------------------------------------------------
# LLM confirmation (one short sentence)
# -----------------------------------------------------------------------------
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
            messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(human, ensure_ascii=False)}],
            temperature=0.1,
        ).strip()
        if txt and len(txt) <= 200:
            return txt
    except Exception:
        pass
    return "Done." if lang not in ("cs", "sk") else "Hotovo."


# -----------------------------------------------------------------------------
# NLU helpers: lightweight intent checks
# -----------------------------------------------------------------------------
def _looks_like_create(msg: str) -> bool:
    """Heuristic robust across CS/SK/EN without regex."""
    m = (msg or "").strip().lower()
    tokens = ["vytvoř", "vytvor", "založ", "zaloz", "přidej", "pridej", "add", "create", "nové", "nove", "new"]
    return any(t in m for t in tokens)

def _contains_known_type(msg: str, asset_types: list[str]) -> bool:
    """Simple case-insensitive substring match for ANY exact catalog label."""
    m = (msg or "").strip().lower()
    for t in asset_types:
        if isinstance(t, str) and t.strip() and t.lower() in m:
            return True
    return False

def _looks_like_explain(msg: str) -> bool:
    m = (msg or "").strip().lower()
    keys = ["explain", "what is", "co je", "vysvětli", "vysvetli", "prosím vysvětli"]
    return any(k in m for k in keys)


# -----------------------------------------------------------------------------
# NLU (planner)
# -----------------------------------------------------------------------------
def _history_tail(state: AssetState, k: int = 5) -> list[dict]:
    try:
        return (state.history or [])[-k:]
    except Exception:
        return []

async def _nlu(deps: GraphDeps, state: AssetState) -> AssetState:
    """
    Map the user input to exactly one intent using the planner.
    CHANGED: flow-lock for ongoing CREATE and first-step guard (list_types when type is unknown).
    """
    raw = (state.message or "").strip()

    # Honor explicit action from router (unchanged)
    if state.action in ("list", "get", "create", "update_meta", "explain", "list_types", "agent_reply"):
        state.intent = cast(Intent, state.action)
        return state

    sel = state.selection or {}

    # ------------------------- Flow-lock while CREATE is in progress -------------------------
    # Allow only `asset.create` to continue slot-filling, EXCEPT a one-off `explain`.
    if (getattr(state, "pending_fields", None) or getattr(state, "create_draft", None)):
        if _looks_like_explain(raw):
            state.intent = cast(Intent, "explain")
            return state

        decision: AgentDecision = await agent_decide(
            deps=deps,
            tab="assets",
            message=raw,
            language=state.locale or "en",
            history=_history_tail(state, 5),
            selection=sel,
            allowed_actions=["asset.create"],   # flow-lock to continue slot-filling
            one_action_only=True,
        )
        state.intent = cast(Intent, "create" if (decision.action or "").strip() == "asset.create" else "agent_reply")
        if getattr(decision, "args", None):
            q = (decision.args or {}).get("question")
            if isinstance(q, str):
                state.question = q
        return state

    # ------------------------- First step for CREATE without a known type --------------------
    create_intent = _looks_like_create(raw)
    if create_intent:
        types = [str(x) for x in (deps.backend_port.list_asset_types() or [])]
        if not _contains_known_type(raw, types):
            state.intent = cast(Intent, "list_types")
            return state
        # If a known type is present, call create directly (the create node will ask next slot).
        decision: AgentDecision = await agent_decide(
            deps=deps,
            tab="assets",
            message=raw,
            language=state.locale or "en",
            history=_history_tail(state, 5),
            selection=sel,
            allowed_actions=["asset.create"],   # constrain to create
            one_action_only=True,
        )
        state.intent = cast(Intent, "create" if (decision.action or "").strip() == "asset.create" else "agent_reply")
        if getattr(decision, "args", None):
            q = (decision.args or {}).get("question")
            if isinstance(q, str):
                state.question = q
        return state

    # ---------------- TYPE-ONLY reply guard (hardening after list_types) ----------------
    # English comment:
    # If the message matches exactly one catalog type label (and we're not already flow-locked),
    # treat it as continuation of CREATE and constrain the planner to asset.create.
    try:
        types = [str(x) for x in (deps.backend_port.list_asset_types() or [])]
    except Exception:
        types = []
    matched = _extract_catalog_type_from_text(raw, types)
    if matched:
        decision: AgentDecision = await agent_decide(
            deps=deps,
            tab="assets",
            message=raw,
            language=state.locale or "en",
            history=_history_tail(state, 5),
            selection=sel,
            allowed_actions=["asset.create"],  # hard constrain
            one_action_only=True,
        )
        state.intent = cast(Intent, "create" if (decision.action or "").strip() == "asset.create" else "agent_reply")
        if getattr(decision, "args", None):
            q = (decision.args or {}).get("question")
            if isinstance(q, str):
                state.question = q
        return state

    # --------------------------------- Default planner path ---------------------------------
    allowed_default = [
        "llm.explain",
        "asset.list",
        "asset.list_types",
        "asset.create",
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

    if getattr(decision, "args", None):
        args = decision.args or {}
        if "question" in args and isinstance(args["question"], str):
            state.question = args["question"]

    return state


# -----------------------------------------------------------------------------
# Read-only nodes
# -----------------------------------------------------------------------------
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

            heading = "Assets" if _lang(state.locale) not in ("cs", "sk") else "Aktiva"
            body = f"{title(heading)}\n\n{bullet_list(lines) if lines else '_No assets found_'}"
            parts = chunk(body, max_chars=1800)

            state.result = _ui_response(parts[0], extra={"agent_events": state.agent_events})
            if len(parts) > 1:
                state.result["messages"].extend(parts[1:])  # type: ignore[index]
            _finish(state, t0, tool="asset.list", ok=True, meta={"count": len(items)})
        except Exception as e:
            _finish(state, t0, tool="asset.list", ok=False, meta={"error": str(e)})
            state.result = _ui_response({"error": "asset.list failed", "reason": str(e)}, extra={"agent_events": state.agent_events})
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
                state.result = _ui_response({"kind": "asset.select.suggest", "preview": preview}, extra={"agent_events": state.agent_events})
                _finish(state, t0, tool="asset.get", ok=True, meta={"suggested": len(preview)})
                return state

            a = deps.backend_port.asset_get(asset_id)
            d = a.model_dump() if hasattr(a, "model_dump") else dict(a)
            state.result = _ui_response({"kind": "asset.get", "asset": d}, extra={"agent_events": state.agent_events})
            _finish(state, t0, tool="asset.get", ok=True, meta={"id": asset_id})
        except Exception as e:
            _finish(state, t0, tool="asset.get", ok=False, meta={"error": str(e)})
            state.result = _ui_response({"error": "asset.get failed", "reason": str(e)}, extra={"agent_events": state.agent_events})
        return state
    return _inner


def _list_asset_types(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        t0 = _event(state, kind="tool.call.start", tool="asset.list_types")
        try:
            types = deps.backend_port.list_asset_types() or []
            heading = "Asset types" if _lang(state.locale) not in ("cs", "sk") else "Typy aktiv"
            content = [title(heading), ""]
            # IMPORTANT: we render catalog labels exactly as provided (English); do not translate.
            content.append(bullet_list([str(x) for x in types]) if types else "_No types found_")
            if types:
                content.extend(["", _hint_create(state.locale, str(types[0]))])
            text = "\n".join(content)
            parts = chunk(text, max_chars=1800)
            state.result = _ui_response(parts[0], extra={"agent_events": state.agent_events})
            if len(parts) > 1:
                state.result["messages"].extend(parts[1:])  # type: ignore[index]

            # ---------- NEW: start slot-filling & flow-lock for CREATE after listing types ----------
            # English comment:
            # Mark that we are entering the CREATE flow with the next missing slot being `assetType`.
            # This ensures the user's next reply (e.g., "Network infrastructure") routes to asset.create.
            state.create_draft = (state.create_draft or {})
            state.create_draft.pop("assetType", None)  # keep unset; the user must choose
            state.pending_fields = ["assetType"]       # key for NLU flow-lock

            _finish(state, t0, tool="asset.list_types", ok=True, meta={"count": len(types)})
        except Exception as e:
            _finish(state, t0, tool="asset.list_types", ok=False, meta={"error": str(e)})
            state.result = _ui_response({"error": "asset.list_types failed", "reason": str(e)}, extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Writes
# -----------------------------------------------------------------------------
def _validate_write_actions(actions: list[dict]) -> tuple[list[dict], Optional[str]]:
    """
    Validate and coerce actions into ChangeSet payloads.
    Returns (validated_actions, error_or_none).

    NOTE: We keep client-side guards aligned to the conversational gating:
      - create_asset requires fields.name, fields.assetType, fields.isPrimary
      - If isPrimary=True → require CIA; if isPrimary=False → CIA must be None/missing
    """
    validated: list[dict] = []
    for a in actions:
        op = (a.get("op") or "").strip()
        try:
            if op == "create_asset":
                payload = {"op": "create_asset", "client_token": a.get("client_token"), "fields": a.get("fields") or {}}
                f = payload["fields"]

                # Basic required fields
                if not (isinstance(f.get("name"), str) and f["name"].strip()):
                    return [], "create_asset requires fields.name"
                if not (isinstance(f.get("assetType"), str) and f["assetType"].strip()):
                    return [], "create_asset requires fields.assetType"
                if not isinstance(f.get("isPrimary"), bool):
                    return [], "create_asset requires fields.isPrimary (bool)"

                # CIA consistency
                c, i, a_ = f.get("confidentiality"), f.get("integrity"), f.get("availability")
                if f["isPrimary"] is True:
                    if not (c and i and a_):
                        return [], "create_asset with isPrimary=True requires CIA (confidentiality, integrity, availability)"
                else:
                    # If isPrimary=False, CIA must be None or absent
                    if any(x is not None for x in (c, i, a_)):
                        return [], "create_asset with isPrimary=False must not contain CIA"

                validated.append(_guard_validate(ActionCreateAsset, payload))  # type: ignore[arg-type]

            elif op == "set_asset_meta":
                fields = (a.get("fields") or {})
                illegal = set(fields.keys()) - {"name", "description"}
                if illegal:
                    return [], f"set_asset_meta: illegal fields for chat update: {sorted(illegal)}"
                payload = {"op": "set_asset_meta", "asset_id": a.get("asset_id"), "fields": fields}
                if not payload["asset_id"]:
                    return [], "set_asset_meta requires asset_id"
                if not payload["fields"]:
                    return [], "set_asset_meta requires at least one field (name/description)"
                validated.append(_guard_validate(ActionSetAssetMeta, payload))  # type: ignore[arg-type]
            else:
                return [], f"unsupported operation '{op}'"
        except Exception as e:
            return [], f"write validation failed for op='{op}': {e}"
    return validated, None


def _extract_create_draft_with_llm(*, deps: GraphDeps, text: str, asset_types: list[str]) -> dict:
    """
    Convert a short request into STRICT JSON draft.
    Enforce:
      - Do not guess assetType; if unknown → null
      - If text contains exactly one catalog label → use that exact label
      - Do not use any catalog label as the name unless explicitly indicated
      - If isPrimary=false → force CIA to null
      - Do not translate CIA or assetType labels (keep canonical EN tokens)
    """
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
        "- Preserve enum tokens and catalog labels exactly (English), do not translate.\n"
        "Output JSON only."
    )
    human = {"user_text": text, "asset_types": asset_types}

    try:
        raw = _llm_text(
            deps=deps,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(human, ensure_ascii=False)}],
            temperature=getattr(deps, "llm_temperature", None) or 0.0,
        ).strip()

        # Tolerate code fences and extract the first JSON object if present.
        if raw.startswith("```"):
            i, j = raw.find("{"), raw.rfind("}")
            if i != -1 and j != -1 and j > i:
                raw = raw[i:j+1]

        obj = json.loads(raw) if raw else {}
        if not isinstance(obj, dict):
            obj = {}

        # Enforce catalog type lock / validity
        if detected_type:
            obj["assetType"] = detected_type
        elif obj.get("assetType") not in asset_types:
            obj["assetType"] = None

        # Prevent "type as name"
        nm = obj.get("name")
        if isinstance(nm, str) and nm.strip().lower() in [t.lower() for t in asset_types]:
            obj["name"] = None

        # Normalize CIA enums
        obj["confidentiality"] = _norm_enum(obj.get("confidentiality"), allowed=CONF_LEVELS)
        obj["integrity"] = _norm_enum(obj.get("integrity"), allowed=TRI_LEVELS)
        obj["availability"] = _norm_enum(obj.get("availability"), allowed=TRI_LEVELS)

        # If isPrimary=false, strip CIA
        if obj.get("isPrimary") is False:
            obj["confidentiality"] = None
            obj["integrity"] = None
            obj["availability"] = None

        return obj
    except Exception:
        return {}


def _create_asset(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    """
    Iterative slot-filling for asset.create (ask-next ONLY inside this node):
      1) Maintain a JSON draft in state.create_draft (merged from LLM + heuristics).
      2) Ask exactly ONE follow-up when something is missing (localized).
      3) Enforce supporting assets (isPrimary=False) never carry CIA.
      4) Build a single ActionCreateAsset and return a prepared changeset.
    """
    def _inner(state: AssetState) -> AssetState:
        raw_types = deps.backend_port.list_asset_types() or []
        types = [str(x) for x in raw_types]  # ensure JSON-serializable & LLM-friendly

        # Merge into current draft
        draft = dict(state.create_draft or {})
        prev_type = draft.get("assetType")
        if state.message:
            inc = _extract_create_draft_with_llm(deps=deps, text=state.message, asset_types=types)

            # TYPE-LOCK: once assetType is set in this flow, do not overwrite it later
            if prev_type and isinstance(inc, dict) and inc.get("assetType") and inc.get("assetType") != prev_type:
                inc.pop("assetType", None)

            # Merge remaining fields (name/isPrimary/CIA/description)
            for k, v in (inc or {}).items():
                if v is not None and (k != "assetType" or not prev_type):
                    draft[k] = v

        # Minimal requirements before submit (FINAL):
        # - Always require: name + assetType + isPrimary
        # - If isPrimary=True, also require all CIA (confidentiality/integrity/availability)
        need_name = not (isinstance(draft.get("name"), str) and draft["name"].strip())
        need_type = not (isinstance(draft.get("assetType"), str) and draft["assetType"].strip())
        need_primary = draft.get("isPrimary") is None
        need_cia = bool(draft.get("isPrimary") is True and any(
            draft.get(k) in (None, "") for k in ("confidentiality", "integrity", "availability")
        ))

        # Edge case: ONLY name missing and the user typed a catalog TYPE label → ask to use it as NAME
        if need_name and not (need_type or need_primary or need_cia):
            raw_msg = (state.message or "").strip()
            if raw_msg and raw_msg.lower() in (t.lower() for t in types):
                confirm_q = (
                    f"Use “{raw_msg}” as the asset name?"
                    if _lang(state.locale) not in ("cs", "sk")
                    else f"Mám použít „{raw_msg}“ jako název aktiva?"
                )
                state.create_draft = draft
                state.pending_fields = ["name"]
                state.actions = None
                state.result = _ui_response(confirm_q)
                return state

        if need_name or need_type or need_primary or need_cia:
            q = _build_next_question(deps=deps, draft=draft, asset_types=types, locale=(state.locale or "en"))
            state.create_draft = draft
            pending = []
            if need_type: pending.append("assetType")
            if need_name: pending.append("name")
            if need_primary: pending.append("isPrimary")
            if need_cia:
                for c in ("confidentiality", "integrity", "availability"):
                    if draft.get(c) in (None, ""):
                        pending.append(c)
            state.pending_fields = pending
            state.actions = None  # nothing to submit this turn
            state.result = _ui_response(q or "Provide the next value.")
            return state

        # Supporting assets must not carry CIA
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

        actions = [{"op": "create_asset", "client_token": f"tmp-{uuid.uuid4().hex[:10]}", "fields": fields}]
        validated, err = _validate_write_actions(actions)
        if err:
            state.result = _ui_response({"error": "validation", "reason": err}, extra={"agent_events": state.agent_events})
            return state

        # Optimistic WS emit (best-effort)
        asset_id = fields.get("id") or fields.get("asset_id")
        hub = getattr(deps, "ws_hub", None)
        if hub and asset_id:
            async def _emit():
                try:
                    item = {
                        "org_id": state.org_id,
                        "asset_id": asset_id,
                        "name": fields.get("name"),
                        "assetType": fields.get("assetType"),
                        "isPrimary": fields.get("isPrimary"),
                        "description": fields.get("description"),
                        "ts": _now_iso(),
                        "actor": state.user_id,
                    }
                    await hub.aemit_asset_created(item)
                except Exception:
                    pass
            _schedule(_emit)

        state.actions = validated
        state.create_draft = None
        state.pending_fields = None
        state.result = _ui_response({"kind": "changeset.prepared", "actions": len(validated)})
        return state
    return _inner


def _update_meta(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    """
    SAFE chat update path: only name/description are modifiable.
    If no explicit items and no selection, propose asset preview.
    """
    def _inner(state: AssetState) -> AssetState:
        items = state.update_items or []

        # Best-effort parse from raw JSON array if present
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
                return _asset_get_or_preview(deps)(state)  # propose a selection
            state.result = _ui_response("Specify `name` and/or `description` to update.")
            return state

        actions: list[dict] = []
        for it in items:
            fields = {"name": it.name, "description": it.description}
            fields = {k: v for k, v in fields.items() if v is not None}
            if fields:
                actions.append({"op": "set_asset_meta", "asset_id": it.asset_id, "fields": fields})

        if not actions:
            state.result = _ui_response("No changes specified (only name/description are allowed here).")
            return state

        validated, err = _validate_write_actions(actions)
        if err:
            state.result = _ui_response({"error": "validation", "reason": err}, extra={"agent_events": state.agent_events})
            return state

        hub = getattr(deps, "ws_hub", None)
        if hub:
            async def _emit():
                try:
                    for a in validated:
                        f = (a.get("fields") or {})
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
            _schedule(_emit)

        state.actions = validated
        state.result = _ui_response({"kind": "changeset.prepared", "actions": len(validated)})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Submit changeset (once per turn)
# -----------------------------------------------------------------------------
async def _submit_changeset_async(deps: GraphDeps, state: AssetState) -> AssetState:
    loc = (state.locale or "en")
    acts = state.actions or []
    if not acts:
        state.result = _ui_response("No actions to apply.", extra={"agent_events": state.agent_events})
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
        state.result = _ui_response({"error": "submit_changeset not available"}, extra={"agent_events": state.agent_events})
        return state

    t0 = _event(state, kind="tool.call.start", tool="submit_changeset", meta={"actions": len(acts)})
    try:
        cs_typed = cs if hasattr(cs, "model_dump") else ChangeSet.model_validate(cs)
        submit_result = await submit_fn(cs_typed, require_success=False)  # type: ignore[misc]
        ok = bool(getattr(submit_result, "submitted", False))

        try:
            cs_dump = cs_typed.model_dump(mode="json") if hasattr(cs_typed, "model_dump") else dict(cs_typed)  # type: ignore[arg-type]
        except Exception:
            cs_dump = {"_error": "cannot_dump_changeset"}

        if ok:
            _finish(state, t0, tool="submit_changeset", ok=True, meta={"actions": len(acts)})
            confirm = _llm_confirm(deps=deps, loc=loc, actions=acts, queued=False)
            state.result = _ui_response(
                confirm,
                extra={
                    "changeset_actions": len(acts),
                    "agent_events": state.agent_events,
                    "messages": [
                        confirm,
                        {"kind": "changeset", "changeset": cs_dump, "ts": _now_iso()},
                        {"kind": "submit_result", "result": {"submitted": True}, "ts": _now_iso()},
                    ],
                },
            )
            state.actions = None  # prevent duplicate re-submit
            return state

        # queued/outbox path
        _finish(state, t0, tool="submit_changeset", ok=True, meta={"queued": True})
        msg_json = {
            "kind": "changeset.queued",
            "tab": "assets",
            "changeset": cs_dump,
            "ts": _now_iso(),
            "org_id": state.org_id,
            "actor": state.user_id,
        }
        confirm = _llm_confirm(deps=deps, loc=loc, actions=acts, queued=True)
        res = _ui_response(confirm, extra={"agent_events": state.agent_events})
        res["messages"].append(msg_json)
        state.result = res
        state.actions = None  # prevent duplicate re-submit
        return state

    except Exception as e:
        _finish(state, t0, tool="submit_changeset", ok=False, meta={"error": str(e)})
        state.result = _ui_response("Submission failed.", extra={"agent_events": state.agent_events})
        state.actions = None  # best-effort cleanup even on failure
        return state


# -----------------------------------------------------------------------------
# Explain (RAG → LLM synth; concise) + mid-creation reminder
# -----------------------------------------------------------------------------
async def _explain_assets_async(deps: GraphDeps, state: AssetState) -> AssetState:
    q = (state.question or state.message or "").strip()
    if not q:
        state.result = _ui_response("Please provide a question about assets.")
        return state

    retriever = None
    try:
        if hasattr(deps, "make_retriever") and callable(deps.make_retriever):
            from workshops_ai.rag.port import Scope
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
            "- Answer briefly (≤120 words) in the user's language if provided; otherwise English.\n"
            "- Ground every claim in provided snippets; include inline [S#] citations.\n"
            "- If unsupported, say you don't have evidence."
        )
        human = {"question": q, "snippets": tagged, "locale": (state.locale or "en")}
        t0 = _event(state, kind="tool.call.start", tool="llm.synthesize", meta={"snippets": len(tagged)})
        try:
            txt = _llm_text(deps=deps, messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(human, ensure_ascii=False)}], temperature=0.2)
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
            txt = _llm_text(deps=deps, messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(human, ensure_ascii=False)}], temperature=0.2)
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
    """
    If we're in the middle of CREATE, append a short reminder indicating which fields are still missing.
    Order mirrors the question order to avoid confusion.
    """
    draft = state.create_draft or {}
    if not (draft or state.pending_fields):
        return
    order = ["assetType", "name", "isPrimary", "confidentiality", "integrity", "availability", "description"]
    missing = [k for k in order if (draft.get(k) in (None, "", []) or k not in draft)]
    if draft.get("isPrimary") is False:
        missing = [k for k in missing if k not in ("confidentiality", "integrity", "availability")]
    if not missing:
        return
    note = (
        f"Do you want to continue? I’m waiting for: {', '.join(missing[:3])}."
        if _lang(state.locale) not in ('cs', 'sk')
        else f"Chceš pokračovat? Aktuálně čekám na: {', '.join(missing[:3])}."
    )
    try:
        state.result["messages"].append(note)  # type: ignore[index]
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Reply-only branch (no tools)
# -----------------------------------------------------------------------------
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
                messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
                temperature=0.3,
            )
        except Exception:
            ans = "How can I help with your assets?"
            if l in ("cs", "sk"):
                ans = "S čím mám v aktivech pomoci?"
        state.result = _ui_response(ans, extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Graph assembly (one tool call per turn; writes → submit)
# -----------------------------------------------------------------------------
def create_graph(deps: GraphDeps):
    """
    Flow:
      start → nlu → route → {agent.reply | list | list_types | get | create | update_meta | explain} → (submit?) → END
    One decisive step per turn; writes always go through submit once.
    """
    graph = StateGraph(AssetState)

    # Async wrappers
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
        if intent == "list":
            return "asset.list"
        if intent == "list_types":
            return "asset.list_types"
        if intent == "get":
            return "asset.get"
        if intent == "create":
            return "asset.create"
        if intent == "update_meta":
            return "asset.update_meta"
        if intent == "explain":
            return "llm.explain"
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

    # After create: if actions exist → submit; else END (we asked a question)
    def _after_create(state: AssetState) -> str:
        return "submit_changeset" if state.actions else "END"

    graph.add_conditional_edges("asset.create", _after_create, {"submit_changeset": "submit_changeset", "END": END})
    graph.add_edge("asset.update_meta", "submit_changeset")

    # Terminal nodes
    for node in ("agent.reply", "asset.list", "asset.list_types", "asset.get", "llm.explain", "submit_changeset"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=deps.checkpointer)


# -----------------------------------------------------------------------------
# Thin runners used by routers (compat)
# -----------------------------------------------------------------------------
async def run_create(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    """Delegate to the unified manager router (keeps one entry for all tabs)."""
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)


async def run_update_meta(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    """Delegate to the unified manager router."""
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)


async def run_explain(*, deps: GraphDeps, text: str, locale: Optional[str] = None) -> dict:
    """Router-facing explain; reuses the internal explain node and returns its UI payload."""
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
