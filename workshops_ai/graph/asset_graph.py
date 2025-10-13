# File: workshops_ai/graph/asset_graph.py
# -----------------------------------------------------------------------------
# LangGraph: ASSETS — LLM-first, slot-driven, one-tool-per-turn
#
# What changed (relative to your previous version):
# - Removed the feature-detection guard (`hasattr(hub, "aemit_asset_event")`).
# - Use the new WsHub emitters directly:
#     • create → hub.aemit_asset_created({...})
#     • update → hub.aemit_asset_updated({...})
#   (still best-effort, scheduled fire-and-forget, but without the guard)
# - For create, we only emit if we already know an `id` for the new asset,
#   because the WS schema requires `asset_id`. Otherwise we skip the optimistic
#   emit and let the UI update after submit.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, Callable, Literal, Optional, cast, TYPE_CHECKING

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from workshops_ai.graph.factory import GraphDeps

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.graph.agent_node import agent_decide, AgentDecision
from workshops_ai.models.changeset import (
    ChangeSet,
    ActionCreateAsset,
    ActionSetAssetMeta,
    build_changeset,
)

# UI formatting helpers (title/bullets/chunk). We add safe fallbacks if unavailable.
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
Intent = Literal["list", "get", "create", "set_meta", "explain", "list_types", "agent_reply"]


class ToolEvent(TypedDict, total=False):
    event_type: str            # "tool.call.start" | "tool.call.success" | "tool.call.fail"
    tool: str
    ok: bool
    duration_ms: int
    meta: dict


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
    selection: Optional[dict] = None      # {"asset_id": "..."} | {"asset_ids": [...]} | {"assets": [...]}
    user_id: Optional[str] = None
    locale: Optional[str] = None           # e.g. "cs", "sk", "en"
    language: Optional[str] = None
    history: Optional[list[dict]] = None   # short rolling window (few turns)

    # Structured inputs
    action: Optional[Intent] = None
    create_items: Optional[list[AssetCreateItem]] = None
    update_items: Optional[list[AssetUpdateItem]] = None
    question: Optional[str] = None         # for explain

    # Working outputs
    intent: Optional[Intent] = None
    actions: Optional[list[dict]] = None
    change: Optional[ChangeSet | dict] = None
    result: Optional[dict] = None

    # Conversational slot-filling
    create_draft: Optional[dict] = None
    pending_fields: Optional[list[str]] = None

    # Telemetry
    agent_events: list[ToolEvent] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Utilities: telemetry, UI payload
# -----------------------------------------------------------------------------
def _event(state: AssetState, *, kind: str, tool: str, ok: bool = True, meta: dict | None = None) -> float:
    """Append a 'tool.call.start' style event and return start timestamp."""
    t0 = time.perf_counter()
    state.agent_events.append({"event_type": kind, "tool": tool, "ok": ok, "meta": meta or {}})
    return t0


def _finish(state: AssetState, t0: float, *, tool: str, ok: bool, meta: dict | None = None) -> None:
    """Append a success/fail tool event with duration in ms."""
    dur = int((time.perf_counter() - t0) * 1000)
    state.agent_events.append(
        {"event_type": ("tool.call.success" if ok else "tool.call.fail"), "tool": tool, "ok": ok, "duration_ms": dur, "meta": meta or {}}
    )


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


def _schedule(coro_or_fn):
    """Fire-and-forget best-effort tasks (e.g., WS emits)."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro_or_fn if asyncio.iscoroutine(coro_or_fn) else coro_or_fn())
    except RuntimeError:
        asyncio.run(coro_or_fn if asyncio.iscoroutine(coro_or_fn) else coro_or_fn())


# -----------------------------------------------------------------------------
# Localization helpers
# -----------------------------------------------------------------------------
def _lang(loc: Optional[str]) -> str:
    return (loc or "en").split("-")[0].lower()


def _hint_create(locale: Optional[str], example_type: str) -> str:
    """Short, localized nudge under the types list."""
    l = _lang(locale)
    if l in ("cs", "sk"):
        return f"Přeješ si, abych rovnou založil typ „{example_type}“?"
    return f"Want me to create a “{example_type}” for you?"


# -----------------------------------------------------------------------------
# Normalizers
# -----------------------------------------------------------------------------
def _parse_level(text: Optional[str | int]) -> Optional[int]:
    """
    Normalize a level to 0..4 if possible (int or first digit 0..4 found in text).
    """
    if text is None:
        return None
    if isinstance(text, int):
        return max(0, min(4, text))
    m = re.search(r"\b([0-4])\b", str(text))
    return int(m.group(1)) if m else None


def _extract_list_limit(text: str, locale: Optional[str]) -> Optional[int]:
    """
    Detect user-requested list truncation (Top/Shorten N).
    Supports simple patterns: "top 10", "first 5", "limit 20",
    CZ/SK variants: "zkrať na 10", "jen 7", "top 15".
    """
    if not text:
        return None
    patterns = [
        r"\btop\s+(\d+)\b",
        r"\bfirst\s+(\d+)\b",
        r"\blimit\s+(\d+)\b",
        r"\b(show|list)\s+(\d+)\b",
        r"\bzkra[ťt]\s*(na)?\s*(\d+)\b",
        r"\bjen\s+(\d+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            # pick the last numeric group
            nums = [int(x) for x in m.groups() if x and x.isdigit()]
            if nums:
                return max(1, nums[-1])
    return None


# -----------------------------------------------------------------------------
# LLM bridge (deps.llm_client only; no direct provider SDKs here)
# -----------------------------------------------------------------------------
def _llm_text(*, deps: GraphDeps, messages: list[dict[str, str]], temperature: float | None = 0.2, model: Optional[str] = None) -> str:
    """
    Minimal, robust bridge for LangChain ChatModel clients.

    Attempts:
    1) LangChain ChatModel .invoke([...BaseMessage...])
    2) Generic callable fallback: client(messages=[...], ...)

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
# Ask-next (slot) question — prompt-driven, locale-aware
# -----------------------------------------------------------------------------
def _build_next_question(
    *, deps: GraphDeps, draft: dict, asset_types: list[str], locale: str
) -> str:
    """
    Ask EXACTLY ONE short follow-up question in the user's language to complete the draft.
    The question is generated by the model from:
      - known draft fields
      - asset type catalog
    """
    lang = _lang(locale)
    system = (
        "You help a user create an asset by asking ONE concise follow-up question.\n"
        f"Language: {lang}.\n"
        "- Ask for the most important missing field next.\n"
        "- If asking about asset type, include the provided catalog as a short bulleted list.\n"
        "- If asking about confidentiality/integrity/availability, require integers 0..4.\n"
        "- If asking whether the asset is primary, request a yes/no answer.\n"
        "- Output only the question text, no preamble, no markdown fences."
    )
    order = ["name", "assetType", "isPrimary", "confidentiality", "integrity", "availability", "description"]
    missing = [k for k in order if (draft.get(k) in (None, "", []) or k not in draft)]
    target = missing[0] if missing else None

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
# LLM confirmation — prompt-driven, locale-aware, no hardcoded phrasing
# -----------------------------------------------------------------------------
def _llm_confirm(*, deps: GraphDeps, loc: str, actions: list[dict], queued: bool = False) -> str:
    """
    Ask the model to produce ONE short confirmation sentence in the user's locale,
    summarizing the applied/queued actions (create/update). No markdown, no preamble.
    """
    lang = _lang(loc)
    system = (
        "You generate ONE short confirmation sentence after a change in an asset system.\n"
        f"Language: {lang}. Keep it under 20 words. Be specific (name/type/fields if present).\n"
        "Use neutral tone. No markdown. No emojis.\n"
        "If queued is true, indicate that the change is queued and will apply later."
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
    return _lang(loc) in ("cs", "sk") and "Hotovo." or "Done."


# -----------------------------------------------------------------------------
# NLU (agent_decide → one intent per turn, no hardcoded 'help' fallback)
# -----------------------------------------------------------------------------
def _history_tail(state: AssetState, k: int = 5) -> list[dict]:
    try:
        return (state.history or [])[-k:]
    except Exception:
        return []


async def _nlu(deps: GraphDeps, state: AssetState) -> AssetState:
    """
    Map the user input to a single intent using the planner (agent_decide).
    - Honors explicit 'action' if provided.
    - Always tries LLM-first. If no actionable tool is chosen, fall back to reply-only.
    """
    raw = (state.message or "").strip()

    if state.action in ("list", "get", "create", "set_meta", "explain", "list_types", "agent_reply"):
        state.intent = cast(Intent, state.action)
        return state

    sel = state.selection or {}
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

    tool_to_intent = {
        "asset.list": "list",
        "asset.list_types": "list_types",
        "asset.create": "create",
        "asset.set_meta": "set_meta",
        "asset.update_meta": "set_meta",
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
    """
    List assets from STATE. By default, returns ALL assets (no local truncation).
    If the user explicitly requested Top/Shorten N, apply that limit.
    Long results are paginated into multiple chat messages using formatting.chunk(...).
    """
    def _inner(state: AssetState) -> AssetState:
        t0 = _event(state, kind="tool.call.start", tool="asset.list")
        try:
            items = deps.backend_port.assets_list(org_id=state.org_id) or []
            # Determine if a limit was requested (e.g., "top 10")
            n = _extract_list_limit(state.message or "", state.locale)
            if n is not None:
                items = items[:n]

            # Format: title + bullet list of "Name (Type)" lines
            lines = []
            for a in items:
                d = a.model_dump() if hasattr(a, "model_dump") else dict(a)
                nm = d.get("name") or ""
                tp = d.get("assetType") or ""
                lines.append(f"{nm} ({tp})" if tp else nm)

            heading = "Assets" if _lang(state.locale) not in ("cs", "sk") else "Aktiva"
            body = f"{title(heading)}\n\n{bullet_list(lines) if lines else '_No assets found_'}"
            parts = chunk(body, max_chars=1800)

            # Emit as multiple chat messages if needed
            messages = parts
            state.result = _ui_response(messages[0], extra={"agent_events": state.agent_events})
            if len(messages) > 1:
                # Append remaining parts as separate messages to the same payload
                state.result["messages"].extend(messages[1:])  # type: ignore[index]
            _finish(state, t0, tool="asset.list", ok=True, meta={"count": len(items)})
        except Exception as e:
            _finish(state, t0, tool="asset.list", ok=False, meta={"error": str(e)})
            state.result = _ui_response({"error": "asset.list failed", "reason": str(e)}, extra={"agent_events": state.agent_events})
        return state
    return _inner


def _asset_get_or_preview(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    """Get a single asset by selection; if none, show a small preview to pick from."""
    def _inner(state: AssetState) -> AssetState:
        asset_id = (state.selection or {}).get("asset_id")
        t0 = _event(state, kind="tool.call.start", tool="asset.get", meta={"asset_id": asset_id})
        try:
            if not asset_id:
                items = deps.backend_port.assets_list(org_id=state.org_id) or []
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
    """
    Return catalogued asset types from the BACKEND (catalogs), no local limits.
    Render with ui/formatting: title + bullets + short localized hint.
    """
    def _inner(state: AssetState) -> AssetState:
        t0 = _event(state, kind="tool.call.start", tool="asset.list_types")
        try:
            types = deps.backend_port.list_asset_types() or []
            # Title
            heading = "Asset types" if _lang(state.locale) not in ("cs", "sk") else "Typy aktiv"
            content = [title(heading), ""]
            # Bullets (full list)
            content.append(bullet_list([str(x) for x in types]) if types else "_No types found_")
            # Hint (use the first type as an inline example)
            if types:
                content.extend(["", _hint_create(state.locale, str(types[0]))])
            text = "\n".join(content)
            # Chunk just in case a very large catalog is present
            parts = chunk(text, max_chars=1800)
            state.result = _ui_response(parts[0], extra={"agent_events": state.agent_events})
            if len(parts) > 1:
                state.result["messages"].extend(parts[1:])  # type: ignore[index]
            _finish(state, t0, tool="asset.list_types", ok=True, meta={"count": len(types)})
        except Exception as e:
            _finish(state, t0, tool="asset.list_types", ok=False, meta={"error": str(e)})
            state.result = _ui_response({"error": "asset.list_types failed", "reason": str(e)}, extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Writes: build typed actions; submit later
# -----------------------------------------------------------------------------
def _validate_write_actions(actions: list[dict]) -> tuple[list[dict], Optional[str]]:
    """
    Validate and coerce action dictionaries into typed payloads expected by ChangeSet.
    Returns (validated_actions, error_or_none).
    """
    validated: list[dict] = []
    for a in actions:
        op = (a.get("op") or "").strip()
        try:
            if op == "create_asset":
                payload = {"op": "create_asset", "client_token": a.get("client_token"), "fields": a.get("fields") or {}}
                if not payload["fields"].get("name"):
                    return [], "create_asset requires fields.name"
                validated.append(_guard_validate(ActionCreateAsset, payload))  # type: ignore[arg-type]
            elif op == "set_asset_meta":
                payload = {"op": "set_asset_meta", "asset_id": a.get("asset_id"), "fields": a.get("fields") or {}}
                if not payload["asset_id"]:
                    return [], "set_asset_meta requires asset_id"
                if not payload["fields"]:
                    return [], "set_asset_meta requires at least one field"
                validated.append(_guard_validate(ActionSetAssetMeta, payload))  # type: ignore[arg-type]
            else:
                return [], f"unsupported operation '{op}'"
        except Exception as e:
            return [], f"write validation failed for op='{op}': {e}"
    return validated, None


def _extract_create_draft(*, deps: GraphDeps, text: str, asset_types: list[str]) -> dict:
    """
    Ask the LLM to convert free-form request into STRICT JSON draft (null when unsure).
    Allowed keys: name, assetType, isPrimary, confidentiality, integrity, availability, description.
    """
    system = (
        "Transform a short asset request into STRICT JSON.\n"
        "Allowed keys: name (str), assetType (str|null), isPrimary (bool|null), "
        "confidentiality (int 0-4|null), integrity (int 0-4|null), availability (int 0-4|null), "
        "description (str|null). If unsure, set null. Output JSON only."
    )
    human = {"user_text": text, "asset_types": asset_types}
    try:
        raw = _llm_text(
            deps=deps,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(human, ensure_ascii=False)}],
            temperature=getattr(deps, "llm_temperature", None) or 0.0,
        )
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return {}
        for k in ("confidentiality", "integrity", "availability"):
            obj[k] = _parse_level(obj.get(k))
        return obj
    except Exception:
        return {}


def _create_asset(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    """
    Iterative slot-filling for asset.create:
      1) Maintain a JSON draft in state.create_draft.
      2) If a required field is missing, ask ONE follow-up (prompt-driven, locale-aware).
      3) Once minimally complete (name + assetType), emit ActionCreateAsset → state.actions.
      4) Optimistic WS emit: asset.created (only if we already know asset `id`).
    """
    def _inner(state: AssetState) -> AssetState:
        # Pull catalog types from backend catalogs (no local file IO)
        types = deps.backend_port.list_asset_types() or []

        # Resume or start a draft; merge extraction from current message (prompt-driven)
        draft = dict(state.create_draft or {})
        if state.message:
            try:
                inc = _extract_create_draft(deps=deps, text=state.message, asset_types=types)
                for k, v in (inc or {}).items():
                    if v is not None:
                        draft[k] = v
            except Exception:
                pass

        # Minimal requirement
        need_name = not (isinstance(draft.get("name"), str) and draft["name"].strip())
        need_type = not (isinstance(draft.get("assetType"), str) and draft["assetType"].strip())

        if need_name or need_type:
            q = _build_next_question(deps=deps, draft=draft, asset_types=types, locale=(state.locale or "en"))
            state.create_draft = draft
            state.pending_fields = [f for f, need in (("name", need_name), ("assetType", need_type)) if need]
            state.actions = None  # prevent submit in this turn
            state.result = _ui_response(q or "Provide the next value.")
            return state

        # Optional fields can remain None; we send only provided values
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
        fields = {k: v for k, v in fields.items() if v is not None}

        actions = [{"op": "create_asset", "client_token": f"tmp-{uuid.uuid4().hex[:10]}", "fields": fields}]
        validated, err = _validate_write_actions(actions)
        if err:
            state.result = _ui_response({"error": "validation", "reason": err}, extra={"agent_events": state.agent_events})
            return state

        # Optimistic WS emit — now WITHOUT the hasattr(...) guard.
        # Only emit if we already know a stable asset id; otherwise skip.
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
                    await hub.aemit_asset_created(item)  # ← direct call (no guard)
                except Exception:
                    # best-effort: never fail the turn on WS errors
                    pass
            _schedule(_emit)

        state.actions = validated
        state.create_draft = None
        state.pending_fields = None

        # IMPORTANT: no confirmation here; we confirm AFTER submission.
        state.result = _ui_response({"kind": "changeset.prepared", "actions": len(validated)})
        return state
    return _inner


def _set_asset_meta(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    """
    Update metadata (asset.set_meta / asset.update_meta unification).
    If nothing is provided and no selection, suggest preview; otherwise emit set_asset_meta.
    Also emits an optimistic 'asset.updated' WS event for each validated item.
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
            if not (state.selection and state.selection.get("asset_id")):
                return _asset_get_or_preview(deps)(state)  # propose selection
            state.result = _ui_response("Specify fields to update.")
            return state

        actions: list[dict] = []
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

        if not actions:
            state.result = _ui_response("No changes specified.")
            return state

        validated, err = _validate_write_actions(actions)
        if err:
            state.result = _ui_response({"error": "validation", "reason": err}, extra={"agent_events": state.agent_events})
            return state

        # Optimistic WS emits — WITHOUT the hasattr(...) guard.
        hub = getattr(deps, "ws_hub", None)
        if hub:
            async def _emit():
                try:
                    for a in validated:
                        f = (a.get("fields") or {})
                        item = {
                            "org_id": state.org_id,
                            "asset_id": a.get("asset_id"),
                            "assetType": f.get("assetType"),
                            "isPrimary": f.get("isPrimary"),
                            "confidentiality": f.get("confidentiality"),
                            "integrity": f.get("integrity"),
                            "availability": f.get("availability"),
                            "description": f.get("description"),
                            "ts": _now_iso(),
                            "actor": state.user_id,
                        }
                        await hub.aemit_asset_updated(item)  # ← direct call (no guard)
                except Exception:
                    # best-effort: never fail the turn on WS errors
                    pass
            _schedule(_emit)

        state.actions = validated
        # IMPORTANT: no confirmation here; we confirm AFTER submission.
        state.result = _ui_response({"kind": "changeset.prepared", "actions": len(validated)})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Submit changeset (once per turn) — returns the single short confirmation
# -----------------------------------------------------------------------------
async def _submit_changeset_async(deps: GraphDeps, state: AssetState) -> AssetState:
    """
    Convert the prepared actions into a ChangeSet and submit via deps.submit_changeset.
    If queued, return a concise 'queued' confirmation; otherwise normal confirmation.
    """
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

        # Dump for side panel
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
            return state

        # queued/outbox path
        _finish(state, t0, tool="submit_changeset", ok=False, meta={"queued": True})
        msg_json = {
            "kind": "changeset.queued",
            "tab": "assets",
            "changeset": cs_dump,
            "ts": _now_iso(),
            "org_id": state.org_id,
            "actor": state.user_id,
        }
        confirm = _llm_confirm(deps=deps, loc=loc, actions=acts, queued=True)
        state.result = _ui_response(confirm, extra={"agent_events": state.agent_events, "messages": [msg_json]})
        return state

    except Exception as e:
        _finish(state, t0, tool="submit_changeset", ok=False, meta={"error": str(e)})
        state.result = _ui_response("Submission failed.", extra={"agent_events": state.agent_events})
        return state


# -----------------------------------------------------------------------------
# Explain (RAG → LLM synth; concise)
# -----------------------------------------------------------------------------
async def _explain_assets_async(deps: GraphDeps, state: AssetState) -> AssetState:
    """
    Optionally run retrieval to ground the answer, then synthesize a short response.
    If retrieval is unavailable, fall back to a direct concise LLM explanation.
    """
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
                return state
        except Exception as e:
            _finish(state, t0, tool="llm.explain", ok=False, meta={"error": str(e)})

    state.result = _ui_response("No evidence available and fallback did not answer.", extra={"agent_events": state.agent_events})
    return state


# -----------------------------------------------------------------------------
# Reply-only branch (no tools) — short, helpful answer in user's language
# -----------------------------------------------------------------------------
def _agent_reply(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    """
    Produce a concise assistant reply with no tool usage.
    This catches greetings, chit-chat, generic questions, etc.
    """
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
      start → nlu → route → {agent.reply | list | list_types | get | create | set_meta | explain} → (submit?) → END
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
    graph.add_node("asset.update_meta", _set_asset_meta(deps))
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
        if intent == "set_meta":
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

    # After create: if actions exist → submit; else end (we asked a question)
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
