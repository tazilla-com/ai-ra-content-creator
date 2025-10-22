# File: workshops_ai/graph/impact_graph.py
# -----------------------------------------------------------------------------
# LangGraph: IMPACTS tab — LLM-first assistant (clarify-first, one-call-per-turn)
#
# This version aligns behavior and UX with the Asset graph and adds important
# hardenings:
#   1) Canonical GET node: "impact.levels.for_asset" (keep "impact.asset_get"
#      as a compatible alias).
#   2) Writes submit exactly once per turn via deps.submit_changeset(...).
#      After submit, we send a short LLM-generated confirmation AND append two
#      structured side-messages:
#        • {"kind":"changeset", ...}
#        • {"kind":"submit_result", ...}  (submitted/queued semantics)
#   3) JSON fallbacks for set/bulk_set/create so callers can post raw JSON.
#   4) Lightweight write validation (levels 0–4, required fields) + coercion to
#      plain dicts to keep LangGraph checkpointer serialization safe.
#   5) Safer UI payload assembly: agent_events are shallow-copied into result to
#      avoid circular references that could cause RecursionError.
#   6) Backend call consistency: assets_list(...) signature tolerant (1-arg
#      or 4-arg) like in Asset graph.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Literal, Optional, cast, List, Dict

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.graph.factory import GraphDeps
from workshops_ai.graph.agent_node import agent_decide, AgentDecision

# Changeset model (single source of truth for write ops)
from workshops_ai.models.changeset import (
    ChangeSet,
    build_changeset,
)

# Plain-dict coercion helpers (avoid recursion / non-serializable state)
from workshops_ai.graph._serde import as_plain_dicts as _as_plain_dicts

# UI formatting helpers (preferred). Provide safe fallbacks if unavailable.
try:
    from workshops_ai.ui.formatting import title, bullet_list, chunk, table  # type: ignore
except Exception:  # pragma: no cover
    def title(text: str) -> str:
        return f"# {text}".strip()

    def bullet_list(items: List[str]) -> str:
        return "\n".join(f"- {x}" for x in items)

    def chunk(text: str, *, max_chars: int = 1800) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        parts, buf, size = [], [], 0
        for line in text.splitlines():
            add = len(line) + 1
            if size + add > max_chars and buf:
                parts.append("\n".join(buf))
                buf, size = [line], add
            else:
                buf.append(line)
                size += add
        if buf:
            parts.append("\n".join(buf))
        return parts

    def table(headers: List[str], rows: List[List[str]]) -> str:
        if not headers:
            return bullet_list([" | ".join(r) for r in rows])
        sep = "| " + " | ".join(["---"] * len(headers)) + " |"
        head = "| " + " | ".join(headers) + " |"
        body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
        return "\n".join([head, sep, body]) if rows else head + "\n" + sep

# LangGraph
try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("langgraph must be installed to use impact_graph") from e

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Types & State (LLM-first planner; clarify-first for missing slots)
# -----------------------------------------------------------------------------
Intent = Literal[
    "clarify", "list", "get", "set", "bulk_set", "create_impact", "explain", "agent_reply"
]


class ImpactItem(BaseModel):
    # Single impact assignment (level is 0..4)
    impact_code: str
    level: int = Field(ge=0, le=4)
    note: Optional[str] = None


class ImpactSetPayload(BaseModel):
    # Impact assignments for one asset
    asset_id: str
    items: List[ImpactItem]


class ImpactBulkPayload(BaseModel):
    # Many assets -> many assignments
    entries: List[ImpactSetPayload]


class ImpactCreateItem(BaseModel):
    # Catalog definition upsert
    code: str
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    weight: Optional[float] = None


class ToolEvent(TypedDict, total=False):
    event_type: str            # "tool.call.start" | "tool.call.success" | "tool.call.fail"
    tool: str
    ok: bool
    duration_ms: int
    meta: dict


class ImpactState(BaseModel):
    # Request context
    org_id: str
    active_tab: str = Field(default="impacts")
    message: Optional[str] = None
    selection: Optional[dict] = None   # {"asset_id": "..."} | ...
    user_id: Optional[str] = None
    locale: Optional[str] = None       # "cs" | "sk" | "en" | ...
    language: Optional[str] = None
    history: Optional[List[dict]] = None  # short rolling window

    # Structured inputs (from planner or API)
    action: Optional[Intent] = None
    set_payload: Optional[ImpactSetPayload] = None
    bulk_payload: Optional[ImpactBulkPayload] = None
    create_items: Optional[List[ImpactCreateItem]] = None
    question: Optional[str] = None

    # Working outputs
    intent: Optional[Intent] = None
    actions: Optional[List[dict]] = None
    change: Optional[ChangeSet | dict] = None
    result: Optional[dict] = None

    # Telemetry
    agent_events: List[ToolEvent] = Field(default_factory=list)

    # Ensure actions are always plain dicts (safe for checkpointer)
    @field_validator("actions", mode="before")
    @classmethod
    def _coerce_actions_to_plain_dicts(cls, v):
        if v is None:
            return None
        try:
            return _as_plain_dicts(list(v))
        except Exception:
            return v


# -----------------------------------------------------------------------------
# Utilities: localization, telemetry, UI payload
# -----------------------------------------------------------------------------
def _lang(locale: Optional[str], fallback: str = "en") -> str:
    """Normalize locale to 'cs' | 'sk' | 'en' (default to 'en')."""
    if not locale:
        return fallback
    l = locale.lower()
    if l.startswith(("cs", "cz")):
        return "cs"
    if l.startswith("sk"):
        return "sk"
    if l.startswith("en"):
        return "en"
    return fallback


def _ui_response(content: str | dict, *, extra: Optional[dict] = None) -> dict:
    """
    Standard UIResponse payload (messages + optional structured data).
    IMPORTANT: never alias agent_events directly to avoid circular refs.
    """
    base = {
        "messages": [content] if isinstance(content, str) else [content],
        "proposals": None,
        "staging": {"needs_commit": False, "batch_id": None, "diff": None},
        "policy": {"emit_changes_enabled": True},
    }
    if extra:
        # Shallow-copy agent_events if present
        ev = extra.get("agent_events")
        if isinstance(ev, list):
            base["agent_events"] = [dict(e) for e in ev]
        # Merge all other extras
        for k, v in extra.items():
            if k == "agent_events":
                continue
            base[k] = v
    return base


def _now_iso() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _event(state: ImpactState, *, kind: str, tool: str, ok: bool = True, meta: dict | None = None) -> float:
    """Start a tool event and return high-precision start time."""
    t0 = time.perf_counter()
    state.agent_events.append({"event_type": kind, "tool": tool, "ok": ok, "meta": meta or {}})
    return t0


def _finish(state: ImpactState, t0: float, *, tool: str, ok: bool, meta: dict | None = None) -> None:
    """Finish a tool event, recording duration and status."""
    dur = int((time.perf_counter() - t0) * 1000)
    state.agent_events.append(
        {"event_type": ("tool.call.success" if ok else "tool.call.fail"), "tool": tool, "ok": ok, "duration_ms": dur, "meta": meta or {}}
    )


# -----------------------------------------------------------------------------
# Backend access helper (align with routers; prefer deps.backend)
# -----------------------------------------------------------------------------
def _backend(deps: GraphDeps):
    return getattr(deps, "backend", None) or getattr(deps, "backend_port", None)


# -----------------------------------------------------------------------------
# LLM bridges
# -----------------------------------------------------------------------------
def _llm_text(
    *,
    deps: GraphDeps,
    messages: List[dict],
    temperature: float | None = 0.2,
    model: Optional[str] = None,
) -> str:
    """
    Minimal robust bridge for LangChain ChatModel clients (e.g., langchain_openai.ChatOpenAI).
    """
    client = getattr(deps, "llm_client", None) or getattr(deps, "llm", None)
    if client is None:
        raise RuntimeError("LLM client is not configured (deps.llm_client is None).")

    temp = temperature if temperature is not None else getattr(deps, "llm_temperature", None) or 0.2

    # LangChain ChatModel via .invoke([...BaseMessage...]) — preferred
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


async def _llm_text_async(*, deps: GraphDeps, messages: List[dict], temperature: float = 0.2) -> str:
    """Async wrapper around _llm_text for environments that expect awaits."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _llm_text(deps=deps, messages=messages, temperature=temperature))


# -----------------------------------------------------------------------------
# Planner (agent_decide → one intent per turn; no hardcoded help/greet)
# -----------------------------------------------------------------------------
def _history_tail(state: ImpactState, k: int = 5) -> List[dict]:
    try:
        return (state.history or [])[-k:]
    except Exception:
        return []


async def _nlu(deps: GraphDeps, state: ImpactState) -> ImpactState:
    """
    Choose a single intent using planner (agent_decide).
    - Honors explicit 'action' if provided.
    - LLM-first: if planner picks no actionable tool, fall back to agent_reply.
    - Slot payloads are parsed into Pydantic models where applicable.
    """
    raw = (state.message or "").strip()

    # Explicit passthrough
    if state.action in ("clarify", "list", "get", "set", "bulk_set", "create_impact", "explain", "agent_reply"):
        state.intent = cast(Intent, state.action)
        return state

    sel = state.selection or {}

    allowed = [
        "impact.list",
        "impact.asset_get",
        "impact.levels.for_asset",   # canonical GET
        "impact.set_level",
        "impact.bulk_set",
        "impact.create_impact",
        "impact.explain",
    ]

    decision: AgentDecision = await agent_decide(
        deps=deps,
        tab="impacts",
        message=raw,
        language=state.locale or "en",
        history=_history_tail(state, 5),
        selection=sel,
        allowed_actions=allowed,
        one_action_only=True,
    )

    # Map tool → internal intent
    tool_to_intent = {
        "impact.list": "list",
        "impact.asset_get": "get",
        "impact.levels.for_asset": "get",
        "impact.set_level": "set",
        "impact.bulk_set": "bulk_set",
        "impact.create_impact": "create_impact",
        "impact.explain": "explain",
    }
    state.intent = cast(Intent, tool_to_intent.get((decision.action or "").strip(), "agent_reply"))

    # Strict args harvesting into Pydantic payloads
    if getattr(decision, "args", None):
        args = decision.args or {}

        if state.intent == "set":
            asset_id = args.get("asset_id") or sel.get("asset_id")
            items = args.get("items")
            if asset_id and isinstance(items, list):
                try:
                    parsed = [ImpactItem(**it) for it in items]
                    state.set_payload = ImpactSetPayload(asset_id=asset_id, items=parsed)
                except Exception:
                    state.set_payload = None

        if state.intent == "bulk_set" and isinstance(args.get("entries"), list):
            try:
                entries: List[ImpactSetPayload] = []
                for e in args["entries"]:
                    if not isinstance(e, dict):
                        continue
                    aid = e.get("asset_id") or sel.get("asset_id")
                    arr = [ImpactItem(**it) for it in (e.get("items") or [])]
                    if aid and arr:
                        entries.append(ImpactSetPayload(asset_id=aid, items=arr))
                if entries:
                    state.bulk_payload = ImpactBulkPayload(entries=entries)
            except Exception:
                state.bulk_payload = None

        if state.intent == "create_impact":
            items_src = []
            if isinstance(args.get("item"), dict):
                items_src.append(args["item"])
            if isinstance(args.get("items"), list):
                items_src.extend([x for x in args["items"] if isinstance(x, dict)])
            if items_src:
                try:
                    state.create_items = [ImpactCreateItem(**x) for x in items_src]
                except Exception:
                    state.create_items = None

        if state.intent == "explain":
            q = args.get("q") or args.get("impactCode") or args.get("impactName")
            if isinstance(q, str) and q.strip():
                state.question = q.strip()

    return state


# -----------------------------------------------------------------------------
# Write validation + plain-dict coercion (prevents serialization issues)
# -----------------------------------------------------------------------------
def _to_plain_dict(obj: Any) -> dict:
    """
    Ensure a JSON-serializable plain dict (avoid Pydantic instances in state.actions).
    """
    try:
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
    except Exception:
        pass
    if isinstance(obj, dict):
        return obj
    # Last resort: JSON round-trip with __dict__ fallback
    return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))


def _validate_actions(actions: List[dict]) -> tuple[List[dict], Optional[str]]:
    """
    Lightweight guardrails for write ops:
      - set_impacts: asset_id present; items non-empty; each item has impact_code and level in 0..4
      - upsert_impacts: each item has code+name
    """
    out: List[dict] = []
    for a in actions:
        op = (a.get("op") or "").strip()

        if op == "set_impacts":
            aid = a.get("asset_id")
            items = a.get("items") or []
            if not aid or not items:
                return [], "set_impacts requires asset_id and non-empty items"
            norm_items = []
            for it in items:
                code = str((it.get("impact_code") or it.get("code") or "")).strip()
                lvl = it.get("level")
                if not code:
                    return [], "impact item missing impact_code"
                try:
                    lvl = int(lvl)
                except Exception:
                    return [], f"level for {code} must be integer 0..4"
                if lvl < 0 or lvl > 4:
                    return [], f"level for {code} must be in 0..4"
                norm_items.append({"impact_code": code, "level": lvl, "note": it.get("note")})
            out.append({"op": "set_impacts", "mode": "merge", "asset_id": aid, "items": norm_items})

        elif op == "upsert_impacts":
            items = a.get("items") or []
            if not items:
                return [], "upsert_impacts requires non-empty items"
            norm = []
            for it in items:
                code = str((it.get("code") or "")).strip()
                name = str((it.get("name") or "")).strip()
                if not code or not name:
                    return [], "each upsert item requires code and name"
                row = {
                    "code": code,
                    "name": name,
                    "category": it.get("category"),
                    "description": it.get("description"),
                    "weight": it.get("weight"),
                }
                norm.append({k: v for k, v in row.items() if v is not None})
            out.append({"op": "upsert_impacts", "items": norm})

        else:
            return [], f"unsupported operation '{op}'"

    return [_to_plain_dict(x) for x in out], None


# -----------------------------------------------------------------------------
# Clarify (ask ONE concise question when a required slot is missing)
# -----------------------------------------------------------------------------
def _clarify_once(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    def _inner(state: ImpactState) -> ImpactState:
        l = _lang(state.locale)

        def _ask(txt_cs: str, txt_sk: str, txt_en: str) -> str:
            return {"cs": txt_cs, "sk": txt_sk}.get(l, txt_en)

        if state.intent == "set":
            if not (state.set_payload and state.set_payload.asset_id):
                # If no selection, suggest a small pick-list; otherwise ask for asset_id.
                if not (state.selection and state.selection.get("asset_id")):
                    return _suggest_select(deps)(state)
                state.result = _ui_response(_ask("Uveď prosím asset_id.", "Uveď prosím asset_id.", "Please specify asset_id."))
                return state
            if not state.set_payload.items:
                state.result = _ui_response(_ask("Uveď items[{impact_code, level}].", "Uveď items[{impact_code, level}].", "Please provide items[{impact_code, level}]."))
                return state

        if state.intent == "bulk_set":
            if not (state.bulk_payload and state.bulk_payload.entries):
                state.result = _ui_response(_ask(
                    "Pošli prosím entries[{asset_id, items[{impact_code, level}]}].",
                    "Pošli prosím entries[{asset_id, items[{impact_code, level}]}].",
                    "Please send entries[{asset_id, items[{impact_code, level}]}].",
                ))
                return state

        if state.intent == "create_impact":
            if not (state.create_items and len(state.create_items) > 0):
                state.result = _ui_response(_ask(
                    "Zadej alespoň jeden dopad {code,name[,category,description,weight]}.",
                    "Zadaj aspoň jeden dopad {code,name[,category,description,weight]}.",
                    "Provide at least one impact {code,name[,category,description,weight]}.",
                ))
                return state

        # If nothing to clarify, just do an agent-style reply (no tools).
        return _agent_reply(deps)(state)
    return _inner


def _suggest_select(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    """If no asset is selected, render a compact preview to help the user pick one."""
    def _inner(state: ImpactState) -> ImpactState:
        t0 = _event(state, kind="tool.call.start", tool="suggest_select")
        try:
            backend = _backend(deps)
            # Support both signatures: assets_list(org_id, ...) and assets_list(org_id=...)
            items = []
            if backend:
                try:
                    items = backend.assets_list(state.org_id, None, None, None) or []
                except TypeError:
                    items = backend.assets_list(org_id=state.org_id) or []
            rows: List[List[str]] = []
            for a in items[:10]:  # preview only; not a catalog list
                d = a.model_dump() if hasattr(a, "model_dump") else dict(a)
                rows.append([d.get("id") or "?", d.get("name") or "-", d.get("assetType") or "-"])
            heading = {"cs": "Vyber prosím aktivum", "sk": "Vyber prosím aktívum", "en": "Please select an asset"}[_lang(state.locale)]
            text = "\n\n".join([title(heading), table(["ID", "Name", "Type"], rows)]) if rows else title(heading)
            state.result = _ui_response(text, extra={"agent_events": state.agent_events})
            _finish(state, t0, tool="suggest_select", ok=True, meta={"count": len(rows)})
        except Exception as e:
            _finish(state, t0, tool="suggest_select", ok=False, meta={"error": str(e)})
            msg = {"cs": f"Nepodařilo se načíst seznam aktiv: {e}", "sk": f"Nepodarilo sa načítať zoznam aktív: {e}", "en": f"Failed to load assets: {e}"}[_lang(state.locale)]
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Read-only nodes (list/get)
# -----------------------------------------------------------------------------
def _list_impacts(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    """
    List the entire impact CATALOG (no hidden truncation).
    Uses ui/formatting (title + bullets). Chunks long outputs.
    """
    def _inner(state: ImpactState) -> ImpactState:
        t0 = _event(state, kind="tool.call.start", tool="impact.list")
        try:
            backend = _backend(deps)
            defs = (backend.impact_defs_list(org_id=state.org_id) if backend else []) or []
            # Build bullet lines "CODE — Name [Category?]"
            lines: List[str] = []
            for row in defs:
                d = row.model_dump() if hasattr(row, "model_dump") else dict(row)
                code = str(d.get("code") or d.get("impact_code") or "").strip()
                name = str(d.get("name") or "").strip()
                cat = str(d.get("category") or "").strip()
                line = f"{code} — {name}" if name else code
                if cat:
                    line += f" [{cat}]"
                lines.append(line)

            heading = {"cs": "Katalog dopadů", "sk": "Katalóg dopadov", "en": "Impact catalog"}[_lang(state.locale)]
            text = "\n\n".join([title(f"{heading} ({len(lines)})"), bullet_list(lines) if lines else "_empty_"])
            parts = chunk(text, max_chars=1800)

            state.result = _ui_response(parts[0], extra={"agent_events": state.agent_events, "impacts_count": len(lines)})
            if len(parts) > 1:
                state.result["messages"].extend(parts[1:])  # type: ignore[index]
            _finish(state, t0, tool="impact.list", ok=True, meta={"count": len(lines)})
        except Exception as e:
            _finish(state, t0, tool="impact.list", ok=False, meta={"error": str(e)})
            msg = {"cs": f"Katalog dopadů se nepodařilo načíst: {e}", "sk": f"Katalóg dopadov sa nepodarilo načítať: {e}", "en": f"Failed to load impact catalog: {e}"}[_lang(state.locale)]
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
        return state
    return _inner


def _asset_get_impacts_or_suggest(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    """
    Show impacts for the selected asset (state/asset_impacts.csv via backend).
    If none is selected, suggest an asset to pick.
    """
    def _inner(state: ImpactState) -> ImpactState:
        asset_id = (state.selection or {}).get("asset_id")
        if not asset_id:
            return _suggest_select(deps)(state)

        t0 = _event(state, kind="tool.call.start", tool="impact.levels.for_asset", meta={"asset_id": asset_id})
        try:
            backend = _backend(deps)
            levels = backend.asset_get_impacts(asset_id) if backend else []  # [{impact_code, level, note?}, ...]
            # Render as bullets: "CODE: level — note?"
            lines = []
            for it in levels:
                code = str(it.get("impact_code") or it.get("code") or "").strip()
                lvl = it.get("level")
                note = str(it.get("note") or "").strip()
                s = f"{code}: {lvl if lvl is not None else '-'}"
                if note:
                    s += f" — {note}"
                lines.append(s)

            heading = {"cs": "Dopady pro aktivum", "sk": "Dopady pre aktívum", "en": "Impacts for asset"}[_lang(state.locale)]
            text = "\n\n".join([title(f"{heading} {asset_id} ({len(lines)})"), bullet_list(lines) if lines else "_none_"])
            parts = chunk(text, max_chars=1800)

            state.result = _ui_response(parts[0], extra={"agent_events": state.agent_events, "asset_id": asset_id, "impacts_count": len(lines)})
            if len(parts) > 1:
                state.result["messages"].extend(parts[1:])  # type: ignore[index]
            _finish(state, t0, tool="impact.levels.for_asset", ok=True, meta={"count": len(lines)})
        except Exception as e:
            _finish(state, t0, tool="impact.levels.for_asset", ok=False, meta={"error": str(e)})
            msg = {"cs": f"Čtení dopadů pro „{asset_id}“ selhalo: {e}", "sk": f"Čítanie dopadov pre „{asset_id}“ zlyhalo: {e}", "en": f"Failed to read impacts for “{asset_id}”: {e}"}[_lang(state.locale)]
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Explain (RAG→LLM; output as title + short bullet points)
# -----------------------------------------------------------------------------
EXPLAIN_SYSTEM = (
    "You are an Impact-domain assistant. Explain impact definitions clearly and concisely. "
    "Prefer the organization's catalog/context when provided."
)

def _get_impacts_retriever(deps: GraphDeps, org_id: Optional[str] = None):
    mk = getattr(deps, "make_retriever", None)
    if callable(mk) and org_id:
        try:
            return mk(org_id, "impacts")
        except Exception:  # pragma: no cover
            logger.debug("make_retriever(org_id, 'impacts') failed", exc_info=True)
    return getattr(deps, "impacts_retriever", None)


async def _explain_node_async(deps: GraphDeps, state: ImpactState) -> ImpactState:
    """Explain impacts via RAG→LLM fallback; wrap as title + short bullet list."""
    query = (state.question or state.message or "").strip() or "impacts overview"
    retriever = _get_impacts_retriever(deps, state.org_id)

    snippets: List[str] = []
    rag_used = False
    if retriever:
        try:
            docs = []
            if hasattr(retriever, "aget_relevant_documents"):
                docs = await retriever.aget_relevant_documents(query)
            elif hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(query)  # type: ignore
            elif callable(retriever):
                maybe = retriever(query)
                docs = await maybe if hasattr(maybe, "__await__") else maybe
            for d in docs or []:
                txt = getattr(d, "page_content", None) or getattr(d, "text", "") or str(d)
                if txt:
                    snippets.append(str(txt))
            rag_used = len(snippets) > 0
        except Exception:
            logger.debug("impacts retriever failed; falling back to LLM-only", exc_info=True)

    sys = EXPLAIN_SYSTEM
    msgs: List[dict] = [{"role": "system", "content": sys}]
    if rag_used:
        ctx = "\n\n---\n".join(snippets[:5])
        msgs.append({"role": "system", "content": f"Context snippets:\n{ctx}"})
    # Ask the model to produce ≤6 crisp bullets, no fluff.
    msgs.append({
        "role": "user",
        "content": json.dumps({
            "task": "Summarize in <=6 short bullets for end users; avoid repetition; be concrete.",
            "query": query,
            "locale": _lang(state.locale),
        }, ensure_ascii=False)
    })
    answer = await _llm_text_async(deps=deps, messages=msgs, temperature=0.2)
    bullets = [ln.strip("-• ").strip() for ln in answer.splitlines() if ln.strip()]
    bullets = bullets if bullets else [answer.strip()]

    heading = {"cs": "Vysvětlení dopadů", "sk": "Vysvetlenie dopadov", "en": "Impacts — explanation"}[_lang(state.locale)]
    text = "\n\n".join([title(f"{heading}: {query}"), bullet_list(bullets)])
    parts = chunk(text, max_chars=1800)

    state.result = _ui_response(parts[0], extra={
        "agent_events": state.agent_events,
        "rag_used": rag_used,
        "sources_count": min(5, len(snippets)) if rag_used else 0,
    })
    if len(parts) > 1:
        state.result["messages"].extend(parts[1:])  # type: ignore[index]
    return state


# -----------------------------------------------------------------------------
# Write nodes (strictly use provided codes/levels; no guessing)
# -----------------------------------------------------------------------------
def _set_impact_level(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    """Set one or more impact levels for a single asset (merge mode)."""
    def _inner(state: ImpactState) -> ImpactState:
        l = _lang(state.locale)

        # JSON fallback: allow {"asset_id":"...", "items":[{impact_code, level, note?}, ...]}
        if not state.set_payload:
            try:
                obj = json.loads(state.message or "{}")
                if isinstance(obj, dict) and obj.get("asset_id") and isinstance(obj.get("items"), list):
                    state.set_payload = ImpactSetPayload(
                        asset_id=obj["asset_id"],
                        items=[ImpactItem(**it) for it in obj["items"]],
                    )
            except Exception:
                pass

        payload = state.set_payload
        if not payload or not payload.items:
            # If no selection, propose asset list; otherwise ask for items only
            if not (state.selection and state.selection.get("asset_id")):
                return _suggest_select(deps)(state)
            msg = {"cs": "Není co nastavit. Potřebuji items[{impact_code, level}].",
                   "sk": "Nie je čo nastaviť. Potrebujem items[{impact_code, level}].",
                   "en": "Nothing to set. Need items[{impact_code, level}]."}[l]
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
            return state

        # Build action and validate
        action = {
            "op": "set_impacts",
            "mode": "merge",
            "asset_id": payload.asset_id,
            "items": [it.model_dump() for it in payload.items],
        }
        validated, err = _validate_actions([action])
        if err:
            state.result = _ui_response(
                {"cs": f"Validace selhala: {err}", "sk": f"Validácia zlyhala: {err}", "en": f"Validation failed: {err}"}[l],
                extra={"agent_events": state.agent_events},
            )
            return state

        state.actions = validated  # already plain dicts via _validate_actions → _to_plain_dict
        # Pre-submit echo; final confirmation is after submit
        last = payload.items[-1]
        echo = {"cs": f"✅ Připraveno nastavit {last.impact_code}={int(last.level)} pro aktivum „{payload.asset_id}“.",
                "sk": f"✅ Pripravené nastaviť {last.impact_code}={int(last.level)} pre aktívum „{payload.asset_id}“.",
                "en": f"✅ Ready to set {last.impact_code}={int(last.level)} for asset “{payload.asset_id}”."}[l]
        state.result = _ui_response(echo, extra={"agent_events": state.agent_events})
        return state
    return _inner


def _bulk_set_impact_levels(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    """Bulk set impact levels across multiple assets (merge mode)."""
    def _inner(state: ImpactState) -> ImpactState:
        l = _lang(state.locale)

        # JSON fallback: allow {"entries":[{"asset_id":"...","items":[{impact_code,level}]} , ...]}
        if not state.bulk_payload:
            try:
                obj = json.loads(state.message or "{}")
                if isinstance(obj, dict) and isinstance(obj.get("entries"), list):
                    entries = []
                    for e in obj["entries"]:
                        if isinstance(e, dict) and e.get("asset_id") and isinstance(e.get("items"), list):
                            entries.append(
                                ImpactSetPayload(
                                    asset_id=e["asset_id"],
                                    items=[ImpactItem(**it) for it in e["items"]],
                                )
                            )
                    if entries:
                        state.bulk_payload = ImpactBulkPayload(entries=entries)
            except Exception:
                pass

        bulk = state.bulk_payload
        if not bulk or not bulk.entries:
            msg = {"cs": "Chybí entries pro hromadné nastavení.",
                   "sk": "Chýbajú entries pre hromadné nastavenie.",
                   "en": "Missing entries for bulk update."}[l]
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
            return state

        actions: List[dict] = []
        for e in bulk.entries:
            if not e.asset_id or not e.items:
                continue
            actions.append({
                "op": "set_impacts",
                "mode": "merge",
                "asset_id": e.asset_id,
                "items": [it.model_dump() for it in e.items],
            })

        if not actions:
            msg = {"cs": "Chybí entries pro hromadné nastavení.",
                   "sk": "Chýbajú entries pre hromadné nastavenie.",
                   "en": "Missing entries for bulk update."}[l]
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
            return state

        validated, err = _validate_actions(actions)
        if err:
            state.result = _ui_response(
                {"cs": f"Validace selhala: {err}", "sk": f"Validácia zlyhala: {err}", "en": f"Validation failed: {err}"}[l],
                extra={"agent_events": state.agent_events},
            )
            return state

        state.actions = validated
        ready = {"cs": "✅ Připraveno k aplikaci (hromadné nastavení).",
                 "sk": "✅ Pripravené na aplikáciu (hromadné nastavenie).",
                 "en": "✅ Ready to apply (bulk set)."}[l]
        state.result = _ui_response(ready, extra={"agent_events": state.agent_events})
        return state
    return _inner


def _create_impact(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    """Create or update impact catalog entries (upsert semantics)."""
    def _inner(state: ImpactState) -> ImpactState:
        l = _lang(state.locale)
        items = state.create_items or []
        if not items:
            # Optional JSON fallback {"items":[{code,name,...}]}
            try:
                raw = json.loads(state.message or "{}")
                src = raw.get("items") or []
                items = [ImpactCreateItem(**x) for x in src]
            except Exception:
                pass

        if not items:
            msg = {"cs": "Zadej alespoň jeden dopad {code,name[,category,description,weight]}.",
                   "sk": "Zadaj aspoň jeden dopad {code,name[,category,description,weight]}.",
                   "en": "Provide at least one impact {code,name[,category,description,weight]}."}[l]
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
            return state

        action = {"op": "upsert_impacts", "items": [it.model_dump() for it in items]}
        validated, err = _validate_actions([action])
        if err:
            state.result = _ui_response(
                {"cs": f"Validace selhala: {err}", "sk": f"Validácia zlyhala: {err}", "en": f"Validation failed: {err}"}[l],
                extra={"agent_events": state.agent_events},
            )
            return state

        state.actions = validated

        echo = {"cs": f"✅ Připraveno přidat/aktualizovat {len(items)} dopadů.",
                "sk": f"✅ Pripravené pridať/aktualizovať {len(items)} dopadov.",
                "en": f"✅ Ready to add/update {len(items)} impact(s)."}[l]
        state.result = _ui_response(echo, extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# LLM-confirmation (short, localized) + unified submit result messages
# -----------------------------------------------------------------------------
async def _llm_confirm_async(
    deps: GraphDeps,
    *,
    locale: Optional[str],
    actions: List[dict],
    submitted: bool,
    fallback_path: Optional[str],
) -> str:
    """
    Ask the LLM to produce a single short confirmation sentence (<= 20 words),
    localized to locale ("cs"/"sk"/"en"). If LLM bridge fails, return a safe fallback.
    """
    l = _lang(locale)
    n_set = sum(1 for a in actions if a.get("op") == "set_impacts")
    n_upsert = sum(1 for a in actions if a.get("op") == "upsert_impacts")
    payload = {
        "locale": l,
        "submitted": bool(submitted),
        "queued": not bool(submitted),
        "counts": {"set": n_set, "upsert": n_upsert},
        "hint": "If queued, mention it briefly; otherwise confirm saved. One sentence; <=20 words.",
    }

    sys = (
        "You generate a single, very short confirmation sentence for end users. "
        "No extra punctuation, no emojis, no technical jargon."
    )
    user = json.dumps(payload, ensure_ascii=False)

    try:
        txt = await _llm_text_async(
            deps=deps,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.1,
        )
        if txt:
            return txt.strip()
    except Exception:
        pass

    # Localized fallback (deterministic)
    if submitted:
        return {"cs": f"Uloženo ({n_set} změn úrovní, {n_upsert} upsertů).",
                "sk": f"Uložené ({n_set} zmien úrovní, {n_upsert} upsertov).",
                "en": f"Saved ({n_set} level updates, {n_upsert} upserts)."}[l]
    else:
        return {"cs": f"Zařazeno do fronty ({n_set} změn, {n_upsert} upsertů).",
                "sk": f"Zaradené do fronty ({n_set} zmien, {n_upsert} upsertov).",
                "en": f"Queued ({n_set} updates, {n_upsert} upserts)."}[l]


# -----------------------------------------------------------------------------
# Submit (single submit per turn) — routers handle on-line vs outbox fallback
# -----------------------------------------------------------------------------
async def _submit_changeset_async(deps: GraphDeps, state: ImpactState) -> ImpactState:
    l = _lang(state.locale)
    acts = state.actions or []
    if not acts:
        state.result = _ui_response({"cs": "Žádné změny k uložení.", "sk": "Žiadne zmeny na uloženie.", "en": "No actions to apply."}[l], extra={"agent_events": state.agent_events})
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
        msg = {"cs": "submit_changeset není k dispozici.", "sk": "submit_changeset nie je k dispozícii.", "en": "submit_changeset not available."}[l]
        state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
        return state

    t0 = _event(state, kind="tool.call.start", tool="submit_changeset", meta={"actions": len(acts)})
    try:
        cs_typed = cs if hasattr(cs, "model_dump") else ChangeSet.model_validate(cs)
        submit_result = await submit_fn(cs_typed, require_success=False)  # factory handles online→OUTBOX fallback
        ok = bool(getattr(submit_result, "submitted", False))
        fallback_path = getattr(submit_result, "fallback_path", None)
        _finish(state, t0, tool="submit_changeset", ok=ok, meta={"actions": len(acts), "fallback_path": str(fallback_path) if fallback_path else None})

        # LLM-powered short confirmation
        confirm = await _llm_confirm_async(
            deps,
            locale=state.locale,
            actions=acts,
            submitted=ok,
            fallback_path=str(fallback_path) if fallback_path else None,
        )

        try:
            cs_dump = cs_typed.model_dump(mode="json") if hasattr(cs_typed, "model_dump") else dict(cs_typed)  # type: ignore[arg-type]
        except Exception:
            cs_dump = {"_error": "cannot_dump_changeset"}

        # Build UI response and append structured side-messages (do not override messages)
        res = _ui_response(confirm, extra={"changeset_actions": len(acts), "agent_events": state.agent_events})
        res["messages"].append({"kind": "changeset", "changeset": cs_dump, "ts": _now_iso()})
        res["messages"].append({"kind": "submit_result", "submitted": ok, "queued": (not ok), "fallback_path": str(fallback_path) if fallback_path else None})
        state.result = res

        # Clear write state to avoid re-entry loops
        state.actions = None
        state.change = None
        return state

    except Exception as e:
        _finish(state, t0, tool="submit_changeset", ok=False, meta={"error": str(e)})
        msg = {"cs": f"Odeslání selhalo: {e}", "sk": f"Odoslanie zlyhalo: {e}", "en": f"Submission failed: {e}"}[l]
        state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
        return state


# -----------------------------------------------------------------------------
# Agent-only reply (no tools) — concise, friendly, domain-aware
# -----------------------------------------------------------------------------
def _agent_reply(deps: GraphDeps) -> Callable[[ImpactState], ImpactState]:
    """
    Produce a concise assistant reply with no tool usage.
    Catches greetings/chitchat/generic questions and keeps the conversation natural.
    """
    def _inner(state: ImpactState) -> ImpactState:
        l = _lang(state.locale)
        system = (
            "You are a helpful assistant for an 'Impacts' tab (risk management). "
            "Be concise (≤80 words), neutral, and specific. "
            "Mention you can list the impact catalog, show impacts for an asset, "
            "set levels (single or bulk), or explain impact concepts—only if relevant. "
            "Do not claim actions were taken."
        )
        user = state.message or ""
        try:
            ans = _llm_text(deps=deps, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.3)
        except Exception:
            ans = {"cs": "Jak mohu s dopady pomoci?", "sk": "Ako môžem s dopadmi pomôcť?", "en": "How can I help with impacts?"}[l]
        state.result = _ui_response(ans, extra={"agent_events": state.agent_events})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Graph assembly
# -----------------------------------------------------------------------------
def create_graph(deps: GraphDeps):
    """
    Flow:
      start → nlu → route → {agent.reply | clarify | list | get | set | bulk_set | create_impact | explain} → (submit?) → END

    - One decisive step per turn; writes always go through a single submit.
    - No hardcoded greet/help/chitchat nodes; agent.reply handles non-tool answers.
    - Catalog lists are not truncated (only chunked for UI size).
    """
    graph = StateGraph(ImpactState)

    # Async node wrappers
    async def _nlu_node(state: ImpactState) -> ImpactState:
        return await _nlu(deps, state)

    async def _submit_node(state: ImpactState) -> ImpactState:
        return await _submit_changeset_async(deps, state)

    async def _explain_node(state: ImpactState) -> ImpactState:
        return await _explain_node_async(deps, state)

    # Nodes
    graph.add_node("nlu", _nlu_node)
    graph.add_node("route", lambda s: s)

    graph.add_node("agent.reply", _agent_reply(deps))
    graph.add_node("clarify", _clarify_once(deps))

    # Read
    graph.add_node("impact.list", _list_impacts(deps))
    graph.add_node("impact.asset_get", _asset_get_impacts_or_suggest(deps))          # alias node (kept for compatibility)
    graph.add_node("impact.levels.for_asset", _asset_get_impacts_or_suggest(deps))  # canonical GET

    # Write
    graph.add_node("impact.set_level", _set_impact_level(deps))
    graph.add_node("impact.bulk_set", _bulk_set_impact_levels(deps))
    graph.add_node("impact.create_impact", _create_impact(deps))

    # Explain
    graph.add_node("impact.explain", _explain_node)

    # Submit
    graph.add_node("submit_changeset", _submit_node)

    # Entry
    graph.set_entry_point("nlu")
    graph.add_edge("nlu", "route")

    # Router
    def _route(state: ImpactState) -> str:
        intent = state.intent or "agent_reply"
        if intent == "clarify":
            return "clarify"
        if intent == "list":
            return "impact.list"
        if intent == "get":
            # Canonical mapping: GET → "impact.levels.for_asset"
            return "impact.levels.for_asset"
        if intent == "set":
            return "impact.set_level"
        if intent == "bulk_set":
            return "impact.bulk_set"
        if intent == "create_impact":
            return "impact.create_impact"
        if intent == "explain":
            return "impact.explain"
        return "agent.reply"

    graph.add_conditional_edges(
        "route",
        _route,
        {
            "agent.reply": "agent.reply",
            "clarify": "clarify",
            "impact.list": "impact.list",
            "impact.asset_get": "impact.asset_get",                # still routable if explicitly requested
            "impact.levels.for_asset": "impact.levels.for_asset",  # canonical GET path
            "impact.set_level": "impact.set_level",
            "impact.bulk_set": "impact.bulk_set",
            "impact.create_impact": "impact.create_impact",
            "impact.explain": "impact.explain",
        },
    )

    # Writes → single submit per turn
    graph.add_edge("impact.set_level", "submit_changeset")
    graph.add_edge("impact.bulk_set", "submit_changeset")
    graph.add_edge("impact.create_impact", "submit_changeset")

    # Terminal nodes (include both GET names for compatibility; canonical is levels.for_asset)
    for node in (
        "agent.reply",
        "clarify",
        "impact.list",
        "impact.asset_get",
        "impact.levels.for_asset",
        "impact.explain",
        "submit_changeset",
    ):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=deps.checkpointer)


# -----------------------------------------------------------------------------
# Thin runners used by routers (compat with Asset graph style)
# -----------------------------------------------------------------------------
async def run_explain(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

async def run_explain_impacts(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    return await run_explain(deps=deps, ui_req=ui_req)

async def run_set_level(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    """Thin runner so the router doesn't assemble confirmations itself."""
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

async def run_bulk_set(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    """Thin runner so the router doesn't assemble confirmations itself."""
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)


__all__ = [
    "ImpactState",
    "create_graph",
    "run_explain",
    "run_explain_impacts",
    "run_set_level",
    "run_bulk_set",
]
