# File: workshops_ai/graph/impact_graph.py
# -----------------------------------------------------------------------------
# Purpose
# LangGraph graph for the IMPACTS tab (SEND-NOW semantics).
#
# Capabilities
# - list_impacts            : read impact catalog (canonical EN codes)
# - asset_get_impacts       : read current impact levels for a selected asset
# - set_impact_level        : set/update impact levels for one asset (CIA scale 0..4)
# - bulk_set_impact_levels  : set/update impact levels across multiple assets
# - explain_impacts (RAG)   : retrieve context snippets for an explanation (optional)
#
# Semantics
# - Write paths are "emit-only": first broadcast schema-validated WS events, then async
#   submit a ChangeSet with an idempotency key (no staging/commit roundtrip).
# - If the submit fails / no backend client is available, JSON is written to outbox.
# - Read paths return immediately.
#
# Output (UIResponse-like dict; manager enforces emit-only policy as well)
# {
#   "messages": ["assistant text..."],
#   "proposals": null,
#   "staging": {"needs_commit": false, "batch_id": null, "diff": null},
#   "policy": {"emit_changes_enabled": true}
# }
#
# Notes
# - Graph is compiled with an injected checkpointer → per-org/tab history.
# - WS events use WsHub.aemit_impact_set_level(...) as per hub contract.
# - ChangeSets use client's submit; fallback writes JSON to /app/workshops_ai/data/outbox.
# - Explain node falls back to LLM if retriever is missing/unavailable.
# - Optional NLU node parses free text into set/bulk_set payloads when possible.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
import uuid
import json
import os
from typing import Any, Literal, Optional, cast, List

from pydantic import BaseModel, Field

from workshops_ai.graph.factory import GraphDeps

try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("langgraph must be installed to use impact_graph") from e

logger = logging.getLogger(__name__)

# Optional LLM (LangChain OpenAI); runtime-safe fallbacks
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------

Intent = Literal["list", "get", "set", "bulk_set", "explain"]


class ImpactItem(BaseModel):
    impact_code: str
    level: int = Field(ge=0, le=4)
    note: Optional[str] = None


class ImpactSetPayload(BaseModel):
    asset_id: str
    items: list[ImpactItem]


class ImpactBulkPayload(BaseModel):
    entries: list[ImpactSetPayload]


class ImpactState(BaseModel):
    # Routing context / inputs
    org_id: str
    active_tab: str = Field(default="impacts")
    message: Optional[str] = None
    selection: Optional[dict] = None  # optionally {"asset_id": "..."}
    user_id: Optional[str] = None
    locale: Optional[str] = None

    # Structured inputs (preferred)
    action: Optional[Intent] = None
    set_payload: Optional[ImpactSetPayload] = None
    bulk_payload: Optional[ImpactBulkPayload] = None
    question: Optional[str] = None  # for explain

    # Working fields
    intent: Optional[Intent] = None
    change: Optional[dict] = None
    result: Optional[dict] = None
    touched_asset_ids: Optional[List[str]] = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _now_iso() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()


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


_LEVEL_VALUE = [5, 20, 50, 70, 100]  # index 0..4 → display/value scale


def _build_changeset(*, kind: str, org_id: str, user_id: Optional[str], data: dict) -> dict:
    cid = str(uuid.uuid4())
    audit = {
        "source": "AI",
        "tags": ["AI", "impact-graph", f"batch-{cid[:8]}"],
        "user_id": user_id,
        "timestamp": _now_iso(),
    }
    # include idempotency key directly in the payload for file-mode naming, too
    return {
        "id": cid,
        "idempotency_key": cid,
        "org_id": org_id,
        "type": kind,
        "audit": audit,
        "data": data,
    }


async def _asubmit_changeset_or_outbox(payload: dict, deps: GraphDeps) -> None:
    """
    Try Tazila client (async/sync). If not present or fails, dump JSON into outbox.
    Outbox path priority:
      settings.tazila.outbox_path → env WORKSHOPS_AI_TAZILA__OUTBOX_PATH → /app/workshops_ai/data/outbox
    """
    # 1) Tazila client
    try:
        from workshops_ai.app.services import tazila_client  # type: ignore
    except Exception:
        tazila_client = None  # type: ignore

    try:
        if tazila_client is not None:
            ikey = payload.get("idempotency_key") or payload.get("id")
            if hasattr(tazila_client, "asubmit_changeset"):
                await tazila_client.asubmit_changeset(payload, idempotency_key=ikey)  # type: ignore[arg-type]
                return
            if hasattr(tazila_client, "submit_changeset"):
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, tazila_client.submit_changeset, payload)  # type: ignore[arg-type]
                return
    except Exception as e:
        logger.warning("Tazila submit failed, falling back to outbox: %s", e)

    # 2) Outbox fallback
    settings = getattr(deps, "settings", None)
    outbox_dir = None
    try:
        outbox_dir = getattr(getattr(settings, "tazila", None), "outbox_path", None)
    except Exception:
        outbox_dir = None
    outbox_dir = outbox_dir or os.getenv("WORKSHOPS_AI_TAZILA__OUTBOX_PATH") or "/app/workshops_ai/data/outbox"

    try:
        os.makedirs(outbox_dir, exist_ok=True)
        cs_id = payload.get("id") or str(uuid.uuid4())
        path = os.path.join(outbox_dir, f"changeset_{cs_id}.json")

        def _write():
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        await asyncio.to_thread(_write)
        logger.info("ChangeSet written to outbox: %s", path)
    except Exception as e:
        logger.warning("Outbox write failed: %s", e)


async def _maybe_mem_set(deps: GraphDeps, key: str, value: Any, *, ttl_sec: Optional[int] = None) -> None:
    """
    Best-effort memory write (supports sync/async impl and optional TTL).
    Tries:
      1) deps.memory_set(key, value, ttl_sec=...) → on TypeError retries without TTL
      2) deps.memory.set(key, value, ttl_sec=...) → on TypeError retries without TTL
    Silently ignores failures.
    """
    candidates = []
    setter = getattr(deps, "memory_set", None)
    if callable(setter):
        candidates.append(setter)
    mem = getattr(deps, "memory", None)
    if mem is not None:
        mset = getattr(mem, "set", None)
        if callable(mset):
            candidates.append(mset)

    for fn in candidates:
        # with TTL
        try:
            res = fn(key, value, ttl_sec=ttl_sec) if ttl_sec is not None else fn(key, value)
            if asyncio.iscoroutine(res):
                await res
            return
        except TypeError:
            # try without TTL
            try:
                res = fn(key, value)
                if asyncio.iscoroutine(res):
                    await res
                return
            except Exception:
                continue
        except Exception:
            continue
    # give up silently


def _get_llm_from_deps(deps: GraphDeps):
    llm = getattr(deps, "llm", None)
    if llm is not None:
        return llm
    if ChatOpenAI is None:
        return None
    # Safe default
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.1, timeout=12, max_retries=1)  # type: ignore[call-arg]


# -----------------------------------------------------------------------------
# Intent
# -----------------------------------------------------------------------------

def _infer_intent(state: ImpactState) -> ImpactState:
    if state.action in ("list", "get", "set", "bulk_set", "explain"):
        state.intent = cast(Intent, state.action)
        return state

    text = (state.message or "").lower()
    if re.search(r"\b(explain|why|context|rag)\b", text):
        state.intent = "explain"
        return state

    if re.search(r"\b(bulk|multiple|all assets)\b", text):
        state.intent = "bulk_set"
        return state

    if re.search(r"\b(set|update|level|impact)\b", text):
        state.intent = "set"
        return state

    if (state.selection or {}).get("asset_id") or re.search(r"\b(get|show|detail)\b", text):
        state.intent = "get"
        return state

    state.intent = "list"
    return state


# -----------------------------------------------------------------------------
# NLU (optional) — parse free text into set/bulk_set payloads
# -----------------------------------------------------------------------------

_NORM_LEVEL_RX = re.compile(r"\b([cia]|confidentiality|integrity|availability|avail)\b[^0-9-]*([0-4])\b", re.I)
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)
_QUOTES_RX = re.compile(r"[\"“”']\s*([^\"“”']+?)\s*[\"“”']")  # reserved if you later map names→IDs

def _norm_txt(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", (s or "")).encode("ascii", "ignore").decode("ascii").lower()


def _derive_set_from_text(message: str, selection_asset_id: Optional[str]) -> Optional[ImpactSetPayload]:
    """
    Regex-only fast path (no LLM). Requires an asset_id from selection or text.
    Examples that work:
      - "set C=3 and A=2 for 7d0e... asset"
      - "C2 I1 A0" (with selected asset)
    """
    t = _norm_txt(message)
    asset_id = selection_asset_id or (_UUID_RE.search(message).group(0) if _UUID_RE.search(message) else None)
    if not asset_id:
        return None

    levels: list[ImpactItem] = []
    for m in _NORM_LEVEL_RX.finditer(t):
        dim, val = m.group(1).lower(), int(m.group(2))
        code = "C" if dim.startswith("c") else "I" if dim.startswith("i") else "A"
        levels.append(ImpactItem(impact_code=code, level=val))
    if not levels:
        return None
    return ImpactSetPayload(asset_id=asset_id, items=levels)


async def _nlu_impacts(deps: GraphDeps, state: ImpactState) -> ImpactState:
    """
    Best-effort NLU that fills set_payload / bulk_payload from natural language.
    - First tries regex-only extraction for common "C/I/A=<0..4>" patterns.
    - If still empty and an LLM is available, asks it for STRICT JSON.
    - Never invents asset IDs; selection.asset_id or explicit UUID must be present.
    """
    raw = (state.message or "").strip()
    if not raw:
        return state

    selection_id = (state.selection or {}).get("asset_id")

    # 1) Fast regex-based extraction (no LLM dependency)
    guess = _derive_set_from_text(raw, selection_id)
    if guess and not state.set_payload:
        state.set_payload = guess
        if not state.action:
            state.action = cast(Intent, "set")

    # 2) LLM-backed parsing if still nothing
    if not state.set_payload and not state.bulk_payload:
        llm = _get_llm_from_deps(deps)
        if llm is not None:
            system = (
                "You are a strict NLU for an 'impacts' assistant. "
                "Extract intent and slots from any language (CS/SK/EN; tolerate typos). "
                "Return ONLY JSON with these optional fields: "
                "{"
                "  action: one of [list,get,set,bulk_set,explain],"
                "  selection: {asset_id},"
                "  set: {asset_id, items:[{impact_code: 'C'|'I'|'A', level: 0..4, note?: string}]},"
                "  bulk: {entries:[{asset_id, items:[{impact_code, level, note?}]}]}"
                "}"
                "Rules:\n"
                "- NEVER invent asset IDs.\n"
                "- If selection.asset_id is provided, you MAY copy it into set.asset_id.\n"
                "- If unsure, leave fields null. Output STRICT JSON, no prose."
            )
            human = {
                "message": raw,
                "selection": {"asset_id": selection_id} if selection_id else None,
            }
            try:
                from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
                resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=json.dumps(human))])  # type: ignore[arg-type]
                txt = (getattr(resp, "content", None) or "").strip()
                i, j = txt.find("{"), txt.rfind("}")
                data = json.loads(txt[i:j + 1] if i != -1 and j != -1 else txt)

                action = data.get("action")
                sel = data.get("selection") or {}
                if sel.get("asset_id") and not selection_id:
                    selection_id = sel.get("asset_id")

                if data.get("set"):
                    s = data["set"]
                    aid = s.get("asset_id") or selection_id
                    items = [ImpactItem(**it) for it in (s.get("items") or [])]
                    if aid and items:
                        state.set_payload = ImpactSetPayload(asset_id=str(aid), items=items)
                        state.action = cast(Intent, "set")
                elif data.get("bulk"):
                    entries = []
                    for e in (data["bulk"].get("entries") or []):
                        aid = e.get("asset_id")
                        items = [ImpactItem(**it) for it in (e.get("items") or [])]
                        if aid and items:
                            entries.append(ImpactSetPayload(asset_id=str(aid), items=items))
                    if entries:
                        state.bulk_payload = ImpactBulkPayload(entries=entries)
                        state.action = cast(Intent, "bulk_set")
                elif action in ("list", "get", "explain"):
                    state.action = cast(Intent, action)
            except Exception as e:
                logger.info("IMPACTS NLU LLM failed: %s", e)

    return state


# -----------------------------------------------------------------------------
# Read-only nodes
# -----------------------------------------------------------------------------

def _list_impacts(deps: GraphDeps):
    def _inner(state: ImpactState) -> ImpactState:
        try:
            impacts = deps.backend_port.list_impacts()
        except Exception as e:
            state.result = _ui_response(f"Failed to read impact catalog: {e}")
            return state

        if not isinstance(impacts, list):
            state.result = _ui_response("Impact catalog is unavailable.")
            return state

        lines = [f"Impact catalog ({len(impacts)}):"]
        for row in impacts[:20]:
            code = row.get("impact_code") or row.get("code") or row.get("EN Code") or "?"
            name = row.get("EN Name") or row.get("name") or row.get("Impact") or ""
            lines.append(f"- {code}: {name}")
        state.result = _ui_response("\n".join(lines))
        return state
    return _inner


def _asset_get_impacts(deps: GraphDeps):
    def _inner(state: ImpactState) -> ImpactState:
        asset_id = (state.selection or {}).get("asset_id")
        if not asset_id:
            state.result = _ui_response("No asset selected. Provide selection.asset_id.")
            return state

        try:
            levels = deps.backend_port.asset_get_impacts(asset_id)
        except Exception as e:
            state.result = _ui_response(f"Failed to read impacts for '{asset_id}': {e}")
            return state

        if not isinstance(levels, list):
            state.result = _ui_response(f"No impact data for asset '{asset_id}'.")
            return state

        summary = [f"Impacts for {asset_id} ({len(levels)} items):"]
        for it in levels[:30]:
            code = it.get("impact_code") or it.get("code")
            lvl = it.get("level")
            summary.append(f"- {code}: {lvl}")
        state.result = _ui_response("\n".join(summary), extra={"asset_id": asset_id, "impacts": levels})
        return state
    return _inner


# -----------------------------------------------------------------------------
# Write nodes — SEND-NOW (set / bulk_set)
# -----------------------------------------------------------------------------

def _mk_ws_payload(*, org_id: str, asset_id: str, it: ImpactItem, actor: Optional[str]) -> dict:
    lvl_idx = int(it.level)
    lvl_idx = 0 if lvl_idx < 0 else 4 if lvl_idx > 4 else lvl_idx
    return {
        "event": "impact.set_level",
        "org_id": org_id,
        "asset_id": asset_id,
        "impact_code": it.impact_code,
        "level_index": lvl_idx,
        "level_value": _LEVEL_VALUE[lvl_idx],
        "note": it.note,
        "actor": actor,
        "ts": _now_iso(),
    }


def _set_impact_level(deps: GraphDeps):
    async def _inner(state: ImpactState) -> ImpactState:
        payload = state.set_payload
        if payload is None:
            # Attempt to assemble from a generic 'update' dict if provided by caller
            update = getattr(state, "update", None) or {}
            try:
                payload = ImpactSetPayload(
                    asset_id=update.get("asset_id") or (state.selection or {}).get("asset_id"),
                    items=[ImpactItem(**it) for it in (update.get("items") or [])],
                )
            except Exception as e:
                state.result = _ui_response(f"Invalid set payload: {e}")
                return state

        if not payload or not payload.asset_id or not payload.items:
            state.result = _ui_response("Missing set payload (asset_id and items[impact_code, level]).")
            return state

        # Schema-validated WS emit per item (await hub)
        for it in payload.items:
            try:
                await deps.ws_hub.aemit_impact_set_level(
                    _mk_ws_payload(
                        org_id=state.org_id,
                        asset_id=payload.asset_id,
                        it=it,
                        actor=state.user_id,
                    )
                )
            except Exception as e:  # pragma: no cover
                logger.warning("WS emit failed for %s/%s: %s", payload.asset_id, it.impact_code, e)

        change = _build_changeset(
            kind="impact.set_levels",
            org_id=state.org_id,
            user_id=state.user_id,
            data={"asset_id": payload.asset_id, "items": [it.model_dump() for it in payload.items]},
        )
        state.change = change
        state.touched_asset_ids = [payload.asset_id]
        return state
    return _inner


def _bulk_set_impact_levels(deps: GraphDeps):
    async def _inner(state: ImpactState) -> ImpactState:
        bulk = state.bulk_payload
        if bulk is None:
            raw = getattr(state, "bulk", None) or {}
            entries = raw.get("entries") or []
            try:
                bulk = ImpactBulkPayload(
                    entries=[
                        ImpactSetPayload(
                            asset_id=e.get("asset_id"),
                            items=[ImpactItem(**it) for it in (e.get("items") or [])],
                        )
                        for e in entries
                    ]
                )
            except Exception as e:
                state.result = _ui_response(f"Invalid bulk payload: {e}")
                return state

        if not bulk or not bulk.entries:
            state.result = _ui_response("Missing bulk payload entries.")
            return state

        # Schema-validated emits for each item in each entry
        for entry in bulk.entries:
            for it in entry.items:
                try:
                    await deps.ws_hub.aemit_impact_set_level(
                        _mk_ws_payload(
                            org_id=state.org_id,
                            asset_id=entry.asset_id,
                            it=it,
                            actor=state.user_id,
                        )
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning("WS emit failed for %s/%s: %s", entry.asset_id, it.impact_code, e)

        change = _build_changeset(
            kind="impact.bulk_set_levels",
            org_id=state.org_id,
            user_id=state.user_id,
            data={
                "entries": [
                    {"asset_id": e.asset_id, "items": [it.model_dump() for it in e.items]}
                    for e in bulk.entries
                ]
            },
        )
        state.change = change
        seen = set()
        touched: List[str] = []
        for e in bulk.entries:
            aid = str(e.asset_id)
            if aid not in seen:
                seen.add(aid)
                touched.append(aid)
        state.touched_asset_ids = touched
        return state
    return _inner


async def _submit_changeset_async(state: ImpactState, deps: GraphDeps) -> ImpactState:
    if state.change:
        await _asubmit_changeset_or_outbox(state.change, deps)
        msg = "Impact levels submitted." if state.intent in ("set", "bulk_set") else "Change submitted."
        extra = {"changeset_id": state.change.get("id")}
        state.result = _ui_response(msg, extra=extra)

        # Best-effort hint after bulk submit (store last touched asset_ids list)
        try:
            if state.intent == "bulk_set" and state.touched_asset_ids:
                key = f"{state.org_id}:impacts.last_touched_asset_ids"
                val = json.dumps(list(state.touched_asset_ids), separators=(",", ":"))
                await _maybe_mem_set(deps, key, val, ttl_sec=24 * 60 * 60)
        except Exception as e:  # best-effort
            logger.info("Failed to remember impacts.last_touched_asset_ids: %s", e)
    else:
        state.result = _ui_response("Nothing to submit.")
    return state


# -----------------------------------------------------------------------------
# LLM fallback helpers (for explain)
# -----------------------------------------------------------------------------

def _build_llm_prompt(question: str, *, locale: Optional[str]) -> str:
    lang = (locale or "").lower()
    return (
        "You are a cybersecurity assistant specializing in business impact analysis (CIA: Confidentiality, "
        "Integrity, Availability). The user asked a question about 'impacts'. "
        f"Language hint: {lang or 'en'}. "
        "Provide a concise, actionable explanation (≤120 words). "
        "If enterprise documents are unavailable, rely on general best-practice knowledge. "
        "Question:\n"
        f"{question}\n"
        "Answer:"
    )


async def _call_llm(llm: Any, prompt: str) -> Optional[str]:
    try:
        if hasattr(llm, "ainvoke") and callable(getattr(llm, "ainvoke")):
            out = await llm.ainvoke(prompt)  # type: ignore[attr-defined]
        elif hasattr(llm, "invoke") and callable(getattr(llm, "invoke")):
            out = await asyncio.to_thread(llm.invoke, prompt)  # type: ignore[attr-defined]
        elif callable(llm):
            maybe = llm(prompt)
            out = await maybe if asyncio.iscoroutine(maybe) else maybe
        else:
            return None

        if out is None:
            return None
        if isinstance(out, str):
            return out.strip()
        if isinstance(out, dict):
            for k in ("content", "text", "output", "answer"):
                v = out.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        txt = getattr(out, "content", None) or getattr(out, "text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
        return str(out)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Explain (RAG) — read-only helper with LLM fallback
# -----------------------------------------------------------------------------

async def _explain_impacts_async(state: ImpactState, deps: GraphDeps) -> ImpactState:
    q = (state.question or state.message or "").strip()
    if not q:
        state.result = _ui_response("Provide a question (state.question or message) to explain impacts.")
        return state

    retriever = None
    try:
        retriever = deps.make_retriever(state.org_id, "impacts")
    except Exception as e:
        logger.info("RAG retriever unavailable: %s", e)

    if retriever:
        try:
            if hasattr(retriever, "asearch"):
                docs = await retriever.asearch(q, k=5)  # type: ignore[attr-defined]
            else:
                docs = retriever.search(q, k=5)  # type: ignore[attr-defined]
        except Exception as e:
            docs = []
            logger.info("RAG search failed, will try LLM fallback: %s", e)

        snippets: list[str] = []
        for d in docs or []:
            txt = (d.get("text") or d.get("page_content") or "").strip()
            src = (d.get("source") or (d.get("metadata") or {}).get("source") or "")
            if txt:
                preview = txt[:400].replace("\n", " ")
                snippets.append(f"- {preview}" + (f" (source: {src})" if src else ""))

        if snippets:
            msg = "Context for your question (top snippets):\n" + "\n".join(snippets[:5])
            state.result = _ui_response(msg, extra={"question": q, "snippets": snippets[:5]})
            return state
    else:
        logger.info("RAG retriever not configured; using LLM fallback if available.")

    llm = getattr(deps, "llm", None)
    if llm:
        prompt = _build_llm_prompt(q, locale=state.locale)
        text = await _call_llm(llm, prompt)
        if text:
            state.result = _ui_response(text, extra={"question": q, "source": "llm_fallback"})
            return state

    state.result = _ui_response(
        "No contextual corpus is available right now and LLM fallback did not return an answer."
    )
    return state


# -----------------------------------------------------------------------------
# Graph assembly
# -----------------------------------------------------------------------------

def create_graph(deps: GraphDeps):
    graph = StateGraph(ImpactState)

    # Nodes
    async def _nlu_node(state: ImpactState) -> ImpactState:
        return await _nlu_impacts(deps, state)

    graph.add_node("nlu_impacts", _nlu_node)
    graph.add_node("route", _infer_intent)
    graph.add_node("list_impacts", _list_impacts(deps))
    graph.add_node("asset_get_impacts", _asset_get_impacts(deps))
    graph.add_node("set_impact_level", _set_impact_level(deps))
    graph.add_node("bulk_set_impact_levels", _bulk_set_impact_levels(deps))

    async def _submit_node(state: ImpactState) -> ImpactState:
        return await _submit_changeset_async(state, deps)

    async def _explain_node(state: ImpactState) -> ImpactState:
        return await _explain_impacts_async(state, deps)

    graph.add_node("submit_changeset", _submit_node)
    graph.add_node("explain_impacts", _explain_node)

    # Entry & routing
    graph.set_entry_point("nlu_impacts")
    graph.add_edge("nlu_impacts", "route")

    def _route(state: ImpactState) -> str:
        intent = state.intent or "list"
        if intent == "list":
            return "list_impacts"
        if intent == "get":
            return "asset_get_impacts"
        if intent == "set":
            return "set_impact_level"
        if intent == "bulk_set":
            return "bulk_set_impact_levels"
        if intent == "explain":
            return "explain_impacts"
        return "list_impacts"

    graph.add_conditional_edges(
        "route",
        _route,
        {
            "list_impacts": "list_impacts",
            "asset_get_impacts": "asset_get_impacts",
            "set_impact_level": "set_impact_level",
            "bulk_set_impact_levels": "bulk_set_impact_levels",
            "explain_impacts": "explain_impacts",
        },
    )

    # Write paths → submit; read/explain paths → end
    graph.add_edge("set_impact_level", "submit_changeset")
    graph.add_edge("bulk_set_impact_levels", "submit_changeset")

    graph.add_edge("list_impacts", END)
    graph.add_edge("asset_get_impacts", END)
    graph.add_edge("explain_impacts", END)
    graph.add_edge("submit_changeset", END)

    # Compile with injected checkpointer for per-org/tab history
    return graph.compile(checkpointer=deps.checkpointer)
