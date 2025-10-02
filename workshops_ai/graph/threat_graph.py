# File: workshops_ai/graph/threat_graph.py
# -----------------------------------------------------------------------------
# Purpose
# LangGraph graph for the THREATS tab (SEND-NOW semantics).
#
# Capabilities
# - list_threats                  : read threat catalog (optionally filtered by asset_type)
# - asset_get_risks               : read current risk probabilities for a selected asset
# - set_risk_probability          : set/update risk probabilities for one asset (0..4)
# - bulk_set_risk_probabilities   : set/update probabilities across multiple assets
# - upsert_threat                 : create/update an org-scoped custom threat (server-side persist)
# - explain_threats (RAG)         : retrieve context snippets; falls back to LLM if RAG missing
#
# Semantics
# - Write paths are "emit-only" + async submit (no staging/commit roundtrip).
# - WS: prefer batch list-shaped event if available; fall back to per-item emitter.
# - Submit: try client (async/sync), fallback to JSON outbox.
#
# Output (UIResponse-like dict)
# {
#   "messages": ["assistant text..."],
#   "proposals": null,
#   "staging": {"needs_commit": false, "batch_id": null, "diff": null},
#   "policy": {"emit_changes_enabled": true}
# }
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Literal, Optional, cast

from pydantic import BaseModel, Field

from workshops_ai.graph.factory import GraphDeps

# Unified DTO + mappers
from workshops_ai.api.schema.threat import AssetRisk
from workshops_ai.agents.utils.mappers import (
    map_asset_risk_to_ws_event,
    map_asset_risk_to_changeset,
)

try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("langgraph must be installed to use threat_graph") from e

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# State & legacy payloads (for back-compat only)
# -----------------------------------------------------------------------------

Intent = Literal["list", "get", "set", "bulk_set", "explain", "upsert"]


class RiskItem(BaseModel):
    threat_code: str
    probability: int = Field(ge=0, le=4)
    note: Optional[str] = None


class RiskSetPayload(BaseModel):
    asset_id: str
    items: list[RiskItem]


class RiskBulkPayload(BaseModel):
    entries: list[RiskSetPayload]


class UpsertThreatPayload(BaseModel):
    threat_code: str
    name: Optional[str] = None
    category: Optional[str] = None
    notes: Optional[str] = None


class ThreatState(BaseModel):
    # Routing context / inputs
    org_id: str
    active_tab: str = Field(default="threats")
    message: Optional[str] = None
    selection: Optional[dict] = None  # e.g., {"asset_id":"...", "asset_type":"..."}
    user_id: Optional[str] = None
    locale: Optional[str] = None

    # Wizard context (consumed in write paths / context)
    organization_profile: Optional[dict] = None
    services: Optional[list[dict]] = None
    existing_assets: Optional[list[dict]] = None
    website_content: Optional[dict] = None
    number_of_assets: Optional[int] = None
    language: Optional[str] = None

    # Structured inputs (preferred)
    action: Optional[Intent] = None

    # Legacy shapes (kept for compatibility with older callers)
    set_payload: Optional[RiskSetPayload] = None
    bulk_payload: Optional[RiskBulkPayload] = None
    upsert_payload: Optional[UpsertThreatPayload] = None

    # NEW: generic unified payload for DTO validation
    payload: Optional[dict] = None

    # For explain
    question: Optional[str] = None

    # Working fields
    intent: Optional[Intent] = None
    change: Optional[dict] = None
    result: Optional[dict] = None
    touched_asset_ids: Optional[List[str]] = None  # hint after bulk submit


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


def _build_changeset(*, kind: str, org_id: str, user_id: Optional[str], data: dict) -> dict:
    cid = str(uuid.uuid4())
    audit = {
        "source": "AI",
        "tags": ["AI", "threat-graph", f"batch-{cid[:8]}"],
        "user_id": user_id,
        "timestamp": _now_iso(),
    }
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
    Try Tazila client (async/sync). If unavailable or fails → JSON outbox.
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
        try:
            res = fn(key, value, ttl_sec=ttl_sec) if ttl_sec is not None else fn(key, value)
            if asyncio.iscoroutine(res):
                await res
            return
        except TypeError:
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


# -----------------------------------------------------------------------------
# Intent inference
# -----------------------------------------------------------------------------

def _infer_intent(state: ThreatState) -> ThreatState:
    if state.action in ("list", "get", "set", "bulk_set", "explain", "upsert"):
        state.intent = cast(Intent, state.action)
        return state

    text = (state.message or "").lower()
    if re.search(r"\b(upsert|create custom threat|new threat)\b", text):
        state.intent = "upsert"
        return state
    if re.search(r"\b(explain|why|context|rag)\b", text):
        state.intent = "explain"
        return state
    if re.search(r"\b(bulk|multiple|all assets)\b", text):
        state.intent = "bulk_set"
        return state
    if re.search(r"\b(set|update|prob(ability)?|risk)\b", text):
        state.intent = "set"
        return state
    if (state.selection or {}).get("asset_id") or re.search(r"\b(get|show|detail)\b", text):
        state.intent = "get"
        return state

    state.intent = "list"
    return state


# -----------------------------------------------------------------------------
# Read-only nodes
# -----------------------------------------------------------------------------

def _list_threats(deps: GraphDeps):
    def _inner(state: ThreatState) -> ThreatState:
        apply_asset_type = (state.selection or {}).get("asset_type")
        try:
            threats = deps.backend_port.list_threats(apply_asset_type=apply_asset_type)
        except Exception as e:
            state.result = _ui_response(f"Failed to read threat catalog: {e}")
            return state

        if not isinstance(threats, list):
            state.result = _ui_response("Threat catalog is unavailable.")
            return state

        hdr = f"Threat catalog ({len(threats)})"
        if apply_asset_type:
            hdr += f" filtered by asset_type='{apply_asset_type}'"
        lines = [hdr + ":"]
        for row in threats[:30]:
            code = row.get("threat_code") or row.get("EN Code") or row.get("code") or "?"
            name = row.get("EN Name") or row.get("name") or row.get("Threat") or ""
            cat = row.get("Threat category") or row.get("category") or ""
            lines.append(f"- {code}: {name}" + (f" [{cat}]" if cat else ""))
        state.result = _ui_response("\n".join(lines))
        return state
    return _inner


def _asset_get_risks(deps: GraphDeps):
    def _inner(state: ThreatState) -> ThreatState:
        asset_id = (state.selection or {}).get("asset_id")
        if not asset_id:
            state.result = _ui_response("No asset selected. Provide selection.asset_id.")
            return state

        try:
            risks = deps.backend_port.asset_get_risks(asset_id)
        except Exception as e:
            state.result = _ui_response(f"Failed to read risks for '{asset_id}': {e}")
            return state

        if not isinstance(risks, list):
            state.result = _ui_response(f"No risk data for asset '{asset_id}'.")
            return state

        summary = [f"Risks for {asset_id} ({len(risks)} items):"]
        for it in risks[:30]:
            code = it.get("threat_code") or it.get("code")
            p = it.get("probability")
            summary.append(f"- {code}: P={p}")
        state.result = _ui_response("\n".join(summary), extra={"asset_id": asset_id, "risks": risks})
        return state
    return _inner


# -----------------------------------------------------------------------------
# WS emit helpers (batch preferred, per-item fallback)
# -----------------------------------------------------------------------------

async def _emit_ws_from_dto(dto: AssetRisk, deps: GraphDeps, org_id: str, actor: Optional[str]) -> None:
    """
    Prefer a single list-shaped event via aemit_risk_set_event(dto-based payload).
    If not available, fall back to per-item aemit_risk_set_probability to preserve UI compatibility.
    """
    hub = getattr(deps, "ws_hub", None)
    if hub is None:
        return

    # 1) Try batch/list-shaped emitter using mapper shape (assetId, assetName, risks[])
    ws_batch = map_asset_risk_to_ws_event(dto)
    if hasattr(hub, "aemit_risk_set_event"):
        try:
            await hub.aemit_risk_set_event(ws_batch)  # type: ignore[attr-defined]
            return
        except Exception as e:  # pragma: no cover
            logger.warning("WS batch emit failed, falling back to per-item: %s", e)

    # 2) Fallback: per-item legacy emitter
    if hasattr(hub, "aemit_risk_set_probability"):
        try:
            asset_id = str(getattr(dto, "assetId", None) or "")
            for r in (dto.risks or []):
                event = {
                    "event": "risk.set_probability",
                    "org_id": org_id,
                    "asset_id": asset_id,
                    "threat_code": r.threatName,   # mapper/UI expect code-like name
                    "probability": int(r.probability),
                    "note": getattr(r, "note", None),
                    "actor": actor,
                    "ts": _now_iso(),
                }
                await hub.aemit_risk_set_probability(event)  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover
            logger.warning("WS per-item emit failed: %s", e)


# -----------------------------------------------------------------------------
# Write nodes — SEND-NOW
# -----------------------------------------------------------------------------

def _adapt_set_payload_to_dto(payload: RiskSetPayload, *, asset_name: Optional[str] = None) -> AssetRisk:
    """
    Minimal adapter for backward compatibility when callers still send RiskSetPayload.
    Builds an AssetRisk DTO so the rest of the write path stays unified.
    """
    risks = [
        {"threatName": it.threat_code, "probability": int(it.probability)}
        for it in payload.items
    ]
    dto_raw = {
        "assetId": payload.asset_id,
        "assetName": asset_name,
        "risks": risks,
    }
    return AssetRisk.model_validate(dto_raw)


def _set_risk_probability(deps: GraphDeps):
    async def _inner(state: ThreatState) -> ThreatState:
        dto: Optional[AssetRisk] = None

        # 1) Preferred: validate provided unified DTO payload
        if state.payload:
            try:
                dto = AssetRisk.model_validate(state.payload)
            except Exception as e:
                state.result = _ui_response(f"Invalid AssetRisk payload: {e}")
                return state

        # 2) Back-compat: build DTO from legacy RiskSetPayload (if present)
        if dto is None:
            payload = state.set_payload
            if payload is None:
                update = getattr(state, "update", None) or {}
                try:
                    items = [RiskItem(**it) for it in (update.get("items") or [])]
                    payload = RiskSetPayload(
                        asset_id=update.get("asset_id") or (state.selection or {}).get("asset_id"),
                        items=items,
                    )
                except Exception as e:
                    state.result = _ui_response(f"Invalid set payload: {e}")
                    return state

            if not payload or not payload.asset_id or not payload.items:
                state.result = _ui_response("Missing set payload (asset_id and items[threat_code, probability]).")
                return state

            # Fetch asset name (optional, for nicer DTO)
            asset_name = None
            try:
                asset = deps.backend_port.asset_get(payload.asset_id)
                asset_name = asset.get("name") if isinstance(asset, dict) else None
            except Exception:
                pass

            dto = _adapt_set_payload_to_dto(payload, asset_name=asset_name)

        # Emit WS event(s) using mapper batch shape (with legacy fallback)
        try:
            await _emit_ws_from_dto(dto, deps, org_id=state.org_id, actor=state.user_id)
        except Exception as e:  # pragma: no cover
            logger.warning("WS emit failed: %s", e)

        # Build changeset via mapper (keeps runtime submit contract)
        try:
            changeset = map_asset_risk_to_changeset(dto, org_id=state.org_id)
        except Exception as e:
            state.result = _ui_response(f"Failed to build ChangeSet: {e}")
            return state

        state.change = changeset
        aid = getattr(dto, "assetId", None)
        state.touched_asset_ids = [str(aid)] if aid else None
        return state
    return _inner


def _bulk_set_risk_probabilities(deps: GraphDeps):
    async def _inner(state: ThreatState) -> ThreatState:
        bulk = state.bulk_payload
        if bulk is None:
            raw = getattr(state, "bulk", None) or {}
            entries = raw.get("entries") or []
            try:
                bulk = RiskBulkPayload(
                    entries=[
                        RiskSetPayload(
                            asset_id=e.get("asset_id"),
                            items=[RiskItem(**it) for it in (e.get("items") or [])],
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

        # Emit per item per asset (legacy-compatible) — keep this path simple
        hub = getattr(deps, "ws_hub", None)
        for entry in bulk.entries:
            for it in entry.items:
                try:
                    if hub and hasattr(hub, "aemit_risk_set_probability"):
                        event = {
                            "event": "risk.set_probability",
                            "org_id": state.org_id,
                            "asset_id": entry.asset_id,
                            "threat_code": it.threat_code,
                            "probability": int(it.probability),
                            "note": it.note,
                            "actor": state.user_id,
                            "ts": _now_iso(),
                        }
                        await hub.aemit_risk_set_probability(event)  # type: ignore[attr-defined]
                except Exception as e:  # pragma: no cover
                    logger.warning("WS emit failed for %s/%s: %s", entry.asset_id, it.threat_code, e)

        # Prepare ChangeSet
        change = _build_changeset(
            kind="risk.bulk_set_probabilities",
            org_id=state.org_id,
            user_id=state.user_id,
            data={
                "entries": [
                    {"asset_id": e.asset_id, "items": [it.model_dump() for it in e.items]}
                    for e in bulk.entries
                ],
                "context": _context_from_wizard(state),
            },
        )
        state.change = change

        # Remember touched asset_ids for UX hints
        seen: set[str] = set()
        touched: list[str] = []
        for e in bulk.entries:
            if e.asset_id and e.asset_id not in seen:
                seen.add(e.asset_id)
                touched.append(e.asset_id)
        state.touched_asset_ids = touched
        return state
    return _inner


def _upsert_threat(deps: GraphDeps):
    async def _inner(state: ThreatState) -> ThreatState:
        payload = state.upsert_payload
        if payload is None:
            raw = getattr(state, "threat", None) or {}
            try:
                payload = UpsertThreatPayload(**raw)
            except Exception as e:
                state.result = _ui_response(f"Invalid upsert payload: {e}")
                return state

        if not payload or not payload.threat_code:
            state.result = _ui_response("Missing upsert_payload.threat_code.")
            return state

        # WS
        hub = getattr(deps, "ws_hub", None)
        if hub:
            try:
                if hasattr(hub, "aemit_threat_upserted"):
                    await hub.aemit_threat_upserted({"org_id": state.org_id, **payload.model_dump()})  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning("WS threat upsert emit failed: %s", e)

        # Submit
        change = _build_changeset(
            kind="threat.upsert",
            org_id=state.org_id,
            user_id=state.user_id,
            data={**payload.model_dump(), "context": _context_from_wizard(state)},
        )
        state.change = change
        return state
    return _inner


async def _submit_changeset_async(state: ThreatState, deps: GraphDeps) -> ThreatState:
    if state.change:
        await _asubmit_changeset_or_outbox(state.change, deps)

        if state.intent in ("set", "bulk_set"):
            msg = "Risk probabilities submitted."
        elif state.intent == "upsert":
            msg = "Threat upsert submitted."
        else:
            msg = "Change submitted."

        state.result = _ui_response(msg, extra={"changeset_id": state.change.get("id")})

        # Best-effort hint after bulk submit
        try:
            if state.intent == "bulk_set" and state.touched_asset_ids:
                key = f"{state.org_id}:threats.last_touched_asset_ids"
                val = json.dumps(list(state.touched_asset_ids), separators=(",", ":"))
                await _maybe_mem_set(deps, key, val, ttl_sec=24 * 60 * 60)
        except Exception:
            pass
    else:
        state.result = _ui_response("Nothing to submit.")
    return state


# -----------------------------------------------------------------------------
# Explain (RAG with LLM fallback)
# -----------------------------------------------------------------------------

def _build_llm_prompt(question: str, *, locale: Optional[str]) -> str:
    lang = (locale or "en").lower()
    return (
        "You are a cybersecurity assistant focused on threats and risk probabilities. "
        f"Language hint: {lang}. Provide a concise, actionable explanation (≤120 words). "
        "When enterprise corpus is unavailable, rely on widely-accepted best practices.\n"
        f"Question:\n{question}\nAnswer:"
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


def _context_from_wizard(state: ThreatState) -> dict:
    return {
        "organization_profile": state.organization_profile,
        "services": state.services,
        "existing_assets": state.existing_assets,
        "website_content": state.website_content,
        "number_of_assets": state.number_of_assets,
        "language": state.language,
    }


async def _explain_threats_async(state: ThreatState, deps: GraphDeps) -> ThreatState:
    q = (state.question or state.message or "").strip()
    if not q:
        state.result = _ui_response("Provide a question (state.question or message) to explain threats/risks.")
        return state

    retriever = None
    try:
        retriever = deps.make_retriever(state.org_id, "threats")
    except Exception as e:
        logger.info("RAG retriever unavailable: %s", e)

    if retriever:
        try:
            if hasattr(retriever, "asearch"):
                docs = await retriever.asearch(q, k=5)  # type: ignore[attr-defined]
            else:
                docs = retriever.search(q, k=5)  # type: ignore[attr-defined]
        except Exception as e:
            logger.info("RAG search failed, trying LLM fallback: %s", e)
            docs = []

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

    llm = getattr(deps, "llm", None)
    if llm:
        prompt = _build_llm_prompt(q, locale=state.locale or state.language)
        text = await _call_llm(llm, prompt)
        if text:
            state.result = _ui_response(text, extra={"question": q, "source": "llm_fallback"})
            return state

    state.result = _ui_response("No contextual corpus available and LLM fallback did not return an answer.")
    return state


# -----------------------------------------------------------------------------
# Graph assembly
# -----------------------------------------------------------------------------

def create_graph(deps: GraphDeps):
    graph = StateGraph(ThreatState)

    # Nodes
    graph.add_node("route", _infer_intent)
    graph.add_node("list_threats", _list_threats(deps))
    graph.add_node("asset_get_risks", _asset_get_risks(deps))
    graph.add_node("set_risk_probability", _set_risk_probability(deps))
    graph.add_node("bulk_set_risk_probabilities", _bulk_set_risk_probabilities(deps))
    graph.add_node("upsert_threat", _upsert_threat(deps))

    async def _submit_node(state: ThreatState) -> ThreatState:
        return await _submit_changeset_async(state, deps)

    async def _explain_node(state: ThreatState) -> ThreatState:
        return await _explain_threats_async(state, deps)

    graph.add_node("submit_changeset", _submit_node)
    graph.add_node("explain_threats", _explain_node)

    # Entry & routing
    graph.set_entry_point("route")

    def _route(state: ThreatState) -> str:
        intent = state.intent or "list"
        if intent == "list":
            return "list_threats"
        if intent == "get":
            return "asset_get_risks"
        if intent == "set":
            return "set_risk_probability"
        if intent == "bulk_set":
            return "bulk_set_risk_probabilities"
        if intent == "upsert":
            return "upsert_threat"
        if intent == "explain":
            return "explain_threats"
        return "list_threats"

    graph.add_conditional_edges(
        "route",
        _route,
        {
            "list_threats": "list_threats",
            "asset_get_risks": "asset_get_risks",
            "set_risk_probability": "set_risk_probability",
            "bulk_set_risk_probabilities": "bulk_set_risk_probabilities",
            "upsert_threat": "upsert_threat",
            "explain_threats": "explain_threats",
        },
    )

    # Write paths → submit; read/explain paths → end
    graph.add_edge("set_risk_probability", "submit_changeset")
    graph.add_edge("bulk_set_risk_probabilities", "submit_changeset")
    graph.add_edge("upsert_threat", "submit_changeset")

    graph.add_edge("list_threats", END)
    graph.add_edge("asset_get_risks", END)
    graph.add_edge("explain_threats", END)
    graph.add_edge("submit_changeset", END)

    # Compile with injected checkpointer for per-org/tab history
    return graph.compile(checkpointer=deps.checkpointer)


__all__ = ["ThreatState", "create_graph"]
