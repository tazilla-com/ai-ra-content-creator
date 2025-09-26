# File: workshops_ai/agents/threat_agent.py
# -----------------------------------------------------------------------------
# ThreatAgent — LangChain Tools agent for the "threats" tab (SEND-NOW semantics)
#
# Capabilities:
#  - list_threats(q?, category?, apply_asset_type?)             → read threat catalog
#  - explain_threats(threat_code, asset_id?)                    → explain threat ↔ asset link
#  - set_risk_probability(asset_id, threat_code, probability, note?)  → set 0..4
#  - bulk_set_risk_probability(items=[...])                     → batch set
#  - upsert_threat(code?, name, category, asset_type_codes[], flags..., description?)
#
# READ path (catalog lookups):
#  - Uses workshops_ai.tools.search_catalogs.* (single source for fuzzy search / explains).
#  - Keeps EN codes canonical; tolerant to CZ/SK aliases via normalizers in the helper.
#
# WRITE path (SEND-NOW):
#  - Optimistic WS broadcast to update UI quickly (schema-validated emitter if available).
#  - Asynchronous ChangeSet submission via asubmit_changeset(...) (no commit step).
#
# Design:
#  - Strict Pydantic v2 validation (fail-closed).
#  - Respect org/user context from UIRequest for audit + idempotency (best-effort).
#  - Defensive against backend variations (Asset model vs. dict fallback).
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import contextvars
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator

# LangChain (Tools agent)
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Project contracts & services
from workshops_ai.settings import get_settings
from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.app.services.ws_hub import WsHub
from workshops_ai.app.services.tazila_client import asubmit_changeset

# ChangeSet (canonical write contract)
from workshops_ai.models.changeset import (
    ChangeSet,
    ActionSetRisks,
    RiskItem,
)

# Backend port (state only: assets etc.)
from workshops_ai.backend import port as backend

# Read-only catalogs (normalize/search/explain)
from workshops_ai.tools import search_catalogs as cat

log = logging.getLogger("workshops_ai.threat_agent")

# Context for org/user propagation into tools & changesets
_THREAT_CTX = contextvars.ContextVar("threat_ctx", default={"org_id": "", "user_id": ""})


# ---------------------------------------------------------------------------
# Strict tool-argument schemas
# ---------------------------------------------------------------------------

class ThreatListArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    q: Optional[str] = Field(None, description="Free-text query (optional)")
    category: Optional[str] = Field(None, description="Threat category code or alias (normalized)")
    apply_asset_type: Optional[str] = Field(
        None, description="Filter threats relevant to a given asset type (EN code)"
    )


class ExplainThreatArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    threat_code: str = Field(..., description="Canonical threat code (EN) or alias (normalized)")
    asset_id: Optional[str] = Field(
        None, description="Optional asset_id to tailor the explanation"
    )


class RiskSetArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    asset_id: str = Field(..., min_length=1)
    threat_code: str = Field(..., min_length=1)
    probability: int = Field(..., ge=0, le=4, description="0..4 scale")
    note: Optional[str] = None

    @field_validator("threat_code")
    @classmethod
    def _strip_code(cls, v: str) -> str:
        return v.strip()


class BulkRiskSetArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    items: list[RiskSetArgs] = Field(..., min_length=1, max_length=500)


class UpsertThreatArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    # If code is None → create; if code exists → upsert/update.
    code: Optional[str] = Field(None, description="Canonical threat code (EN)")
    name: str = Field(..., min_length=2, max_length=200)
    category: str = Field(..., min_length=2, max_length=100, description="Threat category code (EN)")
    asset_type_codes: list[str] = Field(
        default_factory=list,
        description="Relevant asset-type codes (EN); empty = generic"
    )
    intentional: bool = False
    unintentional: bool = False
    environmental: bool = False
    description: Optional[str] = Field(None, max_length=4000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """UTC ISO8601 timestamp for audit records."""
    return datetime.now(timezone.utc).isoformat()


def _truthy(val: Any) -> bool:
    """Coerce common truthy strings to bool (supports 'ano' as yes)."""
    return str(val).strip().lower() in {"1", "true", "t", "yes", "y", "ano"}


async def _ws_emit_risk_set_probability(item: dict) -> None:
    """
    Best-effort WS emit for risk probability change.
    Prefer schema-validated emitter if available, otherwise fallback to generic broadcast.
    """
    try:
        if hasattr(WsHub, "aemit_risk_set_probability"):
            await WsHub.aemit_risk_set_probability(item)  # validated path
        else:
            await WsHub.broadcast("risk.set_probability", item)  # generic path
    except Exception:
        log.exception("WS risk.set_probability emit failed")


def _asset_type_of(a: Any) -> Optional[str]:
    """
    Extract asset type code from various backend shapes (pydantic/dict).
    Accepts attributes like .assetType or dict['assetType'] / ['type'].
    """
    try:
        if hasattr(a, "assetType"):
            return getattr(a, "assetType") or None
        if isinstance(a, dict):
            return a.get("assetType") or a.get("type") or None
    except Exception:
        pass
    return None


def _backend_get_asset(asset_id: str) -> Any:
    """
    Defensive asset getter — supports multiple backend method names and
    never raises if the port does not implement a getter.
    """
    getter_names = ["asset_get", "get_asset", "asset_by_id", "get_asset_by_id"]
    for name in getter_names:
        fn: Optional[Callable] = getattr(backend, name, None)  # type: ignore[attr-defined]
        if callable(fn):
            try:
                return fn(asset_id)
            except Exception:
                log.exception("backend.%s(%s) failed", name, asset_id)
                # Try next candidate instead of stopping early
                continue
    return None


def _mk_changeset_for_risks(*, items_by_asset: dict[str, list[RiskItem]]) -> ChangeSet:
    """
    Build a ChangeSet with one or more ActionSetRisks, grouped by asset_id.
    org_id/user_id are taken from the context set in run().
    """
    ctx = _THREAT_CTX.get()
    org_id = ctx.get("org_id", "")
    user_id = ctx.get("user_id", "")

    ts = _now_iso()
    idem_seed = "|".join(
        [org_id, "risk.set_probability", ts, str(len(items_by_asset))] +
        [f"{aid}:{len(items)}" for aid, items in sorted(items_by_asset.items())]
    )
    idem_key = str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://workshops.ai/idem/{idem_seed}"))

    actions: list[ActionSetRisks] = []
    for asset_id, risk_items in items_by_asset.items():
        actions.append(
            ActionSetRisks(
                op="set_risks",
                mode="merge",
                asset_id=asset_id,
                items=risk_items,
            )
        )

    audit = {
        "tags": ["AI", "batch-threats"],
        "source": "AI",
        "userId": user_id,
        "ts": ts,
    }

    return ChangeSet(
        idempotency_key=idem_key,
        org_id=org_id,
        intent="apply",
        audit=audit,
        actions=actions,
    )


def _to_risk_item(asset_id: str, threat_code: str, probability: int, note: Optional[str]) -> RiskItem:
    """Construct a RiskItem (typed server contract)."""
    return RiskItem(threat_code=threat_code, probability=probability, note=note)


# ---------------------------------------------------------------------------
# Tools (read)
# ---------------------------------------------------------------------------

@tool("list_threats", args_schema=ThreatListArgs)
def tool_list_threats(
    q: Optional[str] = None,
    category: Optional[str] = None,
    apply_asset_type: Optional[str] = None,
) -> dict:
    """
    Return a filtered list of catalog threats (read-only).
    Uses search_catalogs (fuzzy/aliases, respects Asset type × Threat mapping).
    """
    items = cat.list_threats(q=q, category=category, apply_asset_type=apply_asset_type, limit=500)
    return {
        "count": len(items),
        "items": [
            {
                "code": t.get("code"),
                "name": t.get("name"),
                "category": t.get("category"),
                "flags": t.get("flags") or {},
                "asset_types": t.get("asset_types") or [],
                "description": t.get("description"),
            }
            for t in items
        ],
    }


@tool("explain_threats", args_schema=ExplainThreatArgs)
def tool_explain_threats(threat_code: str, asset_id: Optional[str] = None) -> str:
    """
    Explain why a specific threat is relevant; optionally tailored to a given asset.
    Pulls from catalog metadata and asset properties; avoids hallucinations.
    """
    norm_code = cat.normalize_threat_code(threat_code) or threat_code
    t = cat.explain_threat(norm_code)
    if not t:
        return f"Threat '{threat_code}' was not found in the catalog."

    lines: list[str] = [f"Threat: {t.get('code')} — {t.get('name')}"]
    flags = [k for k, v in (t.get("flags") or {}).items() if _truthy(v)]
    if t.get("category"):
        lines.append(f"- Category: {t['category']}")
    if flags:
        lines.append(f"- Nature: {', '.join(flags)}")
    if t.get("asset_types"):
        lines.append(f"- Often mapped to asset types: {', '.join(t['asset_types'])}")

    if asset_id:
        a = _backend_get_asset(asset_id)
        atype = _asset_type_of(a) if a else None
        if atype and t.get("asset_types") and atype in set(t["asset_types"]):
            lines.append(f"- This asset is type '{atype}', which is mapped to this threat.")
        elif atype:
            lines.append(f"- Asset type '{atype}' is not explicitly mapped; consider contextual relevance.")
        else:
            lines.append("- Asset not found or has no type; showing a generic rationale.")

    if t.get("description"):
        desc = str(t["description"]).strip()
        if desc:
            lines.append(f"- Catalog note: {desc[:600]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools (write — SEND-NOW)
# ---------------------------------------------------------------------------

@tool("set_risk_probability", args_schema=RiskSetArgs)
async def tool_set_risk_probability(
    asset_id: str,
    threat_code: str,
    probability: int,
    note: Optional[str] = None,
) -> dict:
    """
    Set probability (0..4) that a threat may materialize for a given asset.
    Emits a WS event and submits a ChangeSet asynchronously.
    """
    norm_code = cat.normalize_threat_code(threat_code) or threat_code

    item_wire = {
        "asset_id": asset_id,
        "threat_code": norm_code,
        "probability": probability,
        **({"note": note} if note else {}),
    }

    # 1) Optimistic WS update so the UI reacts immediately
    await _ws_emit_risk_set_probability(item_wire)

    # 2) Build ChangeSet (single-asset group) using context org/user
    risk_item = _to_risk_item(asset_id, norm_code, probability, note)
    cs = _mk_changeset_for_risks(items_by_asset={asset_id: [risk_item]})
    await asubmit_changeset(cs)

    return {"status": "ok", "applied": item_wire}


@tool("bulk_set_risk_probability", args_schema=BulkRiskSetArgs)
async def tool_bulk_set_risk_probability(items: list[RiskSetArgs]) -> dict:
    """
    Bulk probability updates in a single ChangeSet (grouped per asset_id).
    """
    wire_items: list[dict] = []
    grouped: dict[str, list[RiskItem]] = {}
    for it in items:
        norm_code = cat.normalize_threat_code(it.threat_code) or it.threat_code
        wi = {
            "asset_id": it.asset_id,
            "threat_code": norm_code,
            "probability": it.probability,
            **({"note": it.note} if it.note else {}),
        }
        wire_items.append(wi)
        grouped.setdefault(it.asset_id, []).append(
            _to_risk_item(it.asset_id, norm_code, it.probability, it.note)
        )

    # Optimistic fan-out to WS for all items
    await asyncio.gather(*(_ws_emit_risk_set_probability(wi) for wi in wire_items))

    cs = _mk_changeset_for_risks(items_by_asset=grouped)
    await asubmit_changeset(cs)

    return {"status": "ok", "count": len(wire_items)}


@tool("upsert_threat", args_schema=UpsertThreatArgs)
async def tool_upsert_threat(
    code: Optional[str],
    name: str,
    category: str,
    asset_type_codes: list[str],
    intentional: bool = False,
    unintentional: bool = False,
    environmental: bool = False,
    description: Optional[str] = None,
) -> dict:
    """
    Create or update a threat in the org context (server-side validates/upserts).
    Note: This is a SEND-NOW write — backend remains the source of truth.
    """
    ctx = _THREAT_CTX.get()
    org_id = ctx.get("org_id", "")
    user_id = ctx.get("user_id", "")

    norm_code = cat.normalize_threat_code(code) if code else None
    item = {
        **({"code": norm_code} if norm_code else {}),
        "name": name,
        "category": category,
        "asset_type_codes": asset_type_codes or [],
        "flags": {
            "intentional": bool(intentional),
            "unintentional": bool(unintentional),
            "environmental": bool(environmental),
        },
        **({"description": description} if description else {}),
    }

    # Optimistic WS emit (generic channel, UI can refetch detail)
    try:
        if hasattr(WsHub, "emit"):
            await WsHub.emit("threat.upsert", item)
        else:
            await WsHub.broadcast("threat.upsert", item)
    except Exception:
        log.exception("WS threat.upsert emit failed")

    # Use plural op + array payload; backend validates shape
    cs = ChangeSet(
        idempotency_key=str(uuid.uuid4()),
        org_id=org_id,
        intent="apply",
        audit={"tags": ["AI", "threat-upsert"], "source": "AI", "userId": user_id, "ts": _now_iso()},
        actions=[
            {  # type: ignore[typeddict-item]
                "op": "upsert_threats",
                "items": [item],
            }
        ],  # type: ignore[list-item]
    )
    await asubmit_changeset(cs)

    return {"status": "ok", "upserted": item.get("code", name)}


# ---------------------------------------------------------------------------
# Agent bootstrap
# ---------------------------------------------------------------------------

def _resolve_llm_config() -> tuple[str, int, int]:
    """
    Resolve LLM settings with backward/forward compatibility.
    Prefers nested settings (s.llm.model / s.llm.http_timeout_sec / s.llm.openai_max_retries),
    but supports legacy flat attributes (s.llm_model / s.openai_timeout_sec / s.openai_max_retries).
    Provides safe defaults for local/dev runs.
    """
    s = get_settings()

    # Model
    model = getattr(s, "llm_model", None)
    if not model:
        llm = getattr(s, "llm", None)
        model = getattr(llm, "model", None) if llm else None
    if not model:
        model = "gpt-4o-mini"  # dev default

    # Timeout
    timeout = getattr(s, "openai_timeout_sec", None)
    if timeout is None:
        llm = getattr(s, "llm", None)
        timeout = getattr(llm, "http_timeout_sec", None) if llm else None
    if timeout is None:
        timeout = 60

    # Retries
    retries = getattr(s, "openai_max_retries", None)
    if retries is None:
        llm = getattr(s, "llm", None)
        retries = getattr(llm, "openai_max_retries", None) if llm else None
    if retries is None:
        retries = 2

    return str(model), int(timeout), int(retries)


def _build_executor() -> AgentExecutor:
    """Create an OpenAI Tools agent with our tools and a safe system prompt."""
    model, timeout, retries = _resolve_llm_config()

    s = get_settings()
    llm_cfg = getattr(s, "llm", None)
    api_key = (
        getattr(llm_cfg, "openai_api_key", None)
        or getattr(s, "openai_api_key", None)
    )

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        timeout=timeout,
        max_retries=retries,
        api_key=api_key,  # pull API key from settings
    )

    tools = [
        tool_list_threats,
        tool_explain_threats,
        tool_set_risk_probability,
        tool_bulk_set_risk_probability,
        tool_upsert_threat,
    ]

    # No {tools} variable in the template (avoids KeyError in LC 0.2).
    system = (
        "You are ThreatAgent for the 'threats' tab in WorkshopsAI.\n"
        "RULES:\n"
        "- Detect and use the user's language for messages; keep catalog codes in EN.\n"
        "- Probability scale is integers 0..4 only.\n"
        "- For any write, ALWAYS call the provided tools (do not invent fields).\n"
        "- Be concise; when you change data, say what changed and why.\n"
        "- Do not modify assets/impacts/controls here.\n"
    )

    # Provide explicit context (asset_id/asset_type) so the agent can apply filters.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("system",
             "Context:\n"
             "- org_id: {org_id}\n"
             "- user_id: {user_id}\n"
             "- selection_asset_id: {selection_asset_id}\n"
             "- selection_asset_type: {selection_asset_type}\n"),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run(req: UIRequest) -> UIResponse:
    """
    Entry point for the Manager/Router.
    - Accepts UIRequest, runs the LC agent with our tools, returns UIResponse.
    - SEND-NOW: write tools already perform changes (WS + outbox), so no commit needed.
    """
    if (req.active_tab or "").lower() != "threats":
        # Fail-closed: this agent only handles the 'threats' tab
        return UIResponse(
            messages=[f"This agent handles only the 'threats' tab (got: {req.active_tab})."],
            staging={"needs_commit": False},
            policy={"emit_changes_enabled": True},
        )

    # Set org/user context for downstream tools & changesets
    _THREAT_CTX.set({"org_id": req.org_id or "", "user_id": req.user_id or ""})

    executor = _build_executor()

    # Try to resolve asset_type from backend so the agent can use it in tool calls
    selection_asset_id: Optional[str] = None
    selection_asset_type: Optional[str] = None
    if req.selection and getattr(req.selection, "asset_id", None):
        selection_asset_id = req.selection.asset_id
        try:
            a = _backend_get_asset(selection_asset_id)
            selection_asset_type = _asset_type_of(a) if a else None
        except Exception:
            log.exception("Failed to resolve asset type for asset_id=%s", selection_asset_id)

    try:
        result = await executor.ainvoke(
            {
                "input": req.message or "Help",
                "chat_history": [],
                "org_id": req.org_id,
                "user_id": req.user_id,
                "selection_asset_id": selection_asset_id or "",
                "selection_asset_type": selection_asset_type or "",
            }
        )
        text = (result.get("output") or "").strip() or "Done."
    except Exception as e:
        log.exception("ThreatAgent failed")
        text = f"ThreatAgent error: {e!s}"

    return UIResponse(
        messages=[text],
        staging={"needs_commit": False},
        policy={"emit_changes_enabled": True},
    )


__all__ = [
    "run",
    "tool_list_threats",
    "tool_explain_threats",
    "tool_set_risk_probability",
    "tool_bulk_set_risk_probability",
    "tool_upsert_threat",
]
