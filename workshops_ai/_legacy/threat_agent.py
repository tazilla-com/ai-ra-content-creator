# File: workshops_ai/agents/threat_agent.py
# -----------------------------------------------------------------------------
# ThreatAgent — LangChain Tools agent for the "threats" tab (SEND-NOW semantics)
#
# What's new vs. previous version:
#  1) FAST READ PATH: list-only requests are answered without LLM (no chance to hang).
#  2) STEP TIMEOUTS: each external step is wrapped with its own timeout + telemetry.
#  3) WATCHDOG: prints full thread stack traces ~10s before the manager's timeout.
#  4) Strict, async-safe writes (WS optimistic emit + async ChangeSet submit).
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
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
from workshops_ai.tools.telemetry import telemetry_log

log = logging.getLogger("workshops_ai.threat_agent")

# Context for org/user propagation into tools & changesets
_THREAT_CTX = contextvars.ContextVar("threat_ctx", default={"org_id": "", "user_id": ""})


# -----------------------------------------------------------------------------
# Debug helpers: watchdog + per-step timeout wrappers
# -----------------------------------------------------------------------------

def _now() -> float:
    return time.time()


def _arm_stack_dump(delay_sec: int = 110) -> None:
    """
    If the agent gets stuck, dump all thread stacks to stderr before the router's
    typical 120 s deadline. This is no-op if the process responds earlier.
    """
    def _dump():
        try:
            import sys, faulthandler
            time.sleep(delay_sec)
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            telemetry_log("threat_agent.watchdog_dump", {"delay_sec": delay_sec})
        except Exception:
            # Never crash the agent because of debugging helpers
            pass

    threading.Thread(target=_dump, daemon=True).start()


async def _with_step_timeout(coro, *, step: str, timeout_sec: int):
    """
    Wrap a coroutine with a per-step timeout and structured telemetry.
    Raises asyncio.TimeoutError on timeout; caller decides the fallback.
    """
    t0 = _now()
    telemetry_log("threat_agent.step.start", {"step": step, "timeout_sec": timeout_sec})
    try:
        res = await asyncio.wait_for(coro, timeout=timeout_sec)
        telemetry_log("threat_agent.step.ok", {"step": step, "dt": _now() - t0})
        return res
    except asyncio.TimeoutError:
        telemetry_log("threat_agent.step_timeout", {"step": step, "timeout_sec": timeout_sec})
        raise


# -----------------------------------------------------------------------------
# Strict tool-argument schemas
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truthy(val: Any) -> bool:
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
        "user_id": user_id,
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


# -----------------------------------------------------------------------------
# Tools (read)
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Tools (write — SEND-NOW)
# -----------------------------------------------------------------------------

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

    cs = ChangeSet(
        idempotency_key=str(uuid.uuid4()),
        org_id=org_id,
        intent="apply",
        audit={"tags": ["AI", "threat-upsert"], "source": "AI", "user_id": user_id, "ts": _now_iso()},
        actions=[
            {  # type: ignore[typeddict-item]
                "op": "upsert_threats",
                "items": [item],
            }
        ],  # type: ignore[list-item]
    )
    await asubmit_changeset(cs)

    return {"status": "ok", "upserted": item.get("code", name)}


# -----------------------------------------------------------------------------
# Agent bootstrap
# -----------------------------------------------------------------------------

def _resolve_llm_config() -> tuple[str, int, int, Optional[str]]:
    """
    Resolve LLM settings with backward/forward compatibility.
    Returns (model, timeout_sec, retries, api_key_or_none).
    """
    s = get_settings()

    # Model
    model = getattr(getattr(s, "llm", None), "model", None) or getattr(s, "llm_model", None) or "gpt-4o-mini"

    # Timeout
    timeout = (
        getattr(getattr(s, "llm", None), "call_timeout_sec", None)
        or getattr(s, "openai_timeout_sec", None)
        or 60
    )

    # Retries
    retries = (
        getattr(getattr(s, "llm", None), "openai_max_retries", None)
        or getattr(s, "openai_max_retries", None)
        or 2
    )

    # API key (explicit if present; LangChain will also read env OPENAI_API_KEY)
    api_key = (
        getattr(getattr(s, "llm", None), "openai_api_key", None)
        or getattr(s, "openai_api_key", None)
        or None
    )

    return str(model), int(timeout), int(retries), api_key


def _build_executor() -> AgentExecutor:
    """Create an OpenAI Tools agent with our tools and a safe system prompt."""
    model, timeout, retries, api_key = _resolve_llm_config()

    llm = ChatOpenAI(
        model=model,
        temperature=0.2,
        timeout=timeout,
        max_retries=retries,
        api_key=api_key,  # if None, langchain_openai will fall back to env
    )

    tools = [
        tool_list_threats,
        tool_explain_threats,
        tool_set_risk_probability,
        tool_bulk_set_risk_probability,
        tool_upsert_threat,
    ]

    system = (
        "You are ThreatAgent for the 'threats' tab in WorkshopsAI.\n"
        "RULES:\n"
        "- Detect and use the user's language for messages; keep catalog codes in EN.\n"
        "- Probability scale is integers 0..4 only.\n"
        "- For any write, ALWAYS call the provided tools (do not invent fields).\n"
        "- Be concise; when you change data, say what changed and why.\n"
        "- Do not modify assets/impacts/controls here.\n"
    )

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


# -----------------------------------------------------------------------------
# Short-circuit (fast read) utilities
# -----------------------------------------------------------------------------

def _is_list_only(message: str | None) -> bool:
    """
    Heuristic for 'list mapped threats' style instructions.
    We deliberately avoid LLM for this to guarantee snappy, deterministic replies.
    """
    if not message:
        return False
    m = message.lower()
    # Conservative: look for "list" (or CZ 'seznam') + explicit 'do not change' / 'read-only'.
    return (
        ("list" in m or "seznam" in m or "mapped" in m)
        and ("do not change" in m or "read-only" in m or "bez změn" in m or "bez zmen" in m)
    )


async def _list_threats_fast(asset_id: Optional[str], asset_type: Optional[str]) -> list[str]:
    """
    Pure read path. No LLM. Uses catalogs directly and filters by asset_type when provided.
    """
    items = cat.list_threats(q=None, category=None, apply_asset_type=asset_type, limit=200)
    codes = [t.get("code") for t in items if t.get("code")]
    return codes[:50]  # keep output compact for UI


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

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

    # Arm watchdog ~10s before typical 120 s manager timeout
    _arm_stack_dump(delay_sec=110)

    # Set org/user context for downstream tools & changesets
    _THREAT_CTX.set({"org_id": req.org_id or "", "user_id": req.user_id or ""})

    # Resolve selection context (asset_id → asset_type)
    selection_asset_id: Optional[str] = None
    selection_asset_type: Optional[str] = None
    if req.selection and getattr(req.selection, "asset_id", None):
        selection_asset_id = req.selection.asset_id
        try:
            a = _backend_get_asset(selection_asset_id)
            selection_asset_type = _asset_type_of(a) if a else None
        except Exception:
            log.exception("Failed to resolve asset type for asset_id=%s", selection_asset_id)

    # FAST PATH: list-only requests → bypass LLM completely
    if _is_list_only(req.message):
        try:
            codes = await _with_step_timeout(
                _list_threats_fast(selection_asset_id, selection_asset_type),
                step="list_only",
                timeout_sec=20,
            )
            msg = ", ".join(codes) if codes else "No mapped threats found."
            return UIResponse(
                messages=[msg],
                staging={"needs_commit": False},
                policy={"emit_changes_enabled": True},
            )
        except asyncio.TimeoutError:
            # Fall through to LLM path as a last resort
            telemetry_log("threat_agent.list_only_timeout_fallback", {})

    # LLM path (tools agent)
    executor = _build_executor()

    try:
        # Give the agent a reasonable step timeout that's below manager's ceiling
        result = await _with_step_timeout(
            executor.ainvoke(
                {
                    "input": req.message or "Help",
                    "chat_history": [],
                    "org_id": req.org_id,
                    "user_id": req.user_id,
                    "selection_asset_id": selection_asset_id or "",
                    "selection_asset_type": selection_asset_type or "",
                }
            ),
            step="llm.ainvoke",
            timeout_sec=90,  # independent guard; manager has its own global timeout
        )
        text = (result.get("output") or "").strip() or "Done."
    except asyncio.TimeoutError:
        telemetry_log("threat_agent.invoke_timeout", {"timeout_sec": 90})
        text = "ThreatAgent timed out while processing your request."
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
