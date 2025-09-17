# File: workshops_ai/agents/asset_agent.py
# Role: AssetAgent (tab: "assets") — LangChain-based AI agent (NO BUFFER, send-now semantics)
# Mode: EMIT-ONLY WRITES (always enabled)
# - Capabilities: list/detail (read-only) + NL → (create_asset | set_asset_meta)
# - On every change the agent:
#     1) emits a WS JSON event via WsHub (aemit_change_event)
#     2) builds a single-action ChangeSet and IMMEDIATELY submits it to BE outbox (asubmit_changeset) asynchronously

from __future__ import annotations

import asyncio
import inspect
import os
import re
import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.tools.telemetry import telemetry_log

from workshops_ai.backend.port import BackendPort
from workshops_ai.backend.csv_backend import CsvBackend  # type: ignore

# ChangeSet & outbox
from workshops_ai.models.changeset import (
    ActionCreateAsset,
    ActionSetAssetMeta,
    ChangeSet,
)

from workshops_ai.agents.utils.changeset_builder import build_changeset

# Outbox submit (async, non-blocking for WS/LLM loop)
from workshops_ai.app.services.tazila_client import asubmit_changeset  # type: ignore

# WS change events (raw JSON broadcast via WsHub → /ws)
from workshops_ai.app.services.change_events import aemit_change_event

logger = logging.getLogger(__name__)

# Try to import LangChain/OpenAI at runtime; keep the agent usable without LLM.
_HAS_LLM = True
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.pydantic_v1 import BaseModel as LCBaseModel, Field as LCField
except Exception:  # pragma: no cover
    _HAS_LLM = False
    # Minimal shims so the file imports cleanly even without LangChain installed.
    class LCBaseModel(BaseModel):  # type: ignore
        pass
    def LCField(*args, **kwargs):  # type: ignore
        return None
    ChatOpenAI = object  # type: ignore
    HumanMessage = SystemMessage = object  # type: ignore

Intent = Literal["read", "explain"]
READ_PAGE_LIMIT = 100


# --------------------------------------------------------------------------------------
# LangChain "structured tools" — the AssetAgent allows exactly two actions
# --------------------------------------------------------------------------------------

class CreateAssetArgs(LCBaseModel):
    """Create a new asset via ChangeSet action."""
    name: str = LCField(..., description="Human-readable asset name, e.g., 'CRM'")
    asset_type_code: str = LCField(..., description="Canonical asset type code in EN, e.g., 'Information system'")
    primary: Optional[bool] = LCField(default=None, description="Whether the asset is primary")
    description: Optional[str] = LCField(default=None, description="Short description")
    # Optional foreign keys (UUIDs) if provided by the user
    asset_type_uuid: Optional[str] = LCField(default=None)
    location_uuid: Optional[str] = LCField(default=None)
    owner_uuid: Optional[str] = LCField(default=None)
    supplier_type: Optional[str] = LCField(default=None, description="INTERNAL or EXTERNAL")
    supplier_uuid: Optional[str] = LCField(default=None)
    operator_type: Optional[str] = LCField(default=None, description="INTERNAL or EXTERNAL")
    operator_uuid: Optional[str] = LCField(default=None)
    essential_services_uuids: Optional[list[str]] = LCField(default=None)
    agreement_uuids: Optional[list[str]] = LCField(default=None)

def create_asset_tool(**kwargs) -> dict:
    """
    Tool shim: Prepare a create_asset action (no side effects).
    Return the validated dict of fields that should be used to build the ChangeSet.
    """
    return kwargs
create_asset_tool.name = "create_asset"  # ensure name for LC tool registry
create_asset_tool.args_schema = CreateAssetArgs


class SetAssetMetaArgs(LCBaseModel):
    """Update metadata of an existing asset via ChangeSet action."""
    asset_id: str = LCField(..., description="Target asset id/uuid to update")
    name: Optional[str] = LCField(default=None)
    primary: Optional[bool] = LCField(default=None)
    description: Optional[str] = LCField(default=None)
    location_uuid: Optional[str] = LCField(default=None)
    owner_uuid: Optional[str] = LCField(default=None)
    supplier_type: Optional[str] = LCField(default=None, description="INTERNAL or EXTERNAL")
    supplier_uuid: Optional[str] = LCField(default=None)
    operator_type: Optional[str] = LCField(default=None, description="INTERNAL or EXTERNAL")
    operator_uuid: Optional[str] = LCField(default=None)
    asset_type_uuid: Optional[str] = LCField(default=None)
    asset_type_code: Optional[str] = LCField(default=None)
    essential_services_uuids: Optional[list[str]] = LCField(default=None)
    agreement_uuids: Optional[list[str]] = LCField(default=None)

def set_asset_meta_tool(**kwargs) -> dict:
    """
    Tool shim: Prepare a set_asset_meta action (no side effects).
    Return the validated dict of fields that should be used to build the ChangeSet.
    """
    return kwargs
set_asset_meta_tool.name = "set_asset_meta"
set_asset_meta_tool.args_schema = SetAssetMetaArgs


def _llm() -> ChatOpenAI:
    if not _HAS_LLM:
        raise RuntimeError("LangChain/OpenAI not installed")
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")):
        raise RuntimeError("OPENAI_API_KEY/AZURE_OPENAI_API_KEY not configured")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    timeout = int(os.getenv("OPENAI_TIMEOUT_SEC", "60"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    return ChatOpenAI(model=model, temperature=temperature, timeout=timeout, max_retries=max_retries)  # type: ignore[call-arg]


LC_SYSTEM_PROMPT = """You are AssetAgent for the 'assets' tab (EMIT-ONLY WRITES).
You operate in Czech when the user writes Czech; otherwise English.

GOALS:
1) Always start with ONE short, human-friendly sentence (this will be shown in chat).
2) For changes, use exactly ONE tool call (create_asset or set_asset_meta).
3) For reads (listing/detail) DO NOT call tools and DO NOT print control lines in chat.
   Instead, put the control instruction INSIDE an XML tag on a NEW line:

   <control>READ_LIST [search="<text>"] [type="<asset_type_code>"] [is_primary=true|false]</control>
   or
   <control>READ_DETAIL id="<asset_id>"</control>
   or
   <control>NONE</control>

RULES:
- Tool calls: only create_asset / set_asset_meta; never invent IDs/UUIDs.
- Use canonical EN codes for asset_type_code.
- Never propose impacts/threats/controls here.
- Keep messages concise: one short sentence. The <control> tag is separate and NOT part of the chat output.

IMPORTANT:
- This agent emits ChangeSet JSON and WS events; it doesn’t apply changes directly.
- Reading uses CSV/state only; newly created/updated assets may not immediately appear in lists.
"""


# --------------------------------------------------------------------------------------
# Presentation models for proposals (list/detail)
# --------------------------------------------------------------------------------------

class AssetPreview(BaseModel):
    id: str
    name: str
    assetType: str
    isPrimary: bool | None = None
    confidentiality: int | None = None
    integrity: int | None = None
    availability: int | None = None

class AssetDetail(AssetPreview):
    description: str | None = None
    org_id: str | None = None


# --------------------------------------------------------------------------------------
# Agent entrypoint
# --------------------------------------------------------------------------------------

async def run(req: UIRequest, intent: Intent) -> UIResponse:
    """AI agent: list/detail + NL→Actions (assets-only) with immediate outbox submit."""
    _audit_in(req, intent)

    if intent == "explain":
        return _explain_response()

    msg = (req.message or "").strip()

    # 1) Invoke LLM; if unavailable, gracefully fall back to read-only heuristics.
    try:
        ai_msg = await _ai_invoke(msg)
    except Exception as e:
        telemetry_log("asset_agent.llm_unavailable", {"error": str(e)})
        # Fallback to a simple read control (still respond with a short human sentence)
        parsed = _interpret_ai_read_control(_fallback_read_control(msg))
        return await _handle_read_controls(req, parsed, ai_fallback_text="Here’s what I can show you right now.")

    # 1a) If tools were called, emit WS events and immediately submit a single-action ChangeSet
    tool_calls = _normalize_tool_calls(ai_msg)
    if tool_calls:
        submitted = await _handle_tool_calls(req, tool_calls)
        if submitted > 0:
            return UIResponse(
                messages=[f"Done — {submitted} change(s) queued to the outbox."],
                proposals=None,
                staging={"needs_commit": False},
            )
        # Tools were present but nothing valid was created
        return UIResponse(
            messages=["I couldn't build a valid change from that request. Try: “Create an asset …” or “Update asset …”."],
            proposals=None,
            staging={"needs_commit": False},
        )

    # 1b) No tools: parse hidden control tag and show only the human sentence
    raw_content = (getattr(ai_msg, "content", None) or "").strip()
    control_text, human_text = _extract_control_tag_and_human_text(raw_content)

    if not control_text or control_text.upper().startswith("NONE"):
        # Just a friendly sentence, no read instruction — return it as-is.
        return UIResponse(messages=[human_text or "OK."], proposals=None, staging={"needs_commit": False})

    # Interpret hidden READ_* control and return human sentence + proposals
    parsed = _interpret_ai_read_control(control_text)
    return await _handle_read_controls(req, parsed, ai_fallback_text=(human_text or None))


# --------------------------------------------------------------------------------------
# AI helpers
# --------------------------------------------------------------------------------------

async def _ai_invoke(message: str):
    # Bind tools for schema exposure; we parse tool calls ourselves.
    llm = _llm().bind_tools([create_asset_tool, set_asset_meta_tool])  # type: ignore[attr-defined]
    return await llm.ainvoke(  # type: ignore[call-arg]
        [
            SystemMessage(content=LC_SYSTEM_PROMPT),  # type: ignore[arg-type]
            HumanMessage(content=message or "list assets"),  # type: ignore[arg-type]
        ]
    )

def _normalize_tool_calls(ai_msg: Any) -> list[dict[str, Any]]:
    """
    Normalize tool calls from an AIMessage to a list of {name, args}.
    Works across LC versions.
    """
    calls: list[dict[str, Any]] = []
    try:
        tc = getattr(ai_msg, "tool_calls", None)
        if tc and isinstance(tc, Sequence):
            for c in tc:
                name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)
                args = getattr(c, "args", None) or (c.get("args") if isinstance(c, dict) else None)
                if name and isinstance(args, dict):
                    calls.append({"name": name, "args": args})
    except Exception:
        logger.debug("No tool_calls on AI message; falling back to additional_kwargs.", exc_info=True)

    if not calls:
        try:
            ak = getattr(ai_msg, "additional_kwargs", {}) or {}
            for c in ak.get("tool_calls", []) or []:
                name = c.get("name")
                args = c.get("args") or {}
                if name and isinstance(args, dict):
                    calls.append({"name": name, "args": args})
        except Exception:
            logger.debug("No additional_kwargs.tool_calls present.", exc_info=True)

    return calls


# --------------------------------------------------------------------------------------
# Extract hidden control tag
# --------------------------------------------------------------------------------------

_CONTROL_TAG = re.compile(r"<control>\s*(?P<ctl>.*?)\s*</control>", re.IGNORECASE | re.DOTALL)

def _extract_control_tag_and_human_text(content: str) -> tuple[str | None, str]:
    """
    Returns (control_text_or_none, human_text_without_tag).
    The human text is the content with the <control>...</control> removed and stripped.
    """
    if not content:
        return None, ""
    m = _CONTROL_TAG.search(content)
    control = m.group("ctl").strip() if m else None
    human = (_CONTROL_TAG.sub("", content)).strip()
    return (control if control else None, human)


# --------------------------------------------------------------------------------------
# WS events + immediate single-action ChangeSet submit; returns submitted_count
# --------------------------------------------------------------------------------------

async def _handle_tool_calls(req: UIRequest, tool_calls: Sequence[dict[str, Any]]) -> int:
    submitted = 0

    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("args") or {}

        if name == "create_asset":
            if "name" not in args or "asset_type_code" not in args:
                continue
            client_token = f"tmp-{_slug(args.get('name', 'item'))}"

            # map 'primary' -> 'isPrimary' to match downstream schema
            fields = {k: v for k, v in args.items()}
            if "primary" in fields:
                fields["isPrimary"] = fields.pop("primary")

            action = ActionCreateAsset(
                op="create_asset",
                client_token=client_token,
                fields=fields,
            ).model_dump()

            # 1) WS change event (user sees the change immediately)
            await aemit_change_event(
                "asset.create",
                org_id=req.org_id,
                user_id=req.user_id,
                data=action,
                asset_id=None,
            )

            # 2) Single-action ChangeSet → immediate async submit (non-blocking)
            cs = _make_changeset_one(req, action)
            asyncio.create_task(_submit_now(cs))
            submitted += 1

        elif name == "set_asset_meta":
            asset_id = args.get("asset_id")
            if not asset_id:
                continue

            # map 'primary' -> 'isPrimary' and drop None values
            fields = {k: v for k, v in args.items() if k != "asset_id" and v is not None}
            if "primary" in fields:
                fields["isPrimary"] = fields.pop("primary")
            if not fields:
                continue

            action = ActionSetAssetMeta(
                op="set_asset_meta",
                asset_id=asset_id,
                fields=fields,
            ).model_dump()

            # 1) WS change event
            await aemit_change_event(
                "asset.set_meta",
                org_id=req.org_id,
                user_id=req.user_id,
                data=action,
                asset_id=asset_id,
            )

            # 2) Single-action ChangeSet → immediate async submit
            cs = _make_changeset_one(req, action)
            asyncio.create_task(_submit_now(cs))
            submitted += 1

        # ignore any other tool (tool-gated)
    return submitted


# --------------------------------------------------------------------------------------
# READ control handling
# --------------------------------------------------------------------------------------

_CONTROL_READ_LIST = re.compile(
    r"""^READ_LIST(?:\s+search="(?P<search>[^"]*)")?(?:\s+type="(?P<type>[^"]*)")?(?:\s+is_primary=(?P<primary>true|false))?\s*$""",
    re.IGNORECASE,
)
_CONTROL_READ_DETAIL = re.compile(
    r"""^READ_DETAIL\s+id="(?P<id>[^"]+)"\s*$""",
    re.IGNORECASE,
)

def _interpret_ai_read_control(content: str) -> Optional[tuple[str, dict[str, Any]]]:
    if not content:
        return None
    m = _CONTROL_READ_LIST.match(content)
    if m:
        gd = m.groupdict()
        primary = gd.get("primary")
        return (
            "READ_LIST",
            {
                "search": (gd.get("search") or None),
                "type": (gd.get("type") or None),
                "is_primary": (True if primary == "true" else False if primary == "false" else None),
            },
        )
    m = _CONTROL_READ_DETAIL.match(content)
    if m:
        return ("READ_DETAIL", {"id": m.group("id")})
    return None

async def _handle_read_controls(
    req: UIRequest,
    parsed: Optional[tuple[str, dict[str, Any]]],
    *,
    ai_fallback_text: str | None = None,
) -> UIResponse:
    backend = _get_backend()

    if parsed is None:
        # No recognizable control line; return short hint or LLM text.
        text = ai_fallback_text or "Tell me what you’d like to do with assets (create, update, list, detail)."
        return UIResponse(messages=[text], proposals=None, staging={"needs_commit": False})

    kind, payload = parsed

    if kind == "READ_DETAIL":
        asset_id = payload.get("id") or ""
        if not asset_id:
            return UIResponse(messages=["Please provide an asset ID to show its detail."], proposals=None, staging={"needs_commit": False})
        asset = await _call_async(backend.asset_get, asset_id)
        if asset is None:
            return UIResponse(
                messages=[f"I couldn't find an asset with ID “{asset_id}”."],
                proposals=None,
                staging={"needs_commit": False},
            )
        detail = _to_asset_detail(asset)
        return UIResponse(
            messages=[f"Here’s the asset detail for “{detail.name}”."] ,
            proposals={"asset": detail.model_dump()},
            staging={"needs_commit": False},
        )

    if kind == "READ_LIST":
        filters = Filters(
            search=payload.get("search"),
            type=payload.get("type"),
            is_primary=payload.get("is_primary"),
        )
        items = await _call_async(
            backend.assets_list,
            req.org_id,
            filters.search,
            filters.type,
            filters.is_primary,
        )
        items = _sort_assets(items)[:READ_PAGE_LIMIT]
        previews = [_to_asset_preview(a).model_dump() for a in items]

        parts: list[str] = []
        if filters.search:
            parts.append(f'search="{filters.search}"')
        if filters.type:
            parts.append(f'type="{filters.type}"')
        if filters.is_primary is not None:
            parts.append("primary=true" if filters.is_primary else "primary=false")
        suffix = f" (filters: {', '.join(parts)})" if parts else ""

        return UIResponse(
            messages=[f"I found {len(previews)} item(s){suffix}."],
            proposals={"assets": previews},
            staging={"needs_commit": False},
        )

    # Unknown kind (shouldn't happen)
    return UIResponse(messages=["Nothing to process."], proposals=None, staging={"needs_commit": False})


def _fallback_read_control(message: str) -> str:
    """
    Very small heuristic used only when LLM is unavailable:
    - If message contains an id-like token, prefer READ_DETAIL.
    - Otherwise default to READ_LIST with optional search="...".
    """
    m = message.strip()
    # naive id detection (token with dash or long alnum)
    tokens = re.findall(r"[A-Za-z0-9_-]{6,}", m)
    if tokens:
        return f'READ_DETAIL id="{tokens[0]}"'
    # search:"..." or last word
    search = None
    if "search:" in m.lower():
        search = m.split("search:", 1)[-1].strip()
    elif m and m.lower() not in {"list", "show", "detail"}:
        search = m
    if search:
        return f'READ_LIST search="{search}"'
    return "READ_LIST"


# --------------------------------------------------------------------------------------
# ChangeSet helpers (single-action, send-now)
# --------------------------------------------------------------------------------------

def _make_changeset_one(req: UIRequest, action: dict[str, Any]) -> ChangeSet:
    """Build a ChangeSet with exactly one action using the shared builder to keep schema stable."""
    return build_changeset(
        org_id=req.org_id,
        user_id=req.user_id,
        actions=[action],
        intent="apply",
    )

async def _submit_now(cs: ChangeSet) -> None:
    """
    Fire-and-forget submit; logs telemetry. Exceptions are caught and logged.
    """
    try:
        resp = await asubmit_changeset(cs)
        telemetry_log(
            "changeset.submitted",
            {
                "status": resp.get("status"),
                "mode": resp.get("mode"),
                "path": resp.get("path") or resp.get("id"),
            },
        )
    except Exception as e:
        logger.exception("ChangeSet submit failed")
        telemetry_log("changeset.submit.error", {"error": str(e)})


# --------------------------------------------------------------------------------------
# Non-AI listing/detail helpers (used by READ_* control)
# --------------------------------------------------------------------------------------

@dataclass
class Filters:
    search: str | None
    type: str | None
    is_primary: bool | None

def _to_asset_preview(asset: Any) -> AssetPreview:
    data = asset.model_dump() if hasattr(asset, "model_dump") else dict(asset)
    return AssetPreview(
        id=str(data.get("id") or ""),
        name=str(data.get("name") or ""),
        assetType=str(data.get("assetType") or ""),
        isPrimary=bool(data.get("isPrimary")) if "isPrimary" in data else None,
        confidentiality=_safe_int(data.get("confidentiality")),
        integrity=_safe_int(data.get("integrity")),
        availability=_safe_int(data.get("availability")),
    )

def _to_asset_detail(asset: Any) -> AssetDetail:
    data = asset.model_dump() if hasattr(asset, "model_dump") else dict(asset)
    base = _to_asset_preview(asset).model_dump()
    return AssetDetail(
        **base,
        description=(str(data.get("description")) if data.get("description") is not None else None),
        org_id=str(data.get("org_id")) if data.get("org_id") is not None else None,
    )

def _safe_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None

def _sort_assets(items: Sequence[Any]) -> list[Any]:
    def _key(a: Any) -> tuple[int, str]:
        d = a.model_dump() if hasattr(a, "model_dump") else dict(a)
        primary = 0 if bool(d.get("isPrimary")) else 1
        name = str(d.get("name") or "").casefold()
        return (primary, name)
    return sorted(items, key=_key)

def _get_backend() -> BackendPort:
    return CsvBackend()  # sync → wrap with _call_async

async def _call_async(func, *args, **kwargs):
    # If the target is a coroutine function, await it directly.
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    # Otherwise offload the sync call to a thread to avoid blocking the loop.
    return await asyncio.to_thread(func, *args, **kwargs)

def _audit_in(req: UIRequest, intent: Intent) -> None:
    try:
        telemetry_log(
            "asset_agent_ai.in",
            {
                "org_id": req.org_id,
                "user_id": req.user_id,
                "intent": intent,
                "has_selection": bool(req.selection),
                "message_preview": (req.message[:256] + "…") if req.message and len(req.message) > 256 else req.message,
                "ai_enabled": True,      # always-on if available
                "send_now": True,        # no buffer, immediate submit
            },
        )
    except Exception:
        pass

def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "item"

def _explain_response() -> UIResponse:
    return UIResponse(
        messages=[
            "I’m your AssetAgent. Tell me in plain language what you want to do with assets.",
            "• “Create a CRM asset as primary, type Information system …”",
            "• “Update asset asset-db-001: rename to Customer DB and set primary=false.”",
            "• “List primary Application software containing ‘crm’.”",
            "Every change is broadcast via WS and submitted to the backend outbox as a single-action ChangeSet.",
        ],
        proposals=None,
        staging={"needs_commit": False},
    )
