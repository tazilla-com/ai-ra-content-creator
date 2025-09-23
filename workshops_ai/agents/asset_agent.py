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

# New: context utilities (seed heuristics / summaries)
from workshops_ai.agents.utils.context_enricher import (
    summarize_org_profile,
    extract_seed_candidates,
    dedupe_against_existing,
    limit_results,
)

# Outbox submit (async, non-blocking for WS/LLM loop)
from workshops_ai.app.services.tazila_client import asubmit_changeset  # type: ignore

# WS change events (raw JSON broadcast via WsHub → /ws)
from workshops_ai.app.services.change_events import aemit_change_event

# New: HTTP echo of events (AgentEvent schema)
from workshops_ai.models.agent_event import AgentEvent

# Centralized settings 
from workshops_ai.settings import get_settings

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
_ALLOWED_TOOLS = {"create_asset", "set_asset_meta"}
_REPAIR_HINT = (
    "Your previous output did not match the schema/rules. "
    "Return exactly ONE tool call (create_asset or set_asset_meta) OR a single <control>...</control> line. "
    "When calling a tool, include strictly valid JSON args per the tool schema. "
    "Do not include any extra text besides the one short human sentence and optional <control> tag."
)


# --------------------------------------------------------------------------------------
# Language helpers (base = 'sk' or 'en')
# --------------------------------------------------------------------------------------

def _pref_lang(req: UIRequest) -> str:
    """
    Choose response language: 'sk' or 'en'.
    - 'sk' if req.language startswith 'sk' OR 'cs' (map Czech → Slovak base as requested).
    - 'en' if req.language startswith 'en'.
    - default 'en' if not provided/unknown.
    """
    lang = (getattr(req, "language", None) or "").strip().lower()
    if lang.startswith("sk") or lang.startswith("cs"):
        return "sk"
    if lang.startswith("en"):
        return "en"
    return "en"

_T = {
    "en": {
        "explain_1": "I’m your AssetAgent. Tell me in plain language what you want to do with assets.",
        "explain_2": "• “Create a CRM asset as primary, type Information system …”",
        "explain_3": "• “Update asset asset-db-001: rename to Customer DB and set primary=false.”",
        "explain_4": "• “List primary Application software containing ‘crm’.”",
        "explain_5": "Every change is broadcast via WS and submitted to the backend outbox as a single-action ChangeSet.",
        "hint_generic": "Tell me what you’d like to do with assets (create, update, list, detail).",
        "need_id": "Please provide an asset ID to show its detail.",
        "not_found": "I couldn't find an asset with ID “{id}”.",
        "detail_intro": "Here’s the asset detail for “{name}”.",
        "list_found": "I found {n} item(s){suffix}.",
        "ok": "OK.",
        "done": "Done — {n} change(s) queued to the outbox.",
        "invalid_change": "I couldn't build a valid change from that request. Try: “Create an asset …” or “Update asset …”.",
        "seed_intro": "Here are initial asset candidates based on your services/website.",
    },
    "sk": {
        "explain_1": "Som tvoj AssetAgent. Povedz mi jednoducho, čo chceš s aktívami urobiť.",
        "explain_2": "• „Vytvor CRM ako primárne, typ Information system …“",
        "explain_3": "• „Aktualizuj asset asset-db-001: premenuj na Customer DB a nastav primary=false.“",
        "explain_4": "• „Vypíš primárne Application software obsahujúce ‘crm’.“",
        "explain_5": "Každá zmena sa posiela cez WS a ukladá do outboxu backendu ako single-action ChangeSet.",
        "hint_generic": "Napíš, čo chceš s aktívami spraviť (vytvoriť, upraviť, vypísať, detail).",
        "need_id": "Uveď prosím ID assetu, aby som zobrazil detail.",
        "not_found": "Nenašiel som asset s ID „{id}“.",
        "detail_intro": "Tu je detail assetu „{name}“.",
        "list_found": "Našiel som {n} položiek{suffix}.",
        "ok": "OK.",
        "done": "Hotovo — {n} zmien odoslaných do outboxu.",
        "invalid_change": "Nepodarilo sa vytvoriť platnú zmenu. Skús: „Vytvor asset …“ alebo „Uprav asset …“.",
        "seed_intro": "Tu sú počiatoční kandidáti aktív podľa vašich služieb/webu.",
    },
}

def _t(lang: str, key: str, **kwargs) -> str:
    base = _T.get(lang, _T["en"]).get(key, _T["en"].get(key, key))
    return base.format(**kwargs)


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
    """
    Construct the LLM client strictly from centralized settings.
    """
    if not _HAS_LLM:
        raise RuntimeError("LangChain/OpenAI not installed")

    s = get_settings()
    # Using OpenAI client from env configured by Settings; keys are read by the SDK internally.
    model = getattr(s, "llm_model", None) or "gpt-4o-mini"
    temperature = float(getattr(s, "llm_temperature", 0.0))
    timeout = int(getattr(s, "http_timeout_sec", 60))
    max_retries = int(getattr(s, "openai_max_retries", 3))

    return ChatOpenAI(model=model, temperature=temperature, timeout=timeout, max_retries=max_retries)  # type: ignore[call-arg]


LC_SYSTEM_PROMPT = """You are AssetAgent for the 'assets' tab (EMIT-ONLY WRITES).
Respond strictly in language: {lang}.

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

CONTEXT (use for better suggestions; do not leak raw PII):
<instructions>{user_instructions}</instructions>
<org_profile>{org_profile_summary}</org_profile>
<known_assets_count>{known_assets_count}</known_assets_count>
<website_seed_candidates>{website_seed_candidates}</website_seed_candidates>

IMPORTANT:
- This agent emits ChangeSet JSON and WS events; it doesn’t apply changes directly.
- Reading uses CSV/state only; newly created/updated assets may not immediately appear in lists.
{repair_clause}
"""


# --------------------------------------------------------------------------------------
# Presentation models for proposals (list/detail/seed)
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

    lang = _pref_lang(req)
    msg = (req.message or "").strip()

    if intent == "explain":
        return _explain_response(lang)

    # Build context for prompt (summaries + seed terms)
    org_summary = summarize_org_profile(getattr(req, "organization_profile", None))
    raw_seeds = extract_seed_candidates(getattr(req, "services", None), getattr(req, "website_content", None))
    seeds = dedupe_against_existing(raw_seeds, getattr(req, "existing_assets", None))
    seed_labels = ", ".join([c["name"] for c in seeds]) if seeds else ""

    # 1) Invoke LLM; if unavailable, gracefully fall back to read-only + seed proposals.
    try:
        ai_msg = await _ai_invoke(req, msg, lang, org_summary, seed_labels, repair=False)
        if _needs_repair(ai_msg):
            telemetry_log("asset_agent.repair_retry", {"reason": "schema/rules mismatch"})
            ai_msg = await _ai_invoke(req, msg, lang, org_summary, seed_labels, repair=True)
    except Exception as e:
        telemetry_log("asset_agent.llm_unavailable", {"error": str(e)})
        parsed = _interpret_ai_read_control(_fallback_read_control(msg))
        # Seed proposals on fallback when no explicit READ_DETAIL
        if not parsed or parsed[0] != "READ_DETAIL":
            limit = getattr(req, "number_of_assets", None)
            proposed = limit_results(seeds, limit)
            if proposed:
                return UIResponse(
                    messages=[_t(lang, "seed_intro")],
                    proposals={"seed_assets": proposed},
                    staging={"needs_commit": False},
                )
        return await _handle_read_controls(req, parsed, ai_fallback_text=_t(lang, "hint_generic"))

    # 1a) If tools were called, emit WS events and immediately submit a single-action ChangeSet
    tool_calls = _normalize_tool_calls(ai_msg)
    if tool_calls:
        # enforce exactly ONE tool call
        tool_calls = tool_calls[:1]
        submitted, events = await _handle_tool_calls(req, tool_calls)
        if submitted > 0:
            return UIResponse(
                messages=[_t(lang, "done", n=submitted)],
                proposals=None,
                staging={"needs_commit": False},
                agent_events=events,  # HTTP echo
            )
        return UIResponse(
            messages=[_t(lang, "invalid_change")],
            proposals=None,
            staging={"needs_commit": False},
        )

    # 1b) No tools: parse hidden control tag and show only the human sentence (+ optional seeds)
    raw_content = (getattr(ai_msg, "content", None) or "").strip()
    control_text, human_text = _extract_control_tag_and_human_text(raw_content)

    if not control_text or control_text.upper().startswith("NONE"):
        # Offer seed candidates if any
        limit = getattr(req, "number_of_assets", None)
        proposed = limit_results(seeds, limit)
        if proposed:
            return UIResponse(
                messages=[human_text or _t(lang, "seed_intro")],
                proposals={"seed_assets": proposed},
                staging={"needs_commit": False},
            )
        return UIResponse(messages=[human_text or _t(lang, "ok")], proposals=None, staging={"needs_commit": False})

    # Interpret hidden READ_* control and return human sentence + proposals
    parsed = _interpret_ai_read_control(control_text)
    return await _handle_read_controls(req, parsed, ai_fallback_text=(human_text or None))


# --------------------------------------------------------------------------------------
# AI helpers
# --------------------------------------------------------------------------------------

async def _ai_invoke(
    req: UIRequest,
    message: str,
    lang: str,
    org_profile_summary: str,
    website_seed_labels: str,
    *,
    repair: bool,
):
    # Bind tools for schema exposure; we parse tool calls ourselves.
    llm = _llm().bind_tools([create_asset_tool, set_asset_meta_tool])  # type: ignore[attr-defined]
    sys_prompt = LC_SYSTEM_PROMPT.format(
        lang=lang,
        user_instructions=(getattr(req, "user_instructions", "") or "").strip(),
        org_profile_summary=org_profile_summary or "n/a",
        known_assets_count=len(getattr(req, "existing_assets", []) or []),
        website_seed_candidates=website_seed_labels or "n/a",
        repair_clause=("\nREPAIR:\n" + _REPAIR_HINT if repair else ""),
    )
    return await llm.ainvoke(  # type: ignore[call-arg]
        [
            SystemMessage(content=sys_prompt),  # type: ignore[arg-type]
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

def _needs_repair(ai_msg: Any) -> bool:
    """
    Heuristic schema/rules validation for the AI message.
    Triggers a single repair retry if:
      - more than one tool call,
      - unknown tool name,
      - required tool args missing or invalid by args_schema,
      - content neither has a valid single tool call nor a <control> tag.
    """
    calls = _normalize_tool_calls(ai_msg)
    if calls:
        if len(calls) != 1:
            return True
        name = (calls[0].get("name") or "").strip()
        if name not in _ALLOWED_TOOLS:
            return True
        args = calls[0].get("args") or {}
        try:
            if name == "create_asset":
                # required: name, asset_type_code
                CreateAssetArgs(**args)  # type: ignore[arg-type]
            elif name == "set_asset_meta":
                # required: asset_id and at least one field to update (checked later)
                SetAssetMetaArgs(**args)  # type: ignore[arg-type]
        except Exception:
            return True
        return False

    # No tool calls: check control tag
    content = (getattr(ai_msg, "content", None) or "").strip()
    if not content:
        return True
    control_text, _ = _extract_control_tag_and_human_text(content)
    if control_text is None:
        # Accept pure human text (OK) — but prefer explicit control → do not enforce repair here.
        return False
    # Validate control grammar
    return _interpret_ai_read_control(control_text) is None


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
# WS events + immediate single-action ChangeSet submit; returns (submitted_count, events)
# --------------------------------------------------------------------------------------

async def _handle_tool_calls(req: UIRequest, tool_calls: Sequence[dict[str, Any]]) -> tuple[int, list[AgentEvent]]:
    submitted = 0
    events: list[AgentEvent] = []

    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("args") or {}

        if name == "create_asset":
            # Validate args (schema-level guard)
            try:
                CreateAssetArgs(**args)  # raises if invalid
            except Exception:
                continue

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

            # WS change event (user sees the change immediately)
            await aemit_change_event(
                "asset.create",
                org_id=req.org_id,
                user_id=req.user_id,
                data=action,
                asset_id=None,
            )

            # HTTP echo
            events.append(
                AgentEvent(
                    event="asset.create",
                    org_id=req.org_id,
                    user_id=req.user_id,
                    asset_id=None,
                    data=action,
                    source="AssetAgent",
                    version="v1",
                )
            )

            # Single-action ChangeSet → immediate async submit (non-blocking)
            cs = _make_changeset_one(req, action)
            asyncio.create_task(_submit_now(cs))
            submitted += 1

        elif name == "set_asset_meta":
            # Validate args (schema-level guard)
            try:
                SetAssetMetaArgs(**args)  # raises if invalid
            except Exception:
                continue

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

            # WS change event
            await aemit_change_event(
                "asset.set_meta",
                org_id=req.org_id,
                user_id=req.user_id,
                data=action,
                asset_id=asset_id,
            )

            # HTTP echo
            events.append(
                AgentEvent(
                    event="asset.set_meta",
                    org_id=req.org_id,
                    user_id=req.user_id,
                    asset_id=asset_id,
                    data=action,
                    source="AssetAgent",
                    version="v1",
                )
            )

            # Single-action ChangeSet → immediate async submit
            cs = _make_changeset_one(req, action)
            asyncio.create_task(_submit_now(cs))
            submitted += 1

        # ignore any other tool (tool-gated)
    return submitted, events


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
    lang = _pref_lang(req)

    if parsed is None:
        # No recognizable control line; return short hint or LLM text.
        text = ai_fallback_text or _t(lang, "hint_generic")
        return UIResponse(messages=[text], proposals=None, staging={"needs_commit": False})

    kind, payload = parsed

    if kind == "READ_DETAIL":
        asset_id = payload.get("id") or ""
        if not asset_id:
            return UIResponse(messages=[_t(lang, "need_id")], proposals=None, staging={"needs_commit": False})
        asset = await _call_async(backend.asset_get, asset_id)
        if asset is None:
            return UIResponse(
                messages=[_t(lang, "not_found", id=asset_id)],
                proposals=None,
                staging={"needs_commit": False},
            )
        detail = _to_asset_detail(asset)
        return UIResponse(
            messages=[_t(lang, "detail_intro", name=detail.name)],
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
        # Respect 'number_of_assets' hint (safe cap)
        limit = min(READ_PAGE_LIMIT, int(getattr(req, "number_of_assets", READ_PAGE_LIMIT) or READ_PAGE_LIMIT))
        items = _sort_assets(items)[:limit]
        previews = [_to_asset_preview(a).model_dump() for a in items]

        parts: list[str] = []
        if filters.search:
            parts.append(f'search="{filters.search}"')
        if filters.type:
            parts.append(f'type="{filters.type}"')
        if filters.is_primary is not None:
            parts.append("primary=true" if filters.is_primary else "primary=false")
        suffix = ""
        if parts:
            if lang == "sk":
                suffix = f' (filtre: {", ".join(parts)})'
            else:
                suffix = f' (filters: {", ".join(parts)})'

        return UIResponse(
            messages=[_t(lang, "list_found", n=len(previews), suffix=suffix)],
            proposals={"assets": previews},
            staging={"needs_commit": False},
        )

    # Unknown kind (shouldn't happen)
    return UIResponse(messages=[_t(_pref_lang(req), "ok")], proposals=None, staging={"needs_commit": False})


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
                "has_selection": bool(getattr(req, "selection", None)),
                "message_preview": (req.message[:256] + "…") if req.message and len(req.message) > 256 else req.message,
                "ai_enabled": True,      # always-on if available
                "send_now": True,        # no buffer, immediate submit
                "language": _pref_lang(req),
                "number_of_assets": getattr(req, "number_of_assets", None),
            },
        )
    except Exception:
        pass

def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "item"

def _explain_response(lang: str) -> UIResponse:
    return UIResponse(
        messages=[
            _t(lang, "explain_1"),
            _t(lang, "explain_2"),
            _t(lang, "explain_3"),
            _t(lang, "explain_4"),
            _t(lang, "explain_5"),
        ],
        proposals=None,
        staging={"needs_commit": False},
    )
