# File: workshops_ai/graph/agent_node.py
# Purpose: LLM-driven decision node that:
# - Uses deps.llm_client (+ deps.llm_temperature) to pick exactly ONE action.
# - Performs clarify-first: when a required slot is missing, returns an 'explain' action with a single clarifying question.
# - Returns a strict JSON decision: { "action": "<tool_name>", "args": { ... } } for the graph router.
# - Degrades safely if no LLM is configured (asks for clarification via 'explain').

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, ValidationError


# ------------------------------- Model --------------------------------------- #

class AgentDecision(BaseModel):
    """Router contract: one next action and its arguments."""
    model_config = ConfigDict(extra="forbid")
    action: str = Field(..., description="Selected tool/action name, e.g., 'create_asset'")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the selected action")


# ----------------------------- Prompts & Tools ------------------------------- #

_SYSTEM_BEHAVIOR = (
    "ROLE: You are a conversational assistant for managing assets, impacts, and threats.\n"
    "STYLE: Be brief, natural, and speak the user's language (cs/sk/en).\n"
    "RULES:\n"
    "1) If the user asks for info/explanation, reply textually (action 'explain').\n"
    "2) If the user requests an action, CALL EXACTLY ONE TOOL. Provide ALL required args.\n"
    "   If a required arg is missing, ask ONE short clarifying question (action 'explain'). No loops.\n"
    "3) Never invent IDs/codes/levels. If you need a code (impact/threat), call the corresponding defs.list tool first,\n"
    "   pick the best match, or ask to create a new definition if not found.\n"
    "4) If the user didn't specify an asset but a current asset exists in context, use it; otherwise ask once.\n"
    "5) Levels must be 0..4. If the user speaks relatively (raise/lower), send '+1' or '-1' as level to the tool.\n"
    "6) After a successful tool call, confirm in one sentence: '✅ Done: {what} for {asset} → {value}. Anything else?'\n"
    "7) One user message = at most one tool call (+ at most one defs.list helper if you need a code).\n"
    "8) Output STRICT JSON only: {'action': string, 'args': object}.\n"
)

_TOOLS: list[dict] = [
    {
        "name": "explain",
        "description": "Conversational reply (no writes). Use for explanations or a single clarifying question.",
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "If clarifying, a single short question"},
                "message": {"type": "string", "description": "Optional informational message"}
            },
            "required": []
        },
    },
    # Assets
    {
        "name": "create_asset",
        "description": "Create one or more assets (supports Czech/Slovak prompts).",
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "assetType": {"type": "string"},
                            "isPrimary": {"type": "boolean"}
                        },
                        "required": ["name"]
                    }
                }
            },
            "required": ["items"]
        },
    },
    {
        "name": "set_meta",
        "description": "Update asset metadata (assetType, isPrimary, CIA levels).",
        "schema": {
            "type": "object",
            "properties": {
                "update": {
                    "type": "object",
                    "properties": {
                        "asset_id": {"type": "string"},
                        "assetType": {"type": "string"},
                        "isPrimary": {"type": "boolean"},
                        "confidentiality": {"type": "string", "enum": ["L","M","H"]},
                        "integrity": {"type": "string", "enum": ["L","M","H"]},
                        "availability": {"type": "string", "enum": ["L","M","H"]}
                    },
                    "required": ["asset_id"]
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "asset_id": {"type": "string"},
                            "fields": {"type": "object"}
                        },
                        "required": ["asset_id","fields"]
                    }
                }
            }
        },
    },
    {
        "name": "assets_list",
        "description": "List assets (for selection or search).",
        "schema": {
            "type": "object",
            "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}, "offset": {"type": "integer"}},
            "required": []
        },
    },
    # Impacts defs + set
    {
        "name": "impacts_defs_list",
        "description": "List impact definitions (code/label/category) for lookup.",
        "schema": {
            "type": "object",
            "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}, "offset": {"type": "integer"}},
            "required": []
        },
    },
    {
        "name": "impacts_defs_create",
        "description": "Create an impact definition (code/label/category/description).",
        "schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "label": {"type": "string"},
                "category": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["code", "label"]
        },
    },
    {
        "name": "impacts_set",
        "description": "Assign an impact level (0..4 or +1/-1) to an asset for a code.",
        "schema": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string"},
                "code": {"type": "string"},
                "level": {"oneOf": [{"type": "integer"}, {"type": "string"}], "description": "0..4 or '+1'/'-1'"},
            },
            "required": ["asset_id", "code", "level"]
        },
    },
    # Threats defs + set
    {
        "name": "threats_defs_list",
        "description": "List threat definitions (code/label/category) for lookup.",
        "schema": {
            "type": "object",
            "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}, "offset": {"type": "integer"}},
            "required": []
        },
    },
    {
        "name": "threats_defs_create",
        "description": "Create a threat definition (code/label/category/description).",
        "schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "label": {"type": "string"},
                "category": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["code", "label"]
        },
    },
    {
        "name": "threats_set",
        "description": "Assign a threat level (0..4 or +1/-1) to an asset for a code.",
        "schema": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string"},
                "code": {"type": "string"},
                "level": {"oneOf": [{"type": "integer"}, {"type": "string"}], "description": "0..4 or '+1'/'-1'"},
            },
            "required": ["asset_id", "code", "level"]
        },
    },
]

_ALLOWED_ACTIONS = {t["name"] for t in _TOOLS} | {"explain"}


def _render_tool_catalog() -> str:
    """Compact textual catalog with required args for each tool, to guide the LLM."""
    lines: list[str] = ["TOOLS (call exactly one if action is required):"]
    for t in _TOOLS:
        req = t.get("schema", {}).get("required", []) or []
        req_str = f" (required: {', '.join(req)})" if req else ""
        lines.append(f"- {t['name']}: {t['description']}{req_str}")
    return "\n".join(lines)


# ------------------------------- LLM Bridge ---------------------------------- #

def _to_openai_messages(history: list[dict] | None, system_prompt: str, user_text: str, domain_prompt: Optional[str]) -> list[dict]:
    """Builds a clean message array for Chat Completions APIs."""
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    if domain_prompt:
        msgs.append({"role": "system", "content": domain_prompt})
    if history:
        for m in history:
            r = m.get("role")
            c = m.get("content")
            if r in ("user", "assistant") and isinstance(c, str) and c.strip():
                msgs.append({"role": r, "content": c})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def _parse_json_block(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction:
    - Prefer full-text json.loads
    - Fallback: extract the first { ... } block via regex and parse
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError("LLM did not return valid JSON.")


def _call_llm_json(
    *,
    deps: Any,
    messages: list[dict],
    temperature: float | None,
) -> Dict[str, Any]:
    """
    Calls deps.llm_client (preferred) or deps.llm (legacy) and returns parsed JSON content.
    Tries multiple common client shapes to increase portability.
    """
    client = getattr(deps, "llm_client", None) or getattr(deps, "llm", None)
    if client is None:
        raise RuntimeError("LLM client is not configured.")

    temp = temperature if temperature is not None else getattr(deps, "llm_temperature", None) or 0.2

    # --- OpenAI-like: client.chat.completions.create(...)
    try:
        completions = getattr(getattr(client, "chat", None), "completions", None)
        create = getattr(completions, "create", None)
        if callable(create):
            resp = create(messages=messages, response_format={"type": "json_object"}, temperature=temp)  # type: ignore[arg-type]
            content = resp.choices[0].message.content  # type: ignore[index]
            return _parse_json_block(content)
    except Exception:
        pass

    # --- wrapper: client.chat_completions(messages=..., response_format=..., temperature=...)
    try:
        chat_completions = getattr(client, "chat_completions", None)
        if callable(chat_completions):
            resp = chat_completions(messages=messages, response_format={"type": "json_object"}, temperature=temp)  # type: ignore[arg-type]
            content = getattr(resp, "content", None)
            if isinstance(content, str):
                return _parse_json_block(content)
            try:
                content = resp.choices[0].message.content  # type: ignore[index]
                return _parse_json_block(content)
            except Exception:
                pass
            if hasattr(resp, "as_json"):
                return resp.as_json()  # type: ignore[no-any-return]
    except Exception:
        pass

    # --- generic: client(messages=...) → str
    try:
        text = client(messages=messages, temperature=temp)  # type: ignore[misc]
        if isinstance(text, str):
            return _parse_json_block(text)
    except Exception:
        pass

    raise RuntimeError("Unsupported LLM client interface or invalid response.")


# ------------------------------ Public API ----------------------------------- #

def agent_decide(tab: str, user_text: str, context: dict, deps: Any) -> AgentDecision:
    """
    Decide the single most appropriate action for the user message (tool-first).
    Clarify-first: if required data is missing, return action='explain' with args.question (one concise question).
    Returns AgentDecision for the graph/router.
    """
    language = (context.get("language") or "").lower().strip() or "en"
    domain_prompt: Optional[str] = context.get("domain_prompt")  # optional per-agent addon
    history: List[dict] = context.get("history") or []

    system_prompt = (
        _SYSTEM_BEHAVIOR
        + "\n"
        + _render_tool_catalog()
        + "\n"
        + f"Reply in '{language}'. Output STRICT JSON only."
    )

    messages = _to_openai_messages(history, system_prompt, user_text, domain_prompt)

    # If no LLM is configured, degrade safely to a single clarify question.
    if not getattr(deps, "llm_client", None) and not getattr(deps, "llm", None):
        return AgentDecision(
            action="explain",
            args={
                "question": (
                    "Pro které aktivum? (např. „Customer Database“)"
                    if language.startswith("cs")
                    else "Pre ktoré aktívum? (napr. „Customer Database“)"
                    if language.startswith("sk")
                    else "For which asset? (e.g., “Customer Database”)"
                )
            },
        )

    # Call LLM and parse strict JSON decision
    try:
        raw = _call_llm_json(deps=deps, messages=messages, temperature=getattr(deps, "llm_temperature", None))
        decision = AgentDecision.model_validate(raw)
        if decision.action not in _ALLOWED_ACTIONS:
            return AgentDecision(
                action="explain",
                args={
                    "question": (
                        "Chceš vytvořit/upravit aktivum, nebo nastavit dopad/hrozbu? Prosím upřesni jednu akci."
                        if language.startswith("cs")
                        else "Chceš vytvoriť/upraviť aktívum, alebo nastaviť dopad/hrozbu? Prosím upresni jednu akciu."
                        if language.startswith("sk")
                        else "Do you want to create/update an asset, or set an impact/threat? Please specify one action."
                    )
                },
            )
        return decision
    except (ValidationError, ValueError, RuntimeError):
        return AgentDecision(
            action="explain",
            args={
                "question": (
                    "Potřebuji upřesnit jednu věc: jaká je tvoje cílová akce? (např. 'vytvoř aktivum CRM', 'nastav Finance na 3')"
                    if language.startswith("cs")
                    else "Potrebujem upresniť jednu vec: aká je tvoja cieľová akcia? (napr. 'vytvor aktívum CRM', 'nastav Finance na 3')"
                    if language.startswith("sk")
                    else "One detail please: what is your target action? (e.g., 'create asset CRM', 'set Finance to 3')"
                )
            },
        )
