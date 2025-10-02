# File: workshops_ai/graph/agent_node.py
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class AgentDecision(BaseModel):
    action: str = Field(..., description="Selected tool/action name, e.g., 'assets_list'")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the selected action")

SYSTEM_PROMPT = """You are a helpful risk workshop agent. 
Decide the single most appropriate ACTION to take next from the provided tool list.
Return strictly JSON with fields: action, args.
Prefer conversational follow-ups if the user's request is unclear (then choose 'explain' with a short summary).
Never invent asset IDs; if missing, ask via 'explain' suggesting next step.
"""

def _tool_specs_for_tab(tab: str) -> list[dict]:
    """
    Return minimal JSON schemas of tools per tab.
    We don't bind LangChain Tools zde, jen popíšeme kontrakt, agent se rozhodne a my route-neme na existující uzly.
    """
    common = [
        {"name": "explain", "description": "Explain using available catalog/context; no writes.", "schema": {"type":"object","properties":{"question":{"type":"string"}},"required":[]}},
    ]
    if tab == "assets":
        return common + [
            {"name": "assets_list", "description": "List assets.", "schema": {"type":"object","properties":{},"required":[]}},
            {"name": "asset_get", "description": "Get asset detail by id.", "schema": {"type":"object","properties":{"asset_id":{"type":"string"}},"required":["asset_id"]}},
            {"name": "create_asset", "description": "Create a new asset.", "schema": {"type":"object","properties":{"name":{"type":"string"},"type":{"type":"string"},"is_primary":{"type":"boolean"}},"required":["name"]}},
            {"name": "set_meta", "description": "Set/patch asset metadata.", "schema": {"type":"object","properties":{"asset_id":{"type":"string"},"meta":{"type":"object"}},"required":["asset_id","meta"]}},
        ]
    if tab == "impacts":
        return common + [
            {"name": "list_impacts", "description": "List impact catalog.", "schema": {"type":"object","properties":{},"required":[]}},
            {"name": "asset_get_impacts", "description": "Show CIA/impacts for asset.", "schema": {"type":"object","properties":{"asset_id":{"type":"string"}},"required":["asset_id"]}},
            {"name": "set_impact_level", "description": "Set/update CIA levels for one asset.", "schema": {"type":"object","properties":{"asset_id":{"type":"string"},"conf":{"type":"integer"},"integ":{"type":"integer"},"avail":{"type":"integer"}},"required":["asset_id"]}},
            {"name": "bulk_set_impact_levels", "description": "Bulk set CIA across assets.", "schema": {"type":"object","properties":{"items":{"type":"array","items":{"type":"object"}}},"required":["items"]}},
        ]
    if tab == "threats":
        return common + [
            {"name": "threats_list", "description": "List threats from catalog.", "schema": {"type":"object","properties":{},"required":[]}},
            {"name": "threats_for_asset", "description": "Show threats mapped to one asset.", "schema": {"type":"object","properties":{"asset_id":{"type":"string"}},"required":["asset_id"]}},
            {"name": "upsert_threat", "description": "Create/update a threat.", "schema": {"type":"object","properties":{"asset_id":{"type":"string"},"code":{"type":"string"},"level":{"type":"integer"}},"required":["asset_id","code"]}},
        ]
    return common

def agent_decide(tab: str, user_text: str, context: dict, deps) -> AgentDecision:
    """
    tab      : 'assets' | 'impacts' | 'threats'
    user_text: raw user message
    context  : current state bits (selection, locale, etc.) – může pomoct v promptu
    deps     : GraphDeps (obsahuje llm client)
    """
    tools = _tool_specs_for_tab(tab)
    tool_list_str = "\n".join([f"- {t['name']}: {t['description']}" for t in tools])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nAvailable tools:\n" + tool_list_str},
        {"role": "user", "content": user_text},
    ]

 
    resp = deps.llm_client.chat_completions(messages, response_format={"type":"json_object"}, temperature=deps.llm_temperature or 0.2)

    try:
        data = resp.as_json()  
        return AgentDecision.model_validate(data)
    except Exception:
        return AgentDecision(action="explain", args={"question": "Can you clarify what you want to see or change?"})
