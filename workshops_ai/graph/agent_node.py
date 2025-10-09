# File: workshops_ai/graph/agent_node.py
# Purpose:
#   LLM-driven decision node that:
#     - Uses deps.llm_client with OpenAI-style "functions/tools" to pick EXACTLY ONE tool.
#     - Is tab-aware (assets | impacts | threats | controls supported by registry).
#     - Gets tool schemas from the central ToolRegistry (SSOT).
#     - Enforces allowed actions per tab (if provided) or falls back to registry for the tab.
#     - Strictly validates tool arguments via ToolRegistry.validate_call().
#     - Returns endpoint/op from the registry for uniform routing.
#
# Notes:
#   - If the LLM doesn't pick any tool, we return a natural-language reply (clarify/explain).
#   - No tool execution happens here; this node only decides & validates.
#   - Supports multiple client shapes: OpenAI Chat Completions, OpenAI Responses, LangChain ChatModel.
#   - Integrated lightweight debug telemetry (telemetry_debug.DecideSpan).

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Literal, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from workshops_ai.graph.factory import GraphDeps

from workshops_ai.llm.tools_adapter import sanitize_tools, canonical_name

from pydantic import BaseModel, Field, ConfigDict, ValidationError

logger = logging.getLogger(__name__)

# =============================================================================
# SSOT: Tool Registry
# =============================================================================
from workshops_ai.agents.tools.registry import ToolRegistry

# Optional, best-effort debug telemetry (all calls wrapped in try/except)
try:
    from workshops_ai.telemetry_debug import set_decide_span, finish_decide  # type: ignore
except Exception:  # pragma: no cover
    set_decide_span = None  # type: ignore
    finish_decide = None  # type: ignore


# =============================================================================
# Decision contract (returned to graph)
# =============================================================================
class AgentDecision(BaseModel):
    """
    Router contract:
      - If a tool call is selected and validated, fields (action, args, endpoint, op) are set.
      - If no tool is selected, 'reply' contains a short assistant message (clarification/explanation).
    """
    model_config = ConfigDict(extra="forbid")

    action: Optional[str] = Field(None, description="Canonical tool name selected, e.g., 'asset.create'.")
    args: Dict[str, Any] = Field(default_factory=dict, description="Validated arguments for the tool.")
    endpoint: Optional[str] = Field(None, description="Graph endpoint path resolved from registry.")
    op: Optional[str] = Field(None, description="Canonical op code resolved from registry.")
    reply: Optional[str] = Field(None, description="Assistant message when no tool call is made.")

    # Backward-compatible aliases (read-only)
    @property
    def tool_name(self) -> Optional[str]:  # pragma: no cover
        return self.action

    @property
    def tool_payload(self) -> Dict[str, Any]:  # pragma: no cover
        return self.args


# =============================================================================
# LLM client adapter (OpenAI Chat/Responses + LangChain compatible)
#   Returns (tool_name, tool_args, assistant_text) where only one branch is non-None.
# =============================================================================
def _call_llm_with_tools(
    *,
    deps: Any,
    messages: List[dict],
    functions: List[Dict[str, Any]],
    temperature: Optional[float],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    client = getattr(deps, "llm_client", None) or getattr(deps, "llm", None)
    if client is None:
        raise RuntimeError("LLM client is not configured.")

    temp = 0.2 if temperature is None else float(temperature)
    safe_functions, name_map = sanitize_tools(functions)

    # ---------- OpenAI Chat Completions API ----------
    try:
        completions = getattr(getattr(client, "chat", None), "completions", None)
        create = getattr(completions, "create", None)
        if callable(create):
            # Convert registry tool specs → legacy Chat Completions "functions" param.
            chat_functions = []
            for f in safe_functions:
                fn = f.get("function") if isinstance(f, dict) else None
                if isinstance(fn, dict) and "name" in fn:
                    chat_functions.append({
                        "name": fn["name"],
                        "description": fn.get("description"),
                        "parameters": fn.get("parameters") or {},
                    })

            kwargs = {
                "messages": messages,
                "temperature": temp,
                "function_call": "auto",
            }
            if chat_functions:
                kwargs["functions"] = chat_functions

            model = getattr(getattr(deps, "settings", None), "llm", None)
            model_name = getattr(model, "model", None) if model else None
            if isinstance(model_name, str):
                kwargs["model"] = model_name

            resp = create(**kwargs)  # type: ignore[arg-type]
            msg = resp.choices[0].message  # type: ignore[index]

            # Newer tool_calls (OpenAI 0613+) — map safe -> canonical before returning
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                tc = tool_calls[0]
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None)
                arguments = getattr(fn, "arguments", None)

                if isinstance(name, str) and isinstance(arguments, str):
                    try:
                        args_obj = json.loads(arguments) if arguments.strip() else {}
                    except Exception:
                        args_obj = {}

                    # ⬇️ map back to canonical (e.g., "asset_create" -> "asset.create")
                    name = canonical_name(name, name_map)
                    return name, (args_obj if isinstance(args_obj, dict) else {}), None

            # Older function_call path
            fc = getattr(msg, "function_call", None)
            if fc and getattr(fc, "name", None):
                name = str(fc.name)
                arguments = getattr(fc, "arguments", None)

                if isinstance(arguments, str):
                    try:
                        args_obj = json.loads(arguments) if arguments.strip() else {}
                    except Exception:
                        args_obj = {}
                else:
                    args_obj = {}

                # ⬇️ map back to canonical (e.g., "asset_create" -> "asset.create")
                name = canonical_name(name, name_map)
                return name, (args_obj if isinstance(args_obj, dict) else {}), None

            # No tool used → treat as assistant text
            text = getattr(msg, "content", None)
            return None, None, (str(text) if text else "")
    except Exception as e:
        logger.debug("OpenAI Chat Completions path failed or unavailable: %s", e)

    # ---------- OpenAI Responses API ----------
    try:
        responses = getattr(client, "responses", None)
        create = getattr(responses, "create", None)
        if callable(create):
            # The Responses API expects 'tools=[{type: "function", function: {...}}]'.
            kwargs = {
                "messages": messages,   # Many SDKs accept direct "messages"
                "tools": safe_functions,
                "tool_choice": "auto",
                "temperature": temp,
            }
            model = getattr(getattr(deps, "settings", None), "llm", None)
            model_name = getattr(model, "model", None) if model else None
            if isinstance(model_name, str):
                kwargs["model"] = model_name

            try:
                resp = create(**kwargs)  # type: ignore[arg-type]
            except TypeError:
                # Some SDKs require "input" wrapper
                resp = create(input={"messages": messages}, tools=safe_functions, tool_choice="auto", temperature=temp)  # type: ignore[arg-type]

            # Try to normalize explicit tool_use items
            try:
                items = getattr(resp, "output", None)
                if isinstance(items, list):
                    for it in items:
                        # some SDKs return dicts, some objects
                        t = getattr(it, "type", None) or (it.get("type") if isinstance(it, dict) else None)
                        if t == "tool_use":
                            name = getattr(it, "name", None) or (it.get("name") if isinstance(it, dict) else None)
                            tool_input = getattr(it, "input", None) or (it.get("input") if isinstance(it, dict) else None)
                            if isinstance(name, str) and isinstance(tool_input, dict):
                                # ⬇️ map safe -> canonical (e.g., "asset_create" -> "asset.create")
                                name = canonical_name(name, name_map)
                                return name, tool_input, None

                # Otherwise fallback to assistant text
                text = getattr(resp, "output_text", None)
                if not text and hasattr(resp, "choices"):
                    text = getattr(resp.choices[0].message, "content", None)  # type: ignore[index]
                return None, None, (str(text) if text else "")
            except Exception:
                # Fallback to text on any parsing error
                text = getattr(resp, "output_text", None)
                return None, None, (str(text) if text else "")

    except Exception as e:
        logger.debug("OpenAI Responses path failed or unavailable: %s", e)

       # ---------- LangChain ChatModel .invoke() ----------
    try:
        if hasattr(client, "invoke"):
            lc_messages = messages
            try:
                # Best-effort message conversion (harmless if LC not installed)
                from langchain.schema import HumanMessage, SystemMessage, AIMessage  # type: ignore
                tmp = []
                for m in messages:
                    role = m.get("role")
                    content = m.get("content", "")
                    if role == "system":
                        tmp.append(SystemMessage(content=content))
                    elif role == "user":
                        tmp.append(HumanMessage(content=content))
                    else:
                        tmp.append(AIMessage(content=content))
                lc_messages = tmp
            except Exception:
                pass

            # 🔧 NEW: bind OpenAI-style tools into the LC model if possible
            model_to_use = client
            try:
                bind_tools = getattr(client, "bind_tools", None)
                if callable(bind_tools):
                    # LangChain expects OpenAI tool schema: [{"type":"function","function":{...}}, ...]
                    model_to_use = bind_tools(safe_functions)  # type: ignore[arg-type]
            except Exception:
                pass

            try:
                resp = model_to_use.invoke(lc_messages, temperature=temp)  # type: ignore[misc]
            except TypeError:
                resp = model_to_use.invoke(lc_messages)  # type: ignore[misc]

            # Try common LC tool-call shapes
            # 1) additional_kwargs.tool_calls (preferred in newer LC/OpenAI)
            ak = getattr(resp, "additional_kwargs", {}) or {}
            tcs = ak.get("tool_calls")
            if isinstance(tcs, list) and tcs:
                tc = tcs[0]
                fn = tc.get("function") if isinstance(tc, dict) else None
                name = (fn or {}).get("name") if isinstance(fn, dict) else None
                arguments = (fn or {}).get("arguments") if isinstance(fn, dict) else None
                if isinstance(name, str) and isinstance(arguments, str):
                    try:
                        args_obj = json.loads(arguments) if arguments.strip() else {}
                    except Exception:
                        args_obj = {}
                    name = canonical_name(name, name_map)
                    return name, args_obj, None

            # 2) additional_kwargs.function_call (older LC compat)
            fc = ak.get("function_call")
            if isinstance(fc, dict) and "name" in fc:
                name = fc.get("name")
                arguments = fc.get("arguments", "")
                try:
                    args_obj = json.loads(arguments) if isinstance(arguments, str) and arguments.strip() else (arguments or {})
                except Exception:
                    args_obj = {}
                name = canonical_name(name, name_map)
                return str(name), (args_obj if isinstance(args_obj, dict) else {}), None

            # 3) AIMessage.tool_calls attribute (some LC providers set this)
            try:
                tool_calls_attr = getattr(resp, "tool_calls", None)
                if isinstance(tool_calls_attr, list) and tool_calls_attr:
                    tc = tool_calls_attr[0]
                    fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)
                    name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
                    arguments = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else None)
                    if isinstance(name, str) and isinstance(arguments, str):
                        try:
                            args_obj = json.loads(arguments) if arguments.strip() else {}
                        except Exception:
                            args_obj = {}
                        name = canonical_name(name, name_map)
                        return name, args_obj, None
            except Exception:
                pass

            # Otherwise treat as assistant text
            text = getattr(resp, "content", None) or getattr(resp, "text", None)
            return None, None, (str(text) if text else "")
    except Exception as e:
        logger.debug("LangChain invoke path failed or unavailable: %s", e)



# =============================================================================
# Public API
# =============================================================================
async def agent_decide(
    *,
    deps: GraphDeps,
    tab: Literal["assets", "impacts", "threats", "controls"],
    message: str,
    language: str | None = None,
    locale: str | None = None,
    history: list[dict[str, str]] | None = None,
    selection: dict[str, Any] | None = None,
    allowed_actions: list[str] | None = None,
    one_action_only: bool = True,  # reserved; we already enforce single tool call
) -> AgentDecision:
    """
    Single responsibility:
      - Prepare manager/system/user messages and tool specs
      - Invoke LLM with tool-calling
      - Validate args and resolve registry endpoint/op
      - Emit AgentDecision or a short natural-language reply
      - Emit debug telemetry spans (best-effort, non-fatal)
    """
    lang = (language or locale or "en").strip().lower()
    ToolRegistry.ensure_initialized()
    tab_key = (tab or "").strip().lower()

    # Normalize conversation messages: keep only system/user/assistant
    msgs = list(history or [])
    msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")]
    msgs.append({"role": "user", "content": message})

    # Resolve available tools from registry for the tab
    registry_functions = ToolRegistry.as_openai_functions(tab_key)

    if not registry_functions:
        logger.warning("agent_decide: no tools registered for tab='%s'; reply-only fallback.", tab_key)
        return AgentDecision(reply=_fallback_clarify(lang))

    # If Manager provided a whitelist (allowed_actions), filter by function name.
    # (We expect registry_functions items like {"type": "function", "function": {"name": "...", ...}})
    if allowed_actions:
        allowed = set(a.strip() for a in allowed_actions if a and isinstance(a, str))
        filtered: list[dict[str, Any]] = []
        for f in registry_functions:
            try:
                fn = f.get("function") or {}
                name = fn.get("name")
                if isinstance(name, str) and name in allowed:
                    filtered.append(f)
            except Exception:
                continue
        allowed_functions = filtered or registry_functions  # if filter empties out, fall back to full set (safer)
    else:
        allowed_functions = registry_functions

    # --- NEW: lightweight, non-regex prioritization of tools based on the user message intent ---
    # Goal: bias "create" when the user says "vytvoř/přidej/create/add", etc., without forcing a tool.
    def _fn_name(item: dict[str, Any]) -> str:
        try:
            return str(((item.get("function") or {}).get("name")) or "")
        except Exception:
            return ""

    def _score(name: str, user_msg: str) -> int:
        # NOTE: No regexes. Pure lowercase substring checks.
        msg = user_msg.lower()
        s = 0
        # General tiny bias: prefer mutations over listing when ambiguous
        if ".create" in name:
            s += 10
        if ".update" in name or ".set" in name:
            s += 5

        # Czech/Slovak/English verbs (extend if needed)
        create_tokens = ["vytvoř", "vytvor", "přidej", "pridaj", "add", "create", "new", "založ", "zaloz"]
        list_tokens   = ["ukaž", "ukaz", "vypiš", "vypis", "seznam", "list", "show"]
        explain_tokens= ["vysvětli", "vysvetli", "explain", "co je", "what is"]
        update_tokens = ["přejmenuj", "premen", "rename", "upravit", "uprav", "update", "nastav", "set"]

        if any(t in msg for t in create_tokens):
            if name.endswith(".create"):
                s += 100
            # Slightly demote listing in a create-intent turn
            if name.endswith(".list"):
                s -= 5
        if any(t in msg for t in list_tokens):
            if name.endswith(".list"):
                s += 100
        if any(t in msg for t in explain_tokens):
            if name.endswith(".explain"):
                s += 100
        if any(t in msg for t in update_tokens):
            if (".update" in name) or (".set" in name):
                s += 100
        return s

    try:
        # Stable sort by (-score, original index) to keep registry order as tie-breaker
        _indexed = list(enumerate(allowed_functions))
        _indexed.sort(
            key=lambda pair: (
                -_score(_fn_name(pair[1]), message),
                pair[0],
            )
        )
        allowed_functions = [it for _, it in _indexed]
    except Exception:
        # Never fail because of prioritization — just keep original order
        pass

    # No LLM client available → clarify
    if deps is None or (not getattr(deps, "llm_client", None) and not getattr(deps, "llm", None)):
        return AgentDecision(reply=_fallback_clarify(lang))

    # ---- Debug telemetry: create a DecideSpan before invoking the LLM ----
    _trace = getattr(deps, "telemetry_trace", None)
    _ds = None
    try:
        if _trace is not None and set_decide_span:
            # derive model name (best-effort)
            model_name = None
            try:
                model_cfg = getattr(getattr(deps, "settings", None), "llm", None)
                model_name = getattr(model_cfg, "model", None)
            except Exception:
                pass
            # list of candidate tool names offered to the LLM
            tool_names: list[str] = []
            try:
                tool_names = [((f.get("function") or {}).get("name")) for f in allowed_functions if isinstance(f, dict)]
                tool_names = [t for t in tool_names if t]
            except Exception:
                pass
            # system prompt length from collected 'system' role messages (content only, do not log content)
            system_prompt_len = 0
            try:
                system_prompt_len = sum(len((m.get("content") or "")) for m in msgs if m.get("role") == "system")
            except Exception:
                pass
            _ds = set_decide_span(
                _trace,
                tab=tab_key,
                model=(model_name or "<unknown>"),
                tool_choice="auto",
                tools=tool_names,
                system_prompt=" " * system_prompt_len,  # length only; content intentionally omitted
                user_message=message,
            )
    except Exception:
        _ds = None  # telemetry must never break the flow

    # ---- Invoke LLM with tool-calling ----
    import time as _t
    _started = _t.time()
    try:
        # Clamp/override temperature only if not provided in deps; keep your explicit settings intact.
        _temp = getattr(deps, "llm_temperature", None)
        if _temp is None:
            _temp = 0.15  # slightly lower default → crisper tool choice

        tool_name, tool_args, assistant_text = _call_llm_with_tools(
            deps=deps,
            messages=msgs,
            functions=allowed_functions,
            temperature=_temp,
        )
    except Exception as e:
        logger.warning("agent_decide: LLM invocation failed: %s", e)
        # Finish telemetry with a failure reason
        try:
            if _trace is not None and _ds is not None and finish_decide:
                finish_decide(
                    _trace, _ds,
                    raw_response={},  # raw SDK payload not available here
                    action=None,
                    args=None,
                    reason="llm_invoke_failed",
                    started_at=_started,
                )
        except Exception:
            pass
        return AgentDecision(reply=_fallback_clarify(lang))

    # ---- No tool chosen → return assistant text (clarify/explain) ----
    if not tool_name:
        try:
            if _trace is not None and _ds is not None and finish_decide:
                finish_decide(
                    _trace, _ds,
                    raw_response={},
                    action=None,
                    args=None,
                    reason="no_tool_selected",
                    started_at=_started,
                )
        except Exception:
            pass
        return AgentDecision(reply=assistant_text or _fallback_clarify(lang))

    # ---- Validate arguments strictly via registry ----
    try:
        validated = ToolRegistry.validate_call(tool_name, tool_args or {})
    except ValidationError as ve:
        logger.info("agent_decide: validation failed for tool '%s': %s", tool_name, ve)
        try:
            if _trace is not None and _ds is not None and finish_decide:
                finish_decide(
                    _trace, _ds,
                    raw_response={},
                    action=tool_name,
                    args=(tool_args or {}),
                    reason="args_invalid",
                    started_at=_started,
                )
        except Exception:
            pass
        return AgentDecision(reply=_validation_clarify(lang, tool_name, ve))

    # ---- Resolve endpoint/op for the chosen tool ----
    try:
        endpoint = ToolRegistry.endpoint_for(tool_name)
        op = ToolRegistry.op_for(tool_name)
    except Exception as e:
        logger.error("agent_decide: failed to resolve endpoint/op for '%s': %s", tool_name, e)
        try:
            if _trace is not None and _ds is not None and finish_decide:
                finish_decide(
                    _trace, _ds,
                    raw_response={},
                    action=tool_name,
                    args=validated.model_dump(),
                    reason="endpoint_or_op_unresolved",
                    started_at=_started,
                )
        except Exception:
            pass
        return AgentDecision(reply=_fallback_clarify(lang))

    # ---- Success telemetry ----
    try:
        if _trace is not None and _ds is not None and finish_decide:
            finish_decide(
                _trace, _ds,
                raw_response={},
                action=tool_name,
                args=validated.model_dump(),
                reason=None,
                started_at=_started,
            )
    except Exception:
        pass

    # Prefer action/args (modern); keep properties for compatibility
    return AgentDecision(
        action=tool_name,
        args=validated.model_dump(),
        endpoint=endpoint,
        op=op,
        reply=None,
    )


# =============================================================================
# Localized clarifications
# =============================================================================
def _fallback_clarify(lang: str) -> str:
    if lang.startswith("cs"):
        return "Co chceš udělat? Můžu vytvořit/pojmenovat aktivum, vypsat aktiva, nebo nastavit dopady/hrozby pro konkrétní aktivum."
    if lang.startswith("sk"):
        return "Čo chceš urobiť? Môžem vytvoriť/premenovať aktívum, vypísať aktíva alebo nastaviť dopady/hrozby pre konkrétne aktívum."
    return "What would you like to do? I can create/rename an asset, list assets, or set impacts/threats for a specific asset."


def _validation_clarify(lang: str, tool_name: str, ve: ValidationError) -> str:
    # Keep this short; user should be able to fix arguments quickly.
    if lang.startswith("cs"):
        return f"Nepodařilo se použít nástroj '{tool_name}'. Potřebuji doplnit/opravit argumenty: {ve.errors()}"
    if lang.startswith("sk"):
        return f"Nepodarilo sa použiť nástroj '{tool_name}'. Potrebujem doplniť/upraviť argumenty: {ve.errors()}"
    return f"Couldn't use tool '{tool_name}'. I need different or additional arguments: {ve.errors()}"
