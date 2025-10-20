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
#   - Integrated lightweight debug telemetry (telemetry_debug.DecideSpan) — best effort.
#   - Language policy: the agent must always respond in the user's language (cs/sk/en) inferred from
#     `language` or `locale`. We inject a compact system prompt plus an optional domain prompt
#     to enforce style, one-tool-per-turn, slot-filling, and full-list rules.

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from workshops_ai.graph.factory import GraphDeps

from pydantic import BaseModel, Field, ConfigDict, ValidationError

from workshops_ai.llm.tools_adapter import sanitize_tools, canonical_name
from workshops_ai.agents.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

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
# Prompt loading / rendering helpers
# =============================================================================

_PROMPTS_DIR = Path("workshops_ai/prompts")

def _read_text_safe(path: Path) -> str:
    """Read a UTF-8 text file; return '' if missing or unreadable."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

def _minify_markdown(md: str) -> str:
    """
    Token-economical minifier for Markdown prompts:
      - Remove HTML/MD comments.
      - Collapse excessive blank lines.
      - Strip trailing spaces.
    Keeps headings and list formatting intact.
    """
    if not md:
        return ""
    # Remove <!-- ... --> comments (single line or multi-line)
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)
    # Remove fenced code comments that are entirely commented out (rare)
    md = re.sub(r"```(?:.*?)```", lambda m: m.group(0).rstrip(), md, flags=re.DOTALL)
    # Collapse 3+ blank lines to 2
    md = re.sub(r"\n{3,}", "\n\n", md)
    # Strip trailing spaces per line
    md = "\n".join(line.rstrip() for line in md.splitlines())
    return md.strip()

def _compact_history(history: List[dict[str, str]] | None, max_chars: int = 1200) -> str:
    """
    Produce a compact, language-agnostic history summary suitable for injection into system prompt.
    Keeps only short 'user'/'assistant' messages; drops 'system'.
    """
    if not history:
        return ""
    buf: List[str] = []
    used = 0
    for m in history:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        # Keep first 300 chars from each turn to maintain signal
        snippet = (content[:300] + ("…" if len(content) > 300 else ""))
        line = f"{role}: {snippet}"
        if used + len(line) + 1 > max_chars:
            break
        buf.append(line)
        used += len(line) + 1
    return " | ".join(buf)

def _tools_for_prompt(functions: List[Dict[str, Any]]) -> str:
    """
    Convert OpenAI 'tools' (sanitized) into a compact JSON for {{tools}}:
      [
        {"name": "...", "description": "...", "required": ["..."], "parameters": {...}},
        ...
      ]
    We keep 'parameters' but leave it compact (no whitespace).
    """
    out = []
    for f in functions:
        fn = f.get("function") or {}
        name = fn.get("name")
        desc = fn.get("description")
        params = fn.get("parameters") or {}
        required = []
        try:
            required = list((params.get("required") or []))
        except Exception:
            required = []
        out.append({
            "name": name,
            "description": desc,
            "required": required,
            "parameters": params,  # keep full schema (compact) for validation hints in the model
        })
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))

def _render_system_prompt(
    *,
    tab: str,
    language: str,
    locale: str | None,
    org_id: str | None,
    thread_id: str | None,
    policy: Dict[str, Any] | None,
    selection: Dict[str, Any] | None,
    history_compact: str,
    tools_json: str,
) -> str:
    """
    Load and render the compact system prompt with runtime variables.
    Attempts to load an optional domain prompt for the given `tab` as well.
    Returns a single minified Markdown string.
    """
    sys_md = _read_text_safe(_PROMPTS_DIR / "system.md")
    sys_md = _minify_markdown(sys_md)

    # Optional domain prompt (assets.md, impacts.md, threats.md, controls.md)
    dom_md = _read_text_safe(_PROMPTS_DIR / f"{tab}.md")
    dom_md = _minify_markdown(dom_md)

    # Best-effort variable map
    policy = policy or {}
    selection = selection or {}
    locale = locale or language
    org_id = org_id or ""
    thread_id = thread_id or ""

    def repl(md: str) -> str:
        return (
            md.replace("{{tab}}", tab or "")
              .replace("{{language}}", language or "en")
              .replace("{{locale}}", locale or "")
              .replace("{{org_id}}", org_id or "")
              .replace("{{thread_id}}", thread_id or "")
              .replace("{{history}}", history_compact or "")
              .replace("{{tools}}", tools_json or "[]")
              .replace("{{selection}}", json.dumps(selection, ensure_ascii=False))
              .replace("{{policy.preview_required}}", json.dumps(policy.get("preview_required"), ensure_ascii=False))
              .replace("{{policy.emit_changes_enabled}}", json.dumps(policy.get("emit_changes_enabled"), ensure_ascii=False))
        )

    sys_md = repl(sys_md)
    dom_md = repl(dom_md) if dom_md else ""

    # Final combined prompt: system first, then domain (if any), then a tiny guardrail.
    guard = (
        f"You MUST reply in {language}. Use exactly one tool if needed; "
        "otherwise reply conversationally. Ask at most one concise question if a required slot is missing."
    )

    parts = [p for p in [sys_md, dom_md, guard] if p]
    return "\n\n".join(parts)


# =============================================================================
# Create-flow lock helper
#   🔒 During slot-filling for asset creation, restrict the LLM to *only*
#   `asset.create` to prevent escapes to other tools mid-flow.
#   The detection is tolerant to several selector shapes used by the graph.
# =============================================================================
def _should_lock_to_asset_create(selection: Dict[str, Any] | None) -> bool:
    sel = selection or {}
    # direct boolean flag
    if sel.get("force_to_asset_create") is True:
        return True

    # common "create_flow" container
    cf = sel.get("create_flow")
    if cf is True:
        return True
    if isinstance(cf, dict):
        if cf.get("active") is True:
            return True
        # look for target/op/action hints
        tgt = (cf.get("target") or cf.get("action") or cf.get("op") or cf.get("tool") or "").strip().lower()
        if tgt in {"asset.create", "create_asset"}:
            return True
        # presence of pending_fields typically implies slot-fill in progress
        if isinstance(cf.get("pending_fields"), (list, tuple)) and cf.get("pending_fields"):
            return True

    # top-level hints used by some callers
    pending_action = str(sel.get("pending_action") or sel.get("pending_op") or "").strip().lower()
    if pending_action in {"asset.create", "create_asset"}:
        return True
    if isinstance(sel.get("pending_fields"), (list, tuple)) and sel.get("pending_fields"):
        # treat as create-flow only if an explicit target is creation
        tgt = str(sel.get("pending_target") or sel.get("target") or "").strip().lower()
        if tgt in {"asset.create", "create_asset"}:
            return True

    return False


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

            # Normalize explicit tool_use items if present
            try:
                items = getattr(resp, "output", None)
                if isinstance(items, list):
                    for it in items:
                        t = getattr(it, "type", None) or (it.get("type") if isinstance(it, dict) else None)
                        if t == "tool_use":
                            name = getattr(it, "name", None) or (it.get("name") if isinstance(it, dict) else None)
                            tool_input = getattr(it, "input", None) or (it.get("input") if isinstance(it, dict) else None)
                            if isinstance(name, str) and isinstance(tool_input, dict):
                                name = canonical_name(name, name_map)
                                return name, tool_input, None

                # Otherwise fallback to assistant text
                text = getattr(resp, "output_text", None)
                if not text and hasattr(resp, "choices"):
                    text = getattr(resp.choices[0].message, "content", None)  # type: ignore[index]
                return None, None, (str(text) if text else "")
            except Exception:
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

            # 🔧 Bind OpenAI-style tools into the LC model if possible
            model_to_use = client
            try:
                bind_tools = getattr(client, "bind_tools", None)
                if callable(bind_tools):
                    # LangChain expects OpenAI tool schema: [{"type":"function","function":{...}}, ...]
                    model_to_use = bind_tools(sanitize_tools(functions)[0])  # type: ignore[arg-type]
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
      - Emit optional debug telemetry spans (best-effort, non-fatal)

    ✨ Language & prompt behavior in this version:
      - Compute `lang` from `language` or `locale` (default 'en').
      - Inject compact system prompt with runtime placeholders ({{tools}}, {{language}}, {{history}}, …).
      - Optionally inject a domain prompt for the current `tab` if present.
      - Add a tiny guardrail line (reply in lang, one tool if needed, one concise clarify).
    """
    lang = (language or locale or "en").strip().lower()
    ToolRegistry.ensure_initialized()
    tab_key = (tab or "").strip().lower()
    history = history or []
    msg_len = len((message or "").strip())

    # Resolve available tools from registry for the tab
    registry_functions_all = ToolRegistry.as_openai_functions(tab_key)
    if not registry_functions_all:
        logger.warning("agent_decide: no tools registered for tab='%s'; reply-only fallback.", tab_key)
        return AgentDecision(reply=_fallback_clarify(lang))

    # 🔒 Create-flow lock: if we're in the middle of asset creation slot-fill,
    #     clamp allowed_actions to only ['asset.create'] to avoid tool escape.
    if tab_key == "assets" and _should_lock_to_asset_create(selection):
        allowed_actions = ["asset.create"]

    # If Manager provided a whitelist (allowed_actions), filter by function name.
    if allowed_actions:
        allowed = set(a.strip() for a in allowed_actions if a and isinstance(a, str))
        filtered: list[dict[str, Any]] = []
        for f in registry_functions_all:
            try:
                fn = f.get("function") or {}
                name = fn.get("name")
                if isinstance(name, str) and name in allowed:
                    filtered.append(f)
            except Exception:
                continue
        registry_functions = filtered or registry_functions_all  # fallback to avoid dead-ends
    else:
        registry_functions = registry_functions_all

    # Lightweight priority bias (does NOT force a specific tool)
    def _fn_name(item: dict[str, Any]) -> str:
        try:
            return str(((item.get("function") or {}).get("name")) or "")
        except Exception:
            return ""

    def _score(name: str, user_msg: str) -> int:
        msg = user_msg.lower()
        s = 0
        if ".create" in name:
            s += 10
        if ".update" in name or ".set" in name:
            s += 5
        create_tokens = ["vytvoř", "vytvor", "přidej", "pridaj", "add", "create", "new", "založ", "zaloz"]
        list_tokens   = ["ukaž", "ukaz", "vypiš", "vypis", "seznam", "list", "show"]
        explain_tokens= ["vysvětli", "vysvetli", "explain", "co je", "what is"]
        update_tokens = ["přejmenuj", "premen", "rename", "upravit", "uprav", "update", "nastav", "set"]
        if any(t in msg for t in create_tokens):
            if name.endswith(".create"):
                s += 100
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
        _indexed = list(enumerate(registry_functions))
        _indexed.sort(key=lambda pair: (-_score(_fn_name(pair[1]), message), pair[0]))
        registry_functions = [it for _, it in _indexed]
    except Exception:
        pass  # prioritization must never break the flow

    # ---- Build LLM messages (system + optional domain + history + user) ----
    history_compact = _compact_history(history)
    tools_json = _tools_for_prompt(registry_functions)

    # Best-effort org/thread/policy from deps (optional)
    org_id = getattr(getattr(deps, "context", None), "org_id", None)
    thread_id = getattr(getattr(deps, "context", None), "thread_id", None)
    policy = {}
    try:
        pol = getattr(getattr(deps, "context", None), "policy", None)
        if isinstance(pol, dict):
            policy = pol
    except Exception:
        policy = {}

    system_text = _render_system_prompt(
        tab=tab_key,
        language=lang,
        locale=locale or lang,
        org_id=org_id,
        thread_id=thread_id,
        policy=policy,
        selection=selection or {},
        history_compact=history_compact,
        tools_json=tools_json,
    )

    # Start fresh messages with our unified system prompt
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": system_text}]
    # Add a minimal language guardrail as an extra system line (cheap redundancy)
    msgs.append({
        "role": "system",
        "content": f"Always answer in {lang}. Use exactly one tool if needed; otherwise reply concisely in Markdown.",
    })
    # Append a trimmed version of prior turns so the model has conversational context
    for m in history:
        role = m.get("role")
        if role in ("user", "assistant"):
            msgs.append({"role": role, "content": (m.get("content") or "")})
    # Current user message last
    msgs.append({"role": "user", "content": message})

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
                tool_names = [((f.get("function") or {}).get("name")) for f in registry_functions if isinstance(f, dict)]
                tool_names = [t for t in tool_names if t]
            except Exception:
                pass
            # system prompt length from collected 'system' role messages (content omitted)
            system_prompt_len = len(system_text)
            _ds = set_decide_span(
                _trace,
                tab=tab_key,
                model=(model_name or "<unknown>"),
                tool_choice="auto",
                tools=tool_names,
                system_prompt=" " * system_prompt_len,
                user_message=message,
            )
    except Exception:
        _ds = None  # telemetry must never break the flow

    # ---- Invoke LLM with tool-calling ----
    started = time.perf_counter()
    try:
        # Clamp/override temperature only if not provided in deps; keep your explicit settings intact.
        _temp = getattr(deps, "llm_temperature", None)
        if _temp is None:
            _temp = 0.15  # slightly lower default → crisper tool choice

        tool_name, tool_args, assistant_text = _call_llm_with_tools(
            deps=deps,
            messages=msgs,
            functions=registry_functions,
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
                    started_at=started,
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
                    started_at=started,
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
                    started_at=started,
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
                    started_at=started,
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
                started_at=started,
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
    errs = ve.errors()
    if lang.startswith("cs"):
        return f"Nepodařilo se použít nástroj „{tool_name}“. Potřebuji doplnit/opravit argumenty: {errs}"
    if lang.startswith("sk"):
        return f"Nepodarilo sa použiť nástroj „{tool_name}“. Potrebujem doplniť/upraviť argumenty: {errs}"
    return f"Couldn't use tool '{tool_name}'. I need different or additional arguments: {errs}"
