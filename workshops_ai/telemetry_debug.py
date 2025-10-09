# workshops_ai/telemetry_debug.py
from __future__ import annotations
import json, time, uuid, contextvars
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

_req_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("req_id", default="")

def get_req_id() -> str:
    rid = _req_id_var.get()
    if not rid:
        rid = str(uuid.uuid4())
        _req_id_var.set(rid)
    return rid

def _redact(d: Any) -> Any:
    try:
        s = json.dumps(d, ensure_ascii=False)
    except Exception:
        return d
    # jednoduchá redakce tajemství a dlouhých řetězců
    s = s.replace("sk-", "sk-****")
    return json.loads(s)

@dataclass
class DecideSpan:
    tab: str
    model: str
    tool_choice: str
    tools: List[str]
    system_prompt_len: int
    user_message: str
    llm_latency_ms: int = 0
    raw_response: Optional[Dict[str, Any]] = None
    action: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None  # e.g. "no_tool_selected" / "args_invalid"

@dataclass
class Trace:
    request_id: str
    ui_request: Dict[str, Any]
    decide: Optional[DecideSpan] = None
    ui_response_summary: Optional[str] = None

_last_traces: Dict[str, Trace] = {}

def start_trace(ui_req: Dict[str, Any]) -> Trace:
    rid = get_req_id()
    t = Trace(request_id=rid, ui_request=_redact(ui_req))
    _last_traces[rid] = t
    return t

def set_decide_span(
    trace: Trace, *, tab: str, model: str, tool_choice: str,
    tools: List[str], system_prompt: str, user_message: str
) -> DecideSpan:
    ds = DecideSpan(
        tab=tab, model=model, tool_choice=tool_choice, tools=tools,
        system_prompt_len=len(system_prompt), user_message=user_message
    )
    trace.decide = ds
    return ds

def finish_decide(trace: Trace, ds: DecideSpan, *, raw_response: Dict[str, Any],
                  action: Optional[str], args: Optional[Dict[str, Any]], reason: Optional[str],
                  started_at: float) -> None:
    ds.raw_response = _redact(raw_response)
    ds.action = action
    ds.args = _redact(args) if args else None
    ds.reason = reason
    ds.llm_latency_ms = int((time.time() - started_at) * 1000)

def summarize_ui_response(trace: Trace, ui_resp: Dict[str, Any]) -> None:
    msgs = ui_resp.get("messages", [])
    first = (msgs[0]["content"][:160] + "…") if msgs else "<EMPTY>"
    trace.ui_response_summary = f"messages={len(msgs)} first='{first}'"

def get_last_trace(request_id: str) -> Dict[str, Any]:
    tr = _last_traces.get(request_id)
    return asdict(tr) if tr else {}
