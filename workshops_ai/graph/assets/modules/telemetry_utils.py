# workshops_ai/graph/assets/modules/telemetry_utils.py

from __future__ import annotations
import asyncio, os, time, logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from workshops_ai.diagnostics.json_probe import dump_crash_packet
except Exception:  # pragma: no cover
    dump_crash_packet = None  # type: ignore

try:
    from workshops_ai.telemetry import get_request_id  # type: ignore
except Exception:  # pragma: no cover
    def get_request_id(default: str = "unknown") -> str:  # type: ignore
        return default

DIAG = os.getenv("WSAI_DIAG", "0") in ("1", "true", "True", "yes")

def diag_crashpack(context: str, guarded_obj: Any, exc: BaseException, preview: dict | None = None) -> None:
    if not (DIAG and dump_crash_packet):
        return
    try:
        rid = get_request_id(default="unknown")
    except Exception:
        rid = "unknown"
    path = dump_crash_packet(rid, (preview or {}), guarded_obj, exc)  # type: ignore[misc]
    logger.error("DIAG_CRASHPACK context=%s request_id=%s crashpack=%s", context, rid, path)

def event(state, *, kind: str, tool: str, ok: bool = True, meta: dict | None = None) -> float:
    t0 = time.perf_counter()
    state.agent_events.append({"event_type": kind, "tool": tool, "ok": ok, "meta": meta or {}})
    return t0

def finish(state, t0: float, *, tool: str, ok: bool, meta: dict | None = None) -> None:
    dur = int((time.perf_counter() - t0) * 1000)
    state.agent_events.append({
        "event_type": "tool.call.success" if ok else "tool.call.fail",
        "tool": tool, "ok": ok, "duration_ms": dur, "meta": meta or {}
    })

def schedule(coro_or_fn):
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro_or_fn if asyncio.iscoroutine(coro_or_fn) else coro_or_fn())
    except RuntimeError:
        asyncio.run(coro_or_fn if asyncio.iscoroutine(coro_or_fn) else coro_or_fn())
