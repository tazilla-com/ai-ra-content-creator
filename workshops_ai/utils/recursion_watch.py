# File: workshops_ai/utils/recursion_watch.py
from __future__ import annotations
import sys, threading
from types import FrameType
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional

@dataclass
class CallEvent:
    mod: str
    fn: str
    file: str
    line: int

class RecursionWatch:
    """
    Ultra-light call tracer for detecting runaway recursion loops.
    Collects 'call' events for modules matching given prefixes (default: 'workshops_ai').
    Use .tail_text(n) after an exception to see the last N intra-project calls.
    """
    def __init__(self, *, max_events: int = 6000, module_prefixes: Iterable[str] = ("workshops_ai",)):
        self.max_events = max_events
        self.module_prefixes = tuple(module_prefixes)
        self._buf: list[CallEvent] = []
        self._lock = threading.Lock()
        self._old_sys: Optional[object] = None
        self._old_threading: Optional[object] = None

    def _profile(self, frame: FrameType, event: str, arg):
        if event != "call":
            return
        mod = frame.f_globals.get("__name__", "")
        if self.module_prefixes and not any(mod.startswith(p) for p in self.module_prefixes):
            return
        code = frame.f_code
        ev = CallEvent(mod=mod, fn=code.co_name, file=code.co_filename, line=code.co_firstlineno)
        with self._lock:
            self._buf.append(ev)
            if len(self._buf) > self.max_events:
                del self._buf[: len(self._buf) - self.max_events]

    @contextmanager
    def enabled(self):
        # Trace current thread; also set default for threads created *after* this point
        self._old_sys = sys.getprofile()
        self._old_threading = threading._profile  # type: ignore[attr-defined]
        sys.setprofile(self._profile)
        threading.setprofile(self._profile)
        try:
            yield self
        finally:
            sys.setprofile(self._old_sys)
            threading.setprofile(self._old_threading)

    def tail_text(self, n: int = 200) -> str:
        tail = self._buf[-n:]
        return "\n".join(f"{i+1:>3}. {e.mod}:{e.fn}  ({e.file}:{e.line})" for i, e in enumerate(tail))
