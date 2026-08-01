"""Function-level execution tracer.

Uses ``sys.setprofile`` rather than ``sys.settrace``: the profile hook fires only on call and
return, not on every line, which is cheap enough to leave on during live LLM runs (which are
network-bound anyway). Line-level coverage is not needed here — the audit question is binary,
"was this function ever entered".

No third-party dependency, so it runs inside the production image unchanged.

Usage
-----
    from harness.audit.exec_probe import ExecProbe
    probe = ExecProbe(roots=["workshops_ai"])
    probe.start()
    ...
    probe.dump("/app/logs/audit/exec.json")

or wrap a module entry point::

    python -m harness.audit.exec_probe --out /app/logs/audit/exec.json -m harness.runner chain
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path


class ExecProbe:
    """Records every Python function entered, keyed by file:line:qualname."""

    def __init__(self, roots=("workshops_ai",), base=None):
        self.base = Path(base or os.getcwd()).resolve()
        self.roots = tuple(roots)
        self.hits: dict[str, int] = {}
        self._decided: dict[str, str | None] = {}   # co_filename -> relpath or None
        self._started_at = time.time()
        self._lock = threading.Lock()

    # -- the hot path ---------------------------------------------------------
    def _relpath(self, filename: str):
        try:
            return self._decided[filename]
        except KeyError:
            pass
        rel = None
        try:
            p = Path(filename).resolve()
            r = p.relative_to(self.base).as_posix()
            if any(r == root or r.startswith(root + "/") for root in self.roots):
                rel = r
        except (ValueError, OSError):
            rel = None
        self._decided[filename] = rel
        return rel

    def _profile(self, frame, event, arg):
        if event != "call":
            return
        code = frame.f_code
        rel = self._relpath(code.co_filename)
        if rel is None:
            return
        key = f"{rel}:{code.co_firstlineno}:{getattr(code, 'co_qualname', code.co_name)}"
        h = self.hits
        h[key] = h.get(key, 0) + 1

    # -- control --------------------------------------------------------------
    def start(self):
        threading.setprofile(self._profile)
        sys.setprofile(self._profile)
        return self

    def stop(self):
        sys.setprofile(None)
        threading.setprofile(None)
        return self

    def dump(self, path):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base": str(self.base),
            "roots": list(self.roots),
            "started_at": self._started_at,
            "ended_at": time.time(),
            "n_functions_entered": len(self.hits),
            "hits": dict(sorted(self.hits.items())),
        }
        # merge with an existing dump so several drives accumulate
        if out.exists():
            try:
                prev = json.loads(out.read_text())
                merged = dict(prev.get("hits", {}))
                for k, v in payload["hits"].items():
                    merged[k] = merged.get(k, 0) + v
                payload["hits"] = dict(sorted(merged.items()))
                payload["n_functions_entered"] = len(merged)
                payload["started_at"] = prev.get("started_at", payload["started_at"])
                payload["merged_runs"] = prev.get("merged_runs", 1) + 1
            except Exception:
                pass
        out.write_text(json.dumps(payload, indent=1))
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="run a module under the execution probe")
    ap.add_argument("--out", default="/app/logs/audit/exec.json")
    ap.add_argument("--root", action="append", default=None,
                    help="package root to record (repeatable); default workshops_ai")
    ap.add_argument("-m", "--module", required=True, help="module to run as __main__")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)

    probe = ExecProbe(roots=tuple(args.root or ("workshops_ai",)))
    atexit.register(lambda: (probe.stop(), print(f"[exec_probe] {len(probe.hits)} functions entered -> "
                                                 f"{probe.dump(args.out)}", file=sys.stderr)))
    sys.argv = [args.module] + [a for a in args.rest if a != "--"]
    probe.start()
    try:
        runpy.run_module(args.module, run_name="__main__", alter_sys=True)
    finally:
        probe.stop()


if __name__ == "__main__":
    main()
