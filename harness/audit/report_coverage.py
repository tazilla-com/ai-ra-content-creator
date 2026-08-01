"""Join the AST inventory against one or more exec-probe dumps.

The output is the audit's central table: for every file, how many of its functions were entered
under real traffic. A file at 0/N after a drive that exercised the whole route surface is dead in
a way no import graph can tell you — it may well be imported, and still never run.

    python -m harness.audit.report_coverage \\
        --inventory /app/logs/audit/inventory.json \\
        --exec /app/logs/audit/exec_live.json --exec /app/logs/audit/exec_nollm.json \\
        --out /app/logs/audit/coverage.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_hits(paths):
    hits = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        for k, v in d.get("hits", {}).items():
            hits[k] = hits.get(k, 0) + v
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--exec", action="append", required=True)
    ap.add_argument("--out", default="/app/logs/audit/coverage.json")
    ap.add_argument("--show", type=int, default=60, help="how many files to print")
    args = ap.parse_args(argv)

    inv = json.loads(Path(args.inventory).read_text())["files"]
    hits = load_hits(args.exec)

    # Exec keys are "relpath:co_firstlineno:co_qualname". They CANNOT be joined on line number:
    # for a decorated function CPython sets co_firstlineno to the first decorator line, while the
    # AST reports the `def` line. Joining on lineno silently marks every decorated endpoint, tool
    # and validator as never-run. Join on qualname; keep the line only for reporting.
    entered = {}
    for key, n in hits.items():
        rel, _, rest = key.partition(":")
        lineno, _, qual = rest.partition(":")
        by_file = entered.setdefault(rel, {})
        by_file[qual] = by_file.get(qual, 0) + n

    rows = []
    for rel, meta in sorted(inv.items()):
        fns = meta["functions"]
        got = entered.get(rel, {})
        hit, miss = [], []
        for f in fns:
            (hit if f["qualname"] in got else miss).append(f)
        # a qualname the probe saw that the AST did not produce is a comprehension/lambda frame
        # or a dynamically created function — count it, but do not let it inflate the denominator
        extra = sorted(set(got) - {f["qualname"] for f in fns})
        rows.append({
            "file": rel,
            "loc": meta["loc"],
            "n_functions": len(fns),
            "n_entered": len(hit),
            "pct": round(100 * len(hit) / len(fns), 1) if fns else None,
            "entered_calls": sum(got.values()),
            "never_entered": [f"{f['qualname']}:{f['lineno']}" for f in miss],
            "entered_not_in_ast": extra,
            "module_level_only": len(fns) == 0,
        })

    dead = [r for r in rows if r["n_functions"] and r["n_entered"] == 0]
    partial = [r for r in rows if r["n_entered"] and r["n_entered"] < r["n_functions"]]
    full = [r for r in rows if r["n_functions"] and r["n_entered"] == r["n_functions"]]

    print(f"files: {len(rows)}   "
          f"NO function entered: {len(dead)} ({sum(r['loc'] for r in dead)} loc)   "
          f"partial: {len(partial)}   fully entered: {len(full)}")
    print(f"functions: {sum(r['n_functions'] for r in rows)} total, "
          f"{sum(r['n_entered'] for r in rows)} entered\n")

    print("=== FILES WHERE NOT ONE FUNCTION EVER RAN ===")
    for r in sorted(dead, key=lambda r: -r["loc"])[:args.show]:
        print(f"  {r['loc']:5d} loc  {r['n_functions']:3d} fns   {r['file']}")

    print("\n=== PARTIALLY EXERCISED (biggest unexercised remainder first) ===")
    for r in sorted(partial, key=lambda r: -(r["n_functions"] - r["n_entered"]))[:args.show]:
        print(f"  {r['n_entered']:3d}/{r['n_functions']:3d} fns  {r['loc']:5d} loc  {r['file']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "n_exec_dumps": len(args.exec)}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
