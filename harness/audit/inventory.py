"""Every function and method in the tree, from the AST — the denominator for the exec probe.

Pure stdlib, no import of the application, so it runs anywhere (host or container) and cannot
be fooled by an import that fails.

    python -m harness.audit.inventory --root workshops_ai --out /app/logs/audit/inventory.json
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".venv", "node_modules"}


def walk_functions(path: Path, rel: str):
    """Yield (lineno, qualname, kind, n_lines, is_async) for every def in one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError as exc:
        yield {"qualname": "<syntax-error>", "lineno": getattr(exc, "lineno", 0), "kind": "error",
               "lines": 0, "error": str(exc)}
        return

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qn = f"{prefix}{child.name}"
                yield {
                    "qualname": qn,
                    "lineno": child.lineno,
                    "kind": "async def" if isinstance(child, ast.AsyncFunctionDef) else "def",
                    "lines": (child.end_lineno or child.lineno) - child.lineno + 1,
                    "decorators": [ast.unparse(d)[:80] for d in child.decorator_list],
                }
                yield from visit(child, f"{qn}.<locals>.")
            elif isinstance(child, ast.ClassDef):
                yield from visit(child, f"{prefix}{child.name}.")
            else:
                yield from visit(child, prefix)

    yield from visit(tree, "")


def build(base: Path, roots):
    inv = {}
    for root in roots:
        rootp = base / root
        if not rootp.exists():
            continue
        for p in sorted(rootp.rglob("*.py")):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            rel = p.relative_to(base).as_posix()
            fns = list(walk_functions(p, rel))
            inv[rel] = {
                "loc": sum(1 for _ in p.open("rb")),
                "functions": fns,
                "n_functions": len(fns),
                "function_lines": sum(f.get("lines", 0) for f in fns),
            }
    return inv


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--root", action="append", default=None)
    ap.add_argument("--out", default="/app/logs/audit/inventory.json")
    args = ap.parse_args(argv)

    base = Path(args.base).resolve()
    inv = build(base, args.root or ["workshops_ai"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"base": str(base), "files": inv}, indent=1))

    nf = sum(v["n_functions"] for v in inv.values())
    print(f"{len(inv)} files, {nf} functions, "
          f"{sum(v['loc'] for v in inv.values())} lines -> {out}")


if __name__ == "__main__":
    main()
