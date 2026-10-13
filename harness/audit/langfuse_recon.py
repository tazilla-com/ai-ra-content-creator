"""Read-only reconnaissance of the hosted Langfuse project.

The generators prefer a Langfuse-hosted prompt and fall back to ``wizard/prompts.py`` only when
the fetch fails, so the prompt you can read in the repo is not necessarily the prompt production
sends. This dumps the registry (names, versions, labels, bodies) and aggregates past traces and
observations, which is the only way to see which code paths have ever run against real traffic.

Strictly GET requests. Nothing is created, updated or deleted. Credentials are read from the
environment the application itself uses and are never printed or written to the output file.

    python -m harness.audit.langfuse_recon --out /app/logs/audit/langfuse.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROMPT_NAMES_IN_CODE = [
    "wizard/services", "wizard/assets", "wizard/impacts", "wizard/risks", "wizard/controls",
    "helper/web_search", "workshops/asset-agent-system",
]


def make_get(base, auth):
    def get(path, **params):
        q = urllib.parse.urlencode(params)
        url = f"{base}{path}" + (f"?{q}" if q else "")
        req = urllib.request.Request(
            url, headers={"Authorization": f"Basic {auth}", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return {"__error__": f"HTTP {e.code}", "body": e.read()[:300].decode(errors="replace")}
        except Exception as e:
            return {"__error__": f"{type(e).__name__}: {e}"}
    return get


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/app/logs/audit/langfuse.json")
    ap.add_argument("--trace-pages", type=int, default=3)
    args = ap.parse_args(argv)

    base = (os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com").rstrip("/")
    pk, sk = os.getenv("LANGFUSE_PUBLIC_KEY", ""), os.getenv("LANGFUSE_SECRET_KEY", "")
    print(f"host={base}  enabled={os.getenv('LANGFUSE_ENABLED')}  "
          f"label={os.getenv('LANGFUSE_PROMPT_LABEL')}  keys={'present' if pk and sk else 'MISSING'}")
    if not (pk and sk):
        raise SystemExit("no Langfuse credentials in the environment")

    get = make_get(base, base64.b64encode(f"{pk}:{sk}".encode()).decode())
    out = {"host": base, "prompt_label": os.getenv("LANGFUSE_PROMPT_LABEL")}

    print("\n=== PROMPT REGISTRY ===")
    idx = get("/api/public/v2/prompts", limit=100)
    out["index"] = idx
    names = []
    if "__error__" in idx:
        print("  ", idx)
    else:
        for p in idx.get("data", []):
            names.append(p.get("name"))
            print(f"  - {str(p.get('name')):42s} versions={p.get('versions')} "
                  f"labels={p.get('labels')} tags={p.get('tags')} updated={p.get('lastUpdatedAt')}")

    # always probe the names the code asks for, even if the index did not list them
    detail = {}
    for n in sorted(set(names) | set(PROMPT_NAMES_IN_CODE)):
        d = get(f"/api/public/v2/prompts/{urllib.parse.quote(n, safe='')}")
        detail[n] = d
        if "__error__" in d:
            print(f"\n----- {n}: {d['__error__']} {'(referenced in code but ABSENT)' if n in PROMPT_NAMES_IN_CODE else ''}")
            continue
        body = d.get("prompt")
        text = body if isinstance(body, str) else json.dumps(body, indent=1)
        print(f"\n----- {n}  v{d.get('version')} labels={d.get('labels')} type={d.get('type')} "
              f"chars={len(text)} config={json.dumps(d.get('config'))[:160]}")
        print("      " + text[:1200].replace("\n", "\n      "))
    out["prompts"] = detail

    print("\n\n=== TRACES ===")
    agg_names, newest, oldest, total = {}, None, None, None
    for page in range(1, args.trace_pages + 1):
        tr = get("/api/public/traces", limit=100, page=page)
        if "__error__" in tr:
            print("  ", tr); break
        data = tr.get("data", [])
        total = tr.get("meta", {}).get("totalItems", total)
        if not data:
            break
        for t in data:
            agg_names[t.get("name")] = agg_names.get(t.get("name"), 0) + 1
            ts = t.get("timestamp")
            newest = max(newest, ts) if newest else ts
            oldest = min(oldest, ts) if oldest else ts
    print(f"  totalItems={total}  window={oldest} .. {newest}")
    for n, c in sorted(agg_names.items(), key=lambda x: -x[1]):
        print(f"   {c:5d}  {n}")
    out["trace_names"] = agg_names
    out["trace_window"] = [oldest, newest]
    out["trace_total"] = total

    print("\n=== OBSERVATIONS (what actually executed) ===")
    seen, ototal = {}, None
    for page in range(1, args.trace_pages + 1):
        ob = get("/api/public/observations", limit=100, page=page)
        if "__error__" in ob:
            print("  ", ob); break
        data = ob.get("data", [])
        ototal = ob.get("meta", {}).get("totalItems", ototal)
        if not data:
            break
        for o in data:
            k = f"{o.get('type')}|{o.get('name')}|{o.get('model')}"
            seen[k] = seen.get(k, 0) + 1
    print(f"  totalItems={ototal}")
    for k, c in sorted(seen.items(), key=lambda x: -x[1]):
        print(f"   {c:5d}  {k}")
    out["observations"] = seen
    out["observation_total"] = ototal

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
