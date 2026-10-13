"""Drive the real ASGI app in-process, over its whole route surface, under the execution probe.

Why in-process rather than over HTTP: the probe has to live in the same interpreter as the code
it is recording. Driving the app through ``starlette.testclient`` runs the genuine middleware
stack, dependency graph, routers and WebSocket handler — the same objects uvicorn would serve —
while keeping everything inside one process the probe can see.

Live by default: the wizard steps and the agent make real provider calls, exactly like the rest
of the harness. ``--no-llm`` restricts the run to routes that cost nothing.

    python -m harness.audit.surface_drive --out /app/logs/audit/exec.json
    python -m harness.audit.surface_drive --no-llm --out /app/logs/audit/exec_nollm.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from harness.audit.exec_probe import ExecProbe

SCENARIO = Path("harness/scenarios/sample_org.json")


class Driver:
    def __init__(self, client, log):
        self.c = client
        self.log = log

    def call(self, method, path, *, tag="", **kw):
        started = time.time()
        entry = {"method": method, "path": path, "tag": tag}
        try:
            r = self.c.request(method, path, **kw)
            entry.update(status=r.status_code, ms=round((time.time() - started) * 1000))
            body = r.text or ""
            entry["body_head"] = body[:400]
            try:
                entry["json_keys"] = sorted(r.json().keys()) if body.strip().startswith("{") else None
            except Exception:
                entry["json_keys"] = None
        except Exception as exc:
            entry.update(status=None, ms=round((time.time() - started) * 1000),
                         error=f"{type(exc).__name__}: {exc}"[:500])
        self.log.append(entry)
        note = ""
        if entry.get("status") and not 200 <= entry["status"] < 300:
            note = "  <- " + (entry.get("body_head") or "")[:260].replace("\n", " ")
        print(f"  {entry.get('status')!s:>5}  {method:6s} {path}  {entry.get('tag','')}{note}", flush=True)
        return entry


def enumerate_routes(app):
    out = []
    for r in app.routes:
        path = getattr(r, "path", None)
        methods = sorted(getattr(r, "methods", []) or [])
        kind = type(r).__name__
        out.append({"path": path, "methods": methods, "kind": kind,
                    "name": getattr(r, "name", None),
                    "include_in_schema": getattr(r, "include_in_schema", None)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/app/logs/audit/exec.json")
    ap.add_argument("--requests-out", default="/app/logs/audit/requests.json")
    ap.add_argument("--routes-out", default="/app/logs/audit/routes.json")
    ap.add_argument("--no-llm", action="store_true", help="skip everything that would call a provider")
    ap.add_argument("--evaluation", action="store_true", help="also drive POST /wizard/evaluation (whole chain again)")
    ap.add_argument("--max-assets", type=int, default=4)
    args = ap.parse_args(argv)

    probe = ExecProbe(roots=("workshops_ai",)).start()
    log: list[dict] = []
    routes: list[dict] = []

    try:
        from starlette.testclient import TestClient

        from workshops_ai.app.main import app

        routes = enumerate_routes(app)
        print(f"[surface_drive] {len(routes)} routes registered", flush=True)

        scen = json.loads(SCENARIO.read_text()) if SCENARIO.exists() else {
            "website_content": "A small logistics company.", "organization_profile": {}, "language": "en"}
        website = scen["website_content"]
        org = scen.get("organization_profile") or {}

        with TestClient(app) as client:
            d = Driver(client, log)

            # ---- 1. health / meta: no auth, no cost -------------------------
            for p in ("/health", "/livez", "/readyz", "/health/openai", "/openapi.json", "/docs", "/"):
                d.call("GET", p, tag="meta")

            # ---- 2. every zero-parameter GET the router table declares ------
            for r in routes:
                p = r["path"]
                if not p or "{" in p or "GET" not in (r["methods"] or []):
                    continue
                if p in ("/health", "/livez", "/readyz", "/health/openai", "/openapi.json", "/docs", "/"):
                    continue
                d.call("GET", p, tag="enumerated-get")

            # ---- 3. session + catalog surface -------------------------------
            sid = "audit" + str(int(time.time()))[-8:]
            H = {"X-Session-Id": sid}
            d.call("GET", "/api/catalogs/asset-types", headers=H, tag="catalog")
            d.call("GET", "/api/catalogs/impacts", headers=H, tag="catalog")
            d.call("GET", "/api/catalogs/threats", headers=H, tag="catalog")
            d.call("GET", "/api/catalogs/controls", headers=H, tag="catalog")
            d.call("GET", "/api/catalogs/sectors", headers=H, tag="catalog")
            d.call("GET", "/api/catalogs/scraped-content", headers=H, tag="catalog")

            # ---- 4. the legacy/unclear surfaces: drive them to see what they do
            d.call("POST", "/graph/assets", json={"op": "list"}, headers=H, tag="three-tab")
            d.call("POST", "/graph/threats", json={"op": "list"}, headers=H, tag="three-tab")
            d.call("POST", "/graph/impacts", json={"op": "list"}, headers=H, tag="three-tab")
            d.call("GET", "/graph/assets", headers=H, tag="three-tab")
            d.call("POST", "/api/rag/search", json={"query": "iso 27001"}, headers=H, tag="rag-unmounted")
            # the three-tab chat entry point, with a body its contract actually accepts
            # (UIRequest: org_id + active_tab required, extra="forbid")
            for tab in ("assets", "impacts", "threats", "controls"):
                d.call("POST", "/api/chat", headers=H, tag=f"three-tab-chat:{tab}", timeout=300,
                       json={"org_id": "audit-org", "active_tab": tab, "message": "List what I have."})

            if args.no_llm:
                print("[surface_drive] --no-llm: stopping before provider calls", flush=True)
            else:
                # ---- 5. the wizard chain, threaded through one session ------
                # NOTE the shapes: content goes in `website_content: {content: ...}` (a
                # WebsiteContent model, where `{}` is the magic "no content" value), and the
                # organization profile is validated as api.schema.organization.Organization,
                # which is extra="forbid" and has NO `sector` field -- so the bundled
                # harness scenario's profile cannot be POSTed here unfiltered.
                ORG_FIELDS = {"organizationName", "legalForm", "registeredOffice",
                              "country", "numberOfEmployees", "organizationalUnits", "jobPositions"}
                org_api = {k: v for k, v in org.items() if k in ORG_FIELDS}
                dropped = sorted(set(org) - ORG_FIELDS)
                if dropped:
                    print(f"  [note] dropped from organization_profile (not on Organization): {dropped}",
                          flush=True)
                base = {"website_content": {"content": website}, "organization_profile": org_api,
                        "language": "en"}
                d.call("POST", "/wizard/services", json={**base, "min_services": 1, "max_services": 2},
                       headers=H, tag="wizard", timeout=600)
                d.call("POST", "/wizard/assets",
                       json={**base, "min_assets": 2, "max_assets": args.max_assets},
                       headers=H, tag="wizard", timeout=600)
                d.call("POST", "/wizard/impacts", json={"language": "en"}, headers=H, tag="wizard", timeout=900)
                d.call("POST", "/wizard/risks", json={"language": "en"}, headers=H, tag="wizard", timeout=900)
                d.call("POST", "/wizard/controls", json={"language": "en", "max_controls_per_asset": 3},
                       headers=H, tag="wizard", timeout=900)

                # ---- 6. the live WebSocket agent ---------------------------
                try:
                    import os
                    model = os.getenv("WORKSHOPS_AI_LLM__MODEL", "gpt-5.1-chat").split("#")[0].strip()
                    with client.websocket_connect(f"/ws?session_id={sid}&workshop_id=assets") as ws:
                        frames = []
                        hello = ws.receive_json()
                        frames.append(hello.get("type"))
                        ws.send_json({"type": "ping", "payload": {}})
                        frames.append(ws.receive_json().get("type"))
                        ws.send_json({"type": "chat", "payload": {
                            "message": "Create an asset for the customer database and set confidentiality to PROTECTED.",
                            "assets": [], "model": model}})
                        deadline = time.time() + 300
                        while time.time() < deadline:
                            f = ws.receive_json()
                            frames.append(f.get("type"))
                            if f.get("type") in ("ui_response", "error"):
                                break
                        log.append({"method": "WS", "path": "/ws", "tag": "agent",
                                    "status": 101, "frames": frames})
                        print(f"  {101}  WS     /ws  frames={frames}", flush=True)
                except Exception as exc:
                    log.append({"method": "WS", "path": "/ws", "tag": "agent", "status": None,
                                "error": f"{type(exc).__name__}: {exc}"[:600],
                                "trace": traceback.format_exc()[-1500:]})
                    print(f"  WS FAILED: {type(exc).__name__}: {exc}", flush=True)

                if args.evaluation:
                    d.call("POST", "/wizard/evaluation",
                           json={**base, "max_services": 2, "max_assets": 3, "continue_on_error": True},
                           headers={"X-Session-Id": sid + "b"}, tag="wizard-eval", timeout=1800)
    finally:
        probe.stop()
        p = probe.dump(args.out)
        Path(args.requests_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.requests_out).write_text(json.dumps(log, indent=1, default=str))
        Path(args.routes_out).write_text(json.dumps(routes, indent=1, default=str))
        print(f"\n[surface_drive] {len(probe.hits)} functions entered -> {p}", flush=True)
        print(f"[surface_drive] {len(log)} requests -> {args.requests_out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
