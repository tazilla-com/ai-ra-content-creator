"""Harness CLI — live evidence & verification for the AI-RA content creator.

Subcommands:
  evidence   full Appendix-D suite: chain + agent + timeout fault + token-limit fault + report
  chain      run the full wizard chain once (or N reps) and verify each step
  agent      drive the Asset-workshop agent and assert human-in-the-loop invariants
  fault      induce a real fault: --kind {timeout, token_limit}
  report     re-render report.md from a saved records.json

Runs inside the harness container (see harness/docker-compose.harness.yml):

  docker compose -f docker-compose.yaml -f docker-compose.override.yml \
    -f harness/docker-compose.harness.yml run --rm harness \
    python -m harness.runner evidence --reps 1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness import report
from harness.bootstrap import config_summary, init_runtime, new_session_id, preflight_catalog


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_dir(kind: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path("harness/runs") / f"{ts}_{kind}_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_scenario(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("website_content"):
        raise SystemExit(f"scenario {path} has no 'website_content'")
    return data


def _write(out_dir: Path, records: dict) -> None:
    (out_dir / "records.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(report.render(records), encoding="utf-8")
    # flat results.jsonl: one line per notable record
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for step in (records.get("chain") or {}).get("steps", []):
            fh.write(json.dumps({"record": "chain_step", **step}, ensure_ascii=False, default=str) + "\n")
        if records.get("agent"):
            fh.write(json.dumps({"record": "agent", **records["agent"]}, ensure_ascii=False, default=str) + "\n")
        for name, f in (records.get("faults") or {}).items():
            fh.write(json.dumps({"record": f"fault_{name}", **f}, ensure_ascii=False, default=str) + "\n")


def _base_records(settings, preflight, reps: int) -> dict:
    from harness.prompt_override import served

    meta = {"generated_at": _now(), "reps": reps}
    if served():
        meta["prompts_from_files"] = served()
    return {
        "meta": meta,
        "config": config_summary(settings),
        "preflight": preflight,
    }


async def _do_chain(scenario: dict, args) -> tuple[dict, dict]:
    from harness import steps

    return await steps.run_chain(
        website_content=scenario["website_content"],
        organization_profile=scenario.get("organization_profile"),
        language=scenario.get("language", args.language),
        reasoning_effort=args.reasoning,
        model=args.model,
        max_services=args.max_services,
        max_assets=args.max_assets,
        max_controls_per_asset=args.max_controls,
        session_id=new_session_id("chain"),
    )


async def cmd_chain(args) -> int:
    settings = init_runtime()
    preflight = preflight_catalog(settings)
    scenario = _load_scenario(args.scenario)
    records = _base_records(settings, preflight, args.reps)

    # For >1 rep we keep the last as the reported chain and stash rep statuses.
    rep_status = []
    chain = outputs = None
    for i in range(args.reps):
        chain, outputs = await _do_chain(scenario, args)
        rep_status.append(chain["status"])
    records["chain"] = chain
    records["meta"]["rep_status"] = rep_status

    out_dir = _run_dir("chain")
    _write(out_dir, records)
    _print_summary(out_dir, records)
    return 0 if chain and chain["status"] == "ok" else 2


async def cmd_agent(args) -> int:
    settings = init_runtime()
    preflight = preflight_catalog(settings)
    from harness import steps

    agent = await steps.run_agent(
        message=args.message, model=args.model, expect_proposal=args.expect_proposal,
    )
    records = _base_records(settings, preflight, 1)
    records["agent"] = agent
    out_dir = _run_dir("agent")
    _write(out_dir, records)
    _print_summary(out_dir, records)
    return 0 if agent["status"] == "ok" else 2


async def cmd_fault(args) -> int:
    settings = init_runtime()
    preflight = preflight_catalog(settings)
    scenario = _load_scenario(args.scenario)
    from harness import faults, steps

    records = _base_records(settings, preflight, 1)
    if args.kind == "timeout":
        f = await faults.induce_timeout(
            website_content=scenario["website_content"],
            organization_profile=scenario.get("organization_profile"),
            reasoning_effort=args.reasoning,
            model=args.model,
        )
        records["faults"] = {"timeout": f}
    else:  # token_limit — needs assets + risks, produce them first
        chain, outputs = await steps.run_chain(
            website_content=scenario["website_content"],
            organization_profile=scenario.get("organization_profile"),
            reasoning_effort=args.reasoning,
            model=args.model,
            max_assets=args.max_assets,
            session_id=new_session_id("fault-pre"),
        )
        records["chain"] = chain
        if outputs.get("assets") and outputs.get("risks"):
            f = await faults.induce_token_limit(
                assets=outputs["assets"], asset_risks=outputs["risks"],
                reasoning_effort=args.reasoning, model=args.model,
            )
            records["faults"] = {"token_limit": f}
        else:
            records["faults"] = {"token_limit": {"status": "skipped", "detail": "prechain did not produce assets+risks", "checks": []}}

    out_dir = _run_dir(f"fault_{args.kind}")
    _write(out_dir, records)
    _print_summary(out_dir, records)
    return 0


async def cmd_evidence(args) -> int:
    settings = init_runtime()
    preflight = preflight_catalog(settings)
    scenario = _load_scenario(args.scenario)
    from harness import faults, steps

    records = _base_records(settings, preflight, args.reps)

    # 1. normal operation — full chain
    chain, outputs = await _do_chain(scenario, args)
    records["chain"] = chain

    # 2. user validation — agent
    records["agent"] = await steps.run_agent(
        message=args.agent_message, model=args.model,
    )

    # 3 + 4. real faults
    fault_records: dict[str, Any] = {}
    fault_records["timeout"] = await faults.induce_timeout(
        website_content=scenario["website_content"],
        organization_profile=scenario.get("organization_profile"),
        reasoning_effort=args.reasoning, model=args.model,
    )
    if outputs.get("assets") and outputs.get("risks"):
        fault_records["token_limit"] = await faults.induce_token_limit(
            assets=outputs["assets"], asset_risks=outputs["risks"],
            reasoning_effort=args.reasoning, model=args.model,
        )
    else:
        fault_records["token_limit"] = {
            "status": "skipped",
            "detail": "chain did not produce assets+risks to drive the batch-split fault",
            "checks": [],
        }
    records["faults"] = fault_records

    out_dir = _run_dir("evidence")
    _write(out_dir, records)
    _print_summary(out_dir, records)
    return 0 if chain["status"] == "ok" else 2


def cmd_report(args) -> int:
    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    out = Path(args.records).parent / "report.md"
    out.write_text(report.render(records), encoding="utf-8")
    print(f"[report] rendered {out}")
    return 0


def _print_summary(out_dir: Path, records: dict) -> None:
    chain = records.get("chain") or {}
    agent = records.get("agent") or {}
    faults = records.get("faults") or {}
    print("\n=== harness summary ===")
    if chain:
        print(f"chain: {chain.get('status')} · {chain.get('checks_passed')}/{chain.get('checks_total')} checks "
              f"· {chain.get('total_tokens')} tok · ${chain.get('total_cost_usd')}")
    if agent:
        print(f"agent: {agent.get('status')} · {agent.get('checks_passed')}/{agent.get('checks_total')} checks "
              f"· tools={agent.get('tools_used')}")
    for name, f in faults.items():
        print(f"fault[{name}]: {f.get('status')} — {f.get('detail', '')}")
    print(f"artifacts: {out_dir}")


def _add_common(sp) -> None:
    sp.add_argument("--scenario", default="harness/scenarios/sample_org.json")
    sp.add_argument("--reasoning", default="medium", choices=["minimal", "low", "medium", "high"])
    sp.add_argument("--model", default=None, help="override model (default: configured)")
    sp.add_argument("--language", default="en")
    sp.add_argument("--max-services", type=int, default=3, dest="max_services")
    sp.add_argument("--max-assets", type=int, default=5, dest="max_assets")
    sp.add_argument("--max-controls", type=int, default=5, dest="max_controls")
    sp.add_argument("--prompts-dir", default=None, dest="prompts_dir",
                    help="serve prompts from DIR/<name>.json instead of Langfuse (harness/prompt_override.py)")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(prog="harness.runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evidence", help="full Appendix-D suite + report")
    _add_common(ev)
    ev.add_argument("--reps", type=int, default=1)
    ev.add_argument("--agent-message", default="Create an asset for the customer shipment-tracking database and set its confidentiality to PROTECTED.", dest="agent_message")
    ev.set_defaults(func=cmd_evidence, is_async=True)

    ch = sub.add_parser("chain", help="run the full wizard chain and verify")
    _add_common(ch)
    ch.add_argument("--reps", type=int, default=1)
    ch.set_defaults(func=cmd_chain, is_async=True)

    ag = sub.add_parser("agent", help="drive the agent")
    _add_common(ag)
    ag.add_argument("--message", default="Create an asset for the customer shipment-tracking database and set its confidentiality to PROTECTED.")
    ag.add_argument("--no-expect-proposal", action="store_false", dest="expect_proposal",
                    help="the message asks no change (a question): do not require a proposal")
    ag.set_defaults(func=cmd_agent, is_async=True)

    fa = sub.add_parser("fault", help="induce a real fault")
    _add_common(fa)
    fa.add_argument("--kind", required=True, choices=["timeout", "token_limit"])
    fa.set_defaults(func=cmd_fault, is_async=True)

    rp = sub.add_parser("report", help="re-render report.md from records.json")
    rp.add_argument("--records", required=True)
    rp.set_defaults(func=cmd_report, is_async=False)

    args = p.parse_args()
    if getattr(args, "prompts_dir", None):
        from harness.prompt_override import install

        for name, digest in install(args.prompts_dir).items():
            print(f"prompt from file: {name} sha256:{digest}")
    if getattr(args, "is_async", False):
        return asyncio.run(args.func(args))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
