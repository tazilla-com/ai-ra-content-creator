"""Render harness records into an Appendix-D-shaped, redaction-safe markdown report.

Pure functions (no provider, no ``workshops_ai`` import) so they are unit-testable.
Input is the ``records`` dict assembled by the runner.
"""
from __future__ import annotations

from typing import Any


def _checks_line(checks: list[dict]) -> str:
    if not checks:
        return "—"
    return ", ".join(f"{'✓' if c['passed'] else '✗'} {c['name']}" for c in checks)


def _config_block(config: dict, preflight: dict, meta: dict) -> list[str]:
    return [
        "## Configuration (live)",
        "",
        f"- **Generated (UTC):** {meta.get('generated_at', 'n/a')}",
        f"- **Provider:** `{config.get('provider')}`  ·  "
        f"**wizard model:** `{config.get('wizard_default_model')}`  ·  "
        f"**agent model:** `{config.get('agent_model')}`",
        f"- **Reasoning effort:** `{config.get('reasoning_effort')}`  ·  "
        f"**environment:** `{config.get('environment')}`  ·  "
        f"**throttle:** `{config.get('throttle_enabled')}`",
        f"- **Catalog:** mode `{config.get('catalog_mode')}`"
        + (f" @ `{config.get('catalog_endpoint')}`" if config.get("catalog_endpoint") else "")
        + f"  ·  preflight ok: `{preflight.get('ok')}`",
        "",
        "_No credentials are included in this report._",
        "",
    ]


def _chain_section(chain: dict) -> list[str]:
    lines = [
        "## B.D.1 Normal operation — full wizard chain",
        "",
        f"Session `{chain.get('session_id')}` · overall **{chain.get('status')}** · "
        f"{chain.get('checks_passed')}/{chain.get('checks_total')} checks passed · "
        f"{chain.get('total_tokens')} tokens · ${chain.get('total_cost_usd')}",
        "",
        "| Step | Status | Out | Checks | Tokens | Cost (USD) | Dur (s) |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for s in chain.get("steps", []):
        u = s.get("usage", {})
        lines.append(
            f"| {s['step']} | {'✅' if s['status'] == 'ok' else '❌ ' + (s.get('exc_type') or '')} "
            f"| {s.get('output_count', 0)} "
            f"| {s.get('checks_passed', 0)}/{s.get('checks_total', 0)} "
            f"| {u.get('total_tokens') or '—'} "
            f"| {u.get('cost_usd') if u.get('cost_usd') is not None else '—'} "
            f"| {s.get('duration_sec', '—')} |"
        )
    # detail per step's checks
    lines.append("")
    for s in chain.get("steps", []):
        if s.get("checks"):
            lines.append(f"- **{s['step']}**: {_checks_line(s['checks'])}")
        if s.get("error"):
            lines.append(f"  - error: `{s.get('exc_type')}` — {s.get('error')}")
    lines.append("")
    return lines


def _agent_section(agent: dict) -> list[str]:
    if not agent:
        return []
    lines = [
        "## B.D.2 User validation — agent proposes, human commits",
        "",
        f"Message: _{agent.get('message')}_ · status **{agent.get('status')}** · "
        f"{agent.get('checks_passed')}/{agent.get('checks_total')} checks · "
        f"tools: `{', '.join(agent.get('tools_used') or []) or 'none'}`",
        "",
        f"Proposed **{len(agent.get('generated_assets') or [])}** asset(s), "
        f"**{len(agent.get('deleted_asset_names') or [])}** deletion(s). "
        "The agent has no write path to the Tazilla system of record; changes are "
        "returned as proposals for the client/user to apply.",
        "",
        f"- Checks: {_checks_line(agent.get('checks', []))}",
    ]
    if agent.get("error"):
        lines.append(f"- error: `{agent.get('exc_type')}` — {agent.get('error')}")
    lines.append("")
    return lines


def _fault_section(faults: dict) -> list[str]:
    lines = ["## B.D.3 Provider error / timeout & retry / fallback", ""]
    timeout = faults.get("timeout")
    if timeout:
        lines += [
            "**Timeout (real 1s call timeout on the services step):** "
            f"status `{timeout.get('status')}` — {timeout.get('detail', '')}",
            f"- Checks: {_checks_line(timeout.get('checks', []))}",
            "",
        ]
    token = faults.get("token_limit")
    if token:
        lines += [
            "**Retry / fallback (real tiny per-request token cap on the controls step):** "
            f"status `{token.get('status')}` — {token.get('detail', '')}",
            f"- Checks: {_checks_line(token.get('checks', []))}",
            "",
        ]
    return lines


def _malformed_note() -> list[str]:
    return [
        "## B.D.4 Malformed output",
        "",
        "Malformed model output is guarded by OpenAI/Azure **Structured Outputs**: each "
        "wizard step passes a Pydantic `response_format`, and calculated fields are "
        "restored from the catalog after generation rather than emitted by the model. "
        "The representative differential for this category (a real production incident "
        "root-caused and fixed) is recorded in the project `CHANGELOG.md` "
        "(risks batch-call failures 67% → 0%). Reproducing malformed output live is "
        "non-deterministic and is not forced by this harness.",
        "",
    ]


def _release_section(release: dict | None) -> list[str]:
    lines = ["## B.D.5 Release to production", ""]
    if not release:
        lines += [
            "Collected separately by `harness/release_evidence.py` (host/CI, where the "
            "docker CLI and git are available). See that tool's `release_record.md`.",
            "",
        ]
        return lines
    body = release.get("health", {}).get("body", {}) if isinstance(release.get("health"), dict) else {}
    rev = release.get("source_revision", {})
    lines += [
        f"- Source revision `{rev.get('commit_short')}` (branch `{rev.get('branch')}`)",
        f"- `/health` → status `{body.get('status')}`, version `{body.get('version')}`, "
        f"environment `{body.get('environment')}`",
        "",
    ]
    return lines


def render(records: dict[str, Any]) -> str:
    config = records.get("config", {})
    preflight = records.get("preflight", {})
    meta = records.get("meta", {})
    lines: list[str] = [
        "# Appendix D — Testing and deployment evidence",
        "",
        "Live evidence produced by the `harness/` container against the configured "
        "provider and catalog. Every result below is from a real call; no responses "
        "are mocked.",
        "",
    ]
    lines += _config_block(config, preflight, meta)
    if records.get("chain"):
        lines += _chain_section(records["chain"])
    lines += _agent_section(records.get("agent", {}))
    lines += _fault_section(records.get("faults", {}))
    lines += _malformed_note()
    lines += _release_section(records.get("release"))
    return "\n".join(lines)
