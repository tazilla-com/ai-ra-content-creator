"""Deterministic tests for the harness — no provider or workshops_ai needed.

These exercise the verifiers, the report renderer, the redaction-safe config
summary, and the bundled scenario. The live end-to-end runs (chain/agent/fault)
are triggered explicitly via the container and are not part of this suite.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from harness import report, verify
from harness.bootstrap import config_summary


# --- verifiers ---------------------------------------------------------------

def test_verify_services_pass_and_fail():
    ok = verify.verify_services([{"name": "Freight forwarding"}])
    assert all(c["passed"] for c in ok)
    bad = verify.verify_services([{"name": ""}])
    assert any(not c["passed"] for c in bad)
    assert any(not c["passed"] for c in verify.verify_services([]))


def test_verify_assets_cia_validation():
    good = verify.verify_assets([
        {"name": "DB", "assetType": "Database", "confidentiality": "PROTECTED",
         "integrity": "HIGH", "availability": "HIGH"},
    ])
    assert all(c["passed"] for c in good)
    bad = verify.verify_assets([{"name": "DB", "assetType": "Database", "confidentiality": "NONSENSE"}])
    assert any(c["name"] == "cia_values_valid" and not c["passed"] for c in bad)


def test_verify_risks_probability_bounds():
    # probability is a qualitative likelihood level on a 0..4 integer scale
    good = verify.verify_risks([{"risks": [{"threatName": "Malware", "probability": 3}]}])
    assert all(c["passed"] for c in good)
    bad = verify.verify_risks([{"risks": [{"threatName": "Malware", "probability": 7}]}])
    assert any(c["name"] == "threatName_and_probability_valid" and not c["passed"] for c in bad)


def test_verify_controls_catalog_membership_and_enrichment():
    catalog = {"Access control policy", "Backup"}
    good = verify.verify_controls(
        [{"controls": [{"name": "Access control policy", "category": "A.5"}]}], catalog
    )
    assert all(c["passed"] for c in good)
    off = verify.verify_controls([{"controls": [{"name": "Invented control"}]}], catalog)
    assert any(c["name"] == "names_in_catalog" and not c["passed"] for c in off)
    assert any(c["name"] == "enriched_from_catalog" and not c["passed"] for c in off)


def test_verify_agent_flags_unknown_tools():
    class FakeToolMsg:
        def __init__(self, name):
            self.name = name
    # rename the class so type(m).__name__ == "ToolMessage"
    FakeToolMsg.__name__ = "ToolMessage"
    result = {
        "generated_assets": [{"name": "X"}],
        "deleted_asset_names": [],
        "messages": [FakeToolMsg("create_assets")],
        "error": None,
    }
    checks = verify.verify_agent(result)
    assert all(c["passed"] for c in checks)

    result["messages"] = [FakeToolMsg("write_to_tazilla")]
    checks = verify.verify_agent(result)
    assert any(c["name"] == "only_known_non_writing_tools" and not c["passed"] for c in checks)


# --- config summary redaction ------------------------------------------------

def test_config_summary_excludes_secrets():
    settings = SimpleNamespace(
        llm=SimpleNamespace(provider="azure", model="gpt-5.1-chat", api_key="sk-SECRET-KEY"),
        wizard=SimpleNamespace(default_model="gpt-5.4-mini", reasoning_effort="medium"),
        catalog=SimpleNamespace(mode="http", endpoint="http://mock-tazilla:9000"),
        env="dev",
        openai_throttle=SimpleNamespace(enabled=True),
    )
    summary = config_summary(settings)
    assert "sk-SECRET-KEY" not in json.dumps(summary)
    assert summary["provider"] == "azure"
    assert summary["catalog_endpoint"] == "http://mock-tazilla:9000"


# --- report renderer ---------------------------------------------------------

def test_report_render_smoke():
    records = {
        "meta": {"generated_at": "2026-07-16T00:00:00Z", "reps": 1},
        "config": {"provider": "azure", "wizard_default_model": "gpt-5.4-mini",
                   "agent_model": "gpt-5.1-chat", "reasoning_effort": "medium",
                   "environment": "dev", "throttle_enabled": True,
                   "catalog_mode": "http", "catalog_endpoint": "http://mock-tazilla:9000"},
        "preflight": {"ok": True},
        "chain": {"session_id": "s", "status": "ok", "checks_passed": 10, "checks_total": 10,
                  "total_tokens": 1234, "total_cost_usd": 0.01,
                  "steps": [{"step": "services", "status": "ok", "output_count": 3,
                             "checks_passed": 2, "checks_total": 2, "usage": {"total_tokens": 100, "cost_usd": 0.001},
                             "duration_sec": 1.2, "checks": [{"name": "non_empty", "passed": True, "detail": ""}]}]},
        "agent": {"message": "hi", "status": "ok", "checks_passed": 4, "checks_total": 4,
                  "tools_used": ["create_assets"], "generated_assets": [{"name": "X"}],
                  "deleted_asset_names": [], "checks": []},
        "faults": {"timeout": {"status": "handled", "detail": "AITimeoutError", "checks": []},
                   "token_limit": {"status": "recovered", "detail": "split ok", "checks": []}},
    }
    md = report.render(records)
    assert "Appendix D" in md
    assert "Normal operation" in md
    assert "User validation" in md
    assert "sk-" not in md  # no secrets leaked


def test_bundled_scenario_loads():
    path = Path(__file__).resolve().parents[1] / "scenarios" / "sample_org.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("website_content")
    assert data.get("organization_profile", {}).get("organizationName")
