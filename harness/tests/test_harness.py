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


# --- startup-log redaction (regression: credentials were logged in the clear) -

# These two need workshops_ai importable. Every other test in this file is
# deliberately independent of the application, so they skip rather than fail
# when the package is not on the path.

def _load_settings_module():
    try:
        from workshops_ai import settings as settings_mod
    except Exception:  # pragma: no cover - application not importable here
        return None
    return settings_mod


def test_redacted_dict_hides_every_credential():
    """AppSettings.redacted_dict() must leak no credential, including the provider key.

    Regression: this used to redact six fields and miss llm.api_key (the Azure/OpenAI key) plus
    the four flat legacy aliases, which hold copies of the same values.
    """
    settings_mod = _load_settings_module()
    if settings_mod is None:
        return

    canary = "CANARY-SECRET-VALUE-do-not-log"
    s = settings_mod.AppSettings()
    s.security.jwt_secret = canary
    s.security.api_keys = [canary, canary + "-2"]
    s.catalog.auth_token = canary
    s.llm.api_key = canary
    s.langchain.api_key = canary
    s.langfuse.public_key = canary
    s.langfuse.secret_key = canary
    s.langchain_api_key_flat = canary
    s.langfuse_public_key_flat = canary
    s.langfuse_secret_key_flat = canary
    s.openai_api_key_flat = canary

    rendered = json.dumps(s.redacted_dict(), default=str)
    assert canary not in rendered, "redacted_dict() leaked a credential"
    # and it must still be useful for diagnosis
    assert s.env in rendered


def test_every_secret_path_exists_on_the_model():
    """The explicit secret list must not drift away from the fields it names."""
    settings_mod = _load_settings_module()
    if settings_mod is None:
        return

    s = settings_mod.AppSettings()
    data = s.model_dump()
    for path in settings_mod._SECRET_FIELD_PATHS:
        node = data
        for part in path.split("."):
            assert isinstance(node, dict) and part in node, (
                f"_SECRET_FIELD_PATHS names {path!r}, which is not a field on AppSettings"
            )
            node = node[part]


def test_redact_for_log_handles_pydantic_models():
    """redact_for_log() must not pass a model through untouched.

    Regression: it walked Mapping/list/tuple only, so a BaseModel hit the leaf branch and came back
    unredacted. That is how the startup log printed every credential.
    """
    settings_mod = _load_settings_module()
    if settings_mod is None:
        return
    from workshops_ai.tools.telemetry import redact_for_log

    canary = "CANARY-SECRET-VALUE-do-not-log"
    s = settings_mod.AppSettings()
    s.llm.api_key = canary
    s.security.jwt_secret = canary

    assert canary not in json.dumps(redact_for_log(s), default=str)


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
