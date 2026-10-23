"""Deterministic tests for the harness — no provider or workshops_ai needed.

These exercise the verifiers, the report renderer, the redaction-safe config
summary, and the bundled scenario. The live end-to-end runs (chain/agent/fault)
are triggered explicitly via the container and are not part of this suite.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
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


# --- health probes (regression: each probe imported the app and wrote a Redis session) -----------

_HEALTHCHECK = Path(__file__).resolve().parents[2] / "docker" / "healthcheck.py"


def _probe(port: int) -> int:
    env = {"APP_HOST": "127.0.0.1", "APP_PORT": str(port), "PATH": os.environ.get("PATH", "")}
    return subprocess.run([sys.executable, str(_HEALTHCHECK)], env=env, timeout=10).returncode


def test_healthcheck_probes_livez_without_importing_the_app():
    """docker/healthcheck.py must answer from /livez alone and import nothing of workshops_ai.

    Regression: it imported workshops_ai.settings, which made every probe cost about 0.4 s of CPU.
    """
    import http.server
    import threading

    status = {"code": 200}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status["code"] if self.path == "/livez" else 404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        assert _probe(port) == 0
        status["code"] = 503
        assert _probe(port) == 1
    finally:
        server.shutdown()
        server.server_close()
    assert _probe(port) == 1  # nothing listening

    imported = subprocess.run(
        [sys.executable, "-c", "import runpy, sys; runpy.run_path(sys.argv[1]); "
         "print(sorted(m for m in sys.modules if m.startswith('workshops_ai')))", str(_HEALTHCHECK)],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    assert imported == "[]", imported


def test_session_middleware_gives_probes_no_session():
    """Health probes must get no session: no Redis write, no cookie.

    Regression: every cookie-less request got a new sess:<id> key with an 8 h TTL, so each probe
    wrote one. Needs workshops_ai and starlette importable, like the redaction tests.
    """
    try:
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient
        from workshops_ai.app.middlewares import session_middleware as sm
    except Exception:  # pragma: no cover - application not importable here
        return

    writes = []

    class FakeRedis:
        async def get(self, key):
            return None

        async def set(self, key, value, ex=None):
            writes.append(key)

    async def fake_get_redis():
        return FakeRedis()

    async def ok(request):
        return PlainTextResponse("OK")

    original = sm.get_redis
    sm.get_redis = fake_get_redis
    try:
        app = Starlette(routes=[Route(p, ok) for p in ("/livez", "/readyz", "/health", "/wizard/state")])
        app.add_middleware(sm.SessionMiddleware, secure_cookie=False)
        client = TestClient(app)
        for path in ("/livez", "/readyz", "/health"):
            response = client.get(path)
            assert response.status_code == 200 and "set-cookie" not in response.headers, path
        assert writes == []
        response = client.get("/wizard/state")
        assert "wsai.sid" in response.headers.get("set-cookie", "")
        assert len(writes) == 1 and writes[0].startswith("sess:")
    finally:
        sm.get_redis = original


# --- dependencies (regression: every image build resolved the whole stack afresh) -----------------

_REPO = Path(__file__).resolve().parents[2]


def _entries(name: str) -> list[str]:
    lines = (_REPO / name).read_text(encoding="utf-8").splitlines()
    return [s for s in (line.split("#", 1)[0].strip() for line in lines) if s]


def test_constraints_pin_every_requirement():
    """constraints.txt must pin every package requirements.txt names, to a version it allows.

    Regression: requirements.txt pinned 3 of its 38 lines, so each build could change most of the stack.
    """
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version

    pins = {}
    for entry in _entries("constraints.txt"):
        name, sep, version = entry.partition("==")
        assert sep and version and not set(version) & set("<>=!~,;"), f"not an exact pin: {entry}"
        pins[canonicalize_name(name)] = Version(version)
    for entry in _entries("requirements.txt"):
        req = Requirement(entry)
        pin = pins.get(canonicalize_name(req.name))
        assert pin is not None, f"{req.name} has no pin in constraints.txt"
        allowed = req.specifier.contains(pin, prereleases=True)
        assert allowed, f"{req.name}=={pin} is outside {req.specifier}"


# --- cost (regression: every cost_usd was 1000x too low) ------------------------------------------

def test_openai_cost_uses_prices_per_1k_tokens():
    """openai_pricing holds USD per 1K tokens, as env.example and every price table give it.

    Regression: the calculator divided by 1M, so every cost_usd was 1000x too low. Needs workshops_ai
    importable, like the redaction tests.
    """
    try:
        from workshops_ai.wizard.cost_calculator import calculate_openai_cost
    except Exception:  # pragma: no cover - application not importable here
        return

    pricing = {"gpt-5.4-mini": {"prompt": 0.00075, "completion": 0.0045}}  # $0.75 / $4.50 per 1M
    cost = calculate_openai_cost(1_000_000, 1_000_000, "gpt-5.4-mini", pricing)
    assert abs(cost - 5.25) < 1e-9, cost
