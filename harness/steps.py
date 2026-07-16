"""Drive the real wizard generators and the Asset-workshop agent — live.

Every function here calls the application's own generators/agent against the
configured provider and catalog. Domain objects (Service, Asset, ...) are threaded
between steps exactly as the wizard chains them. ``workshops_ai`` is imported
lazily so the pure helpers/verifiers remain importable without the app.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from harness import verify
from harness.bootstrap import catalog_control_names, new_session_id


def usage_dict(metrics: Any) -> dict:
    return {
        "prompt_tokens": getattr(metrics, "prompt_tokens", None),
        "completion_tokens": getattr(metrics, "completion_tokens", None),
        "total_tokens": getattr(metrics, "total_tokens", None),
        "cost_usd": getattr(metrics, "cost_usd", None),
        "model": getattr(metrics, "model", None),
    }


def _dump_list(items, limit: Optional[int] = None) -> list:
    out = [it.model_dump(mode="json") if hasattr(it, "model_dump") else it for it in (items or [])]
    return out if limit is None else out[:limit]


async def _run_step(name: str, coro, verify_fn) -> tuple[dict, Any]:
    started = time.time()
    try:
        output, metrics, _debug = await coro
        checks = verify_fn(output) if verify_fn else []
        passed, total = verify.summarize(checks)
        return (
            {
                "step": name,
                "status": "ok",
                "duration_sec": round(time.time() - started, 3),
                "usage": usage_dict(metrics),
                "output_count": len(output or []),
                "checks": checks,
                "checks_passed": passed,
                "checks_total": total,
                "error": None,
                "exc_type": None,
            },
            output,
        )
    except BaseException as exc:  # noqa: BLE001 - any failure is recorded as evidence
        return (
            {
                "step": name,
                "status": "error",
                "duration_sec": round(time.time() - started, 3),
                "usage": {},
                "output_count": 0,
                "checks": [],
                "checks_passed": 0,
                "checks_total": 0,
                "error": str(exc)[:1500],
                "exc_type": type(exc).__name__,
            },
            None,
        )


def _chain_result(session_id: str, steps: list[dict]) -> dict:
    total_cost = sum((s["usage"].get("cost_usd") or 0) for s in steps)
    total_tokens = sum((s["usage"].get("total_tokens") or 0) for s in steps)
    return {
        "kind": "chain",
        "session_id": session_id,
        "steps": steps,
        "status": "ok" if all(s["status"] == "ok" for s in steps) else "error",
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
        "checks_passed": sum(s["checks_passed"] for s in steps),
        "checks_total": sum(s["checks_total"] for s in steps),
    }


async def run_chain(
    *,
    website_content: str,
    organization_profile: Optional[dict] = None,
    language: str = "en",
    reasoning_effort: str = "medium",
    model: Optional[str] = None,
    min_services: int = 1,
    max_services: int = 3,
    min_assets: int = 3,
    max_assets: int = 5,
    max_controls_per_asset: int = 5,
    session_id: Optional[str] = None,
) -> tuple[dict, dict]:
    """Run services -> assets -> impacts -> risks -> controls, verifying each step.

    Returns ``(result_dict, outputs)`` where ``outputs`` holds the produced domain
    objects for reuse (e.g. by the token-limit fault, to avoid regenerating).
    """
    from workshops_ai.settings import get_settings
    from workshops_ai.wizard.asset_generator import AssetGenerator
    from workshops_ai.wizard.control_generator import ControlGenerator
    from workshops_ai.wizard.impact_generator import ImpactGenerator
    from workshops_ai.wizard.risk_generator import RiskGenerator
    from workshops_ai.wizard.service_generator import ServiceGenerator

    session_id = session_id or new_session_id("chain")
    steps: list[dict] = []
    outputs: dict = {"session_id": session_id}

    r, services = await _run_step(
        "services",
        ServiceGenerator(model=model, reasoning_effort=reasoning_effort).generate_services(
            website_content=website_content,
            language=language,
            min_services=min_services,
            max_services=max_services,
            organization_profile=organization_profile,
            session_id=session_id,
        ),
        verify.verify_services,
    )
    steps.append(r)
    if r["status"] != "ok":
        return _chain_result(session_id, steps), outputs
    outputs["services"] = services

    r, assets = await _run_step(
        "assets",
        AssetGenerator(model=model, reasoning_effort=reasoning_effort).generate_assets(
            website_content=website_content,
            language=language,
            min_assets=min_assets,
            max_assets=max_assets,
            services=services,
            organization_profile=organization_profile,
            session_id=session_id,
        ),
        verify.verify_assets,
    )
    steps.append(r)
    if r["status"] != "ok":
        return _chain_result(session_id, steps), outputs
    outputs["assets"] = assets

    r, impacts = await _run_step(
        "impacts",
        ImpactGenerator(model=model, reasoning_effort=reasoning_effort).generate_asset_impacts(
            assets=assets,
            language=language,
            website_content=website_content,
            organization_profile=organization_profile,
            services=services,
            session_id=session_id,
        ),
        verify.verify_impacts,
    )
    steps.append(r)
    if r["status"] != "ok":
        return _chain_result(session_id, steps), outputs
    outputs["impacts"] = impacts

    r, risks = await _run_step(
        "risks",
        RiskGenerator(model=model, reasoning_effort=reasoning_effort).generate_asset_risks(
            assets=assets,
            language=language,
            website_content=website_content,
            organization_profile=organization_profile,
            services=services,
            asset_impacts=impacts,
            session_id=session_id,
        ),
        verify.verify_risks,
    )
    steps.append(r)
    if r["status"] != "ok":
        return _chain_result(session_id, steps), outputs
    outputs["risks"] = risks

    cat_names = catalog_control_names(get_settings(), session_id)
    r, controls = await _run_step(
        "controls",
        ControlGenerator(model=model, reasoning_effort=reasoning_effort).generate_asset_controls(
            assets=assets,
            asset_risks=risks,
            language=language,
            max_controls_per_asset=max_controls_per_asset,
            website_content=website_content,
            session_id=session_id,
        ),
        lambda out: verify.verify_controls(out, cat_names),
    )
    steps.append(r)
    outputs["controls"] = controls
    return _chain_result(session_id, steps), outputs


async def run_agent(
    *,
    message: str,
    model: Optional[str] = None,
    input_assets: Optional[list] = None,
    workshop_id: str = "assets",
    session_id: Optional[str] = None,
) -> dict:
    """Drive the Asset-workshop agent and assert human-in-the-loop invariants."""
    from workshops_ai.graph.assets.agent import invoke_asset_agent
    from workshops_ai.settings import get_settings

    settings = get_settings()
    session_id = session_id or new_session_id("agent")
    model = model or settings.llm.model

    valid_types = None
    try:
        from workshops_ai.wizard.catalog_provider import get_catalog_provider

        provider = get_catalog_provider()
        try:
            valid_types = list(provider.get_catalog_names("asset_types", session_id=session_id))
        except TypeError:
            valid_types = list(provider.get_catalog_names("asset_types"))
    except Exception:
        valid_types = None

    started = time.time()
    try:
        result = await invoke_asset_agent(
            message=message,
            session_id=session_id,
            workshop_id=workshop_id,
            input_assets=input_assets or [],
            valid_asset_types=valid_types,
            model=model,
        )
        checks = verify.verify_agent(result)
        passed, total = verify.summarize(checks)
        tool_names = [
            getattr(m, "name", None)
            for m in result.get("messages", [])
            if type(m).__name__ == "ToolMessage"
        ]
        return {
            "kind": "agent",
            "status": "ok" if not result.get("error") else "error",
            "duration_sec": round(time.time() - started, 3),
            "message": message,
            "generated_assets": _dump_list(result.get("generated_assets"), limit=10),
            "deleted_asset_names": result.get("deleted_asset_names") or [],
            "tools_used": [t for t in tool_names if t],
            "checks": checks,
            "checks_passed": passed,
            "checks_total": total,
            "error": result.get("error"),
        }
    except BaseException as exc:  # noqa: BLE001
        return {
            "kind": "agent",
            "status": "error",
            "duration_sec": round(time.time() - started, 3),
            "message": message,
            "error": str(exc)[:1500],
            "exc_type": type(exc).__name__,
            "checks": [],
            "checks_passed": 0,
            "checks_total": 0,
        }
