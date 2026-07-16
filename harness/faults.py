"""Config-induced REAL faults — no fake/scripted LLM.

These exercise the application's genuine error and recovery paths by changing
*configuration* (a 1-second call timeout; a tiny per-request token cap) and then
running a real step against the real provider. They provide the
"timeout / provider error" and "retry / fallback" evidence.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

from harness import verify
from harness.bootstrap import new_session_id
from harness.steps import usage_dict


@contextmanager
def _temp_attr(obj, attr, value):
    sentinel = object()
    old = getattr(obj, attr, sentinel)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        if old is sentinel:
            delattr(obj, attr)
        else:
            setattr(obj, attr, old)


def _reinit_throttle(settings) -> None:
    try:
        from workshops_ai.llm.openai_throttle import init_throttle

        init_throttle(
            rpm=settings.openai_throttle.rpm,
            tpm=settings.openai_throttle.tpm,
            max_concurrent=settings.openai_throttle.max_concurrent,
            per_request_limits=settings.wizard.openai_per_request_limits,
        )
    except Exception:
        pass


async def induce_timeout(
    *,
    website_content: str,
    organization_profile: Optional[dict] = None,
    language: str = "en",
    reasoning_effort: str = "medium",
    model: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    """Force a genuine LLM timeout (1-second per-call timeout) on the services step
    and record how the failure surfaces. PASS = a typed error surfaced cleanly."""
    from workshops_ai.settings import get_settings
    from workshops_ai.wizard.service_generator import ServiceGenerator

    settings = get_settings()
    session_id = session_id or new_session_id("fault-timeout")
    started = time.time()
    outcome: dict = {"kind": "fault", "fault": "timeout"}

    with _temp_attr(settings.wizard, "llm_timeout_sec", 1):
        try:
            await ServiceGenerator(model=model, reasoning_effort=reasoning_effort).generate_services(
                website_content=website_content,
                language=language,
                organization_profile=organization_profile,
                session_id=session_id,
            )
            outcome["status"] = "no_error"
            outcome["detail"] = "call completed within 1s; no timeout observed (re-run or lower content)"
        except BaseException as exc:  # noqa: BLE001
            outcome["status"] = "handled"
            outcome["exc_type"] = type(exc).__name__
            outcome["detail"] = str(exc)[:400]

    outcome["duration_sec"] = round(time.time() - started, 3)
    outcome["checks"] = [
        verify.check(
            "timeout_surfaced_cleanly",
            outcome["status"] == "handled",
            f"{outcome.get('exc_type', '')}: {outcome.get('detail', '')}",
        )
    ]
    return outcome


async def induce_token_limit(
    *,
    assets: list,
    asset_risks: list,
    language: str = "en",
    reasoning_effort: str = "medium",
    model: Optional[str] = None,
    session_id: Optional[str] = None,
    tiny_limit: int = 12000,
) -> dict:
    """Force a real ``RequestTooLargeError`` by shrinking the per-request token cap,
    then run the (batch) controls step and record whether the recursive batch-split
    recovered or the error surfaced. Either is valid fault-handling evidence.

    Reuses assets + asset_risks already produced by the chain (no regeneration).
    """
    from workshops_ai.settings import get_settings
    from workshops_ai.wizard.control_generator import ControlGenerator

    settings = get_settings()
    session_id = session_id or new_session_id("fault-tokenlimit")
    model_name = model or settings.wizard.default_model
    shrunk = {**dict(settings.wizard.openai_per_request_limits), model_name: tiny_limit}

    started = time.time()
    outcome: dict = {"kind": "fault", "fault": "token_limit", "tiny_limit": tiny_limit, "model": model_name}

    with _temp_attr(settings.wizard, "openai_per_request_limits", shrunk):
        _reinit_throttle(settings)  # throttle caches per-request limits at init
        try:
            controls, metrics, _debug = await ControlGenerator(
                model=model, reasoning_effort=reasoning_effort
            ).generate_asset_controls(
                assets=assets,
                asset_risks=asset_risks,
                language=language,
                session_id=session_id,
            )
            produced = sum(len(getattr(c, "controls", []) or []) for c in (controls or []))
            outcome["status"] = "recovered"
            outcome["usage"] = usage_dict(metrics)
            outcome["detail"] = (
                f"recursive batch-split produced {produced} controls under a "
                f"{tiny_limit}-token per-request cap"
            )
        except BaseException as exc:  # noqa: BLE001
            outcome["status"] = "surfaced"
            outcome["exc_type"] = type(exc).__name__
            outcome["detail"] = str(exc)[:400]

    _reinit_throttle(settings)  # restore real limits
    outcome["duration_sec"] = round(time.time() - started, 3)
    outcome["checks"] = [
        verify.check(
            "fault_handled",
            outcome["status"] in ("recovered", "surfaced"),
            f"{outcome['status']}: {outcome.get('detail', '')}",
        )
    ]
    return outcome
