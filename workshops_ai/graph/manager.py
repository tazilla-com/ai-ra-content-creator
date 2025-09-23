# File: workshops_ai/graph/manager.py
# Purpose:
#   Minimal-yet-enterprise router for Workshop Agents.
#   For MVP it supports only active_tab == "assets" and dispatches to AssetAgent.
#
# Design notes:
# - Contract-first: route_chat(UIRequest) -> UIResponse (strict Pydantic types).
# - Tool-gating by active_tab (security-by-design; no cross-tab calls).
# - Lightweight intent detector (read | propose | change | explain) – conservative default is "read".
# - Per-request telemetry (best-effort), structured messages in user locale (English/Slovak handled by agent).
# - Defensive imports (avoid cycles). If the agent isn’t wired yet, return a graceful UIResponse.
# - Assets are **EMIT-ONLY**: the agent may emit WS events and write ChangeSets to an outbox;
#   there are no direct writes to a production backend in this MVP.
#
# NOTE (pass-through):
# - The router forwards the UIRequest object **as-is** to the agent; it does not reconstruct,
#   strip, or transform fields like language/number_of_assets/policy/etc.

from __future__ import annotations

from typing import Any, Final, Literal, TypedDict, cast

from pydantic import ValidationError

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.tools.telemetry import telemetry_log
from workshops_ai.tools.memory import memory_get, memory_set

# --------------------------------------------------------------------------------------
# Types & constants
# --------------------------------------------------------------------------------------

Intent = Literal["read", "propose", "change", "explain"]

class ToolGate(TypedDict):
    tools: list[str]          # human-readable names of allowed tools
    writes_allowed: bool      # whether staging/emit/writes are permitted

TOOL_GATES: Final[dict[str, ToolGate]] = {
    # EMIT-ONLY for assets: allow read tools + emit-capable tools
    "assets":   {"tools": ["assets_list", "asset_get", "create_asset", "set_asset_meta"], "writes_allowed": True},
    "impacts":  {"tools": ["list_impacts", "asset_get_impacts", "staging_*"], "writes_allowed": True},
    "threats":  {"tools": ["list_threats", "asset_get_risks", "staging_*"], "writes_allowed": True},
    "controls": {"tools": ["list_controls", "get_control_coverage", "staging_*"], "writes_allowed": True},
}

DEFAULT_RESPONSE: Final[UIResponse] = UIResponse(
    messages=["Nothing to process."],
    proposals=None,
    staging={"needs_commit": False},
)

# --------------------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------------------

async def route_chat(req: UIRequest) -> UIResponse:
    """
    Main router entrypoint.
    For the 'assets' MVP the agent works in EMIT-ONLY mode (WS + outbox); it is not side-effect-free.

    IMPORTANT:
    - This function forwards the incoming `UIRequest` instance **unchanged** to the agent.
      No field is dropped or rewritten (incl. language, number_of_assets, policy, website_content, etc.).
    """
    _audit_in(req)

    # Defensive shape validation (API layer already validated; this is extra safety).
    if not isinstance(req, UIRequest):
        try:
            req = UIRequest.model_validate(cast(Any, req))
        except ValidationError:
            return _fail("Input does not match UIRequest contract.", None)

    gate = TOOL_GATES.get(req.active_tab)
    if gate is None:
        # Unknown tab – reply explicitly and do not attempt any tools.
        return UIResponse(
            messages=[f"Tab “{req.active_tab}” is not supported yet. Please use the Assets tab."],
            proposals=None,
            staging={"needs_commit": False},
        )

    intent = _infer_intent(req.message)

    # Tool-gating: only dispatch to the agent bound to the active_tab
    if req.active_tab == "assets":
        return await _dispatch_assets(req, intent)

    # Known but not implemented tabs – reply clearly, no silent fallthrough.
    return UIResponse(
        messages=[f"Tab “{req.active_tab}” will be available in the next iteration. "
                  f"Currently only Assets are supported (emit-only; changes go to outbox and WS)."],
        proposals=None,
        staging={"needs_commit": False},
    )

# --------------------------------------------------------------------------------------
# Dispatchers
# --------------------------------------------------------------------------------------

async def _dispatch_assets(req: UIRequest, intent: Intent) -> UIResponse:
    """Call AssetAgent with tool-gating and safe fallbacks (EMIT-ONLY semantics)."""
    # Persist lightweight per-tab memory (best-effort, non-blocking).
    try:
        memory_set(req.org_id, "assets.last_intent", intent)
        if (aid := _selection_asset_id(req)) is not None:
            memory_set(req.org_id, "assets.last_selected_asset_id", aid)
    except Exception:
        pass

    try:
        # Lazy import to avoid early cycles if modules are scaffolded later.
        from workshops_ai.agents.asset_agent import run as asset_run  # type: ignore
    except Exception as e:
        # Agent not wired yet – do not 500; respond gracefully.
        telemetry_log("manager.asset_agent_not_available", {"error": str(e)})
        return UIResponse(
            messages=[
                "AssetAgent is not available yet (module not loaded). "
                "Please check `workshops_ai/agents/asset_agent.py` (function `run`)."
            ],
            proposals=None,
            staging={"needs_commit": False},
        )

    # IMPORTANT: do NOT force read-only anymore; pass the inferred intent through.
    try:
        # Pass the **original req object** unchanged.
        result = await asset_run(req, intent)
    except Exception as e:
        telemetry_log("manager.asset_agent_runtime_error", {"error": str(e)})
        return _fail("AssetAgent execution failed. Please check agent logs.", req, details=str(e))

    # Defensive shape validation for the outgoing payload
    try:
        return UIResponse.model_validate(result)
    except ValidationError as ve:
        telemetry_log("manager.response_validation_error", {"errors": ve.errors()})
        return _fail("Agent response failed UIResponse validation.", req, details=ve.errors())

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _infer_intent(message: str | None) -> Intent:
    """
    Lightweight intent heuristics (Czech/English keywords).
    Conservative default is 'read'.
    """
    if not message:
        return "read"
    m = message.lower()

    # Explain/help
    if any(k in m for k in ("explain", "help", "nápověda", "vysvětli", "proč", "jak funguje")):
        return "explain"

    # Propose (suggestions)
    if any(k in m for k in ("navrh", "navrhni", "suggest", "proposal", "doporuč")):
        return "propose"

    # Change/update
    if any(k in m for k in ("change", "update", "upravit", "změň", "nastav", "set ")):
        return "change"

    # Read/list/detail
    if any(k in m for k in ("list", "seznam", "vyjmenuj", "show", "detail", "zobraz", "najdi")):
        return "read"

    return "read"

def _selection_asset_id(req: UIRequest) -> str | None:
    try:
        sel = req.selection or {}
        aid = sel.get("asset_id")  # type: ignore[call-arg]
        return cast(str | None, aid)
    except Exception:
        return None

def _fail(msg: str, req: UIRequest | None, *, details: Any | None = None) -> UIResponse:
    try:
        telemetry_log("manager.fail", {
            "message": msg,
            "org_id": getattr(req, "org_id", None) if req else None,
            "details": details
        })
    except Exception:
        pass
    return UIResponse(messages=[msg], proposals=None, staging={"needs_commit": False})

def _audit_in(req: UIRequest) -> None:
    try:
        telemetry_log(
            "manager.route_in",
            {
                "org_id": req.org_id,
                "user_id": req.user_id,
                "active_tab": req.active_tab,
                "has_selection": bool(req.selection),
                "staging": req.staging,
                # extra context (optional) for observability:
                "language": getattr(req, "language", None),
                "number_of_assets": getattr(req, "number_of_assets", None),
                "emit_changes_enabled": getattr(getattr(req, "policy", None), "emit_changes_enabled", True),
            },
        )
    except Exception:
        pass
