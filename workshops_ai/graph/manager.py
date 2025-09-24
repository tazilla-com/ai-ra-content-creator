# File: workshops_ai/graph/manager.py
# -----------------------------------------------------------------------------
# Purpose
# Minimal-yet-enterprise router for Workshop Agents.
# For now it supports two tabs and dispatches accordingly:
#   - active_tab == "assets"  → AssetAgent (emit-only)
#   - active_tab == "impacts" → ImpactAgent (LangChain tools, send-now)
#
# Design notes
# - Contract-first: route_chat(UIRequest) -> UIResponse (strict Pydantic types).
# - Tool-gating by active_tab (security-by-design; no cross-tab calls).
# - Lightweight intent detector (read | propose | change | explain) for assets path
#   to keep backward compatibility with your AssetAgent signature.
# - Per-request telemetry (best-effort), structured messages in the agent.
# - Defensive imports (avoid cycles). If an agent isn’t wired yet, return a graceful UIResponse.
# - Both tabs operate in EMIT-ONLY / SEND-NOW semantics (WS optimistic + submit ChangeSet).
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Final, Literal, TypedDict, cast

from pydantic import ValidationError

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.tools.telemetry import telemetry_log
from workshops_ai.tools.memory import memory_get, memory_set

# -----------------------------------------------------------------------------
# Types & constants
# -----------------------------------------------------------------------------

Intent = Literal["read", "propose", "change", "explain"]


class ToolGate(TypedDict):
    tools: list[str]     # human-readable names of allowed tools
    writes_allowed: bool # whether emit/writes are permitted


TOOL_GATES: Final[dict[str, ToolGate]] = {
    "assets":   {"tools": ["assets_list", "asset_get", "create_asset", "set_asset_meta"], "writes_allowed": True},
    # Impacts = send-now tools; no staging/commit in UI
    "impacts":  {"tools": ["list_impacts", "asset_get_impacts", "set_impact_level", "bulk_set_impact_levels", "explain_impacts"], "writes_allowed": True},
    "threats":  {"tools": ["list_threats", "asset_get_risks"], "writes_allowed": False},
    "controls": {"tools": ["list_controls", "get_control_coverage"], "writes_allowed": False},
}

DEFAULT_RESPONSE: Final[UIResponse] = UIResponse(
    messages=["Nothing to process."],
    proposals=None,
    staging={"needs_commit": False},
)

# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------

async def route_chat(req: UIRequest) -> UIResponse:
    """
    Main router entrypoint.
    IMPORTANT:
    - This function forwards the incoming `UIRequest` instance **unchanged** to the agents.
      No field is dropped or rewritten (incl. locale, selection, policy, etc.).
    """
    _audit_in(req)

    # Defensive shape validation (API layer already validated; this is extra safety).
    if not isinstance(req, UIRequest):
        try:
            req = UIRequest.model_validate(cast(Any, req))
        except ValidationError:
            return _fail("Input does not match UIRequest contract.", None)

    gate = TOOL_GATES.get((req.active_tab or "").lower())
    if gate is None:
        # Unknown tab – reply explicitly and do not attempt any tools.
        return UIResponse(
            messages=[f"Tab “{req.active_tab}” is not supported yet. Use Assets or Impacts."],
            proposals=None,
            staging={"needs_commit": False},
        )

    active_tab = (req.active_tab or "").lower()

    # Route by tab (tool-gating is enforced by only constructing the specific agent)
    if active_tab == "assets":
        intent = _infer_intent(req.message)  # keep old AssetAgent signature compatibility
        return await _dispatch_assets(req, intent)

    if active_tab == "impacts":
        return await _dispatch_impacts(req)

    # Known but not implemented tabs – reply clearly, no silent fallthrough.
    return UIResponse(
        messages=[f"Tab “{req.active_tab}” will be available in the next iteration."],
        proposals=None,
        staging={"needs_commit": False},
    )

# -----------------------------------------------------------------------------
# Dispatchers
# -----------------------------------------------------------------------------

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
        # AssetAgent historical entrypoint was a module-level `run(req, intent)`.
        from workshops_ai.agents.asset_agent import run as asset_run  # type: ignore
    except Exception as e:
        telemetry_log("manager.asset_agent_not_available", {"error": str(e)})
        return UIResponse(
            messages=[
                "AssetAgent is not available (module not loaded). "
                "Check `workshops_ai/agents/asset_agent.py` (function `run`)."
            ],
            proposals=None,
            staging={"needs_commit": False},
            policy={"emit_changes_enabled": True},  # emit-only tab
        )

    try:
        result = await asset_run(req, intent)  # pass the original req object unchanged
    except Exception as e:
        telemetry_log("manager.asset_agent_runtime_error", {"error": str(e)})
        return _fail("AssetAgent execution failed. Please check agent logs.", req, details=str(e))

    # Defensive shape validation for the outgoing payload
    try:
        resp = UIResponse.model_validate(result)
        # Ensure optimistic UI policy is on
        policy = dict(getattr(resp, "policy", {}) or {})
        policy["emit_changes_enabled"] = True
        resp.policy = policy  # type: ignore[attr-defined]
        return resp
    except ValidationError as ve:
        telemetry_log("manager.response_validation_error", {"errors": ve.errors()})
        return _fail("Agent response failed UIResponse validation.", req, details=ve.errors())


async def _dispatch_impacts(req: UIRequest) -> UIResponse:
    """
    Call ImpactAgent (LangChain tools, SEND-NOW).
    ImpactAgent sets policy.emit_changes_enabled=True in its own response as well.
    """
    # Persist lightweight per-tab memory (best-effort)
    try:
        memory_set(req.org_id, "impacts.last_message", (req.message or "")[:200])
        if (aid := _selection_asset_id(req)) is not None:
            memory_set(req.org_id, "impacts.last_selected_asset_id", aid)
    except Exception:
        pass

    # Lazy imports to avoid import cycles
    try:
        from workshops_ai.agents.impact_agent import ImpactAgent  # class-based, LangChain tools
    except Exception as e:
        telemetry_log("manager.impact_agent_not_available", {"error": str(e)})
        return UIResponse(
            messages=[
                "ImpactAgent is not available (module not loaded). "
                "Check `workshops_ai/agents/impact_agent.py` (class `ImpactAgent`)."
            ],
            proposals=None,
            staging={"needs_commit": False},
            policy={"emit_changes_enabled": True},
        )

    # Obtain backend port (defensive multi-try; adjust to your actual provider)
    backend = _get_backend_port()
    if backend is None:
        telemetry_log("manager.backend_port_missing", {})
        return UIResponse(
            messages=["Backend port is not available. Please ensure the CSV/API backend is initialized."],
            proposals=None,
            staging={"needs_commit": False},
            policy={"emit_changes_enabled": True},
        )

    # WsHub for broadcasts
    try:
        from workshops_ai.app.services.ws_hub import WsHub
    except Exception as e:
        telemetry_log("manager.ws_hub_missing", {"error": str(e)})
        return UIResponse(
            messages=["WsHub is not available. Please check `workshops_ai/app/services/ws_hub.py`."],
            proposals=None,
            staging={"needs_commit": False},
            policy={"emit_changes_enabled": True},
        )

    # Construct agent and run
    try:
        agent = ImpactAgent(backend_port=backend, ws_hub=WsHub)
        result = await agent.run(req)  # agent handles LangChain + tools routing
    except Exception as e:
        telemetry_log("manager.impact_agent_runtime_error", {"error": str(e)})
        return _fail("ImpactAgent execution failed. Please check agent logs.", req, details=str(e))

    # Validate outgoing payload & ensure optimistic policy
    try:
        resp = UIResponse.model_validate(result)
        policy = dict(getattr(resp, "policy", {}) or {})
        policy["emit_changes_enabled"] = True
        resp.policy = policy  # type: ignore[attr-defined]
        return resp
    except ValidationError as ve:
        telemetry_log("manager.response_validation_error", {"errors": ve.errors()})
        return _fail("Agent response failed UIResponse validation.", req, details=ve.errors())

# -----------------------------------------------------------------------------
# Backend resolver (defensive; adjust to your project structure)
# -----------------------------------------------------------------------------

def _get_backend_port() -> Any | None:
    """
    Try to obtain a BackendPort instance in a defensive way.
    Adjust the import path to your actual backend provider.
    """
    # 1) Preferred provider: workshops_ai.backend.provider.get_backend_port()
    try:
        from workshops_ai.backend.provider import get_backend_port  # type: ignore
        return get_backend_port()
    except Exception:
        pass

    # 2) CSV backend with a factory: workshops_ai.backend.csv_backend.get_backend_port()
    try:
        from workshops_ai.backend.csv_backend import get_backend_port  # type: ignore
        return get_backend_port()
    except Exception:
        pass

    # 3) Global singleton style: workshops_ai.backend.port.backend_port
    try:
        from workshops_ai.backend.port import backend_port  # type: ignore
        return backend_port
    except Exception:
        pass

    return None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _infer_intent(message: str | None) -> Intent:
    """
    Lightweight intent heuristics (Czech/English keywords).
    Conservative default is 'read'.
    """
    if not message:
        return "read"
    m = message.lower()

    # Explain/help
    if any(k in m for k in ("explain", "help", "nápověda", "vysvětli", "prečo", "preco", "ako funguje", "why")):
        return "explain"

    # Propose (suggestions)
    if any(k in m for k in ("navrh", "navrhni", "suggest", "proposal", "doporuč", "recommend")):
        return "propose"

    # Change/update
    if any(k in m for k in ("change", "update", "upravit", "zmeni", "změň", "nastav", "set ")):
        return "change"

    # Read/list/detail
    if any(k in m for k in ("list", "seznam", "vyjmenuj", "show", "detail", "zobraz", "najdi", "read", "view")):
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
                "locale": getattr(req, "locale", None),
                "emit_changes_enabled": getattr(getattr(req, "policy", None), "emit_changes_enabled", True),
            },
        )
    except Exception:
        pass
