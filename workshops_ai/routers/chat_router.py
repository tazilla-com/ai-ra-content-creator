# File: workshops_ai/routers/chat_router.py
from __future__ import annotations

"""
Chat Router
-----------
- Auth-protected POST /chat endpoint used by the UI.
- Resolves a configurable "AI manager/router" callable and invokes it.
- Normalizes heterogeneous manager outputs into a stable ChatResponse.

Key features:
- Optional authentication dependency (API key / JWT); aligns with main app.
- Telemetry hook (best-effort, no-op if module missing).
- Timeout guard and friendly error mapping.
- Passthrough `policy` field for UI features (e.g., emit_changes_enabled).
"""

import asyncio
import importlib
import inspect
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict

from workshops_ai.settings import get_settings

# --- Security dependency (API key / JWT) --------------------------------------
# Prefer central path; fall back to legacy path; dev runs without auth.
try:
    from workshops_ai.app.security import require_auth  # type: ignore[reportMissingImports]
    _AUTH_DEPS = [Depends(require_auth)]
except Exception:  # pragma: no cover
    try:
        from workshops_ai.app.deps.auth import require_auth  # type: ignore[reportMissingImports]
        _AUTH_DEPS = [Depends(require_auth)]
    except Exception:  # pragma: no cover
        _AUTH_DEPS = []

# --- Telemetry (best-effort, optional) ----------------------------------------
try:
    from workshops_ai.tools.telemetry import telemetry_log  # type: ignore
except Exception:  # pragma: no cover
    def telemetry_log(event: str, payload: dict) -> None:  # type: ignore
        return

settings = get_settings()
router = APIRouter(dependencies=_AUTH_DEPS)

# ------------------------------------------------------------------------------
# UI contract models
# ------------------------------------------------------------------------------

ActiveTab = Literal["assets", "impacts", "threats", "controls"]


class Selection(BaseModel):
    asset_id: Optional[str] = None


class ChatRequest(BaseModel):
    org_id: str
    active_tab: ActiveTab
    selection: Selection = Field(default_factory=Selection)
    message: str = ""
    staging: bool = False
    user_id: str
    model_config = ConfigDict(extra="ignore")


class ChatResponse(BaseModel):
    messages: List[str]
    proposals: Optional[Dict[str, Any]] = None
    staging: Dict[str, Any] = Field(default_factory=lambda: {"needs_commit": False})
    # UI expects policy passthrough (e.g. {"emit_changes_enabled": true})
    policy: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"emit_changes_enabled": True}
    )


# ------------------------------------------------------------------------------
# Manager/Router resolver
# ------------------------------------------------------------------------------

# Default spec:
#   workshops_ai/graph/manager.py -> async def route_chat(UIRequest) -> UIResponse
_DEFAULT_SPEC = "workshops_ai.graph.manager:route_chat"

# Prefer central settings; fall back to default
_SPEC = getattr(settings, "ai_manager_spec", None) or _DEFAULT_SPEC

# Fallback candidates if settings/default fails (kept for compatibility)
_FALLBACKS = [
    "workshops_ai.agents.manager:get_manager",
    "workshops_ai.agents.router:router",
    "workshops_ai.agents.manager:router",
    "workshops_ai.agents.manager:manager",
]

_manager_singleton: Any = None


def _load_from_spec(spec: str) -> Any:
    mod_name, _, attr_str = spec.partition(":")
    if not mod_name:
        raise RuntimeError("Invalid AI manager spec (missing module).")
    module = importlib.import_module(mod_name)
    candidates = [a.strip() for a in (attr_str or "").split(",") if a.strip()] or [
        "route_chat",
        "get_manager",
        "router",
        "manager",
    ]
    last_err: Exception | None = None
    for name in candidates:
        try:
            obj = getattr(module, name)
        except Exception as e:
            last_err = e
            continue
        # Factory pattern: get_manager() -> instance
        if name == "get_manager" and callable(obj):
            return obj()
        return obj
    raise RuntimeError(
        f"Could not resolve any of {candidates} in {mod_name}. Last error: {last_err}"
    )


def _resolve_manager() -> Any:
    """
    Resolve and memoize the manager/router object or callable according to
    settings.ai_manager_spec, with compatibility fallbacks.
    """
    global _manager_singleton
    if _manager_singleton is not None:
        return _manager_singleton

    last_err: Exception | None = None

    # 1) Settings/default spec
    try:
        _manager_singleton = _load_from_spec(_SPEC)
        return _manager_singleton
    except Exception as e:
        last_err = e

    # 2) Fallbacks
    for spec in _FALLBACKS:
        try:
            _manager_singleton = _load_from_spec(spec)
            return _manager_singleton
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Manager/Router not found (spec='{_SPEC}'). Last error: {last_err}")


# ------------------------------------------------------------------------------
# Invocation helpers
# ------------------------------------------------------------------------------

async def _invoke_manager(manager: Any, payload: dict) -> Any:
    """
    Accepts:
      - async function (e.g., `route_chat`)       -> await directly
      - object with .ainvoke(payload)             -> await
      - object with .invoke(payload)              -> run in thread
      - plain callable returning awaitable/result -> await if needed
    """
    # async def route_chat(...)
    if inspect.iscoroutinefunction(manager):
        return await manager(payload)

    # object with .ainvoke/.invoke
    ainvoke = getattr(manager, "ainvoke", None)
    if callable(ainvoke):
        return await ainvoke(payload)

    invoke = getattr(manager, "invoke", None)
    if callable(invoke):
        return await asyncio.to_thread(invoke, payload)

    # plain callable
    if callable(manager):
        res = manager(payload)
        return await res if inspect.isawaitable(res) else res

    raise TypeError(
        "Manager/Router is not invocable (expected async function, ainvoke/invoke, or callable)."
    )


def _as_list_messages(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        return [v if isinstance(v, str) else str(v) for v in val]
    return [str(val)]


def _normalize_to_chat_response(res: Any) -> ChatResponse:
    """
    Accept Pydantic UIResponse / dict / string, return ChatResponse.
    Preserves `policy` if present; otherwise sets a safe default.
    """
    # Pydantic model (e.g., UIResponse) – prefer model_dump
    if hasattr(res, "model_dump"):
        res = res.model_dump()

    if isinstance(res, dict):
        policy = res.get("policy")
        if not isinstance(policy, dict):
            policy = {"emit_changes_enabled": True}
        return ChatResponse(
            messages=_as_list_messages(
                res.get("messages") or res.get("message") or res.get("text")
            ),
            proposals=res.get("proposals"),
            staging=res.get("staging") or {"needs_commit": False},
            policy=policy,
        )

    return ChatResponse(
        messages=_as_list_messages(res),
        staging={"needs_commit": False},
        policy={"emit_changes_enabled": True},
    )


# ------------------------------------------------------------------------------
# Route
# ------------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # 1) Input hardening – length guard (from centralized settings)
    max_len = int(getattr(settings.chat, "max_message_len", 0) or 0)
    if max_len and len(req.message) > max_len:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Message too long (>{max_len} chars).",
        )

    # 2) Audit/telemetry (best-effort)
    try:
        telemetry_log(
            "chat.request",
            {
                "org_id": req.org_id,
                "user_id": req.user_id,
                "active_tab": req.active_tab,
                "staging": req.staging,
            },
        )
    except Exception:
        pass

    # 3) Resolve & invoke manager/router with timeout
    try:
        manager = _resolve_manager()
    except Exception as e:
        return ChatResponse(
            messages=[f"Agent not available: {e}"],
            staging={"needs_commit": False},
            policy={"emit_changes_enabled": True},
        )

    try:
        timeout = int(getattr(settings.chat, "timeout_sec", 0) or 0) or None
        raw = await asyncio.wait_for(
            _invoke_manager(manager, req.model_dump()),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return ChatResponse(
            messages=[f"Timed out after {getattr(settings.chat, 'timeout_sec', 0)}s"],
            staging={"needs_commit": False},
            policy={"emit_changes_enabled": True},
        )
    except Exception as e:
        return ChatResponse(
            messages=[f"Agent failed: {type(e).__name__}: {e}"],
            staging={"needs_commit": False},
            policy={"emit_changes_enabled": True},
        )

    return _normalize_to_chat_response(raw)
