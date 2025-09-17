# File: workshops_ai/routers/chat_router.py
from __future__ import annotations

import asyncio
import importlib
import inspect
import os
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field, ConfigDict

# Telemetry (best-effort, no-op if missing)
try:
    from workshops_ai.tools.telemetry import telemetry_log  # type: ignore
except Exception:  # pragma: no cover
    def telemetry_log(event: str, payload: dict) -> None:  # type: ignore
        return

router = APIRouter()

# ---------------------------------------------------------------------------
# UI contract models
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Manager/Router resolver
# ---------------------------------------------------------------------------

# Default to YOUR manager:
#   workshops_ai/graph/manager.py -> async def route_chat(UIRequest) -> UIResponse
_DEFAULT_SPEC = "workshops_ai.graph.manager:route_chat"

# Optional env override:
#   AI_MANAGER="module:attr"   e.g. AI_MANAGER="workshops_ai.graph.manager:route_chat"
_SPEC = os.getenv("AI_MANAGER", _DEFAULT_SPEC)

# Fallback candidates if env/default fails (kept for compatibility)
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
        "route_chat", "get_manager", "router", "manager"
    ]
    last_err: Exception | None = None
    for name in candidates:
        try:
            obj = getattr(module, name)
        except Exception as e:
            last_err = e
            continue
        # get_manager() factory
        if name == "get_manager" and callable(obj):
            return obj()
        return obj
    raise RuntimeError(f"Could not resolve any of {candidates} in {mod_name}. Last error: {last_err}")

def _resolve_manager() -> Any:
    global _manager_singleton
    if _manager_singleton is not None:
        return _manager_singleton

    last_err: Exception | None = None

    # 1) Env/default spec
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

# ---------------------------------------------------------------------------
# Invocation helpers
# ---------------------------------------------------------------------------

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

    raise TypeError("Manager/Router is not invocable (expected async function, ainvoke/invoke, or callable).")

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
    """
    # Pydantic model (e.g., UIResponse) – prefer model_dump
    if hasattr(res, "model_dump"):
        res = res.model_dump()

    if isinstance(res, dict):
        return ChatResponse(
            messages=_as_list_messages(res.get("messages") or res.get("message") or res.get("text")),
            proposals=res.get("proposals"),
            staging=res.get("staging") or {"needs_commit": False},
        )

    return ChatResponse(messages=_as_list_messages(res), staging={"needs_commit": False})

# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # audit/telemetry best-effort
    try:
        telemetry_log("chat.request", {
            "org_id": req.org_id, "user_id": req.user_id, "active_tab": req.active_tab, "staging": req.staging
        })
    except Exception:
        pass

    # resolve & invoke your manager/router
    try:
        manager = _resolve_manager()
    except Exception as e:
        return ChatResponse(messages=[f"Agent not available: {e}"], staging={"needs_commit": False})

    try:
        raw = await _invoke_manager(manager, req.model_dump())
    except Exception as e:
        return ChatResponse(messages=[f"Agent failed: {type(e).__name__}: {e}"], staging={"needs_commit": False})

    return _normalize_to_chat_response(raw)
