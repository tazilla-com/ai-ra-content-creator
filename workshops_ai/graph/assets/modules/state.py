# File: workshops_ai/graph/assets/modules/state.py
from __future__ import annotations

from typing import Optional, Literal, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

try:
    # Pydantic v2
    from pydantic import ConfigDict  # type: ignore
    _PYDANTIC_V2 = True
except Exception:  # pragma: no cover
    # Fallback pro Pydantic v1
    ConfigDict = dict  # type: ignore
    _PYDANTIC_V2 = False

# ---------------------------------------------------------------------------
# Intents / action names
# ---------------------------------------------------------------------------
Intent = Literal[
    "list",
    "get",
    "create",
    "update_meta",
    "set_meta",
    "explain",
    "list_types",
    "agent_reply",
]


class ToolEvent(TypedDict, total=False):
    """Minimal telemetry envelope for tool calls logged to the UI stream."""
    event_type: str
    tool: str
    ok: bool
    duration_ms: int
    meta: dict


class AssetCreateItem(BaseModel):
    """Structured create payload (used by tool runners and tests)."""
    id: Optional[str] = None
    name: str
    assetType: Optional[str] = None
    isPrimary: Optional[bool] = None
    confidentiality: Optional[str] = None
    integrity: Optional[str] = None
    availability: Optional[str] = None
    description: Optional[str] = None


class AssetUpdateItem(BaseModel):
    """
    Structured update payload.

    Povolené změny:
    - vždy: name, description
    - u primárních aktiv navíc: confidentiality, integrity, availability (CIA)
    """
    asset_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    confidentiality: Optional[str] = None  # NEW
    integrity: Optional[str] = None        # NEW
    availability: Optional[str] = None     # NEW


class AssetState(BaseModel):
    """
    Conversation state pro Assets agenta.
    Fields s `exclude=True` se nevrací klientovi.
    """
    if _PYDANTIC_V2:
        model_config = ConfigDict(validate_assignment=False, arbitrary_types_allowed=True)
    else:  # pragma: no cover
        class Config:
            validate_assignment = False
            arbitrary_types_allowed = True

    # Request context
    org_id: str
    active_tab: str = Field(default="assets")
    message: str | None = None
    selection: dict | None = None
    user_id: str | None = None
    locale: str | None = None
    language: str | None = None
    history: list[dict] | None = None
    staging: bool | None = None
    once_flags: dict | None = None

    # Structured inputs
    action: Literal[
        "list",
        "get",
        "create",
        "update_meta",
        "set_meta",
        "explain",
        "list_types",
        "agent_reply",
    ] | None = None
    create_items: list["AssetCreateItem"] | None = None
    update_items: list["AssetUpdateItem"] | None = None
    question: str | None = None

    # Planner output
    intent: Literal[
        "list",
        "get",
        "create",
        "update_meta",
        "set_meta",
        "explain",
        "list_types",
        "agent_reply",
    ] | None = None

    # Non-persisted runtime (internal)
    actions: list[dict] | None = Field(default=None, exclude=True)
    has_actions: bool = False
    change: dict | None = Field(default=None, exclude=True)
    result: dict | None = Field(default=None, exclude=True)

    # CREATE flow (slot filling)
    create_draft: dict | None = None
    pending_fields: list[str] | None = None
    create_flow: bool = False

    # Telemetry — UI-only
    agent_events: list["ToolEvent"] = Field(default_factory=list, exclude=True)
