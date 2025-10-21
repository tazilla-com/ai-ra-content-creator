# workshops_ai/graph/assets/modules/state.py

from __future__ import annotations

from typing import Optional, Literal, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict  # v2
    _PYDANTIC_V2 = True
except Exception:  # pragma: no cover
    ConfigDict = dict  # type: ignore
    _PYDANTIC_V2 = False


Intent = Literal["list", "get", "create", "update_meta", "explain", "list_types", "agent_reply"]


class ToolEvent(TypedDict, total=False):
    event_type: str
    tool: str
    ok: bool
    duration_ms: int
    meta: dict


class AssetCreateItem(BaseModel):
    id: Optional[str] = None
    name: str
    assetType: Optional[str] = None
    isPrimary: Optional[bool] = None
    confidentiality: Optional[str] = None
    integrity: Optional[str] = None
    availability: Optional[str] = None
    description: Optional[str] = None


class AssetUpdateItem(BaseModel):
    asset_id: str
    name: Optional[str] = None
    description: Optional[str] = None


class AssetState(BaseModel):
    if _PYDANTIC_V2:
        model_config = ConfigDict(validate_assignment=False, arbitrary_types_allowed=True)
    else:
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
    action: Literal["list","get","create","update_meta","explain","list_types","agent_reply"] | None = None
    create_items: list["AssetCreateItem"] | None = None
    update_items: list["AssetUpdateItem"] | None = None
    question: str | None = None

    # Working outputs
    intent: Literal["list","get","create","update_meta","explain","list_types","agent_reply"] | None = None

    # Non-persisted runtime
    actions: list[dict] | None = Field(default=None, exclude=True)
    has_actions: bool = False
    change: dict | None = Field(default=None, exclude=True)
    result: dict | None = Field(default=None, exclude=True)

    # CREATE flow
    create_draft: dict | None = None
    pending_fields: list[str] | None = None
    create_flow: bool = False

    # Telemetry — UI-only
    agent_events: list["ToolEvent"] = Field(default_factory=list, exclude=True)
