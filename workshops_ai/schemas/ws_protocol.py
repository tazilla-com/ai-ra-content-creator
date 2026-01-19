from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, StrictBool, StrictStr, HttpUrl, field_validator
from pydantic.config import ConfigDict


WorkshopId = Literal["assets", "impacts", "threats", "controls"]
JSONDict = Dict[str, Any]


class AssetInput(BaseModel):
    """
    Asset sent FROM Tazilla TO agent.
    Represents existing assets in the system.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: Optional[UUID] = None
    name: StrictStr
    description: Optional[StrictStr] = None
    isPrimary: bool = True
    confidentiality: Optional[Literal["PUBLIC", "INTERNAL", "PROTECTED", "STRICTLY_PROTECTED"]] = None
    integrity: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    availability: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    assetType: StrictStr
    selected: bool = False


class AssetOutput(BaseModel):
    """
    Asset sent FROM agent TO Tazilla.
    Includes reasoning for educational purposes.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: Optional[UUID] = None
    name: StrictStr
    description: Optional[StrictStr] = None
    isPrimary: bool = True
    confidentiality: Optional[Literal["PUBLIC", "INTERNAL", "PROTECTED", "STRICTLY_PROTECTED"]] = None
    integrity: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    availability: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    assetType: StrictStr
    reasoning: Optional[StrictStr] = Field(default=None, description="Agent's explanation for CIA levels")


class OrganizationalUnit(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: StrictStr
    description: Optional[StrictStr] = None


class JobPosition(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: StrictStr
    description: Optional[StrictStr] = None


class Organization(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    organizationName: StrictStr
    legalForm: Optional[StrictStr] = None
    registeredOffice: Optional[StrictStr] = None
    country: Optional[StrictStr] = None
    numberOfEmployees: Optional[int] = None
    organizationalUnits: List[OrganizationalUnit] = Field(default_factory=list)
    jobPositions: List[JobPosition] = Field(default_factory=list)


class JsonPatch(BaseModel):
    """JSON Patch operation (RFC 6902)"""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    op: Literal["add", "remove", "replace"]
    path: StrictStr = Field(..., description="JSON Pointer, e.g., '/assets/-' or '/assets/0/name'")
    value: Optional[Any] = Field(default=None, description="Value for add/replace operations")


class ChatPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: StrictStr = Field(..., min_length=1)
    assets: List[AssetInput] = Field(default_factory=list)
    organization: Optional[Organization] = None
    model: StrictStr

    @field_validator("message")
    @classmethod
    def _message_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("message cannot be empty")
        return v.strip()


class ClientMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: Literal["chat", "ping", "sync_history"]
    payload: Optional[JSONDict] = None


class HelloPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ok: bool = True
    session_id: StrictStr
    workshop_id: WorkshopId
    message: StrictStr = "Hello! I'm ready to help with your assets."


class ToolStatusPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tool: StrictStr
    status: Literal["started", "completed", "failed"]
    message: Optional[StrictStr] = None


class UIResponsePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: StrictStr
    patches: Optional[List[JsonPatch]] = None
    assets: Optional[List[AssetOutput]] = None
    tools_used: List[StrictStr] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: StrictStr
    timestamp: datetime
    assets: Optional[List[AssetOutput]] = None


class HistorySyncPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    messages: List[ConversationMessage]
    total_messages: int
    returned_messages: int


class ErrorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: StrictStr
    code: Literal[
        "BUSY",
        "VALIDATION_ERROR",
        "AGENT_ERROR",
        "TIMEOUT",
        "RATE_LIMITED",
        "SESSION_EXPIRED",
        "TOOL_FAILED"
    ]
    details: Optional[Any] = None


class AgentMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: Literal["hello", "pong", "tool_status", "ui_response", "history_sync", "error"]
    payload: JSONDict = Field(default_factory=dict)


__all__ = [
    "WorkshopId",
    "JSONDict",
    "AssetInput",
    "AssetOutput",
    "Organization",
    "OrganizationalUnit",
    "JobPosition",
    "JsonPatch",
    "ClientMessage",
    "ChatPayload",
    "AgentMessage",
    "HelloPayload",
    "ToolStatusPayload",
    "UIResponsePayload",
    "ConversationMessage",
    "HistorySyncPayload",
    "ErrorPayload",
]
