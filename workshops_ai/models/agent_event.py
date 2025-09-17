# File: workshops_ai/models/agent_event.py
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone

class AgentEvent(BaseModel):
    event: str               # např. "asset.create" | "asset.set_meta"
    org_id: str
    user_id: str
    asset_id: Optional[str] = None
    data: dict[str, Any]
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "AssetAgent"
    version: str = "v1"
