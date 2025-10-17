# File: workshops_ai/schemas/chat.py
# Purpose:
#   Backward-compatible Pydantic v2 schemas ChatIn/ChatOut for the legacy /chat endpoint.
#   They map 1:1 to the newer UIRequest/UIResponse contract (schemas/ui_contract.py).
#
# Characteristics:
# - Strict (extra="forbid"), whitespace strip, safe JSON serialization.
# - Helper conversions: to_ui()/from_ui() bridging the new UI contract.
# - Minimal but extendable typing for "selection" and "proposals".
#
# Notes:
# - Newer code should prefer UIRequest/UIResponse directly (see schemas/ui_contract.py).
#   This file exists for older imports and compatibility.

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Common aliases ---

ActiveTab = Literal["assets", "impacts", "threats"]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        ser_json_inf_nan=False,
    )


# ======================
# INPUT (ChatIn)
# ======================

class ChatSelection(StrictModel):
    """Optional UI selection. For the Assets tab we typically expect asset_id."""
    asset_id: str | None = None


class ChatIn(StrictModel):
    """
    Input for POST /chat (legacy name).
    Back-compatible equivalent of UIRequest.
    """
    org_id: str = Field(..., description="Organization (tenant) ID.")
    active_tab: ActiveTab = Field(..., description="Active UI tab.")
    selection: ChatSelection | dict[str, Any] | None = Field(
        default=None,
        description="Optional selection; for assets often { asset_id }.",
    )
    message: str | None = Field(
        default=None,
        description="Optional user message in natural language.",
    )
    staging: bool = Field(
        default=True,
        description="Whether actions run in staging mode (no commit).",
    )
    user_id: str = Field(..., description="User ID for audit/telemetry.")

    # --- Conversion to the new contract ---
    def to_ui(self) -> "UIRequest":  # type: ignore[name-defined]
        from workshops_ai.schemas.ui_contract import UIRequest  # lazy import
        # UIRequest allows selection as dict[str, Any] | None — pass through as-is
        selection = self.selection.model_dump() if isinstance(self.selection, BaseModel) else self.selection
        return UIRequest(
            org_id=self.org_id,
            active_tab=self.active_tab,
            selection=selection or None,
            message=self.message,
            staging=self.staging,
            user_id=self.user_id,
        )

    @classmethod
    def from_ui(cls, ui: "UIRequest") -> "ChatIn":  # type: ignore[name-defined]
        selection = ui.selection or {}
        # Heuristic: if it's a plain dict, try to pull asset_id
        asset_id = None
        try:
            asset_id = selection.get("asset_id")  # type: ignore[assignment]
        except Exception:
            pass
        return cls(
            org_id=ui.org_id,
            active_tab=ui.active_tab,
            selection=ChatSelection(asset_id=asset_id),
            message=ui.message,
            staging=ui.staging,
            user_id=ui.user_id,
        )


# ======================
# OUTPUT (ChatOut)
# ======================

class StagingOut(StrictModel):
    batch_id: str | None = Field(default=None, description="Staging batch ID, if opened.")
    diff: dict[str, Any] | None = Field(default=None, description="Staging diff preview.")
    needs_commit: bool = Field(default=False, description="Whether an explicit commit is needed.")


class PolicyOut(StrictModel):
    """
    Optional policy mirror for UI (if the new UIResponse defines it).
    For AssetAgent we commonly expose 'emit-only' information.
    """
    emit_changes_enabled: bool = Field(default=True, description="Emit-only write mode is enabled.")


class ChatOut(StrictModel):
    """
    Output for /chat (legacy name).
    Back-compatible equivalent of UIResponse.
    """
    messages: list[str] = Field(default_factory=list, description="Messages for the UI (localized).")
    proposals: dict[str, Any] | None = Field(
        default=None,
        description="Optional proposals to render in UI (if the agent produces them).",
    )
    staging: StagingOut = Field(default_factory=StagingOut)
    policy: PolicyOut | None = Field(
        default=None,
        description="Optional policy info (present if UIResponse supports it).",
    )

    # --- Conversion from/to the new contract ---
    @classmethod
    def from_ui(cls, ui: "UIResponse") -> "ChatOut":  # type: ignore[name-defined]
        from workshops_ai.schemas.ui_contract import UIResponse  # noqa: F401 (readability hint)

        staging = ui.staging or {}
        # Safely read optional policy (UIResponse may not have it in older versions)
        ui_policy = getattr(ui, "policy", None)
        policy_obj: PolicyOut | None = None
        if ui_policy:
            if isinstance(ui_policy, dict):
                policy_obj = PolicyOut(**ui_policy)
            else:
                # Assume Pydantic model or object with attributes
                emit = getattr(ui_policy, "emit_changes_enabled", True)
                policy_obj = PolicyOut(emit_changes_enabled=bool(emit))

        return cls(
            messages=list(ui.messages or []),
            proposals=ui.proposals,
            staging=StagingOut(
                batch_id=staging.get("batch_id") if isinstance(staging, dict) else getattr(staging, "batch_id", None),
                diff=staging.get("diff") if isinstance(staging, dict) else getattr(staging, "diff", None),
                needs_commit=(
                    staging.get("needs_commit")
                    if isinstance(staging, dict)
                    else bool(getattr(staging, "needs_commit", False))
                ),
            ),
            policy=policy_obj,
        )

    def to_ui(self) -> "UIResponse":  # type: ignore[name-defined]
        from workshops_ai.schemas.ui_contract import UIResponse  # lazy import

        payload: dict[str, Any] = {
            "messages": self.messages,
            "proposals": self.proposals,
            "staging": self.staging.model_dump(),
        }

        # Pass policy only if UIResponse supports it (backward-safe)
        try:
            model_fields = getattr(UIResponse, "model_fields", {})  # Pydantic v2
            if "policy" in model_fields and self.policy is not None:
                payload["policy"] = self.policy.model_dump()
        except Exception:
            # Silently ignore if the target model doesn't support policy
            pass

        return UIResponse(**payload)


# ======================
# OpenAPI examples
# ======================
# Set OpenAPI examples FIRST…
ChatIn.model_config["json_schema_extra"] = {  # type: ignore[index]
    "examples": [
        {
            "org_id": "demo-org",
            "active_tab": "assets",
            "selection": {"asset_id": "asset-001"},
            "message": "List primary assets",
            "staging": True,
            "user_id": "demo-user",
        }
    ]
}

ChatOut.model_config["json_schema_extra"] = {  # type: ignore[index]
    "examples": [
        {
            "messages": ["Found 5 assets."],
            "proposals": None,
            "staging": {"batch_id": None, "diff": None, "needs_commit": False},
            "policy": {"emit_changes_enabled": True},
        }
    ]
}

# …then rebuild so FastAPI includes the examples in the generated OpenAPI schema.
ChatIn.model_rebuild()
ChatOut.model_rebuild()