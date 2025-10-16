# File: workshops_ai/schemas/chat.py
# Purpose:
#   Backward-compatible Pydantic v2 schemas ChatIn/ChatOut for the legacy /chat endpoint.
#   They map 1:1 to the newer UIRequest/UIResponse contract (schemas/ui_contract.py).
#
# Characteristics:
# - Strict (extra="forbid"), whitespace strip, safe JSON serialization.
# - Helper conversions: to_ui()/from_ui() bridging the new UI contract.
# - A strict helper _to_ui_request(...) to accept a dict payload and return UIRequest,
#   normalizing language/locale, selection.asset_id, and carrying staging/session_id/conn_id.
#
# Notes:
# - Newer code should prefer UIRequest/UIResponse directly (see schemas/ui_contract.py).
#   This file exists for older imports and compatibility.

from __future__ import annotations

from typing import Any, Literal, Optional

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
        default=False,
        description="Whether actions run in staging mode (no commit); default False for safety.",
    )
    user_id: str = Field(..., description="User ID for audit/telemetry.")

    # Newly surfaced transport/session hints (strictly forwarded to UIRequest):
    session_id: str | None = Field(
        default=None,
        description="Conversation/session/thread id; if None, router may derive a stable value.",
    )
    conn_id: str | None = Field(
        default=None,
        description="Best-effort transport connection id (e.g., WS connection).",
    )

    # Language hints (we unify to UIRequest.language; locale works as a fallback):
    language: str | None = Field(
        default=None,
        description="Preferred language code (e.g., 'cs-CZ', 'en').",
    )
    locale: str | None = Field(
        default=None,
        description="Optional locale hint; used only if language is not provided.",
    )

    # --- Conversion to the new contract ---
    def to_ui(self) -> "UIRequest":  # type: ignore[name-defined]
        """
        Convert ChatIn → UIRequest:
        - selection is normalized to dict | None
        - language is unified: prefer `language`, else `locale`
        - staging/session_id/conn_id are passed through without guessing
        """
        from workshops_ai.schemas.ui_contract import UIRequest  # lazy import

        # Normalize selection to a dict with optional asset_id, or None
        selection: dict[str, Any] | None
        if isinstance(self.selection, BaseModel):
            sel_dict = self.selection.model_dump()
            selection = sel_dict if any(v is not None for v in sel_dict.values()) else None
        elif isinstance(self.selection, dict):
            # Keep only safe keys; currently we standardize on 'asset_id'
            asset_id = self.selection.get("asset_id")
            selection = {"asset_id": asset_id} if asset_id else None
        else:
            selection = None

        # Unify language/locale (do not guess if both are None)
        eff_lang = self.language or self.locale

        return UIRequest(
            org_id=self.org_id,
            active_tab=self.active_tab,
            selection=selection,
            message=self.message,
            staging=self.staging,
            user_id=self.user_id,
            session_id=self.session_id,
            conn_id=self.conn_id,
            language=eff_lang,
        )

    @classmethod
    def from_ui(cls, ui: "UIRequest") -> "ChatIn":  # type: ignore[name-defined]
        """
        Convert UIRequest → ChatIn:
        - selection normalized to ChatSelection (asset_id only)
        - language propagated to ChatIn.language (locale remains None here)
        - carries staging/session_id/conn_id
        """
        selection = ui.selection or {}
        # Heuristic: if it's a plain dict, try to pull asset_id
        asset_id: Optional[str] = None
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
            session_id=getattr(ui, "session_id", None),
            conn_id=getattr(ui, "conn_id", None),
            language=getattr(ui, "language", None),
            locale=None,
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
# Helper: strict payload → UIRequest
# ======================

def _to_ui_request(payload: dict[str, Any]) -> "UIRequest":  # type: ignore[name-defined]
    """
    Strictly parse a dict payload into UIRequest with normalization:

    - language/locale: prefer 'language'; if missing, fall back to 'locale';
      remove 'locale' key before validation since UIRequest is strict (extra='forbid').
    - selection: ensure a dict with only 'asset_id' (if present); ignore unknown keys.
    - staging/session_id/conn_id: pass them through verbatim if provided.
    - accept_language (legacy): if present and 'language' is not set, map it to language.

    If direct UIRequest validation fails, bridge via ChatIn (legacy) and convert with .to_ui().
    """
    from workshops_ai.schemas.ui_contract import UIRequest  # lazy import

    # Work on a shallow copy to avoid mutating caller's dict.
    data = dict(payload or {})

    # Normalize accept_language → language (only if language not explicitly provided)
    al = data.get("accept_language")
    if isinstance(al, str) and al.strip() and not data.get("language"):
        data["language"] = al.strip()[:256]
    data.pop("accept_language", None)  # drop unknown key for strict models

    # Unify language/locale (keep only language for UIRequest)
    if not data.get("language") and isinstance(data.get("locale"), str):
        data["language"] = (data.get("locale") or "").strip()[:256] or None
    data.pop("locale", None)

    # Normalize selection to a dict with asset_id if possible
    sel = data.get("selection")
    if isinstance(sel, BaseModel):
        sel = sel.model_dump()
    if isinstance(sel, dict):
        asset_id = sel.get("asset_id")
        data["selection"] = {"asset_id": asset_id} if asset_id else None
    elif sel is None:
        pass
    else:
        # Unknown selection shape → drop to avoid validation failure
        data["selection"] = None

    # Ensure staging/session_id/conn_id keys are preserved if present;
    # (no-op here since we don't alter them; just documenting intent)
    # data.get("staging"); data.get("session_id"); data.get("conn_id")

    # Validate directly against UIRequest first
    try:
        return UIRequest.model_validate(data)
    except Exception:
        # Fallback: validate as ChatIn (legacy) and bridge to UIRequest
        legacy = ChatIn.model_validate(data)
        return legacy.to_ui()


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
            "staging": False,
            "user_id": "demo-user",
            "session_id": "thread-abc",
            "conn_id": "WO5JZvgTYvhqpUTJ_Cr11Fls",
            "language": "cs-CZ",
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
