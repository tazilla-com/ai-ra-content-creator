# File: workshops_ai/utils/submit_feedback.py
# Helper to apply SubmitOrFallbackResult to staging state and build a user-facing chat message.

from typing import Any, Optional
from workshops_ai.app.services.outbox_writer import SubmitOrFallbackResult
from pydantic import BaseModel


def _as_dict(obj: Any) -> dict:
    """Pydantic | dict -> dict (best-effort)."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json", by_alias=True, exclude_none=True)
    return dict(obj) if isinstance(obj, dict) else {}


def _actions(payload: dict) -> list[dict]:
    acts = payload.get("actions")
    return acts if isinstance(acts, list) else []


def _first_asset_create(payload: dict) -> dict[str, Optional[str]]:
    """
    Try to extract a friendly description from the first create-asset action
    to enrich chat messages (e.g., name + type).
    """
    for a in _actions(payload):
        op = str(a.get("op") or "").strip()
        if op in ("asset.create", "create_asset"):
            fields = a.get("fields") or {}
            return {
                "name": fields.get("name"),
                "assetType": fields.get("assetType"),
            }
    return {"name": None, "assetType": None}


def apply_submit_outcome_and_message(
    *,
    state: Any,                         # graph/router state; MUST expose boolean 'staging'
    changeset: Any,                     # ChangeSet (Pydantic) or dict you just submitted
    result: SubmitOrFallbackResult,     # returned from deps.submit_changeset(...)
) -> str:
    """
    Sets `state.staging` (a boolean) and returns a human-friendly chat message.

    Rules:
      - submitted == True  -> staging = False (already sent online)
      - queued_outbox      -> staging = True  (saved to outbox; pending commit/retry)
      - empty/invalid      -> staging = False (nothing pending)
    NOTE:
      `state.staging` must be a bool (the AssetState model validates it strictly).
    """

    # Compute small details used in messages (safe even if dict is not a ChangeSet)
    payload = _as_dict(changeset)
    acts = _actions(payload)
    act_count = len(acts)
    asset_hint = _first_asset_create(payload)
    asset_label = None
    if asset_hint["name"] and asset_hint["assetType"]:
        asset_label = f"‘{asset_hint['name']}’ ({asset_hint['assetType']})"
    elif asset_hint["name"]:
        asset_label = f"‘{asset_hint['name']}’"

    # 1) Online submit succeeded
    if result.submitted:
        state.staging = False
        sr = getattr(result, "submit_result", None)
        base = f"Submitted {act_count} change(s) successfully"
        via = f" via {getattr(sr, 'transport', None) or 'backend'}" if sr else ""
        detail = (
            f" (status {getattr(sr, 'status_code', None)})" if getattr(sr, "status_code", None) is not None else ""
        )
        subject = f" for asset {asset_label}" if asset_label else ""
        return f"✅ {base}{via}{detail}{subject}."

    # 2) Special reasons: empty / invalid
    rcode = (result.reason or {}).get("code") if result.reason else None
    if rcode == "empty_changeset":
        state.staging = False
        return "ℹ️ Nothing to submit: the change set contains no actions."
    if rcode == "invalid_changeset":
        state.staging = False
        return "⚠️ The change set is invalid and was not saved. Please fix the payload and try again."

    # 3) Fallback: queued to OUTBOX (most important UX case)
    # Prefer the explicit fallback_path on the result; fall back to reason.path if needed.
    path = None
    try:
        if getattr(result, "fallback_path", None):
            path = str(result.fallback_path)
    except Exception:
        path = None
    if path is None and result.reason:
        path = (result.reason or {}).get("path")

    submit_err = (result.reason or {}).get("submit_error") if result.reason else None

    # IMPORTANT: keep `staging` strictly boolean – AssetState expects bool
    state.staging = True

    subject = f" for asset {asset_label}" if asset_label else ""
    where = f"Saved to Outbox at {path}" if path else "Saved to Outbox"
    why = f"; online submit failed: {submit_err}" if submit_err else ""
    hint = "You can retry sending it or keep it in staging."
    return f"📝 Created a draft{subject} and {where}{why}. {hint}"
