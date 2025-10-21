# workshops_ai/graph/assets/tools/runners.py

from __future__ import annotations
from typing import Optional
from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.graph.factory import GraphDeps
from workshops_ai.graph.assets.modules.state import AssetState
from workshops_ai.graph.assets.modules.write_utils import validate_write_actions, apply_actions_async
from workshops_ai.graph.assets.modules.enums_json import CONF_LEVELS, TRI_LEVELS
from workshops_ai.graph.assets.modules.text_utils import lang

async def run_create(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

async def run_update_meta(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    from workshops_ai.graph.manager import route_chat
    return await route_chat(ui_req)

async def run_set_meta(*, deps: GraphDeps, ui_req: UIRequest) -> UIResponse:
    args: dict = ((ui_req.body or {}).get("args") or {})
    asset_id: str = str(args.get("assetId") or "").strip()
    if not asset_id:
        return UIResponse(
            messages=[{"error": "validation", "reason": "assetId is required"}],
            proposals=None,
            staging={"needs_commit": False, "batch_id": None, "diff": None},
            policy={"emit_changes_enabled": True},
        )

    try:
        existing = deps.backend_port.asset_get(asset_id)
    except Exception as ex:
        return UIResponse(
            messages=[{"error": "backend", "reason": f"Failed to fetch asset: {ex}"}],
            proposals=None,
            staging={"needs_commit": False, "batch_id": None, "diff": None},
            policy={"emit_changes_enabled": True},
        )
    if not existing:
        return UIResponse(
            messages=[{"error": "not_found", "asset_id": asset_id}],
            proposals=None,
            staging={"needs_commit": False, "batch_id": None, "diff": None},
            policy={"emit_changes_enabled": True},
        )

    def _get(obj, *keys):
        for k in keys:
            if isinstance(obj, dict) and k in obj:
                return obj[k]
            if hasattr(obj, k):
                try:
                    return getattr(obj, k)
                except Exception:
                    pass
        return None

    is_primary = bool(_get(existing, "isPrimary", "primary", "is_primary"))

    CONF_MAP = {"PUBLIC": 1, "INTERNAL": 2, "PROTECTED": 3, "STRICTLY_PROTECTED": 4}
    TRI_MAP  = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

    def _policy_to_int(slot: str, value):
        if value is None: return None
        s = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        if slot == "confidentiality": return CONF_MAP.get(s)
        if slot in ("integrity", "availability"): return TRI_MAP.get(s)
        return None

    c_raw = args.get("confidentiality", None)
    i_raw = args.get("integrity", None)
    a_raw = args.get("availability", None)

    cia_fields: dict[str, int | None] = {}
    if "confidentiality" in args: cia_fields["confidentiality"] = _policy_to_int("confidentiality", c_raw)
    if "integrity" in args:        cia_fields["integrity"] = _policy_to_int("integrity", i_raw)
    if "availability" in args:     cia_fields["availability"] = _policy_to_int("availability", a_raw)

    if cia_fields and not is_primary:
        return UIResponse(
            messages=[{"error": "policy", "reason": "CIA cannot be set for a supporting asset (isPrimary=false).", "asset_id": asset_id}],
            proposals=None,
            staging={"needs_commit": False, "batch_id": None, "diff": None},
            policy={"emit_changes_enabled": True},
        )

    for k, v in list(cia_fields.items()):
        if k in args and args.get(k) is not None and v is None:
            msg = f"{k} must be one of: " + ("PUBLIC/INTERNAL/PROTECTED/STRICTLY_PROTECTED" if k=="confidentiality" else "LOW/MEDIUM/HIGH")
            return UIResponse(
                messages=[{"error": "validation", "reason": msg}],
                proposals=None,
                staging={"needs_commit": False, "batch_id": None, "diff": None},
                policy={"emit_changes_enabled": True},
            )

    if not cia_fields:
        return UIResponse(
            messages=["No changes to apply."],
            proposals=None,
            staging={"needs_commit": False, "batch_id": None, "diff": None},
            policy={"emit_changes_enabled": True},
        )

    actions: list[dict] = [{"op": "set_asset_cia", "asset_id": asset_id, "fields": cia_fields}]

    dummy_state = AssetState(
        org_id=ui_req.org_id,
        active_tab="assets",
        message="",
        selection={"asset_id": asset_id},
        user_id=ui_req.user_id or "unknown",
        locale=(ui_req.headers or {}).get("accept-language") if hasattr(ui_req, "headers") else "en",
        agent_events=[],
    )

    validated, err = validate_write_actions(actions)
    if err:
        dummy_state.result = UIResponse.model_validate(
            {"messages":[{"error":"validation","reason":err}],"proposals":None,"staging":{"needs_commit":False,"batch_id":None,"diff":None},"policy":{"emit_changes_enabled":True}}
        )
        return UIResponse.model_validate(dummy_state.result)

    dummy_state = await apply_actions_async(deps=deps, state=dummy_state, acts=validated)
    return UIResponse.model_validate(dummy_state.result or {"messages":["No result."],"proposals":None,"staging":{"needs_commit":False,"batch_id":None,"diff":None},"policy":{"emit_changes_enabled":True}})

async def run_explain(*, deps: GraphDeps, text: str, locale: Optional[str] = None) -> dict:
    state = AssetState(
        org_id=getattr(getattr(deps, "settings", object()), "org_id", "demo-org"),
        active_tab="assets",
        message=text,
        question=text,
        locale=locale or "en",
        agent_events=[],
    )
    from workshops_ai.graph.assets.modules.explain import explain_assets_async
    state = await explain_assets_async(deps, state)
    return state.result or {"ok": False, "error": "no_result"}
