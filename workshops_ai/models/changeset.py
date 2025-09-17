# /app/models/changeset.py
from __future__ import annotations
from typing import Literal, List, Optional, Dict, Any
from pydantic import BaseModel, Field, StrictStr, StrictInt, StrictFloat

# ---- položky
class ImpactItem(BaseModel):
    impact_code: StrictStr
    level: StrictInt = Field(ge=0, le=4)
    note: Optional[str] = None

class RiskItem(BaseModel):
    threat_code: StrictStr
    probability: StrictInt = Field(ge=0, le=4)
    note: Optional[str] = None

class ControlUpsert(BaseModel):
    control_code: StrictStr
    rationale: Optional[str] = None
    expected_coverage: Optional[StrictFloat] = Field(default=None, ge=0.0, le=1.0)

# ---- akce
class ActionSetImpacts(BaseModel):
    op: Literal["set_impacts"]
    mode: Literal["merge", "replace"] = "merge"
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None  # "client:<token>"
    items: List[ImpactItem]

class ActionSetRisks(BaseModel):
    op: Literal["set_risks"]
    mode: Literal["merge", "replace"] = "merge"
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None
    items: List[RiskItem]

class ActionUpsertControls(BaseModel):
    op: Literal["upsert_controls"]
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None
    items: List[ControlUpsert]

class ActionRemoveControls(BaseModel):
    op: Literal["remove_controls"]
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None
    codes: List[StrictStr]

class ActionSetAssetMeta(BaseModel):
    op: Literal["set_asset_meta"]
    asset_id: Optional[StrictStr] = None
    asset_ref: Optional[StrictStr] = None
    fields: Dict[str, Any]

class ActionCreateAsset(BaseModel):
    op: Literal["create_asset"]
    client_token: StrictStr
    fields: Dict[str, Any]  # name, asset_type_uuid/asset_type_code, primary, ...

Action = (
    ActionCreateAsset
    | ActionSetAssetMeta
    | ActionSetImpacts
    | ActionSetRisks
    | ActionUpsertControls
    | ActionRemoveControls
)

# ---- kořen
class ChangeSet(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    idempotency_key: StrictStr
    org_id: StrictStr
    intent: Literal["preview", "apply"] = "apply"
    audit: Dict[str, Any]
    actions: List[Action]
