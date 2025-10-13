# File: workshops_ai/agents/tools/registry.py
# Purpose:
#   Canonical tool registry (SSOT) with strict Pydantic v2 argument schemas,
#   aligned to backend DTOs and graph ops. Produces OpenAI-style tool specs
#   (a.k.a. "functions") for prompting and validates payloads.
#
# What changed in this version:
#   - Added **slot metadata** (`slots`) and **ask_order** to tools that need it,
#     so the LLM can drive slot-filling with one concise follow-up question.
#   - Kept tool names/ops as-is (do NOT rename) to avoid breaking routers/WS.
#   - No pseudo “help/chitchat” tools; only concrete, callable tools live here.
#
# Exposed domains and ops (must match your routers/graphs):
#   Assets:
#     - asset.list
#     - asset.list_types
#     - asset.get
#     - asset.ask_next
#     - asset.create
#     - asset.update_meta
#     - asset.explain
#     - asset.set_meta
#   Impacts:
#     - impact.list
#     - impact.levels.for_asset
#     - impact.set_level
#     - impact.bulk_set
#     - impact.explain
#   Threats:
#     - threat.list
#     - threat.set_probability
#     - threat.bulk_set
#     - threat.explain
#   Controls:
#     - control.list
#     - control.explain
#
from __future__ import annotations

import json
import re
from typing import Any, ClassVar, Dict, Literal, Optional, List, get_args

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

# =============================================================================
# Shared primitives and constants
# =============================================================================

# UUID v7 (loose) validator: 8-4-4-4-12, version "7", variant [89ab]
_UUID_V7_RX = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-7[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)

def _is_uuid_v7_like(s: str) -> bool:
    return bool(_UUID_V7_RX.match(s or ""))


# Enumerations (kept in EN; do not translate canonical codes)
ConfidentialityLevel = Literal['PUBLIC', 'INTERNAL', 'PROTECTED', 'STRICTLY_PROTECTED']
TriadLevel           = Literal['LOW', 'MEDIUM', 'HIGH']

AssetType = Literal[
    "Application software", "Cloud services", "Data storage",
    "Electronic data/Business process", "Other", "Information system",
    "IoT", "Location", "Mobile devices", "Laptop", "Operating system",
    "Paper data", "PC", "Staff", "Portable data media",
    "Server infrastructure", "Network infrastructure",
    "Virtualization platform", "Equipment, machines"
]

ImpactCategory = Literal[
    "Availability", "Clients", "Confidentiality", "Finance", "Integrity",
    "Legislation", "Reputation", "People", "Other"
]

ThreatCategory = Literal[
    "Physical threats", "Natural threats", "Compromise of functions or services",
    "Human actions", "Technical failures and infrastructure disruptions",
    "Organizational threats", "Legal and regulatory threats",
    "Violation of privacy", "Other"
]


# =============================================================================
# Slot metadata (declarative; used only in prompts for better LLM UX)
# =============================================================================

class SlotDef(BaseModel):
    """Declarative description of a single slot to guide LLM questions (prompt-only)."""
    model_config = ConfigDict(extra="forbid")

    required: bool = Field(False, description="Whether this slot is required before calling the tool.")
    autofill: bool = Field(False, description="Whether the model may infer/default this slot.")
    label: StrictStr = Field(..., description="Human-readable label (EN).")
    hint: Optional[StrictStr] = Field(None, description="Short guidance or example value.")
    enum: Optional[List[StrictStr]] = Field(None, description="Optional guidance enum (not enforced at runtime).")

SlotsMap = Dict[str, SlotDef]


# =============================================================================
# Reusable selectors and DTO-aligned argument models
# =============================================================================

class AssetSelector(BaseModel):
    """
    Asset selection by ID or by exact name.
    At least one of (assetId, assetName) must be provided.
    """
    model_config = ConfigDict(extra="forbid")
    assetId: Optional[StrictStr] = Field(None, description="UUID v7 of the asset, if known.")
    assetName: Optional[StrictStr] = Field(None, description="Exact asset name, if ID is unknown.")

    @field_validator("assetId")
    @classmethod
    def _check_uuid_v7(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not _is_uuid_v7_like(v):
            raise ValueError("assetId must be a UUID v7")
        return v

    @field_validator("assetName")
    @classmethod
    def _non_empty_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v2 = v.strip()
        if not v2:
            raise ValueError("assetName must be a non-empty string")
        return v2

    @model_validator(mode="after")
    def _at_least_one(self):
        if not (self.assetId or self.assetName):
            raise ValueError("Provide either assetId or assetName")
        return self


# ---------- Assets ----------

class AssetListArgs(BaseModel):
    """List assets with optional substring filter on name."""
    model_config = ConfigDict(extra="forbid")
    query: Optional[StrictStr] = Field(None, description="Optional substring filter on asset name.")

    @field_validator("query")
    @classmethod
    def _trim(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v2 = v.strip()
        return v2 or None

class AssetCreateArgs(BaseModel):
    """
    Create a new asset.
    - isPrimary defaults to True (primary asset). If False (supporting asset),
      confidentiality/integrity/availability must remain null.
    """
    model_config = ConfigDict(extra="forbid")

    # Core fields
    name: StrictStr = Field(..., description="Localized asset name.")
    description: Optional[StrictStr] = Field(None, description="Localized description.")
    isPrimary: StrictBool = Field(True, description="Primary (business-critical) vs supporting asset.")
    assetType: AssetType = Field(..., description="Catalog value (fixed code list).")

    # CIA (primary assets only) — string policy levels
    confidentiality: Optional[ConfidentialityLevel] = Field(None, description="Confidentiality policy level.")
    integrity: Optional[TriadLevel] = Field(None, description="Integrity policy level.")
    availability: Optional[TriadLevel] = Field(None, description="Availability policy level.")

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("name must be a non-empty string")
        return v2

    @field_validator("description")
    @classmethod
    def _desc_trim(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return v.strip() or None

    @model_validator(mode="after")
    def _cia_rule(self):
        if self.isPrimary is False:
            if any(x is not None for x in (self.confidentiality, self.integrity, self.availability)):
                raise ValueError(
                    "For supporting assets (isPrimary=False), confidentiality/integrity/availability must be null"
                )
        return self

class AssetSetMetaArgs(BaseModel):
    """
    Update metadata of an existing asset (broader set incl. CIA numeric levels).
    All fields optional except assetId.
    """
    model_config = ConfigDict(extra="forbid")

    assetId: StrictStr = Field(..., description="UUID v7 of the existing asset.")
    assetType: Optional[AssetType] = None
    isPrimary: Optional[StrictBool] = None
    confidentiality: Optional[StrictInt] = Field(None, ge=0, le=4, description="Confidentiality 0..4")
    integrity: Optional[StrictInt] = Field(None, ge=0, le=4, description="Integrity 0..4")
    availability: Optional[StrictInt] = Field(None, ge=0, le=4, description="Availability 0..4")
    description: Optional[StrictStr] = None

    @field_validator("assetId")
    @classmethod
    def _uuid_v7_only(cls, v: str) -> str:
        if not _is_uuid_v7_like(v):
            raise ValueError("assetId must be UUID v7")
        return v

    @field_validator("description")
    @classmethod
    def _trim_or_none(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v2 = v.strip()
        return v2 or None

class AssetUpdateMetaArgs(BaseModel):
    """
    Update asset name/description (no isPrimary/assetType changes here).
    """
    model_config = ConfigDict(extra="forbid")
    assetId: StrictStr = Field(..., description="UUID v7 of the existing asset.")
    name: Optional[StrictStr] = Field(None, description="New localized name.")
    description: Optional[StrictStr] = Field(None, description="New localized description.")

    @field_validator("assetId")
    @classmethod
    def _uuid_v7_only(cls, v: str) -> str:
        if not _is_uuid_v7_like(v):
            raise ValueError("assetId must be UUID v7")
        return v

    @field_validator("name", "description")
    @classmethod
    def _trim_non_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v2 = v.strip()
        if not v2:
            raise ValueError("empty strings are not allowed")
        return v2

class AssetExplainArgs(AssetSelector):
    """Explain an asset (by ID or name), leveraging retrieval if available."""
    pass

class AssetGetArgs(BaseModel):
    """
    Detail lookup; either assetId or assetName (soft validation here because
    UI selection may inject ID outside of payload).
    """
    model_config = ConfigDict(extra="forbid")
    assetId: Optional[StrictStr] = None
    assetName: Optional[StrictStr] = None


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Impacts ----------

class ImpactListArgs(BaseModel):
    """List available impact definitions (catalog)."""
    model_config = ConfigDict(extra="forbid")

class ImpactLevelsForAssetArgs(AssetSelector):
    """Return current impact levels assigned to the asset."""
    pass

class ImpactSetLevelArgs(AssetSelector):
    """
    Set impact level (0..4) for a specific impact on an asset.
    """
    model_config = ConfigDict(extra="forbid")
    impactName: StrictStr = Field(..., description="Exact impact name as defined in catalog.")
    impactLevel: StrictInt = Field(..., ge=0, le=4, description="Impact level 0..4")

    @field_validator("impactName")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("impactName must be a non-empty string")
        return v2

class ImpactBulkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    impactName: StrictStr = Field(..., description="Exact impact name as defined in catalog.")
    impactLevel: StrictInt = Field(..., ge=0, le=4, description="Impact level 0..4")

    @field_validator("impactName")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("impactName must be a non-empty string")
        return v2

class ImpactBulkSetArgs(AssetSelector):
    model_config = ConfigDict(extra="forbid")
    items: List[ImpactBulkItem] = Field(..., min_length=1, description="One or more impact updates.")

class ImpactExplainArgs(BaseModel):
    """Explain an impact concept by its exact name (not category)."""
    model_config = ConfigDict(extra="forbid")
    impactName: StrictStr = Field(..., description="Exact impact name to explain.")

    @field_validator("impactName")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("impactName must be a non-empty string")
        return v2


# ---------- Threats ----------

class ThreatListArgs(BaseModel):
    """List available threats (catalog)."""
    model_config = ConfigDict(extra="forbid")

class ThreatSetProbabilityArgs(AssetSelector):
    """Set probability (0..4) for a threat (by threatName) on an asset."""
    model_config = ConfigDict(extra="forbid")
    threatName: StrictStr = Field(..., description="Exact threat name as defined in catalog.")
    probability: StrictInt = Field(..., ge=0, le=4, description="Probability 0..4")

    @field_validator("threatName")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("threatName must be a non-empty string")
        return v2

class ThreatBulkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threatName: StrictStr = Field(..., description="Exact threat name.")
    probability: StrictInt = Field(..., ge=0, le=4, description="Probability 0..4")

    @field_validator("threatName")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("threatName must be a non-empty string")
        return v2

class ThreatBulkSetArgs(AssetSelector):
    model_config = ConfigDict(extra="forbid")
    items: List[ThreatBulkItem] = Field(..., min_length=1, description="One or more threat probability updates.")

class ThreatExplainArgs(BaseModel):
    """Explain a threat concept by its exact name."""
    model_config = ConfigDict(extra="forbid")
    threatName: StrictStr = Field(..., description="Exact threat name to explain.")

    @field_validator("threatName")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("threatName must be a non-empty string")
        return v2


# ---------- Controls ----------

class ControlListArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

class ControlExplainArgs(BaseModel):
    """Explain a control by its catalog name."""
    model_config = ConfigDict(extra="forbid")
    controlName: StrictStr = Field(..., description="Exact catalog control name to explain.")

    @field_validator("controlName")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("controlName must be a non-empty string")
        return v2


# =============================================================================
# ToolSpec & Registry
# =============================================================================

class ToolSpec(BaseModel):
    """Canonical tool specification (registry row)."""
    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field(..., description="Canonical tool name, e.g., 'asset.create'.")
    description: StrictStr = Field(..., description="Short action description for the model.")
    args_model: type[BaseModel] = Field(..., description="Pydantic v2 model class for the arguments.")
    tab: Literal["assets", "impacts", "threats", "controls"] = Field(..., description="UI tab / domain.")
    endpoint: StrictStr = Field(..., description="Graph endpoint path where this tool is executed.")
    op: StrictStr = Field(..., description="Canonical op code (used by router/WS/ChangeSet).")

    # Slot-driven guidance for LLM (prompt-only; ignored by runtime validation)
    slots: Optional[SlotsMap] = Field(default=None, description="Declarative slot metadata for prompts.")
    ask_order: Optional[List[StrictStr]] = Field(default=None, description="Preferred order of slot questions.")

    def json_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)  # cleaner payloads
        return schema

    def as_openai_function(self) -> dict[str, Any]:
        """
        Produce OpenAI-style tool spec:
          {"type":"function","function":{"name", "description", "parameters", "x-slots"?, "x-ask-order"?}}
        """
        fn = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }
        if self.slots is not None:
            fn["function"]["x-slots"] = {k: v.model_dump(mode="json") for k, v in self.slots.items()}
        if self.ask_order is not None:
            fn["function"]["x-ask-order"] = list(self.ask_order)
        return fn

    def validate_payload(self, payload: dict[str, Any]) -> BaseModel:
        return self.args_model.model_validate(payload)


class ToolRegistry:
    """In-memory registry for all tools across tabs (SSOT)."""
    _tools: ClassVar[List[ToolSpec]] = []

    # --------------------------- Construction -------------------------------

    @classmethod
    def _build_default(cls) -> List[ToolSpec]:
        tools: List[ToolSpec] = []

        # ===== Assets =====
        tools.append(ToolSpec(
            name="asset.list",
            description="List assets (optionally filtered by text query).",
            args_model=AssetListArgs,
            tab="assets",
            endpoint="/graph/assets",
            op="asset.list",
            # Listing needs no slots.
        ))
        tools.append(ToolSpec(
            name="asset.list_types",
            description="List available asset types from catalog (full list, no pagination).",
            args_model=NoArgs,
            tab="assets",
            endpoint="/graph/assets",
            op="asset.list_types",
        ))
        tools.append(ToolSpec(
            name="asset.get",
            description="Get asset detail by selection (prefer UI selection.asset_id when available).",
            args_model=AssetGetArgs,
            tab="assets",
            endpoint="/graph/assets",
            op="asset.get",
        ))
        tools.append(ToolSpec(
            name="asset.ask_next",
            description="Ask one concise follow-up question to complete asset creation/update (slot-filling helper).",
            args_model=NoArgs,
            tab="assets",
            endpoint="/graph/assets",
            op="asset.ask_next",
        ))
        tools.append(ToolSpec(
            name="asset.create",
            description="Create a new asset including isPrimary and assetType (CIA levels only for primary).",
            args_model=AssetCreateArgs,
            tab="assets",
            endpoint="/graph/assets",
            op="asset.create",
            slots={
                "name": SlotDef(required=True,  autofill=False, label="Asset name",  hint="e.g., CRM, Data Warehouse"),
                "assetType": SlotDef(required=True, autofill=True,  label="Asset type",
                                     hint="Catalog value", enum=list(get_args(AssetType))),
                "isPrimary": SlotDef(required=False, autofill=True, label="Primary asset", hint="true/false"),
                "confidentiality": SlotDef(required=False, autofill=True, label="Confidentiality (policy)",
                                           hint="PUBLIC/INTERNAL/PROTECTED/STRICTLY_PROTECTED",
                                           enum=list(get_args(ConfidentialityLevel))),
                "integrity": SlotDef(required=False, autofill=True, label="Integrity (policy)",
                                     hint="LOW/MEDIUM/HIGH", enum=list(get_args(TriadLevel))),
                "availability": SlotDef(required=False, autofill=True, label="Availability (policy)",
                                        hint="LOW/MEDIUM/HIGH", enum=list(get_args(TriadLevel))),
                "description": SlotDef(required=False, autofill=True, label="Description", hint="Short purpose/notes"),
            },
            ask_order=["name", "assetType", "isPrimary", "confidentiality", "integrity", "availability", "description"],
        ))
        tools.append(ToolSpec(
            name="asset.update_meta",
            description="Update asset name/description (no isPrimary/assetType change here).",
            args_model=AssetUpdateMetaArgs,
            tab="assets",
            endpoint="/graph/assets",
            op="asset.update_meta",
            slots={
                "assetId": SlotDef(required=True,  autofill=False, label="Asset ID", hint="UUID v7"),
                "name":    SlotDef(required=False, autofill=False, label="New name", hint="If renaming"),
                "description": SlotDef(required=False, autofill=True, label="New description", hint="Short notes"),
            },
            ask_order=["assetId", "name", "description"],
        ))
        tools.append(ToolSpec(
            name="asset.explain",
            description="Explain an asset (by ID or exact name).",
            args_model=AssetExplainArgs,
            tab="assets",
            endpoint="/graph/assets",
            op="asset.explain",
            slots={
                "assetId": SlotDef(required=False, autofill=False, label="Asset ID", hint="UUID v7"),
                "assetName": SlotDef(required=False, autofill=False, label="Asset name", hint="Exact name"),
            },
            ask_order=["assetId", "assetName"],
        ))
        tools.append(ToolSpec(
            name="asset.set_meta",
            description="Update asset metadata (assetType, CIA numeric levels, description, isPrimary).",
            args_model=AssetSetMetaArgs,
            tab="assets",
            endpoint="/graph/assets",
            op="asset.set_meta",
            slots={
                "assetId": SlotDef(required=True,  autofill=False, label="Asset ID", hint="UUID v7"),
                "assetType": SlotDef(required=False, autofill=True, label="Asset type",
                                     hint="Catalog value", enum=list(get_args(AssetType))),
                "isPrimary": SlotDef(required=False, autofill=False, label="Primary asset", hint="true/false"),
                "confidentiality": SlotDef(required=False, autofill=True, label="Confidentiality (0..4)", hint="Integer 0..4"),
                "integrity":       SlotDef(required=False, autofill=True, label="Integrity (0..4)",       hint="Integer 0..4"),
                "availability":    SlotDef(required=False, autofill=True, label="Availability (0..4)",    hint="Integer 0..4"),
                "description":     SlotDef(required=False, autofill=True, label="Description",            hint="Short notes"),
            },
            ask_order=["assetId", "assetType", "isPrimary", "confidentiality", "integrity", "availability", "description"],
        ))

        # ===== Impacts =====
        tools.append(ToolSpec(
            name="impact.list",
            description="List available impact definitions from catalog (full list).",
            args_model=ImpactListArgs,
            tab="impacts",
            endpoint="/graph/impacts",
            op="impact.list",
        ))
        tools.append(ToolSpec(
            name="impact.levels.for_asset",
            description="Show current impact levels for a specific asset.",
            args_model=ImpactLevelsForAssetArgs,
            tab="impacts",
            endpoint="/graph/impacts",
            op="impact.levels.for_asset",
            slots={
                "assetId":   SlotDef(required=True,  autofill=False, label="Asset ID",   hint="UUID v7 or provide name"),
                "assetName": SlotDef(required=False, autofill=False, label="Asset name", hint="Exact name if ID unknown"),
            },
            ask_order=["assetId", "assetName"],
        ))
        tools.append(ToolSpec(
            name="impact.set_level",
            description="Set a single impact level (0..4) for an asset (by impactName).",
            args_model=ImpactSetLevelArgs,
            tab="impacts",
            endpoint="/graph/impacts",
            op="impact.set_level",
            slots={
                "assetId":     SlotDef(required=False, autofill=False, label="Asset ID",     hint="UUID v7"),
                "assetName":   SlotDef(required=False, autofill=False, label="Asset name",   hint="Exact name if ID unknown"),
                "impactName":  SlotDef(required=True,  autofill=False, label="Impact name",  hint="Exact catalog name (e.g., Reputation)"),
                "impactLevel": SlotDef(required=True,  autofill=False, label="Impact level", hint="Integer 0..4"),
            },
            ask_order=["assetId", "assetName", "impactName", "impactLevel"],
        ))
        tools.append(ToolSpec(
            name="impact.bulk_set",
            description="Bulk set multiple impact levels (0..4) for an asset.",
            args_model=ImpactBulkSetArgs,
            tab="impacts",
            endpoint="/graph/impacts",
            op="impact.bulk_set",
            slots={
                "assetId":   SlotDef(required=False, autofill=False, label="Asset ID",   hint="UUID v7"),
                "assetName": SlotDef(required=False, autofill=False, label="Asset name", hint="Exact name if ID unknown"),
                "items":     SlotDef(required=True,  autofill=False, label="Impact items", hint="List of {impactName, impactLevel}"),
            },
            ask_order=["assetId", "assetName", "items"],
        ))
        tools.append(ToolSpec(
            name="impact.explain",
            description="Explain an impact concept (by impactName).",
            args_model=ImpactExplainArgs,
            tab="impacts",
            endpoint="/graph/impacts",
            op="impact.explain",
            slots={
                "impactName": SlotDef(required=True, autofill=False, label="Impact name", hint="Exact catalog name"),
            },
            ask_order=["impactName"],
        ))

        # ===== Threats =====
        tools.append(ToolSpec(
            name="threat.list",
            description="List available threats from catalog (full list).",
            args_model=ThreatListArgs,
            tab="threats",
            endpoint="/graph/threats",
            op="threat.list",
        ))
        tools.append(ToolSpec(
            name="threat.set_probability",
            description="Set threat probability (0..4) for an asset's threat (by threatName).",
            args_model=ThreatSetProbabilityArgs,
            tab="threats",
            endpoint="/graph/threats",
            op="threat.set_probability",
            slots={
                "assetId":    SlotDef(required=False, autofill=False, label="Asset ID",    hint="UUID v7"),
                "assetName":  SlotDef(required=False, autofill=False, label="Asset name",  hint="Exact name if ID unknown"),
                "threatName": SlotDef(required=True,  autofill=False, label="Threat name", hint="Exact catalog name (e.g., Phishing)"),
                "probability":SlotDef(required=True,  autofill=False, label="Probability", hint="Integer 0..4"),
            },
            ask_order=["assetId", "assetName", "threatName", "probability"],
        ))
        tools.append(ToolSpec(
            name="threat.bulk_set",
            description="Bulk set multiple threat probabilities (0..4) for an asset.",
            args_model=ThreatBulkSetArgs,
            tab="threats",
            endpoint="/graph/threats",
            op="threat.bulk_set",
            slots={
                "assetId":   SlotDef(required=False, autofill=False, label="Asset ID",   hint="UUID v7"),
                "assetName": SlotDef(required=False, autofill=False, label="Asset name", hint="Exact name if ID unknown"),
                "items":     SlotDef(required=True,  autofill=False, label="Threat items", hint="List of {threatName, probability}"),
            },
            ask_order=["assetId", "assetName", "items"],
        ))
        tools.append(ToolSpec(
            name="threat.explain",
            description="Explain a threat concept (by threatName).",
            args_model=ThreatExplainArgs,
            tab="threats",
            endpoint="/graph/threats",
            op="threat.explain",
            slots={
                "threatName": SlotDef(required=True, autofill=False, label="Threat name", hint="Exact catalog name"),
            },
            ask_order=["threatName"],
        ))

        # ===== Controls =====
        tools.append(ToolSpec(
            name="control.list",
            description="List controls from catalog (full list).",
            args_model=ControlListArgs,
            tab="controls",
            endpoint="/graph/controls",
            op="control.list",
        ))
        tools.append(ToolSpec(
            name="control.explain",
            description="Explain a control (by catalog name).",
            args_model=ControlExplainArgs,
            tab="controls",
            endpoint="/graph/controls",
            op="control.explain",
            slots={
                "controlName": SlotDef(required=True, autofill=False, label="Control name", hint="Exact catalog name"),
            },
            ask_order=["controlName"],
        ))

        return tools

    # --------------------------- API ----------------------------------------

    @classmethod
    def ensure_initialized(cls) -> None:
        if not cls._tools:
            cls._tools = cls._build_default()

    @classmethod
    def all(cls) -> List[ToolSpec]:
        cls.ensure_initialized()
        return list(cls._tools)

    @classmethod
    def by_tab(cls, tab: Literal["assets", "impacts", "threats", "controls"]) -> List[ToolSpec]:
        cls.ensure_initialized()
        return [t for t in cls._tools if t.tab == tab]

    @classmethod
    def get(cls, name: str) -> Optional[ToolSpec]:
        cls.ensure_initialized()
        for t in cls._tools:
            if t.name == name:
                return t
        return None

    # ------------------------ Prompt helpers --------------------------------

    @classmethod
    def as_openai_functions(
        cls,
        tab: Optional[Literal["assets", "impacts", "threats", "controls"]] = None
    ) -> list[dict[str, Any]]:
        cls.ensure_initialized()
        selected = cls.by_tab(tab) if tab else cls.all()
        return [t.as_openai_function() for t in selected]

    @classmethod
    def as_prompt_json(
        cls,
        tab: Optional[Literal["assets", "impacts", "threats", "controls"]] = None
    ) -> str:
        funs = cls.as_openai_functions(tab)
        return json.dumps(funs, separators=(",", ":"))

    # ----------------------- Validation / Routing ----------------------------

    @classmethod
    def validate_call(cls, tool_name: str, payload: dict[str, Any]) -> BaseModel:
        spec = cls.get(tool_name)
        if spec is None:
            # Construct a pydantic-style ValidationError for uniform handling
            raise ValidationError.from_exception_data(
                "ToolNotFound",
                [{"loc": ("tool",), "msg": f"Unknown tool: {tool_name}", "type": "value_error"}],
            )
        return spec.validate_payload(payload)

    @classmethod
    def endpoint_for(cls, tool_name: str) -> str:
        spec = cls.get(tool_name)
        if spec is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return spec.endpoint

    @classmethod
    def op_for(cls, tool_name: str) -> str:
        spec = cls.get(tool_name)
        if spec is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return spec.op
