# File: workshops_ai/agents/tools/registry.py
# Purpose: Canonical tool registry and strict Pydantic v2 argument schemas
#          aligned with backend DTOs (the “source of truth” you provided).
#          Generates OpenAI-style tool/function specs for prompts
#          and maps canonical tool names → graph endpoints.
#
# Conventions:
# - Tool names: "<domain>.<verb>[.<qualifier>]" (dot-separated, canonical).
# - Args models: camelCase fields to match backend DTOs; extra fields forbidden.
# - Validation: strict types, UUID v7 check where applicable.
# - Tabs/domains: "assets", "impacts", "threats", "controls".
#
# This module is declarative (no business logic).

from __future__ import annotations

import json
import re
from typing import Any, ClassVar, Literal, Optional, List

from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr, ConfigDict, ValidationError, field_validator, model_validator

# -----------------------------------------------------------------------------
# Shared primitives and constants
# -----------------------------------------------------------------------------

# UUID v7 (loose) validator: 8-4-4-4-12, version "7", variant [89ab]
_UUID_V7_RX = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-7[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)

def _is_uuid_v7_like(s: str) -> bool:
    return bool(_UUID_V7_RX.match(s))

# Enumerations taken verbatim from your DTOs (“must not be translated”)
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
    "Physical threats",
    "Natural threats",
    "Compromise of functions or services",
    "Human actions",
    "Technical failures and infrastructure disruptions",
    "Organizational threats",
    "Legal and regulatory threats",
    "Violation of privacy",
    "Other",
]

# -----------------------------------------------------------------------------
# Reusable selectors
# -----------------------------------------------------------------------------

class AssetSelector(BaseModel):
    """
    Either assetId (UUID v7) or assetName must be provided.
    Prefer assetId when available.
    """
    model_config = ConfigDict(extra="forbid")
    assetId: Optional[StrictStr] = Field(None, description="UUID v7 of the asset, if known.")
    assetName: Optional[StrictStr] = Field(None, description="Exact asset name when ID is not yet assigned.")

    @field_validator("assetId")
    @classmethod
    def _check_uuid_v7(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not _is_uuid_v7_like(v):
            raise ValueError("assetId must be UUID v7")
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

# -----------------------------------------------------------------------------
# Asset tools (args) — aligned to Asset DTO
# -----------------------------------------------------------------------------

class AssetListArgs(BaseModel):
    """List assets with optional text filter (applied to name)."""
    model_config = ConfigDict(extra="forbid")
    query: Optional[StrictStr] = Field(None, description="Optional substring filter on asset name.")

    @field_validator("query")
    @classmethod
    def _non_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v2 = v.strip()
        return v2 or None

class AssetCreateArgs(BaseModel):
    """
    Create a new asset.
    - isPrimary defaults to True (primary asset). If set to False (supporting),
      confidentiality/integrity/availability MUST be null.
    """
    model_config = ConfigDict(extra="forbid")

    # Core fields
    name: StrictStr = Field(..., description="Localized asset name.")
    description: Optional[StrictStr] = Field(None, description="Localized asset description.")
    isPrimary: StrictBool = Field(
        True,
        description="Primary (business-critical) vs supporting asset (immutable after creation)."
    )
    assetType: AssetType = Field(..., description="Fixed code list; must be one of predefined values.")

    # CIA (primary assets only)
    confidentiality: Optional[ConfidentialityLevel] = Field(
        None, description="ISO 27001 confidentiality level (primary assets only)."
    )
    integrity: Optional[TriadLevel] = Field(
        None, description="ISO 27001 integrity level (primary assets only)."
    )
    availability: Optional[TriadLevel] = Field(
        None, description="ISO 27001 availability level (primary assets only)."
    )

    # Trim and require non-empty name
    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("name must be a non-empty string")
        return v2

    # Normalize description: trim; convert empty to None
    @field_validator("description")
    @classmethod
    def _desc_trim(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return v.strip() or None

    # Cross-field rule (runs after all fields are parsed):
    # If the asset is supporting (isPrimary == False), all CIA fields must be None.
    @model_validator(mode="after")
    def _cia_rule(self):
        if self.isPrimary is False:
            if any(x is not None for x in (self.confidentiality, self.integrity, self.availability)):
                raise ValueError(
                    "For supporting assets (isPrimary=False), confidentiality/integrity/availability must be null"
                )
        return self


class AssetUpdateMetaArgs(BaseModel):
    """
    Update asset metadata (rename/change description).
    - isPrimary is immutable and not part of this tool.
    - assetType is not changed here either.
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
    def _trim_and_non_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v2 = v.strip()
        if not v2:
            raise ValueError("empty strings are not allowed")
        return v2

class AssetExplainArgs(AssetSelector):
    """Explain an asset (by ID or by name), using domain knowledge/RAG if available."""
    pass

# -----------------------------------------------------------------------------
# Impact tools (args) — aligned to Impact/AssetImpact DTOs
# -----------------------------------------------------------------------------

class ImpactListArgs(BaseModel):
    """List available impact definitions (catalog)."""
    model_config = ConfigDict(extra="forbid")

class ImpactCreateArgs(BaseModel):
     """
     Create a new impact definition in catalog (logical definition via ChangeSet).
     - code: canonical identifier (usually the EN Name used as code in CSVs).
     - name: human label; defaults to `code` if omitted.
     - category: optional category code (must match catalog taxonomy when validated downstream).
     """
     model_config = ConfigDict(extra="forbid")
     code: StrictStr = Field(..., description="Canonical impact code (unique).")
     name: Optional[StrictStr] = Field(None, description="Human-readable name; defaults to code.")
     category: Optional[StrictStr] = Field(None, description="Impact category code (optional).")
     description: Optional[StrictStr] = Field(None, description="Optional description/notes.")
 
     @field_validator("code", "name", "category", "description")
     @classmethod
     def _trim(cls, v: Optional[str]) -> Optional[str]:
         if v is None:
             return v
         v2 = v.strip()
         if not v2:
             # allow empty only when field is optional and not provided
             return None if v is not None else v
         return v2

class ImpactLevelsForAssetArgs(AssetSelector):
    """Return current impact levels assigned to the asset."""
    pass

class ImpactSetLevelArgs(AssetSelector):
    """
    Set impact level (0..4) for a specific impact on an asset.
    Uses impactName (exact) as per DTO.
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

# -----------------------------------------------------------------------------
# Threat tools (args) — aligned to Threat/AssetRisk DTOs
# -----------------------------------------------------------------------------

class ThreatListArgs(BaseModel):
    """List available threats (catalog)."""
    model_config = ConfigDict(extra="forbid")

class ThreatCreateArgs(BaseModel):
     """
     Create a new threat definition in catalog (logical definition via ChangeSet).
     - code: canonical identifier (usually the EN Name used as code in CSVs).
     - name: human label; defaults to `code` if omitted.
     - category: optional category code.
     - flags: optional booleans (intentional/unintentional/environmental).
     """
     model_config = ConfigDict(extra="forbid")
     code: StrictStr = Field(..., description="Canonical threat code (unique).")
     name: Optional[StrictStr] = Field(None, description="Human-readable threat name; defaults to code.")
     category: Optional[StrictStr] = Field(None, description="Threat category code (optional).")
     description: Optional[StrictStr] = Field(None, description="Optional description/notes.")
     intentional: Optional[StrictBool] = Field(None, description="Whether the threat is intentional.")
     unintentional: Optional[StrictBool] = Field(None, description="Whether the threat is unintentional.")
     environmental: Optional[StrictBool] = Field(None, description="Whether the threat is environmental.")
 
     @field_validator("code", "name", "category", "description")
     @classmethod
     def _trim(cls, v: Optional[str]) -> Optional[str]:
         if v is None:
             return v
         v2 = v.strip()
         if not v2:
             return None if v is not None else v
         return v2

class ThreatSetProbabilityArgs(AssetSelector):
    """
    Set probability 0..4 for a threat (identified by threatName) on an asset.
    """
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

# -----------------------------------------------------------------------------
# Controls (catalog) — aligned to PlannedControl/AssetControls DTOs
# Minimal toolset for now: list/explain (planning endpoint may be separate).
# -----------------------------------------------------------------------------

class ControlListArgs(BaseModel):
    """List controls from catalog."""
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

# -----------------------------------------------------------------------------
# ToolSpec & Registry
# -----------------------------------------------------------------------------

class ToolSpec(BaseModel):
    """Canonical tool specification."""
    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field(..., description="Canonical tool name, e.g., 'asset.create'.")
    description: StrictStr = Field(..., description="Short action description for the model.")
    args_model: type[BaseModel] = Field(..., description="Pydantic v2 model class for the arguments.")
    tab: Literal["assets", "impacts", "threats", "controls"] = Field(..., description="UI tab / domain.")
    endpoint: StrictStr = Field(..., description="Graph endpoint path where this tool is executed.")
    op: StrictStr = Field(..., description="Canonical op code (used by router/WS/ChangeSet).")

    def json_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)  # cleaner payloads
        return schema

    def as_openai_function(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }

    def validate_payload(self, payload: dict[str, Any]) -> BaseModel:
        return self.args_model.model_validate(payload)

class ToolRegistry:
    """In-memory registry for all tools across tabs."""
    _tools: ClassVar[List[ToolSpec]] = []

    # --------------------------- Construction -------------------------------

    @classmethod
    def _build_default(cls) -> List[ToolSpec]:
        tools: List[ToolSpec] = []

        # --- Assets ---
        tools.append(ToolSpec(
            name="asset.list",
            description="List assets (optionally filtered by text query).",
            args_model=AssetListArgs,
            tab="assets",
            endpoint="/graph/assets",
            op="asset.list",
        ))
        tools.append(ToolSpec(
            name="asset.create",
            description="Create a new asset, including isPrimary and assetType.",
            args_model=AssetCreateArgs,
            tab="assets",
            endpoint="/graph/assets",
            op="asset.create",
        ))
        tools.append(ToolSpec(
            name="asset.update_meta",
            description="Update asset name/description (isPrimary and assetType are immutable here).",
            args_model=AssetUpdateMetaArgs,
            tab="assets",
            endpoint="/graph/assets",
            op="asset.update_meta",
        ))
        tools.append(ToolSpec(
            name="asset.explain",
            description="Explain an asset (ID or name).",
            args_model=AssetExplainArgs,
            tab="assets",
            endpoint="/graph/assets",
            op="asset.explain",
        ))

        # --- Impacts ---
        tools.append(ToolSpec(
            name="impact.list",
            description="List available impact definitions from catalog.",
            args_model=ImpactListArgs,
            tab="impacts",
            endpoint="/graph/impacts",
            op="impact.list",
        ))
        tools.append(ToolSpec(
           name="impact.create",
           description="Create a new impact definition (catalog) via ChangeSet.",
           args_model=ImpactCreateArgs,
           tab="impacts",
           endpoint="/graph/impacts",
           op="impact.create",
        ))
        tools.append(ToolSpec(
            name="impact.levels.for_asset",
            description="Show current impact levels for a specific asset.",
            args_model=ImpactLevelsForAssetArgs,
            tab="impacts",
            endpoint="/graph/impacts",
            op="impact.levels.for_asset",
        ))
        tools.append(ToolSpec(
            name="impact.set_level",
            description="Set a single impact level (0..4) for an asset (by impactName).",
            args_model=ImpactSetLevelArgs,
            tab="impacts",
            endpoint="/graph/impacts",
            op="impact.set_level",
        ))
        tools.append(ToolSpec(
            name="impact.bulk_set",
            description="Bulk set multiple impact levels (0..4) for an asset.",
            args_model=ImpactBulkSetArgs,
            tab="impacts",
            endpoint="/graph/impacts",
            op="impact.bulk_set",
        ))
        tools.append(ToolSpec(
            name="impact.explain",
            description="Explain an impact concept (by impactName).",
            args_model=ImpactExplainArgs,
            tab="impacts",
            endpoint="/graph/impacts",
            op="impact.explain",
        ))

        # --- Threats ---
        tools.append(ToolSpec(
            name="threat.list",
            description="List available threats from catalog.",
            args_model=ThreatListArgs,
            tab="threats",
            endpoint="/graph/threats",
            op="threat.list", 
        ))
        tools.append(ToolSpec(
           name="threat.create",
           description="Create a new threat definition (catalog) via ChangeSet.",
           args_model=ThreatCreateArgs,
           tab="threats",
           endpoint="/graph/threats",
           op="threat.create",
        ))
        tools.append(ToolSpec(
            name="threat.set_probability",
            description="Set threat probability (0..4) for an asset's threat (by threatName).",
            args_model=ThreatSetProbabilityArgs,
            tab="threats",
            endpoint="/graph/threats",
            op="threat.set_probability",
        ))
        tools.append(ToolSpec(
            name="threat.bulk_set",
            description="Bulk set multiple threat probabilities (0..4) for an asset.",
            args_model=ThreatBulkSetArgs,
            tab="threats",
            endpoint="/graph/threats",
            op="threat.bulk_set",
        ))
        tools.append(ToolSpec(
            name="threat.explain",
            description="Explain a threat concept (by threatName).",
            args_model=ThreatExplainArgs,
            tab="threats",
            endpoint="/graph/threats",
            op="threat.explain",
        ))

        # --- Controls (catalog) ---
        tools.append(ToolSpec(
            name="control.list",
            description="List controls from catalog.",
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
    def as_openai_functions(cls, tab: Optional[Literal["assets", "impacts", "threats", "controls"]] = None) -> list[dict[str, Any]]:
        selected = cls.by_tab(tab) if tab else cls.all()
        return [t.as_openai_function() for t in selected]

    @classmethod
    def as_prompt_json(cls, tab: Optional[Literal["assets", "impacts", "threats", "controls"]] = None) -> str:
        funs = cls.as_openai_functions(tab)
        return json.dumps(funs, separators=(",", ":"))

    # ----------------------- Validation / Routing ----------------------------

    @classmethod
    def validate_call(cls, tool_name: str, payload: dict[str, Any]) -> BaseModel:
        spec = cls.get(tool_name)
        if spec is None:
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
