"""
Asset creation, update, and listing tools.
"""

from __future__ import annotations

from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field


class CreateAssetsInput(BaseModel):
    """Input for create_assets tool."""
    description: str = Field(
        ...,
        description=(
            "Detailed description of what assets to create. "
            "Include context that helps determine CIA classification. "
            "Can describe single asset or multiple assets."
        ),
        examples=[
            "IT infrastructure for hospital: EHR system for patient records (contains sensitive health data), patient portal for appointments, PACS for medical imaging",
            "E-commerce platform: payment gateway processing credit cards, inventory management, customer database with personal info",
            "Single database server for storing internal company documents, accessed only by employees"
        ]
    )


class UpdateAssetsInput(BaseModel):
    """Input for update_assets tool."""
    asset_names: list[str] = Field(
        ...,
        description="Names of assets to update. Must match existing asset names exactly.",
        min_length=1,
        examples=[
            ["Asset Name 1"],
            ["Asset Name 1", "Asset Name 2"],
            ["Asset Name 1", "Asset Name 2", "Asset Name 3"]
        ]
    )
    updates: dict = Field(
        ...,
        description=(
            "Fields to change. Can update: name, description, assetType, "
            "confidentiality, integrity, availability, isPrimary. "
            "Cannot update: id"
        ),
        examples=[
            {"confidentiality": "HIGH"},
            {"description": "Updated description", "availability": "HIGH"},
            {"confidentiality": "PROTECTED", "integrity": "HIGH", "availability": "MEDIUM"}
        ]
    )


VALID_UPDATE_FIELDS = {
    "name", "description", "assetType",
    "confidentiality", "integrity", "availability",
    "isPrimary"
}


@tool
def list_assets(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]
) -> Command:
    """
    List all current assets in the system.

    CRITICAL: ALWAYS use this tool when user asks about their assets!

    USE THIS TOOL WHEN:
    - User asks: "aké mám assety?", "what assets do I have?", "show my assets"
    - User asks: "koľko assetov mám?", "how many assets?"
    - User wants to see current state before updating
    - You need to verify asset names before calling update_assets

    NEVER guess or make up asset names! Always call this tool to get facts.

    Returns:
    - List of all assets with their properties (name, description, CIA levels, etc)
    - Empty list if no assets exist yet
    """
    input_assets = state.get("input_assets", [])

    if not input_assets:
        content = "No assets found in the system. You can create new assets if needed."
        return Command(
            update={
                "tools_used": state.get("tools_used", []) + ["list_assets"],
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
            }
        )

    # Format assets for display
    import json
    assets_summary = []
    for i, asset in enumerate(input_assets, 1):
        name = asset.get("name", "Unnamed")
        desc = asset.get("description", "No description")
        conf = asset.get("confidentiality", "Not set")
        integ = asset.get("integrity", "Not set")
        avail = asset.get("availability", "Not set")
        asset_type = asset.get("assetType", "Not set")
        is_primary = asset.get("isPrimary", False)

        assets_summary.append(
            f"{i}. **{name}**\n"
            f"   - Description: {desc}\n"
            f"   - Type: {asset_type}\n"
            f"   - Confidentiality: {conf}\n"
            f"   - Integrity: {integ}\n"
            f"   - Availability: {avail}\n"
            f"   - Primary: {is_primary}"
        )

    content = f"Found {len(input_assets)} asset(s):\n\n" + "\n\n".join(assets_summary)

    return Command(
        update={
            "tools_used": state.get("tools_used", []) + ["list_assets"],
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
        }
    )


@tool
def create_assets(
    description: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]
) -> Command:
    """
    Create one or more assets with FULL CIA classification.

    WHEN TO USE THIS TOOL:
    - User confirms your proposed assets → create them as proposed
    - User modifies your proposals BEFORE creation → create with modifications
    - DO NOT use update_assets on proposals that don't exist yet!

    Example: You propose A, B, C. User says "create A and B, change B integrity to HIGH, skip C"
    → Call create_assets ONCE with A (as proposed) and B (with HIGH integrity)
    → Do NOT call update_assets (B doesn't exist yet to update!)

    CRITICAL REQUIREMENTS:
    - PRIMARY assets (databases, applications, data) MUST have CIA levels
    - Supporting assets (servers, network, locations, staff) must NOT have CIA (leave as None)
    - Asset type MUST be from valid_asset_types in your state

    BEFORE CALLING THIS TOOL, ensure you have context for CIA classification:
    - Check organization info in state (industry, size helps determine CIA)
    - If industry-specific: call lookup_knowledge first for best practices
    - If company-specific: call scrape_company_info first
    - If still unclear: ASK the user before creating!

    QUESTIONS TO ASK USER IF CONTEXT IS MISSING:
    - "Aké dáta bude tento systém obsahovať?" (What data will it store?)
    - "Je tento systém kritický pre dennú prevádzku?" (Is it critical for operations?)
    - "Kto bude mať prístup k týmto dátam?" (Who will access this data?)
    - "Čo sa stane ak systém nebude dostupný hodinu/deň?" (Impact of downtime?)

    BATCH CAPABILITY:
    - Describe multiple assets in one call for efficiency
    - "IT infrastructure: servers, databases, and applications for hospital"
    - Tool will generate appropriate assets with individual CIA levels

    HOW TO WRITE GOOD DESCRIPTIONS:
    - Correct: "Database storing customer personal data and payment info" → Clear CIA context
    - Correct: "Internal wiki for company policies, non-sensitive" → Clear LOW sensitivity
    - Wrong: "Database" → Too vague, unclear CIA
    - Wrong: "Server" → No context for classification

    CIA QUICK REFERENCE:
    - Confidentiality: PUBLIC < INTERNAL < PROTECTED < STRICTLY_PROTECTED
    - Integrity: LOW < MEDIUM < HIGH (impact of data corruption)
    - Availability: LOW < MEDIUM < HIGH (acceptable downtime)

    Returns Command that updates state with generated assets.
    """
    from workshops_ai.graph.assets.tools.generator import generate_assets_with_structured_output

    organization = state.get("organization", {})
    existing_assets = state.get("input_assets", [])
    valid_types = state.get("valid_asset_types", [])

    if not valid_types:
        return Command(
            update={
                "error": "No valid asset types available. Cannot create assets."
            }
        )

    result = generate_assets_with_structured_output(
        description=description,
        organization=organization,
        existing_assets=existing_assets,
        valid_types=valid_types,
    )

    assets_dicts = [asset.model_dump() for asset in result.assets]

    # Return Command with ToolMessage (required for LangGraph)
    asset_names = [a["name"] for a in assets_dicts]
    return Command(
        update={
            "generated_assets": assets_dicts,
            "tools_used": state.get("tools_used", []) + ["create_assets"],
            "messages": [
                ToolMessage(
                    content=f"Created {len(assets_dicts)} assets: {', '.join(asset_names)}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
def update_assets(
    asset_names: list[str],
    name: str | None = None,
    description: str | None = None,
    confidentiality: str | None = None,
    integrity: str | None = None,
    availability: str | None = None,
    assetType: str | None = None,
    isPrimary: bool | None = None,
    *,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]
) -> Command:
    """
    Update one or more existing assets.

    IMPORTANT: You have full access to input_assets in your state - check there for current values!

    USE THIS TOOL WHEN:
    - User wants to change CIA levels: "nastav [Asset Name] na HIGH availability"
    - User wants to rename asset: "premenuj [Asset Name] na [New Name]"
    - User wants to change description or type
    - User wants to batch update: "všetky vybrané assety na PROTECTED"

    PARTIAL UPDATES - CRITICAL:
    - Only provide parameters you want to CHANGE
    - Other fields remain unchanged automatically
    - Example: update_assets(["Asset Name"], integrity="HIGH")
      → Only integrity changes, confidentiality and availability stay the same
    - DO NOT ask user for unchanged values - just update what they requested!

    VALID UPDATE PARAMETERS:
    - name: Asset name (string)
    - description: Asset description (string)
    - assetType: Must be from valid_asset_types in state
    - confidentiality: PUBLIC | INTERNAL | PROTECTED | STRICTLY_PROTECTED
    - integrity: LOW | MEDIUM | HIGH
    - availability: LOW | MEDIUM | HIGH
    - isPrimary: true | false

    CANNOT UPDATE:
    - id: Asset ID is immutable

    BATCH UPDATES:
    - Pass multiple asset names in asset_names list
    - Same updates applied to ALL specified assets
    - Example: update_assets(["Asset Name 1", "Asset Name 2"], availability="HIGH")

    ASSET NAME MATCHING:
    - Asset names must match EXACTLY (case-sensitive)
    - Check input_assets in state for correct names
    - If asset not found, error is returned

    PARTIAL SUCCESS:
    - If some assets found and some not, updates found ones
    - Reports which assets were not found

    Returns Command that updates state with modified assets.
    """
    # Build updates dict from non-None parameters
    updates = {}
    if name is not None:
        updates["name"] = name
    if description is not None:
        updates["description"] = description
    if confidentiality is not None:
        updates["confidentiality"] = confidentiality
    if integrity is not None:
        updates["integrity"] = integrity
    if availability is not None:
        updates["availability"] = availability
    if assetType is not None:
        updates["assetType"] = assetType
    if isPrimary is not None:
        updates["isPrimary"] = isPrimary

    if not updates:
        error_msg = "No fields to update. Provide at least one field: name, description, confidentiality, integrity, availability, assetType, isPrimary"
        return Command(
            update={
                "error": error_msg,
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]
            }
        )

    input_assets = state.get("input_assets", [])
    valid_types = state.get("valid_asset_types", [])

    invalid_fields = set(updates.keys()) - VALID_UPDATE_FIELDS
    if invalid_fields:
        error_msg = f"Invalid update fields: {invalid_fields}. Valid: {VALID_UPDATE_FIELDS}"
        return Command(
            update={
                "error": error_msg,
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]
            }
        )

    if "assetType" in updates and updates["assetType"] not in valid_types:
        error_msg = f"Invalid assetType: {updates['assetType']}. Must be one of: {valid_types}"
        return Command(
            update={
                "error": error_msg,
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]
            }
        )

    cia_validation = {
        "confidentiality": {"PUBLIC", "INTERNAL", "PROTECTED", "STRICTLY_PROTECTED"},
        "integrity": {"LOW", "MEDIUM", "HIGH"},
        "availability": {"LOW", "MEDIUM", "HIGH"}
    }
    for field, valid_values in cia_validation.items():
        if field in updates and updates[field] is not None and updates[field] not in valid_values:
            error_msg = f"Invalid {field}: {updates[field]}. Must be one of: {valid_values} or None (for supporting assets)"
            return Command(
                update={
                    "error": error_msg,
                    "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]
                }
            )

    updated_assets = []
    not_found = []

    for asset_name in asset_names:
        asset = next((a for a in input_assets if a.get("name") == asset_name), None)

        if not asset:
            not_found.append(asset_name)
            continue

        updated_asset = {**asset, **updates}
        updated_assets.append(updated_asset)

    if not_found and not updated_assets:
        error_msg = f"No assets found with names: {not_found}. Check input_assets for correct names."
        return Command(
            update={
                "error": error_msg,
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]
            }
        )

    # Build success message
    success_msg = f"Updated {len(updated_assets)} assets"
    if not_found:
        success_msg += f". Warning: Some assets not found: {not_found}"

    return Command(
        update={
            "generated_assets": updated_assets,
            "tools_used": state.get("tools_used", []) + ["update_assets"],
            "messages": [
                ToolMessage(
                    content=success_msg,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )
