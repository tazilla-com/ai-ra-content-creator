"""
Asset generation using structured output for CIA classification.
"""

from __future__ import annotations

import logging
from typing import Optional, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from workshops_ai.llm.client import make_llm_client_from_settings
from workshops_ai.settings import get_settings

logger = logging.getLogger(__name__)


# Extended Asset model for generation (includes reasoning field)
class GeneratedAsset(BaseModel):
    """
    Asset model for generation - based on workshops_ai.api.schema.asset.Asset
    Extended with reasoning field for agent explanation.
    """
    id: Optional[UUID] = Field(
        None,
        description="Always null for new assets - backend will generate UUID v7 after save"
    )

    name: str = Field(
        ...,
        description="LOCALIZED asset name in user's language (Slovak/Czech/English based on conversation)"
    )

    description: Optional[str] = Field(
        None,
        description="LOCALIZED asset description in user's language (Slovak/Czech/English based on conversation)"
    )

    isPrimary: bool = Field(
        True,
        description="Primary (business-critical) or supporting asset. For supporting: CIA must be null"
    )

    confidentiality: Optional[Literal["PUBLIC", "INTERNAL", "PROTECTED", "STRICTLY_PROTECTED"]] = Field(
        None,
        description="Confidentiality level - ALWAYS IN ENGLISH, never translate"
    )

    integrity: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = Field(
        None,
        description="Integrity level - ALWAYS IN ENGLISH, never translate"
    )

    availability: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = Field(
        None,
        description="Availability level - ALWAYS IN ENGLISH, never translate"
    )

    assetType: str = Field(
        ...,
        description="Asset type from valid_asset_types list - ALWAYS IN ENGLISH, never translate"
    )

    reasoning: str = Field(
        ...,
        description="LOCALIZED explanation in user's language (Slovak/Czech/English) why this CIA classification"
    )


class CreateAssetsOutput(BaseModel):
    """Output from asset generation."""
    assets: list[GeneratedAsset]
    summary: str = Field(..., description="Brief summary in user's language (Slovak/Czech/English)")


def generate_assets_with_structured_output(
    description: str,
    organization: dict,
    existing_assets: list[dict],
    valid_types: list[str],
    services: list[dict] | None = None,
) -> CreateAssetsOutput:
    """
    Use structured output to generate validated asset output.

    Args:
        description: What assets to create
        organization: Company context
        existing_assets: Current assets (avoid duplicates)
        valid_types: Valid asset type values

    Returns:
        CreateAssetsOutput with generated assets
    """
    settings = get_settings()
    llm = make_llm_client_from_settings(settings)

    llm_with_structure = llm.with_structured_output(CreateAssetsOutput)

    # Build services context if available
    services_text = ""
    if services:
        service_names = [s.get("name", "Unknown") for s in services]
        services_text = (
            "\nBusiness services this organization provides:\n"
            + "\n".join(f"- {name}" for name in service_names)
            + "\n\nUse these services as context when creating assets - assets should support these business services.\n"
        )

    prompt = f"""
Generate assets based on this description:
{description}

Organization context:
- Industry: {organization.get('industry', 'Unknown')}
- Size: {organization.get('size', 'Unknown')}
- Name: {organization.get('name', 'Unknown')}
{services_text}
Valid asset types (use ONLY these - always in English):
{', '.join(valid_types)}

Existing assets (avoid duplicates):
{[a.get('name') for a in existing_assets]}

CRITICAL: Create EXACTLY the assets mentioned in the description — no more, no less.
- Include BOTH primary assets (with CIA) AND supporting assets (with CIA = null) if described
- If description mentions "servers as supporting infrastructure", create them as isPrimary=false
- Do NOT invent extra assets beyond what the description lists

CIA Classification Rules:
- **If description includes specific CIA values** (e.g., "with PROTECTED confidentiality, HIGH integrity"):
  → Use EXACTLY those values, do NOT change them!
- **If description doesn't specify CIA values**:
  → Infer appropriate values from organization context and asset description

LOCALIZATION RULES (CRITICAL):
1. **LOCALIZE these fields** in the same language as the description (Slovak/Czech/English):
   - name: Asset name
   - description: Asset description
   - reasoning: Explanation for CIA classification

2. **NEVER translate these fields** (always English):
   - confidentiality: PUBLIC, INTERNAL, PROTECTED, STRICTLY_PROTECTED
   - integrity: LOW, MEDIUM, HIGH
   - availability: LOW, MEDIUM, HIGH
   - assetType: Must be exactly from valid_asset_types list above

3. **id field**: Always null (backend generates UUID after save)

Requirements:
- Every PRIMARY asset MUST have CIA classification
- For supporting assets (isPrimary=false): set CIA to null
- Include localized reasoning for each classification
- Use appropriate asset type from the list (in English)
"""

    logger.info(
        "Generating assets with structured output",
        extra={
            "description_length": len(description),
            "organization_industry": organization.get("industry"),
            "valid_types_count": len(valid_types),
            "existing_assets_count": len(existing_assets)
        }
    )

    result: CreateAssetsOutput = llm_with_structure.invoke(prompt)

    # Post-processing: enforce CIA rules
    for asset in result.assets:
        if asset.isPrimary:
            # Primary assets MUST have all CIA fields
            missing = []
            if asset.confidentiality is None:
                missing.append("confidentiality")
            if asset.integrity is None:
                missing.append("integrity")
            if asset.availability is None:
                missing.append("availability")
            if missing:
                logger.warning(
                    f"Primary asset '{asset.name}' missing CIA fields: {missing}. "
                    "Setting defaults (INTERNAL/MEDIUM/MEDIUM)."
                )
                if asset.confidentiality is None:
                    asset.confidentiality = "INTERNAL"
                if asset.integrity is None:
                    asset.integrity = "MEDIUM"
                if asset.availability is None:
                    asset.availability = "MEDIUM"
        else:
            # Supporting assets MUST NOT have CIA (strip if LLM set them)
            if asset.confidentiality or asset.integrity or asset.availability:
                logger.warning(
                    f"Supporting asset '{asset.name}' had CIA values set — clearing."
                )
                asset.confidentiality = None
                asset.integrity = None
                asset.availability = None

    logger.info(
        "Assets generated",
        extra={"assets_count": len(result.assets)}
    )

    return result
