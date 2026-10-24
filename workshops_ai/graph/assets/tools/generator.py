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
    model: str | None = None,
    user_message: str | None = None,
) -> tuple[CreateAssetsOutput, list[GeneratedAsset]]:
    """
    Use structured output to generate validated asset output.

    Args:
        description: What assets to create
        organization: Company context (api.schema Organization fields)
        existing_assets: Current assets (avoid duplicates)
        valid_types: Valid asset type values
        services: Business services from the wizard
        model: The model of the request; the configured default when None
        user_message: The user's message, whose language the assets take

    Returns:
        CreateAssetsOutput with the assets whose type is in the catalog, and the assets left out for their type
    """
    settings = get_settings()
    llm = make_llm_client_from_settings(settings, override_model=model)

    llm_with_structure = llm.with_structured_output(CreateAssetsOutput)

    # Build services context if available
    services_text = ""
    if services:
        service_names = [s.get("name", "Unknown") for s in services]
        services_text = (
            "\nBusiness services this organization provides:\n"
            + "\n".join(f"- {name}" for name in service_names)
            + "\n"
        )

    # The fields of api.schema Organization; the payload carries no industry or size
    organization = organization or {}
    org_lines = [
        f"- {label}: {organization[key]}"
        for key, label in (("organizationName", "Name"), ("legalForm", "Legal form"),
                           ("country", "Country"), ("numberOfEmployees", "Employees"))
        if organization.get(key)
    ]
    organization_text = ("\nOrganization:\n" + "\n".join(org_lines) + "\n") if org_lines else ""

    existing_names = [a.get("name") for a in existing_assets if a.get("name")]
    # The agent may write the description in another language than the user's; the user's message decides
    language_hint = f': "{user_message[:300]}"' if user_message else " (the language of the description)"

    # One type per line: a type name may contain a comma ("Equipment, machines")
    prompt = f"""
Create exactly the assets described below, no more and no fewer.

Description:
{description}
{organization_text}{services_text}
Asset types in Tazilla's catalogue; give each asset exactly one of these names:
{chr(10).join(f"- {t}" for t in valid_types)}

Existing assets, which must not be created again:
{chr(10).join(f"- {n}" for n in existing_names) or "- none"}

Rules:
- Primary assets (isPrimary true) are information together with the business activity that uses it; each has
  confidentiality, integrity and availability. Supporting assets (isPrimary false) are what primary assets run on
  or depend on, such as systems, applications, servers, networks, devices, premises or staff; their CIA values
  are null.
- Use the CIA values the description gives exactly. Infer the missing ones from what the asset holds and what
  its loss would cause.
- Write name, description, reasoning and summary in the language of the user's message{language_hint}. Keep the
  CIA values and the asset type names exactly as listed.
- id is null.
"""

    logger.info(
        "Generating assets with structured output",
        extra={
            "description_length": len(description),
            "organization_name": organization.get("organizationName"),
            "valid_types_count": len(valid_types),
            "existing_assets_count": len(existing_assets)
        }
    )

    result: CreateAssetsOutput = llm_with_structure.invoke(prompt)

    # The type must be a catalog name: matched without regard to case and surrounding spaces, and given the
    # catalog's own spelling; an asset with another type is left out and reported to the agent
    catalog_types = {t.strip().casefold(): t for t in valid_types}
    accepted: list[GeneratedAsset] = []
    rejected: list[GeneratedAsset] = []
    for asset in result.assets:
        asset_type = catalog_types.get((asset.assetType or "").strip().casefold())
        if asset_type is None:
            logger.warning(f"Asset '{asset.name}' has type '{asset.assetType}', not in the catalog; left out")
            rejected.append(asset)
            continue
        asset.assetType = asset_type
        accepted.append(asset)
    result.assets = accepted

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
        extra={"assets_count": len(result.assets), "rejected_count": len(rejected)}
    )

    return result, rejected
