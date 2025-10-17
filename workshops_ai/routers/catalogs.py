# workshops_ai/routers/catalogs.py
"""
FastAPI router for catalog endpoints.

Provides read-only access to CSV-based catalogs:
- /api/catalogs/impacts - Impact catalog with categories and weights
- /api/catalogs/threats - Threat catalog with categories and types
- /api/catalogs/controls - Control catalog with categories
- /api/catalogs/controls/coverage - Control-to-Threat coverage mappings
- /api/catalogs/asset-types/threats - Asset type to Threat priors

All endpoints support language parameter (en/sk/cs) for localized names.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict

from workshops_ai.wizard.catalog_loader import (
    load_impact_catalog,
    load_threat_catalog,
    load_control_catalog,
    load_control_threat_coverage,
    load_asset_type_threat_priors,
)
from workshops_ai.api.schema.impact import Impact
from workshops_ai.api.schema.threat import Threat
from workshops_ai.api.schema.control import PlannedControl

logger = logging.getLogger(__name__)

# Router with /api prefix
router = APIRouter(
    prefix="/api/catalogs",
    tags=["catalogs"],
)


# =============================================================================
# Response Models
# =============================================================================

class ImpactCatalogResponse(BaseModel):
    """Response for impact catalog endpoint."""
    model_config = ConfigDict(extra="forbid")

    language: str = Field(..., description="Language code (en/sk/cs)")
    total: int = Field(..., description="Total number of impacts")
    impacts: List[Impact] = Field(..., description="List of impacts with categories and weights")


class ThreatCatalogResponse(BaseModel):
    """Response for threat catalog endpoint."""
    model_config = ConfigDict(extra="forbid")

    language: str = Field(..., description="Language code (en/sk/cs)")
    total: int = Field(..., description="Total number of threats")
    threats: List[Threat] = Field(..., description="List of threats with categories and types")


class ControlCatalogResponse(BaseModel):
    """Response for control catalog endpoint."""
    model_config = ConfigDict(extra="forbid")

    language: str = Field(..., description="Language code (en/sk/cs)")
    total: int = Field(..., description="Total number of controls")
    controls: List[PlannedControl] = Field(
        ...,
        description="List of controls (without costs/threats - populated by AI)"
    )


class ControlThreatCoverage(BaseModel):
    """Control to Threat coverage entry."""
    model_config = ConfigDict(extra="forbid")

    control: str = Field(..., description="Control name")
    threat: str = Field(..., description="Threat name")
    coverage: int = Field(..., ge=0, le=100, description="Coverage percentage (0-100)")


class ControlThreatCoverageResponse(BaseModel):
    """Response for control-threat coverage endpoint."""
    model_config = ConfigDict(extra="forbid")

    total: int = Field(..., description="Total number of control-threat mappings")
    coverage: List[ControlThreatCoverage] = Field(
        ...,
        description="List of control → threat coverage mappings"
    )


class AssetTypeThreatPrior(BaseModel):
    """Asset type to Threat prior mapping."""
    model_config = ConfigDict(extra="forbid")

    asset_type: str = Field(..., description="Asset type")
    threats: List[str] = Field(..., description="List of relevant threat names")


class AssetTypeThreatPriorsResponse(BaseModel):
    """Response for asset type threat priors endpoint."""
    model_config = ConfigDict(extra="forbid")

    total: int = Field(..., description="Total number of asset types")
    priors: List[AssetTypeThreatPrior] = Field(
        ...,
        description="List of asset type → threat priors"
    )


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/impacts",
    response_model=ImpactCatalogResponse,
    summary="Get Impact catalog",
    description="Retrieve the official Impact catalog with categories, weights, and localized names"
)
async def get_impact_catalog(
    language: str = Query(
        default="en",
        regex="^(en|sk|cs)$",
        description="Language code for localized names (en/sk/cs)"
    )
) -> ImpactCatalogResponse:
    """
    Get Impact catalog with localized names and descriptions.

    The catalog includes:
    - Impact name (localized)
    - Description (localized)
    - Weight (importance/severity multiplier)
    - Category (Availability, Clients, Confidentiality, Finance, etc.)

    Args:
        language: Language code (en/sk/cs)

    Returns:
        ImpactCatalogResponse with list of impacts

    Raises:
        HTTPException: If catalog loading fails
    """
    try:
        impacts = load_impact_catalog(language=language)

        logger.info(
            "impact_catalog_retrieved",
            extra={
                "language": language,
                "count": len(impacts),
            }
        )

        return ImpactCatalogResponse(
            language=language,
            total=len(impacts),
            impacts=impacts
        )

    except Exception as e:
        logger.error(
            "impact_catalog_load_failed",
            extra={
                "language": language,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load impact catalog: {str(e)}"
        )


@router.get(
    "/threats",
    response_model=ThreatCatalogResponse,
    summary="Get Threat catalog",
    description="Retrieve the official Threat catalog with categories, types, and localized names"
)
async def get_threat_catalog(
    language: str = Query(
        default="en",
        regex="^(en|sk|cs)$",
        description="Language code for localized names (en/sk/cs)"
    )
) -> ThreatCatalogResponse:
    """
    Get Threat catalog with localized names and descriptions.

    The catalog includes:
    - Threat name (localized)
    - Description (localized)
    - Category (Physical, Natural, Human actions, etc.)
    - Type flags (unintentional, intentional, environmental)

    Args:
        language: Language code (en/sk/cs)

    Returns:
        ThreatCatalogResponse with list of threats

    Raises:
        HTTPException: If catalog loading fails
    """
    try:
        threats = load_threat_catalog(language=language)

        logger.info(
            "threat_catalog_retrieved",
            extra={
                "language": language,
                "count": len(threats),
            }
        )

        return ThreatCatalogResponse(
            language=language,
            total=len(threats),
            threats=threats
        )

    except Exception as e:
        logger.error(
            "threat_catalog_load_failed",
            extra={
                "language": language,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load threat catalog: {str(e)}"
        )


@router.get(
    "/controls",
    response_model=ControlCatalogResponse,
    summary="Get Control catalog",
    description="Retrieve the official Control catalog with categories and localized names"
)
async def get_control_catalog(
    language: str = Query(
        default="en",
        regex="^(en|sk|cs)$",
        description="Language code for localized names (en/sk/cs)"
    )
) -> ControlCatalogResponse:
    """
    Get Control catalog with localized names and descriptions.

    The catalog includes:
    - Control name (localized)
    - Description/notes (localized)
    - Category (ISO 27001 aligned - 17 categories)

    Note: Costs and threat coverage are NOT included in the catalog.
    These are populated by the AI based on asset context.

    Args:
        language: Language code (en/sk/cs)

    Returns:
        ControlCatalogResponse with list of controls

    Raises:
        HTTPException: If catalog loading fails
    """
    try:
        controls = load_control_catalog(language=language)

        logger.info(
            "control_catalog_retrieved",
            extra={
                "language": language,
                "count": len(controls),
            }
        )

        return ControlCatalogResponse(
            language=language,
            total=len(controls),
            controls=controls
        )

    except Exception as e:
        logger.error(
            "control_catalog_load_failed",
            extra={
                "language": language,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load control catalog: {str(e)}"
        )


@router.get(
    "/controls/coverage",
    response_model=ControlThreatCoverageResponse,
    summary="Get Control-to-Threat coverage mappings",
    description="Retrieve control effectiveness against specific threats (0-100% coverage)"
)
async def get_control_threat_coverage(
    control: Optional[str] = Query(
        None,
        description="Filter by specific control name (optional)"
    ),
    threat: Optional[str] = Query(
        None,
        description="Filter by specific threat name (optional)"
    ),
) -> ControlThreatCoverageResponse:
    """
    Get Control → Threat coverage mappings.

    Each mapping indicates how effective a control is against a specific threat
    (0-100% coverage). These are baseline values from the catalog; the AI may
    adjust them based on asset context.

    Args:
        control: Optional control name filter (exact match)
        threat: Optional threat name filter (exact match)

    Returns:
        ControlThreatCoverageResponse with filtered coverage mappings

    Raises:
        HTTPException: If catalog loading fails
    """
    try:
        # Load full coverage mapping
        coverage_dict = load_control_threat_coverage()

        # Build flat list
        coverage_list = []
        for ctrl_name, threat_dict in coverage_dict.items():
            # Filter by control if specified
            if control and ctrl_name != control:
                continue

            for threat_name, cov_value in threat_dict.items():
                # Filter by threat if specified
                if threat and threat_name != threat:
                    continue

                coverage_list.append(ControlThreatCoverage(
                    control=ctrl_name,
                    threat=threat_name,
                    coverage=cov_value
                ))

        logger.info(
            "control_threat_coverage_retrieved",
            extra={
                "total_mappings": len(coverage_list),
                "control_filter": control,
                "threat_filter": threat,
            }
        )

        return ControlThreatCoverageResponse(
            total=len(coverage_list),
            coverage=coverage_list
        )

    except Exception as e:
        logger.error(
            "control_threat_coverage_load_failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load control-threat coverage: {str(e)}"
        )


@router.get(
    "/asset-types/threats",
    response_model=AssetTypeThreatPriorsResponse,
    summary="Get Asset Type to Threat priors",
    description="Retrieve relevant threats for each asset type (used for smart threat selection)"
)
async def get_asset_type_threat_priors(
    asset_type: Optional[str] = Query(
        None,
        description="Filter by specific asset type (optional)"
    )
) -> AssetTypeThreatPriorsResponse:
    """
    Get Asset Type → Threat prior mappings.

    These priors indicate which threats are most relevant for each asset type,
    used by the risk generator to pre-filter threats before LLM evaluation.

    Args:
        asset_type: Optional asset type filter (exact match)

    Returns:
        AssetTypeThreatPriorsResponse with filtered priors

    Raises:
        HTTPException: If catalog loading fails
    """
    try:
        # Load priors
        priors_dict = load_asset_type_threat_priors()

        # Build list
        priors_list = []
        for at, threats in priors_dict.items():
            # Filter by asset type if specified
            if asset_type and at != asset_type:
                continue

            priors_list.append(AssetTypeThreatPrior(
                asset_type=at,
                threats=threats
            ))

        logger.info(
            "asset_type_threat_priors_retrieved",
            extra={
                "total_asset_types": len(priors_list),
                "asset_type_filter": asset_type,
            }
        )

        return AssetTypeThreatPriorsResponse(
            total=len(priors_list),
            priors=priors_list
        )

    except Exception as e:
        logger.error(
            "asset_type_threat_priors_load_failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load asset type threat priors: {str(e)}"
        )
