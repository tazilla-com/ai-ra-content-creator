# workshops_ai/routers/catalogs.py
"""
FastAPI router for catalog endpoints.

Provider Pattern:
Uses configurable catalog provider (CSV or Tazilla API) based on CATALOG__MODE:
- mode=file: CSVCatalogProvider (dev/mockup, ignores session_id)
- mode=http: TazillaAPICatalogProvider (production, requires session_id)

Endpoints:
- GET /api/catalogs/impacts - Impact catalog with categories
- GET /api/catalogs/threats - Threat catalog with categories, types, and asset type associations
- GET /api/catalogs/controls - Control catalog with categories
- GET /api/catalogs/asset-types - Asset type catalog with dependencies
- GET /api/catalogs/impact-categories - Impact category catalog
- GET /api/catalogs/threat-categories - Threat category catalog
- GET /api/catalogs/control-categories - Control category catalog

All endpoints support:
- language parameter (en/sk/cs) for localized names
- X-Session-Id header for client-specific catalogs (required for Tazilla API mode)
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException, status, Header
from pydantic import BaseModel, Field, ConfigDict

from workshops_ai.wizard.catalog_provider import get_catalog_provider
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
    """Response for impact catalog endpoint (localized by Tazilla backend)."""
    model_config = ConfigDict(extra="forbid")

    total: int = Field(..., description="Total number of impacts")
    impacts: List[Impact] = Field(..., description="List of impacts with categories")


class ThreatCatalogResponse(BaseModel):
    """Response for threat catalog endpoint (localized by Tazilla backend)."""
    model_config = ConfigDict(extra="forbid")

    total: int = Field(..., description="Total number of threats")
    threats: List[Threat] = Field(..., description="List of threats with categories, types, and asset type associations")


class ControlCatalogResponse(BaseModel):
    """Response for control catalog endpoint (localized by Tazilla backend)."""
    model_config = ConfigDict(extra="forbid")

    total: int = Field(..., description="Total number of controls")
    controls: List[PlannedControl] = Field(
        ...,
        description="List of controls (without costs/threats - populated by AI)"
    )


class AssetTypeItem(BaseModel):
    """Single asset type item."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Asset type name (localized)")
    description: str = Field(..., description="Asset type notes/description (localized)")
    isDependent: bool = Field(..., description="Whether this asset type is dependent on other assets")


class AssetTypeCatalogResponse(BaseModel):
    """Response for asset type catalog endpoint (localized by Tazilla backend)."""
    model_config = ConfigDict(extra="forbid")

    total: int = Field(..., description="Total number of asset types")
    assetTypes: List[AssetTypeItem] = Field(..., description="List of asset types")


class CategoryItem(BaseModel):
    """Single category item."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Category name (localized by backend)")
    description: str = Field(..., description="Category description/notes (localized by backend)")


class CategoryCatalogResponse(BaseModel):
    """Response for category catalog endpoints (localized by Tazilla backend)."""
    model_config = ConfigDict(extra="forbid")

    total: int = Field(..., description="Total number of categories")
    categories: List[CategoryItem] = Field(..., description="List of categories")


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/impacts",
    response_model=ImpactCatalogResponse,
    summary="Get Impact catalog",
    description="Retrieve the official Impact catalog with categories and names (localized by Tazilla backend)"
)
async def get_impact_catalog(
    x_session_id: Optional[str] = Header(
        None,
        alias="X-Session-Id",
        description="Session ID for client-specific catalogs (optional for CSV provider)"
    )
) -> ImpactCatalogResponse:
    """
    Get Impact catalog (already localized by Tazilla backend).

    The catalog includes:
    - Impact name (localized by backend)
    - Description (localized by backend)
    - Category (Availability, Clients, Confidentiality, Finance, etc.)

    Args:
        x_session_id: Session ID for client-specific catalogs

    Returns:
        ImpactCatalogResponse with list of impacts

    Raises:
        HTTPException: If catalog loading fails
    """
    try:
        provider = get_catalog_provider()
        impacts = await provider.load_impact_catalog(
            session_id=x_session_id
        )

        logger.info(
            "impact_catalog_retrieved",
            extra={
                "session_id": x_session_id,
                "count": len(impacts),
            }
        )

        return ImpactCatalogResponse(
            total=len(impacts),
            impacts=impacts
        )

    except Exception as e:
        logger.error(
            "impact_catalog_load_failed",
            extra={
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
    description="Retrieve the official Threat catalog with categories, types, asset type associations, and names (localized by Tazilla backend)"
)
async def get_threat_catalog(
    x_session_id: Optional[str] = Header(
        None,
        alias="X-Session-Id",
        description="Session ID for client-specific catalogs (optional for CSV provider)"
    )
) -> ThreatCatalogResponse:
    """
    Get Threat catalog (already localized by Tazilla backend).

    The catalog includes:
    - Threat name (localized by backend)
    - Description (localized by backend)
    - Category (Physical, Natural, Human actions, etc.)
    - Type flags (unintentional, intentional, environmental)
    - Asset types (list of asset type names this threat is relevant for)

    Args:
        x_session_id: Session ID for client-specific catalogs

    Returns:
        ThreatCatalogResponse with list of threats

    Raises:
        HTTPException: If catalog loading fails
    """
    try:
        provider = get_catalog_provider()
        threats = await provider.load_threat_catalog(
            session_id=x_session_id
        )

        logger.info(
            "threat_catalog_retrieved",
            extra={
                "session_id": x_session_id,
                "count": len(threats),
            }
        )

        return ThreatCatalogResponse(
            total=len(threats),
            threats=threats
        )

    except Exception as e:
        logger.error(
            "threat_catalog_load_failed",
            extra={
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
    description="Retrieve the official Control catalog with categories and names (localized by Tazilla backend)"
)
async def get_control_catalog(
    x_session_id: Optional[str] = Header(
        None,
        alias="X-Session-Id",
        description="Session ID for client-specific catalogs (optional for CSV provider)"
    )
) -> ControlCatalogResponse:
    """
    Get Control catalog (already localized by Tazilla backend).

    The catalog includes:
    - Control name (localized by backend)
    - Description/notes (localized by backend)
    - Category (ISO 27001 aligned - 17 categories)

    Note: Costs and threat coverage are NOT included in the catalog.
    These are populated by the AI based on asset context.

    Args:
        x_session_id: Session ID for client-specific catalogs

    Returns:
        ControlCatalogResponse with list of controls

    Raises:
        HTTPException: If catalog loading fails
    """
    try:
        provider = get_catalog_provider()
        controls = await provider.load_control_catalog(
            session_id=x_session_id
        )

        logger.info(
            "control_catalog_retrieved",
            extra={
                "session_id": x_session_id,
                "count": len(controls),
            }
        )

        return ControlCatalogResponse(
            total=len(controls),
            controls=controls
        )

    except Exception as e:
        logger.error(
            "control_catalog_load_failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load control catalog: {str(e)}"
        )


@router.get(
    "/asset-types",
    response_model=AssetTypeCatalogResponse,
    summary="Get Asset Type catalog",
    description="Retrieve the official Asset Type catalog (localized by Tazilla backend)"
)
async def get_asset_type_catalog(
    x_session_id: Optional[str] = Header(
        None,
        alias="X-Session-Id",
        description="Session ID for client-specific catalogs (optional for CSV provider)"
    )
) -> AssetTypeCatalogResponse:
    """
    Get Asset Type catalog (already localized by Tazilla backend).

    The catalog includes:
    - Asset type name (localized by backend)
    - Description/notes (localized by backend)
    - isDependent flag (whether asset depends on other assets)

    Args:
        x_session_id: Session ID for client-specific catalogs

    Returns:
        AssetTypeCatalogResponse with list of asset types

    Raises:
        HTTPException: If catalog loading fails
    """
    try:
        provider = get_catalog_provider()
        asset_types_data = await provider.load_asset_type_catalog(
            session_id=x_session_id
        )

        # Convert dict format to Pydantic models
        asset_types = [
            AssetTypeItem(**item)
            for item in asset_types_data
        ]

        logger.info(
            "asset_type_catalog_retrieved",
            extra={
                "session_id": x_session_id,
                "count": len(asset_types),
            }
        )

        return AssetTypeCatalogResponse(
            total=len(asset_types),
            assetTypes=asset_types
        )

    except Exception as e:
        logger.error(
            "asset_type_catalog_load_failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load asset type catalog: {str(e)}"
        )


@router.get(
    "/impact-categories",
    response_model=CategoryCatalogResponse,
    summary="Get Impact Categories catalog",
    description="Retrieve the official Impact Categories catalog with localized names"
)
async def get_impact_category_catalog(
    x_session_id: Optional[str] = Header(
        None,
        alias="X-Session-Id",
        description="Session ID for client-specific catalogs (optional for CSV provider)"
    )
) -> CategoryCatalogResponse:
    """
    Get Impact Categories catalog (already localized by Tazilla backend).

    Args:
        x_session_id: Session ID for client-specific catalogs

    Returns:
        CategoryCatalogResponse with list of impact categories

    Raises:
        HTTPException: If catalog loading fails
    """
    try:
        provider = get_catalog_provider()
        categories_data = await provider.load_impact_category_catalog(
            session_id=x_session_id
        )

        categories = [CategoryItem(**item) for item in categories_data]

        logger.info(
            "impact_category_catalog_retrieved",
            extra={
                "session_id": x_session_id,
                "count": len(categories),
            }
        )

        return CategoryCatalogResponse(
            total=len(categories),
            categories=categories
        )

    except Exception as e:
        logger.error(
            "impact_category_catalog_load_failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load impact category catalog: {str(e)}"
        )


@router.get(
    "/threat-categories",
    response_model=CategoryCatalogResponse,
    summary="Get Threat Categories catalog",
    description="Retrieve the official Threat Categories catalog with localized names"
)
async def get_threat_category_catalog(
    x_session_id: Optional[str] = Header(
        None,
        alias="X-Session-Id",
        description="Session ID for client-specific catalogs (optional for CSV provider)"
    )
) -> CategoryCatalogResponse:
    """
    Get Threat Categories catalog (already localized by Tazilla backend).

    Args:
        x_session_id: Session ID for client-specific catalogs

    Returns:
        CategoryCatalogResponse with list of threat categories

    Raises:
        HTTPException: If catalog loading fails
    """
    try:
        provider = get_catalog_provider()
        categories_data = await provider.load_threat_category_catalog(
            session_id=x_session_id
        )

        categories = [CategoryItem(**item) for item in categories_data]

        logger.info(
            "threat_category_catalog_retrieved",
            extra={
                "session_id": x_session_id,
                "count": len(categories),
            }
        )

        return CategoryCatalogResponse(
            total=len(categories),
            categories=categories
        )

    except Exception as e:
        logger.error(
            "threat_category_catalog_load_failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load threat category catalog: {str(e)}"
        )


@router.get(
    "/control-categories",
    response_model=CategoryCatalogResponse,
    summary="Get Control Categories catalog",
    description="Retrieve the official Control Categories catalog with localized names"
)
async def get_control_category_catalog(
    x_session_id: Optional[str] = Header(
        None,
        alias="X-Session-Id",
        description="Session ID for client-specific catalogs (optional for CSV provider)"
    )
) -> CategoryCatalogResponse:
    """
    Get Control Categories catalog (already localized by Tazilla backend).

    Args:
        x_session_id: Session ID for client-specific catalogs

    Returns:
        CategoryCatalogResponse with list of control categories

    Raises:
        HTTPException: If catalog loading fails
    """
    try:
        provider = get_catalog_provider()
        categories_data = await provider.load_control_category_catalog(
            session_id=x_session_id
        )

        categories = [CategoryItem(**item) for item in categories_data]

        logger.info(
            "control_category_catalog_retrieved",
            extra={
                "session_id": x_session_id,
                "count": len(categories),
            }
        )

        return CategoryCatalogResponse(
            total=len(categories),
            categories=categories
        )

    except Exception as e:
        logger.error(
            "control_category_catalog_load_failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load control category catalog: {str(e)}"
        )


@router.get(
    "/sectors",
    response_model=CategoryCatalogResponse,
    summary="Get Sectors catalog",
    description="Retrieve the official Sectors catalog with localized names"
)
async def get_sector_catalog(
    x_session_id: Optional[str] = Header(
        None,
        alias="X-Session-Id",
        description="Session ID for client-specific catalogs (optional for CSV provider)"
    )
) -> CategoryCatalogResponse:
    """
    Get Sectors catalog (already localized by Tazilla backend).

    Args:
        x_session_id: Session ID for client-specific catalogs

    Returns:
        CategoryCatalogResponse with list of sectors

    Raises:
        HTTPException: If catalog loading fails
    """
    try:
        provider = get_catalog_provider()
        sectors_data = await provider.load_sector_catalog(
            session_id=x_session_id
        )

        sectors = [CategoryItem(**item) for item in sectors_data]

        logger.info(
            "sector_catalog_retrieved",
            extra={
                "session_id": x_session_id,
                "count": len(sectors),
            }
        )

        return CategoryCatalogResponse(
            total=len(sectors),
            categories=sectors
        )

    except Exception as e:
        logger.error(
            "sector_catalog_load_failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load sector catalog: {str(e)}"
        )
