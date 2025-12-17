# workshops_ai/wizard/catalog_provider.py
"""
Catalog provider pattern for switching between CSV and Tazilla API backends.

Architecture:
- CatalogProvider: Abstract interface
- CSVCatalogProvider: Loads from local CSV files (dev/mockup)
- TazillaAPICatalogProvider: Fetches from Tazilla API (production)
- get_catalog_provider(): Factory function based on settings
"""
from __future__ import annotations

import httpx
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from fastapi import HTTPException, status

from workshops_ai.api.schema.impact import Impact
from workshops_ai.api.schema.threat import Threat
from workshops_ai.api.schema.control import PlannedControl

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Provider Interface
# =============================================================================

class CatalogProvider(ABC):
    """Abstract base class for catalog providers."""

    @abstractmethod
    async def load_impact_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Impact]:
        """Load impact catalog (localized by Tazilla backend)."""
        pass

    @abstractmethod
    async def load_threat_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Threat]:
        """Load threat catalog (localized by Tazilla backend)."""
        pass

    @abstractmethod
    async def load_control_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[PlannedControl]:
        """Load control catalog (localized by Tazilla backend)."""
        pass

    @abstractmethod
    async def load_asset_type_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Load asset type catalog (localized by Tazilla backend)."""
        pass

    @abstractmethod
    async def load_impact_category_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Load impact category catalog (localized by Tazilla backend)."""
        pass

    @abstractmethod
    async def load_threat_category_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Load threat category catalog (localized by Tazilla backend)."""
        pass

    @abstractmethod
    async def load_control_category_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Load control category catalog (localized by Tazilla backend)."""
        pass

    @abstractmethod
    async def load_sector_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Load sector catalog with localization."""
        pass


# =============================================================================
# CSV Provider (Dev/Mockup)
# =============================================================================

class CSVCatalogProvider(CatalogProvider):
    """
    Provider that loads catalogs from local CSV files.

    Ignores session_id - all clients see the same mockup data.
    Uses English ("en") as the fixed language for CSV files.
    """

    async def load_impact_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Impact]:
        """Load impact catalog from CSV file (English only)."""
        from workshops_ai.wizard.catalog_loader import load_impact_catalog as _load

        # CSV provider ignores session_id and uses fixed English language
        return _load(language="en", session_id=None)

    async def load_threat_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Threat]:
        """Load threat catalog from CSV file (English only)."""
        from workshops_ai.wizard.catalog_loader import load_threat_catalog as _load

        return _load(language="en", session_id=None)

    async def load_control_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[PlannedControl]:
        """Load control catalog from CSV file (English only)."""
        from workshops_ai.wizard.catalog_loader import load_control_catalog as _load

        return _load(language="en", session_id=None)

    async def load_asset_type_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Load asset type catalog from CSV file (English only)."""
        from workshops_ai.wizard.catalog_loader import load_asset_type_catalog as _load

        return _load(language="en", session_id=None)

    async def load_impact_category_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Load impact category catalog from CSV file (English only)."""
        from workshops_ai.wizard.catalog_loader import load_impact_category_catalog as _load

        return _load(language="en", session_id=None)

    async def load_threat_category_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Load threat category catalog from CSV file (English only)."""
        from workshops_ai.wizard.catalog_loader import load_threat_category_catalog as _load

        return _load(language="en", session_id=None)

    async def load_control_category_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Load control category catalog from CSV file (English only)."""
        from workshops_ai.wizard.catalog_loader import load_control_category_catalog as _load

        return _load(language="en", session_id=None)

    async def load_sector_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Load sector catalog from CSV file (English only)."""
        from workshops_ai.wizard.catalog_loader import load_sector_catalog as _load

        return _load(language="en", session_id=None)


# =============================================================================
# Tazilla API Provider (Production)
# =============================================================================

class TazillaAPICatalogProvider(CatalogProvider):
    """
    Provider that fetches catalogs from Tazilla API.

    Sends session_id to get client-specific customized catalogs.
    """

    def __init__(
        self,
        endpoint: str,
        auth_token: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize Tazilla API provider.

        Args:
            endpoint: Base URL for Tazilla API
            auth_token: Bearer token for authentication (optional for cluster-internal calls)
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout

        # Build headers - only add Authorization if token is provided
        headers = {
            "Content-Type": "application/json",
        }
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        # Create async HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=headers
        )

        logger.info(
            "tazilla_catalog_provider_initialized",
            extra={
                "endpoint": self.endpoint,
                "timeout": timeout,
                "auth_enabled": auth_token is not None,
            }
        )

    async def _fetch_catalog(
        self,
        catalog_type: str,
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Generic method to fetch catalog from Tazilla API.

        Args:
            catalog_type: Catalog type (impacts, threats, controls, asset-types)
            session_id: Session ID for client-specific catalogs

        Returns:
            API response JSON (already localized by Tazilla backend)

        Raises:
            HTTPException: If API request fails
        """
        if not session_id:
            raise ValueError(
                f"session_id is required for Tazilla API provider "
                f"(catalog_type={catalog_type})"
            )

        url = f"{self.endpoint}/catalogs/{catalog_type}"
        headers = {"X-Session-Id": session_id}

        logger.debug(
            "tazilla_catalog_fetch_started",
            extra={
                "catalog_type": catalog_type,
                "session_id": session_id,
                "url": url,
            }
        )

        try:
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()

            logger.info(
                "tazilla_catalog_fetched",
                extra={
                    "catalog_type": catalog_type,
                    "session_id": session_id,
                    "total": data.get("total", 0),
                }
            )

            return data

        except httpx.TimeoutException as e:
            logger.error(
                "tazilla_catalog_timeout",
                extra={
                    "catalog_type": catalog_type,
                    "session_id": session_id,
                    "timeout_sec": self.timeout,
                }
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Tazilla API timeout after {self.timeout}s (catalog: {catalog_type})"
            )
        except httpx.ConnectError as e:
            logger.error(
                "tazilla_catalog_connection_failed",
                extra={
                    "catalog_type": catalog_type,
                    "session_id": session_id,
                    "error": str(e),
                }
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Cannot connect to Tazilla API (catalog: {catalog_type})"
            )
        except httpx.HTTPStatusError as e:
            logger.error(
                "tazilla_catalog_http_error",
                extra={
                    "catalog_type": catalog_type,
                    "session_id": session_id,
                    "status_code": e.response.status_code,
                    "response_preview": e.response.text[:200] if hasattr(e.response, 'text') else None,
                }
            )
            # Propagate original status code from Tazilla API
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Tazilla API error (catalog: {catalog_type}): {e.response.text[:200]}"
            )
        except Exception as e:
            logger.error(
                "tazilla_catalog_unexpected_error",
                extra={
                    "catalog_type": catalog_type,
                    "session_id": session_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error fetching catalog: {type(e).__name__}"
            )

    async def load_impact_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Impact]:
        """Fetch impact catalog from Tazilla API (already localized)."""
        from pydantic import ValidationError

        data = await self._fetch_catalog("impacts", session_id)

        # Parse response into Impact objects with validation
        try:
            impacts = [Impact(**item) for item in data.get("impacts", [])]
            return impacts
        except ValidationError as e:
            logger.error(
                "tazilla_catalog_validation_failed",
                extra={
                    "catalog_type": "impacts",
                    "session_id": session_id,
                    "error": str(e),
                    "errors": e.errors(),
                }
            )
            raise ValueError(f"Invalid impact data from Tazilla API: {str(e)}") from e

    async def load_threat_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Threat]:
        """Fetch threat catalog from Tazilla API (already localized)."""
        from pydantic import ValidationError

        data = await self._fetch_catalog("threats", session_id)

        # Parse response into Threat objects with validation
        try:
            threats = [Threat(**item) for item in data.get("threats", [])]
            return threats
        except ValidationError as e:
            logger.error(
                "tazilla_catalog_validation_failed",
                extra={
                    "catalog_type": "threats",
                    "session_id": session_id,
                    "error": str(e),
                    "errors": e.errors(),
                }
            )
            raise ValueError(f"Invalid threat data from Tazilla API: {str(e)}") from e

    async def load_control_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[PlannedControl]:
        """Fetch control catalog from Tazilla API (already localized)."""
        from pydantic import ValidationError

        data = await self._fetch_catalog("controls", session_id)

        # Parse response into PlannedControl objects with validation
        try:
            controls = [PlannedControl(**item) for item in data.get("controls", [])]
            return controls
        except ValidationError as e:
            logger.error(
                "tazilla_catalog_validation_failed",
                extra={
                    "catalog_type": "controls",
                    "session_id": session_id,
                    "error": str(e),
                    "errors": e.errors(),
                }
            )
            raise ValueError(f"Invalid control data from Tazilla API: {str(e)}") from e

    async def load_asset_type_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch asset type catalog from Tazilla API (already localized)."""
        data = await self._fetch_catalog("asset-types", session_id)

        # Return raw dicts (matching CSV provider format)
        # Note: No Pydantic validation since we return dicts, not typed models
        asset_types = data.get("assetTypes", [])

        # Basic structure validation
        for idx, item in enumerate(asset_types):
            if not isinstance(item, dict):
                logger.error(
                    "tazilla_catalog_validation_failed",
                    extra={
                        "catalog_type": "asset-types",
                        "session_id": session_id,
                        "error": f"Item at index {idx} is not a dict",
                    }
                )
                raise ValueError(f"Invalid asset type data at index {idx}: expected dict")

            # Check required fields
            required = ["name", "description", "isDependent"]
            missing = [f for f in required if f not in item]
            if missing:
                logger.error(
                    "tazilla_catalog_validation_failed",
                    extra={
                        "catalog_type": "asset-types",
                        "session_id": session_id,
                        "error": f"Missing fields: {missing}",
                        "item_index": idx,
                    }
                )
                raise ValueError(f"Asset type at index {idx} missing fields: {missing}")

        return asset_types

    async def load_impact_category_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Fetch impact category catalog from Tazilla API (already localized)."""
        data = await self._fetch_catalog("impact-categories", session_id)

        categories = data.get("categories", [])

        # Basic validation
        for idx, item in enumerate(categories):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid category at index {idx}: expected dict")
            required = ["name", "description"]
            missing = [f for f in required if f not in item]
            if missing:
                raise ValueError(f"Category at index {idx} missing fields: {missing}")

        return categories

    async def load_threat_category_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Fetch threat category catalog from Tazilla API (already localized)."""
        data = await self._fetch_catalog("threat-categories", session_id)

        categories = data.get("categories", [])

        # Basic validation
        for idx, item in enumerate(categories):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid category at index {idx}: expected dict")
            required = ["name", "description"]
            missing = [f for f in required if f not in item]
            if missing:
                raise ValueError(f"Category at index {idx} missing fields: {missing}")

        return categories

    async def load_control_category_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Fetch control category catalog from Tazilla API (already localized)."""
        data = await self._fetch_catalog("control-categories", session_id)

        categories = data.get("categories", [])

        # Basic validation
        for idx, item in enumerate(categories):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid category at index {idx}: expected dict")
            required = ["name", "description"]
            missing = [f for f in required if f not in item]
            if missing:
                raise ValueError(f"Category at index {idx} missing fields: {missing}")

        return categories

    async def load_sector_catalog(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Fetch sector catalog from Tazilla API (already localized)."""
        data = await self._fetch_catalog("sectors", session_id)

        sectors = data.get("sectors", [])

        for idx, item in enumerate(sectors):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid sector at index {idx}: expected dict")
            required = ["name", "description"]
            missing = [f for f in required if f not in item]
            if missing:
                raise ValueError(f"Sector at index {idx} missing fields: {missing}")

        return sectors

    async def close(self):
        """Close HTTP client connection."""
        await self.client.aclose()


# =============================================================================
# Factory Function
# =============================================================================

_provider_instance: Optional[CatalogProvider] = None


def get_catalog_provider() -> CatalogProvider:
    """
    Get catalog provider instance based on settings.

    Returns singleton instance configured from CATALOG__MODE setting:
    - "file": CSVCatalogProvider (dev/mockup)
    - "http": TazillaAPICatalogProvider (production)

    Returns:
        CatalogProvider instance
    """
    global _provider_instance

    if _provider_instance is not None:
        return _provider_instance

    from workshops_ai.settings import get_settings

    settings = get_settings()
    mode = settings.catalog.mode

    if mode == "file":
        logger.info("Using CSV catalog provider (mockup mode)")
        _provider_instance = CSVCatalogProvider()

    elif mode == "http":
        if not settings.catalog.endpoint:
            raise ValueError(
                "CATALOG__ENDPOINT is required when CATALOG__MODE=http"
            )

        logger.info(
            "Using Tazilla API catalog provider",
            extra={
                "endpoint": settings.catalog.endpoint,
                "timeout": settings.catalog.timeout_sec,
                "auth_enabled": settings.catalog.auth_token is not None,
            }
        )
        _provider_instance = TazillaAPICatalogProvider(
            endpoint=settings.catalog.endpoint,
            auth_token=settings.catalog.auth_token,
            timeout=settings.catalog.timeout_sec,
        )

    else:
        raise ValueError(f"Invalid CATALOG__MODE: {mode} (expected 'file' or 'http')")

    return _provider_instance


def reset_provider():
    """Reset provider instance (for testing)."""
    global _provider_instance
    _provider_instance = None
