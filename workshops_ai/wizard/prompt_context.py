# workshops_ai/wizard/prompt_context.py
"""
Centralized prompt context builder for all wizard generators.

Handles:
- Loading catalogs from provider (CSV or API)
- Caching catalogs in session state with micro-TTL
- Building standardized prompt sections
- Formatting catalog data for prompts
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from workshops_ai.api.schema.service import Service
from workshops_ai.wizard.catalog_provider import get_catalog_provider
from workshops_ai.wizard.session import get_wizard_state, save_wizard_state

logger = logging.getLogger(__name__)


class PromptContextBuilder:
    """
    Centralized builder for prompt contexts.

    Handles catalog loading with session-based caching and micro-TTL.
    All wizard generators should use this instead of building sections themselves.
    """

    # Default catalog cache TTL (5 minutes)
    DEFAULT_CATALOG_TTL_SEC = 300

    def __init__(self, session_id: Optional[str] = None, catalog_ttl_sec: Optional[int] = None):
        """
        Initialize builder.

        Args:
            session_id: Session ID for session-based caching and API catalog fetching
            catalog_ttl_sec: Custom TTL for catalog cache (default: 300 seconds)
        """
        self.session_id = session_id
        self.catalog_ttl_sec = catalog_ttl_sec or self.DEFAULT_CATALOG_TTL_SEC
        self.provider = get_catalog_provider()

    async def _get_session_state(self) -> Dict[str, Any]:
        """Get current session state."""
        if not self.session_id:
            return {}
        return await get_wizard_state(self.session_id)

    async def _save_session_state(self, state: Dict[str, Any]) -> None:
        """Save updated session state."""
        if not self.session_id:
            return
        await save_wizard_state(self.session_id, state)

    def _is_catalog_fresh(self, cached_catalog: Dict[str, Any]) -> bool:
        """
        Check if cached catalog is still fresh based on TTL.

        Args:
            cached_catalog: Cached catalog dict with 'cached_at' and 'ttl_seconds'

        Returns:
            True if fresh, False if expired
        """
        if not cached_catalog or 'cached_at' not in cached_catalog:
            return False

        try:
            cached_at = datetime.fromisoformat(cached_catalog['cached_at'])
            ttl_seconds = cached_catalog.get('ttl_seconds', self.catalog_ttl_sec)

            # Make both timezone-aware for comparison
            now = datetime.now(timezone.utc)
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=timezone.utc)

            age_seconds = (now - cached_at).total_seconds()
            is_fresh = age_seconds < ttl_seconds

            logger.debug(
                f"Catalog freshness check: age={age_seconds:.1f}s, ttl={ttl_seconds}s, fresh={is_fresh}"
            )

            return is_fresh
        except Exception as e:
            logger.warning(f"Failed to check catalog freshness: {e}")
            return False

    async def _get_cached_catalog(
        self,
        catalog_type: str,
        loader_func,
    ) -> Any:
        """
        Get catalog with session-based caching and micro-TTL.

        Args:
            catalog_type: Type of catalog (impacts, threats, controls, etc.)
            loader_func: Async function to load catalog if not cached/expired

        Returns:
            Cached or freshly loaded catalog data
        """
        state = await self._get_session_state()

        # Get cached_catalogs section
        cached_catalogs = state.get('cached_catalogs', {})
        cached_entry = cached_catalogs.get(catalog_type)

        # Check if cache is fresh
        if cached_entry and self._is_catalog_fresh(cached_entry):
            logger.debug(f"Using cached catalog from session: {catalog_type}")
            # Deserialize catalog data
            return self._deserialize_catalog(catalog_type, cached_entry['data'])

        # Load fresh data
        logger.debug(f"Loading fresh catalog: {catalog_type}")
        data = await loader_func(session_id=self.session_id)

        # Cache in session state
        if self.session_id:
            cached_catalogs[catalog_type] = {
                'data': self._serialize_catalog(data),
                'cached_at': datetime.now(timezone.utc).isoformat(),
                'ttl_seconds': self.catalog_ttl_sec,
            }
            state['cached_catalogs'] = cached_catalogs
            await self._save_session_state(state)
            logger.debug(f"Cached catalog in session: {catalog_type}")

        return data

    def _serialize_catalog(self, data: Any) -> Any:
        """
        Serialize catalog data for session storage.

        Converts Pydantic models to dicts.
        """
        if isinstance(data, list):
            return [
                item.model_dump() if hasattr(item, 'model_dump') else item
                for item in data
            ]
        return data

    def _deserialize_catalog(self, catalog_type: str, cached_data: Any) -> Any:
        """
        Deserialize catalog data from session storage.

        Reconstructs Pydantic models from dicts.
        """
        if catalog_type == "impacts":
            from workshops_ai.api.schema.impact import Impact
            return [Impact(**item) for item in cached_data]

        elif catalog_type == "threats":
            from workshops_ai.api.schema.threat import Threat
            return [Threat(**item) for item in cached_data]

        elif catalog_type == "controls":
            from workshops_ai.api.schema.control import PlannedControl
            return [PlannedControl(**item) for item in cached_data]

        else:
            # For simple dicts (sectors, asset_types, categories)
            return cached_data

    async def build_sections(
        self,
        organization_profile: Optional[Dict[str, Any]] = None,
        services: Optional[List[Service]] = None,
        website_content: Optional[str] = None,
        user_instructions: Optional[str] = None,
        previous_services: Optional[List[Service]] = None,
        include_catalogs: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Build all prompt sections.

        Args:
            organization_profile: Organization metadata
            services: List of services
            website_content: Website text content
            user_instructions: User refinement instructions
            previous_services: Previous services for refinement
            include_catalogs: List of catalogs to include (e.g., ["impacts", "threats", "sectors"])
                            If None and session_id exists, ALL catalogs will be loaded automatically.
                            If [] (empty list), NO catalogs will be loaded.
                            If specific list, only those catalogs will be loaded.

        Returns:
            Dict of section names to formatted content
        """
        sections = {}

        # Auto-load all catalogs if include_catalogs is None and we have a session
        # This allows Langfuse prompts to reference any catalog without code changes
        if include_catalogs is None and self.session_id:
            include_catalogs = ["sectors", "impacts", "threats", "controls", "asset_types"]
            logger.debug(f"Auto-loading all catalogs for session {self.session_id}")

        # Organization profile section
        if organization_profile:
            profile_json = json.dumps(organization_profile, ensure_ascii=False, indent=2)
            sections["organization_profile_section"] = (
                f"Organization profile (optional JSON):\n{profile_json}\n"
            )
        else:
            sections["organization_profile_section"] = ""

        # Services section
        if services:
            services_json = json.dumps(
                [s.model_dump(mode='json') for s in services],
                ensure_ascii=False,
                indent=2
            )
            sections["services_section"] = (
                f"Services (optional JSON array):\n{services_json}\n"
            )
        else:
            sections["services_section"] = ""

        # Website content section
        if website_content:
            sections["website_content_section"] = (
                f"Website content:\n{website_content}\n"
            )
        else:
            sections["website_content_section"] = ""

        # User instructions section (for refinement)
        if user_instructions:
            sections["user_instructions_section"] = (
                f"User refinement instructions:\n{user_instructions}\n"
            )
        else:
            sections["user_instructions_section"] = ""

        # Previous services section (for refinement)
        if previous_services:
            prev_json = json.dumps(
                [s.model_dump(mode='json') for s in previous_services],
                ensure_ascii=False,
                indent=2
            )
            sections["previous_services_section"] = (
                f"Previous services:\n{prev_json}\n"
            )
        else:
            sections["previous_services_section"] = ""

        # Load and format catalog sections if requested
        if include_catalogs:
            for catalog_type in include_catalogs:
                section_content = await self._build_catalog_section(catalog_type)
                sections[f"{catalog_type}_catalog_section"] = section_content

        return sections

    async def _build_catalog_section(self, catalog_type: str) -> str:
        """
        Build formatted catalog section for prompts.

        Args:
            catalog_type: Type of catalog (impacts, threats, controls, sectors, asset_types)

        Returns:
            Formatted catalog section content
        """
        if catalog_type == "impacts":
            impacts = await self._get_cached_catalog(
                "impacts",
                self.provider.load_impact_catalog
            )
            # Format impacts with descriptions
            catalog_json = json.dumps([
                {
                    "name": imp.name,
                    "description": imp.description,
                }
                for imp in impacts
            ], ensure_ascii=False, indent=2)
            return f"Impact Catalog (with descriptions):\n{catalog_json}\n"

        elif catalog_type == "threats":
            threats = await self._get_cached_catalog(
                "threats",
                self.provider.load_threat_catalog
            )
            # Format threats with descriptions and asset types
            catalog_json = json.dumps([
                {
                    "name": thr.name,
                    "description": thr.description,
                    "category": thr.category,
                    "assetTypes": thr.assetTypes,
                }
                for thr in threats
            ], ensure_ascii=False, indent=2)
            return f"Threat Catalog (with descriptions):\n{catalog_json}\n"

        elif catalog_type == "controls":
            controls = await self._get_cached_catalog(
                "controls",
                self.provider.load_control_catalog
            )
            # Format controls with descriptions
            catalog_json = json.dumps([
                {
                    "name": ctrl.name,
                    "description": ctrl.description,
                    "category": ctrl.category,
                }
                for ctrl in controls
            ], ensure_ascii=False, indent=2)
            return f"Control Catalog (with descriptions):\n{catalog_json}\n"

        elif catalog_type == "sectors":
            sectors_data = await self._get_cached_catalog(
                "sectors",
                self.provider.load_sector_catalog
            )
            # Format sectors with descriptions
            catalog_json = json.dumps(sectors_data, ensure_ascii=False, indent=2)
            return f"Sector Catalog:\n{catalog_json}\n"

        elif catalog_type == "asset_types":
            asset_types_data = await self._get_cached_catalog(
                "asset_types",
                self.provider.load_asset_type_catalog
            )
            # Format asset types
            catalog_json = json.dumps(asset_types_data, ensure_ascii=False, indent=2)
            return f"Asset Type Catalog:\n{catalog_json}\n"

        else:
            logger.warning(f"Unknown catalog type: {catalog_type}")
            return ""

    async def get_catalog_names(self, catalog_type: str) -> List[str]:
        """
        Get just the names from a catalog (for backward compatibility).

        Args:
            catalog_type: Type of catalog

        Returns:
            List of catalog item names
        """
        if catalog_type == "impacts":
            impacts = await self._get_cached_catalog(
                "impacts",
                self.provider.load_impact_catalog
            )
            return [imp.name for imp in impacts]

        elif catalog_type == "threats":
            threats = await self._get_cached_catalog(
                "threats",
                self.provider.load_threat_catalog
            )
            return [thr.name for thr in threats]

        elif catalog_type == "controls":
            controls = await self._get_cached_catalog(
                "controls",
                self.provider.load_control_catalog
            )
            return [ctrl.name for ctrl in controls]

        elif catalog_type == "sectors":
            sectors_data = await self._get_cached_catalog(
                "sectors",
                self.provider.load_sector_catalog
            )
            return [sec["name"] for sec in sectors_data]

        else:
            logger.warning(f"Unknown catalog type: {catalog_type}")
            return []

    async def clear_catalog_cache(self):
        """Clear all cached catalogs for current session."""
        if not self.session_id:
            return

        state = await self._get_session_state()
        if 'cached_catalogs' in state:
            del state['cached_catalogs']
            await self._save_session_state(state)
            logger.info(f"Cleared catalog cache for session {self.session_id}")
