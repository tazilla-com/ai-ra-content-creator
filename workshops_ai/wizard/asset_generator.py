# workshops_ai/wizard/asset_generator.py
"""
Asset generation using LangChain and OpenAI.

Generates IT and business assets from services and website content,
with ISO 27001 classification (CIA ratings, primary/supporting distinction).
"""
from __future__ import annotations

import json
import logging
import asyncio
from typing import List, Tuple, Dict, Any, Optional, Literal, get_args, get_origin
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pydantic import ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from workshops_ai.settings import get_settings
from workshops_ai.api.schema.asset import Asset
from workshops_ai.wizard.schemas import Service
from workshops_ai.wizard.prompts import (
    ASSETS_SYSTEM_PROMPT,
    ASSETS_HUMAN_PROMPT_TEMPLATE,
)
from workshops_ai.wizard.cost_calculator import create_usage_metrics, UsageMetrics
from workshops_ai.wizard.exceptions import (
    AIResponseError,
    AITimeoutError,
    ValidationRetriesExhausted,
    ModelNotAllowedError,
)

logger = logging.getLogger(__name__)


class AssetGenerator:
    """
    Generates IT and business assets from services and website content.

    Follows ISO 27001 classification:
    - Primary assets: Business-critical data/processes (must have CIA ratings)
    - Supporting assets: Infrastructure/systems (CIA must be null)
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initialize AssetGenerator with specified model.

        Args:
            model: OpenAI model name (defaults to settings if not provided)

        Raises:
            ModelNotAllowedError: If model not in allowed list
        """
        settings = get_settings()
        self.settings = settings

        # Resolve model
        self.model_name = model or settings.llm.model

        # Validate model against allowed list
        allowed_models = settings.wizard.allowed_models
        if self.model_name not in allowed_models:
            raise ModelNotAllowedError(self.model_name, allowed_models)

        # Initialize LangChain ChatOpenAI
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=settings.llm.temperature,
            openai_api_key=settings.llm.api_key,
            timeout=settings.wizard.llm_timeout_sec,
            max_retries=0,  # We handle retries ourselves with tenacity
        )

        logger.info(
            "AssetGenerator initialized",
            extra={
                "model": self.model_name,
                "temperature": settings.llm.temperature,
                "timeout_sec": settings.wizard.llm_timeout_sec,
            }
        )

    @staticmethod
    def _allowed_asset_types_from_model() -> List[str]:
        """
        Extract allowed asset types from Asset model Literal annotation.

        Returns:
            List of asset type strings from Asset.assetType Literal
        """
        typ = Asset.__annotations__.get("assetType")
        if typ is None:
            return []

        # Handle Literal type
        if get_origin(typ) is Literal:
            return list(get_args(typ))

        # Handle nested Literal (e.g., Optional[Literal[...]])
        for arg in get_args(typ):
            if get_origin(arg) is Literal:
                return list(get_args(arg))

        return []

    def _build_prompt_sections(
        self,
        website_content: str,
        services: Optional[List[Service]],
        organization_profile: Optional[Dict[str, Any]],
        user_instructions: Optional[str],
        previous_assets: Optional[List[Asset]],
    ) -> Dict[str, str]:
        """
        Build prompt sections from inputs.

        Args:
            website_content: Website text content
            services: Optional services from previous step
            organization_profile: Optional org metadata
            user_instructions: Optional user refinement instructions
            previous_assets: Optional previous assets for refinement

        Returns:
            Dict of section names to content
        """
        sections = {}

        # Organization profile section
        if organization_profile:
            profile_text = "\n".join([
                f"{key}: {value}"
                for key, value in organization_profile.items()
            ])
            sections["organization_profile_section"] = (
                f"Organization Profile:\n{profile_text}\n"
            )
        else:
            sections["organization_profile_section"] = ""

        # Services section (JSON)
        if services:
            services_json = json.dumps(
                [s.model_dump() for s in services],
                indent=2,
                ensure_ascii=False
            )
            sections["services_section"] = (
                f"Services (use these to identify related assets):\n{services_json}\n"
            )
        else:
            sections["services_section"] = ""

        # User instructions section
        if user_instructions:
            sections["user_instructions_section"] = (
                f"User Instructions (PRIORITY):\n{user_instructions}\n"
            )
        else:
            sections["user_instructions_section"] = ""

        # Previous assets section (for refinement)
        if previous_assets:
            assets_json = json.dumps(
                [a.model_dump() for a in previous_assets],
                indent=2,
                ensure_ascii=False
            )
            sections["previous_assets_section"] = (
                f"Previous Assets (refine these):\n{assets_json}\n"
            )
        else:
            sections["previous_assets_section"] = ""

        # Website content section (truncate if needed)
        max_length = self.settings.wizard.max_content_length
        truncated_content = website_content[:max_length]
        if len(website_content) > max_length:
            truncated_content += "\n... [content truncated]"

        sections["website_content_section"] = (
            f"Website Content:\n{truncated_content}\n"
        )

        return sections

    def _build_full_prompt_text(
        self,
        language: str,
        number: int,
        asset_types: List[str],
        sections: Dict[str, str]
    ) -> str:
        """
        Build full prompt text for debugging.

        Returns truncated version for logging.
        """
        human_prompt = ASSETS_HUMAN_PROMPT_TEMPLATE.format(
            number=number,
            language=language,
            asset_types=", ".join(asset_types),
            **sections
        )

        full_prompt = f"{ASSETS_SYSTEM_PROMPT}\n\n{human_prompt}"

        # Truncate for logging
        max_log_length = 500
        if len(full_prompt) > max_log_length:
            return full_prompt[:max_log_length] + "... [truncated for logging]"
        return full_prompt

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ValidationError),
        reraise=True,
    )
    async def _call_llm_with_retry(
        self,
        prompt: ChatPromptTemplate,
        **kwargs
    ) -> Tuple[str, UsageMetrics]:
        """
        Call LLM with automatic retry on validation errors.

        Args:
            prompt: LangChain ChatPromptTemplate
            **kwargs: Prompt variables

        Returns:
            Tuple of (response text, usage metrics)

        Raises:
            ValidationError: After retries exhausted
            AITimeoutError: If request times out
            AIResponseError: If response cannot be parsed
        """
        try:
            # Format messages
            messages = prompt.format_messages(**kwargs)

            # Invoke LLM (timeout already configured in ChatOpenAI constructor)
            response = await self.llm.ainvoke(messages)

            # Extract response text
            response_text = response.content

            # Extract usage from response metadata
            usage = response.response_metadata.get("token_usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Create usage metrics (with pricing from settings)
            metrics = create_usage_metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=self.model_name,
                pricing=self.settings.wizard.openai_pricing
            )

            logger.info(
                "LLM call successful",
                extra={
                    "model": self.model_name,
                    "tokens": metrics.total_tokens,
                    "cost_usd": metrics.cost_usd,
                }
            )

            return response_text, metrics

        except Exception as e:
            # Check if it's a timeout error from OpenAI/httpx
            error_msg = str(e).lower()
            if "timeout" in error_msg or "timed out" in error_msg:
                logger.error(
                    "LLM timeout",
                    extra={
                        "model": self.model_name,
                        "timeout_sec": self.settings.wizard.llm_timeout_sec,
                        "error": str(e),
                    }
                )
                raise AITimeoutError(
                    timeout_sec=self.settings.wizard.llm_timeout_sec,
                    model=self.model_name
                ) from e

            # Other errors
            logger.error(
                "LLM call failed",
                extra={
                    "model": self.model_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            raise

    def _parse_and_validate_assets(
        self,
        response_text: str,
        allowed_asset_types: List[str]
    ) -> List[Asset]:
        """
        Parse JSON response and validate assets.

        Enforces ISO 27001 rules:
        - Primary assets (isPrimary=true): CIA ratings required
        - Supporting assets (isPrimary=false): CIA must be null

        Args:
            response_text: Raw LLM response
            allowed_asset_types: List of allowed asset type strings

        Returns:
            List of validated Asset objects

        Raises:
            ValidationError: If response invalid or assets don't match schema
            AIResponseError: If JSON cannot be parsed
        """
        try:
            # Try to extract JSON array from response
            # Handle cases where LLM includes extra text
            response_text = response_text.strip()

            # Find JSON array boundaries
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]")

            if start_idx == -1 or end_idx == -1:
                raise AIResponseError(
                    "No JSON array found in response",
                    model=self.model_name,
                    details={"response_preview": response_text[:200]}
                )

            json_text = response_text[start_idx:end_idx + 1]

            # Parse JSON
            assets_data = json.loads(json_text)

            if not isinstance(assets_data, list):
                raise AIResponseError(
                    "Response is not a JSON array",
                    model=self.model_name
                )

            # Validate each asset
            assets = []
            for idx, asset_dict in enumerate(assets_data):
                try:
                    # Validate asset type
                    asset_type = asset_dict.get("assetType")
                    if asset_type and asset_type not in allowed_asset_types:
                        logger.warning(
                            f"Invalid assetType '{asset_type}' at index {idx}, skipping"
                        )
                        continue

                    # Validate isPrimary + CIA logic
                    is_primary = asset_dict.get("isPrimary", True)

                    if is_primary:
                        # Primary assets: Ensure CIA ratings are set
                        if not asset_dict.get("confidentiality"):
                            logger.warning(
                                f"Primary asset at index {idx} missing confidentiality, setting to INTERNAL"
                            )
                            asset_dict["confidentiality"] = "INTERNAL"
                        if not asset_dict.get("integrity"):
                            logger.warning(
                                f"Primary asset at index {idx} missing integrity, setting to MEDIUM"
                            )
                            asset_dict["integrity"] = "MEDIUM"
                        if not asset_dict.get("availability"):
                            logger.warning(
                                f"Primary asset at index {idx} missing availability, setting to MEDIUM"
                            )
                            asset_dict["availability"] = "MEDIUM"
                    else:
                        # Supporting assets: Ensure CIA is null
                        asset_dict["confidentiality"] = None
                        asset_dict["integrity"] = None
                        asset_dict["availability"] = None

                    asset = Asset(**asset_dict)
                    assets.append(asset)

                except ValidationError as e:
                    logger.error(
                        f"Asset validation failed at index {idx}",
                        extra={"error": str(e), "data": asset_dict}
                    )
                    # Continue with other assets instead of failing completely
                    continue

            return assets

        except json.JSONDecodeError as e:
            raise AIResponseError(
                f"Failed to parse JSON: {str(e)}",
                model=self.model_name,
                details={"response_preview": response_text[:200]}
            ) from e

    async def generate_assets(
        self,
        website_content: str,
        language: str = "en",
        number_of_assets: int = 10,
        services: Optional[List[Service]] = None,
        organization_profile: Optional[Dict[str, Any]] = None,
        user_instructions: Optional[str] = None,
        previous_assets: Optional[List[Asset]] = None,
    ) -> Tuple[List[Asset], UsageMetrics, Dict[str, Any]]:
        """
        Generate assets from website content and services.

        Args:
            website_content: Website text content
            language: Output language (en/sk/cs)
            number_of_assets: Number of assets to generate (1-50)
            services: Optional services from previous step
            organization_profile: Optional organization metadata
            user_instructions: Optional user refinement instructions
            previous_assets: Optional previous assets for refinement

        Returns:
            Tuple of (assets list, usage metrics, debug info dict)

        Raises:
            ValidationRetriesExhausted: If validation fails after all retries
            AITimeoutError: If LLM request times out
            AIResponseError: If response cannot be parsed
        """
        # Clamp number of assets to valid range
        number_of_assets = max(1, min(50, number_of_assets))

        # Extract allowed asset types from Asset model
        allowed_asset_types = self._allowed_asset_types_from_model()
        if not allowed_asset_types:
            raise RuntimeError("Failed to extract asset types from Asset model")

        # Build prompt sections
        sections = self._build_prompt_sections(
            website_content=website_content,
            services=services,
            organization_profile=organization_profile,
            user_instructions=user_instructions,
            previous_assets=previous_assets,
        )

        # Build full prompt for debugging
        prompt_text = self._build_full_prompt_text(
            language=language,
            number=number_of_assets,
            asset_types=allowed_asset_types,
            sections=sections
        )

        # Create LangChain prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", ASSETS_SYSTEM_PROMPT),
            ("human", ASSETS_HUMAN_PROMPT_TEMPLATE),
        ])

        # Call LLM with retry logic
        try:
            response_text, usage_metrics = await self._call_llm_with_retry(
                prompt=prompt,
                number=number_of_assets,
                language=language,
                asset_types=", ".join(allowed_asset_types),
                **sections
            )

            # Parse and validate
            assets = self._parse_and_validate_assets(
                response_text=response_text,
                allowed_asset_types=allowed_asset_types
            )

            # Limit to requested number
            assets = assets[:number_of_assets]

            # Build debug info
            debug_info = {
                "prompt_text": prompt_text,
                "response_text": response_text[:500],  # Truncate for logging
                "assets_count": len(assets),
                "allowed_asset_types": allowed_asset_types,
            }

            logger.info(
                "Assets generated successfully",
                extra={
                    "count": len(assets),
                    "model": self.model_name,
                    "tokens": usage_metrics.total_tokens,
                }
            )

            return assets, usage_metrics, debug_info

        except ValidationError as e:
            # Retries exhausted
            raise ValidationRetriesExhausted(
                retries=self.settings.wizard.llm_max_retries,
                errors=[str(e)]
            ) from e
