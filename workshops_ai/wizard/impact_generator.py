# workshops_ai/wizard/impact_generator.py
"""
Impact evaluation generator using LangChain and OpenAI.

Generates impact evaluations (severity 0-4) for each asset based on
official Impact catalog from CSV.
"""
from __future__ import annotations

import json
import logging
import asyncio
from typing import List, Tuple, Dict, Any, Optional
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
from workshops_ai.api.schema.impact import ImpactEvaluation, AssetImpact
from workshops_ai.wizard.schemas import Service
from workshops_ai.wizard.prompts import (
    IMPACTS_SYSTEM_PROMPT,
    IMPACTS_HUMAN_PROMPT_TEMPLATE,
)
from workshops_ai.wizard.catalog_loader import load_impact_catalog, get_impact_names
from workshops_ai.wizard.cost_calculator import create_usage_metrics, UsageMetrics
from workshops_ai.wizard.exceptions import (
    AIResponseError,
    AITimeoutError,
    ValidationRetriesExhausted,
    ModelNotAllowedError,
)

logger = logging.getLogger(__name__)


class ImpactGenerator:
    """
    Generates impact evaluations for assets based on Impact catalog.

    Uses catalog-based approach:
    - Loads impacts from CSV
    - Per-asset LLM evaluation
    - Filters level 0 (not applicable)
    - Preserves catalog order
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initialize ImpactGenerator with specified model.

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
            "ImpactGenerator initialized",
            extra={
                "model": self.model_name,
                "temperature": settings.llm.temperature,
                "timeout_sec": settings.wizard.llm_timeout_sec,
            }
        )

    def _build_prompt_sections(
        self,
        organization_profile: Optional[Dict[str, Any]],
        services: Optional[List[Service]],
        user_instructions: Optional[str],
    ) -> Dict[str, str]:
        """
        Build prompt sections from inputs.

        Args:
            organization_profile: Optional org metadata
            services: Optional services for context
            user_instructions: Optional user refinement instructions

        Returns:
            Dict of section names to content
        """
        sections = {}

        # Organization profile section
        if organization_profile:
            profile_json = json.dumps(organization_profile, ensure_ascii=False)
            sections["organization_profile_section"] = (
                f"Organization profile (optional JSON):\n{profile_json}\n"
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
                f"Services (optional JSON array):\n{services_json}\n"
            )
        else:
            sections["services_section"] = ""

        # User instructions section
        if user_instructions:
            sections["user_instructions_section"] = (
                f"Additional user revision requests (optional): {user_instructions}\n"
            )
        else:
            sections["user_instructions_section"] = ""

        return sections

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

    def _parse_and_validate_impacts(
        self,
        response_text: str,
        allowed_impact_names: List[str],
        ordered_catalog_names: List[str],
    ) -> List[ImpactEvaluation]:
        """
        Parse JSON response and validate impact evaluations.

        Filters out level 0 (not applicable) and preserves catalog order.

        Args:
            response_text: Raw LLM response
            allowed_impact_names: List of allowed impact names
            ordered_catalog_names: List of catalog names in order

        Returns:
            List of validated ImpactEvaluation objects (level 1-4 only)

        Raises:
            AIResponseError: If JSON cannot be parsed
        """
        try:
            # Try to extract JSON array from response
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
            evaluations_data = json.loads(json_text)

            if not isinstance(evaluations_data, list):
                raise AIResponseError(
                    "Response is not a JSON array",
                    model=self.model_name
                )

            # Validate each evaluation
            refined: List[ImpactEvaluation] = []
            for idx, ev_dict in enumerate(evaluations_data):
                try:
                    if not isinstance(ev_dict, dict):
                        continue

                    impact_name = str(ev_dict.get("impactName", "")).strip()
                    impact_level = int(ev_dict.get("impactLevel", 0))

                    # Validate impact name
                    if impact_name not in allowed_impact_names:
                        logger.warning(
                            f"Invalid impactName '{impact_name}' at index {idx}, skipping"
                        )
                        continue

                    # Filter out level 0 (not applicable)
                    if 1 <= impact_level <= 4:
                        refined.append(ImpactEvaluation(
                            impactName=impact_name,
                            impactLevel=impact_level
                        ))

                except (ValueError, ValidationError) as e:
                    logger.error(
                        f"Impact evaluation validation failed at index {idx}",
                        extra={"error": str(e), "data": ev_dict}
                    )
                    continue

            # Preserve catalog order
            order_index = {n: i for i, n in enumerate(ordered_catalog_names)}
            final_evals = sorted(
                refined,
                key=lambda x: order_index.get(x.impactName, 10**9)
            )

            return final_evals

        except json.JSONDecodeError as e:
            raise AIResponseError(
                f"Failed to parse JSON: {str(e)}",
                model=self.model_name,
                details={"response_preview": response_text[:200]}
            ) from e

    async def generate_asset_impacts(
        self,
        assets: List[Asset],
        language: str = "en",
        organization_profile: Optional[Dict[str, Any]] = None,
        services: Optional[List[Service]] = None,
        user_instructions: Optional[str] = None,
    ) -> Tuple[List[AssetImpact], UsageMetrics, Dict[str, Any]]:
        """
        Generate impact evaluations for each asset.

        Processes one asset at a time, evaluating all impacts from catalog.

        Args:
            assets: List of assets to evaluate
            language: Output language (en/sk/cs)
            organization_profile: Optional organization metadata
            services: Optional services for context
            user_instructions: Optional user refinement instructions

        Returns:
            Tuple of (asset impacts list, aggregated usage metrics, debug info dict)

        Raises:
            ValidationRetriesExhausted: If validation fails after all retries
            AITimeoutError: If LLM request times out
            AIResponseError: If response cannot be parsed
        """
        # Load impact catalog for language
        ordered_catalog_names = get_impact_names(language)
        allowed_names_set = set(ordered_catalog_names)

        # Build shared prompt sections
        sections = self._build_prompt_sections(
            organization_profile=organization_profile,
            services=services,
            user_instructions=user_instructions,
        )

        # Create LangChain prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", IMPACTS_SYSTEM_PROMPT),
            ("human", IMPACTS_HUMAN_PROMPT_TEMPLATE),
        ])

        results: List[AssetImpact] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        last_response_text = ""
        last_prompt_text = ""

        # Process each asset individually
        for asset in assets:
            try:
                # Single-asset JSON
                assets_json = json.dumps([asset.model_dump()], ensure_ascii=False)

                # Format prompt for debug preview
                messages = prompt.format_messages(
                    assets_json=assets_json,
                    ordered_catalog_names_json=json.dumps(ordered_catalog_names, ensure_ascii=False),
                    language=language,
                    **sections
                )
                last_prompt_text = str(messages)

                # Call LLM with retry logic
                response_text, usage_metrics = await self._call_llm_with_retry(
                    prompt=prompt,
                    assets_json=assets_json,
                    ordered_catalog_names_json=json.dumps(ordered_catalog_names, ensure_ascii=False),
                    language=language,
                    **sections
                )

                # Accumulate usage
                total_prompt_tokens += usage_metrics.prompt_tokens
                total_completion_tokens += usage_metrics.completion_tokens
                last_response_text = response_text

                # Parse and validate
                evaluations = self._parse_and_validate_impacts(
                    response_text=response_text,
                    allowed_impact_names=list(allowed_names_set),
                    ordered_catalog_names=ordered_catalog_names,
                )

                # Create AssetImpact
                results.append(AssetImpact(
                    assetId=asset.id,
                    assetName=asset.name,
                    impacts=evaluations
                ))

                logger.info(
                    f"Impact evaluation completed for asset '{asset.name}'",
                    extra={
                        "asset_id": str(asset.id) if asset.id else None,
                        "impacts_count": len(evaluations),
                    }
                )

            except AITimeoutError:
                # Re-raise timeout errors immediately - don't continue with other assets
                raise
            except (AIResponseError, ValidationRetriesExhausted):
                # Re-raise AI-specific errors immediately
                raise
            except Exception as e:
                logger.error(
                    f"Impact evaluation failed for asset '{asset.name}'",
                    extra={
                        "asset_id": str(asset.id) if asset.id else None,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                # For other unexpected errors, continue with other assets
                results.append(AssetImpact(
                    assetId=asset.id,
                    assetName=asset.name,
                    impacts=[]
                ))

        # Create aggregated usage metrics
        aggregated_metrics = create_usage_metrics(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            model=self.model_name,
            pricing=self.settings.wizard.openai_pricing
        )

        # Build debug info
        debug_info = {
            "assets_processed": len(assets),
            "catalog_impacts_count": len(ordered_catalog_names),
            "last_prompt_preview": last_prompt_text[:500],
            "last_response_preview": last_response_text[:500],
        }

        logger.info(
            "Impact evaluations generated successfully",
            extra={
                "assets_count": len(assets),
                "total_tokens": aggregated_metrics.total_tokens,
                "cost_usd": aggregated_metrics.cost_usd,
            }
        )

        return results, aggregated_metrics, debug_info
