# workshops_ai/wizard/risk_generator.py
"""
Risk evaluation generator using LangChain and OpenAI.

Generates risk evaluations (probability 0-4) for each asset based on
official Threat catalog from CSV.
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
from workshops_ai.api.schema.impact import AssetImpact
from workshops_ai.api.schema.threat import AssetRisk, Risk
from workshops_ai.wizard.schemas import Service
from workshops_ai.wizard.prompts import (
    COMBINED_RISK_SYSTEM_PROMPT,
    COMBINED_RISK_HUMAN_PROMPT_TEMPLATE,
)
from workshops_ai.wizard.catalog_loader import (
    load_threat_catalog,
    get_threat_names,
    load_asset_type_threat_priors,
)
from workshops_ai.wizard.cost_calculator import create_usage_metrics, UsageMetrics
from workshops_ai.wizard.exceptions import (
    AIResponseError,
    AITimeoutError,
    ValidationRetriesExhausted,
    ModelNotAllowedError,
)

logger = logging.getLogger(__name__)


class RiskGenerator:
    """
    Generates risk evaluations for assets based on Threat catalog.

    Uses catalog-based approach:
    - Loads threats from CSV
    - Per-asset LLM evaluation
    - Filters probability 0 (not applicable)
    - Preserves catalog order
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initialize RiskGenerator with specified model.

        Args:
            model: OpenAI model name (defaults to settings if not provided)

        Raises:
            ModelNotAllowedError: If model not in allowed list
        """
        from workshops_ai.llm.wizard_client import create_wizard_llm_client

        settings = get_settings()
        self.settings = settings

        self.llm = create_wizard_llm_client(model)
        self.model_name = self.llm.model_name

        logger.info(
            "RiskGenerator initialized",
            extra={
                "model": self.model_name,
                "temperature": settings.llm.temperature,
                "timeout_sec": settings.wizard.llm_timeout_sec,
            }
        )

    def _build_prompt_sections(
        self,
        website_content: str,
        organization_profile: Optional[Dict[str, Any]],
        services: Optional[List[Service]],
        asset_impacts: Optional[List[AssetImpact]],
        user_instructions: Optional[str],
    ) -> Dict[str, str]:
        """
        Build prompt sections from inputs.

        Args:
            website_content: Website text content
            organization_profile: Optional org metadata
            services: Optional services for context
            asset_impacts: Optional asset impacts for context
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

        # Asset impacts section (JSON)
        if asset_impacts:
            impacts_json = json.dumps(
                [ai.model_dump() for ai in asset_impacts],
                indent=2,
                ensure_ascii=False
            )
            sections["asset_impacts_section"] = (
                f"Asset impacts (optional JSON array):\n{impacts_json}\n"
            )
        else:
            sections["asset_impacts_section"] = ""

        # Website content section (truncate if needed, omit if empty/disabled)
        if website_content:
            max_length = self.settings.wizard.max_content_length
            truncated_content = website_content[:max_length]
            if len(website_content) > max_length:
                truncated_content += "\n... [content truncated]"

            sections["website_content_section"] = (
                f"Website Content:\n{truncated_content}\n"
            )
        else:
            sections["website_content_section"] = ""

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

    def _parse_and_validate_risks(
        self,
        response_text: str,
        allowed_threat_names: List[str],
        ordered_catalog_names: List[str],
    ) -> List[Risk]:
        """
        Parse JSON response and validate risk evaluations.

        Filters out probability 0 (not applicable) and preserves catalog order.

        Args:
            response_text: Raw LLM response
            allowed_threat_names: List of allowed threat names
            ordered_catalog_names: List of catalog names in order

        Returns:
            List of validated Risk objects (probability 1-4 only)

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
            refined: List[Risk] = []
            for idx, ev_dict in enumerate(evaluations_data):
                try:
                    if not isinstance(ev_dict, dict):
                        continue

                    threat_name = str(ev_dict.get("threatName", "")).strip()
                    probability = int(ev_dict.get("probability", 0))

                    # Validate threat name
                    if threat_name not in allowed_threat_names:
                        logger.warning(
                            f"Invalid threatName '{threat_name}' at index {idx}, skipping"
                        )
                        continue

                    # Filter out probability 0 (not applicable)
                    if 1 <= probability <= 4:
                        refined.append(Risk(
                            threatName=threat_name,
                            probability=probability
                        ))

                except (ValueError, ValidationError) as e:
                    logger.error(
                        f"Risk evaluation validation failed at index {idx}",
                        extra={"error": str(e), "data": ev_dict}
                    )
                    continue

            # Preserve catalog order
            order_index = {n: i for i, n in enumerate(ordered_catalog_names)}
            final_evals = sorted(
                refined,
                key=lambda x: order_index.get(x.threatName, 10**9)
            )

            return final_evals

        except json.JSONDecodeError as e:
            raise AIResponseError(
                f"Failed to parse JSON: {str(e)}",
                model=self.model_name,
                details={"response_preview": response_text[:200]}
            ) from e

    def _select_relevant_threats(
        self,
        asset: Asset,
        all_threat_names: List[str],
        priors: Dict[str, List[str]],
        max_threats: int = 20,
    ) -> List[str]:
        """
        Select relevant threats for an asset using AssetType→Threat priors.

        If priors exist for the asset type, use them (limited to max_threats).
        Otherwise, use all threats (but still limit to max_threats for performance).

        Args:
            asset: Asset to evaluate
            all_threat_names: All threat names from catalog
            priors: AssetType→Threat mapping
            max_threats: Maximum number of threats to evaluate (default 20)

        Returns:
            List of relevant threat names (subset of catalog)
        """
        # Get relevant threats from priors
        relevant = priors.get(asset.assetType, [])

        if relevant:
            # Use priors, limit to max_threats
            selected = relevant[:max_threats]
            logger.info(
                f"Using {len(selected)} prior threats for asset type '{asset.assetType}'",
                extra={
                    "asset_name": asset.name,
                    "asset_type": asset.assetType,
                    "threats_count": len(selected),
                }
            )
        else:
            # No priors, use first max_threats from catalog
            selected = all_threat_names[:max_threats]
            logger.warning(
                f"No priors for asset type '{asset.assetType}', using first {len(selected)} catalog threats",
                extra={
                    "asset_name": asset.name,
                    "asset_type": asset.assetType,
                    "threats_count": len(selected),
                }
            )

        return selected

    async def generate_asset_risks(
        self,
        assets: List[Asset],
        language: str = "en",
        min_threats_per_asset: Optional[int] = None,
        max_threats_per_asset: Optional[int] = None,
        website_content: str = "",
        organization_profile: Optional[Dict[str, Any]] = None,
        services: Optional[List[Service]] = None,
        asset_impacts: Optional[List[AssetImpact]] = None,
        user_instructions: Optional[str] = None,
        use_priors: bool = True,
    ) -> Tuple[List[AssetRisk], UsageMetrics, Dict[str, Any]]:
        """
        Generate risk evaluations for each asset.

        Processes one asset at a time, evaluating relevant threats from catalog.
        Uses AssetType→Threat priors for smart threat selection (if available).

        Args:
            assets: List of assets to evaluate
            language: Output language (en/sk/cs)
            min_threats_per_asset: Minimum threats per asset (optional, if not provided, no minimum enforced)
            max_threats_per_asset: Maximum threats per asset (optional, if not provided, no maximum enforced)
            website_content: Website text content (optional)
            organization_profile: Optional organization metadata
            services: Optional services for context
            asset_impacts: Optional asset impacts for context
            user_instructions: Optional user refinement instructions
            use_priors: Whether to use AssetType→Threat priors for selection (default True)

        Returns:
            Tuple of (asset risks list, aggregated usage metrics, debug info dict)

        Raises:
            ValidationRetriesExhausted: If validation fails after all retries
            AITimeoutError: If LLM request times out
            AIResponseError: If response cannot be parsed
        """

        # Build threats count constraint text for prompt
        if min_threats_per_asset is not None and max_threats_per_asset is not None:
            threats_constraint = f"Evaluate between {min_threats_per_asset} and {max_threats_per_asset} most relevant threats."
        elif min_threats_per_asset is not None:
            threats_constraint = f"Evaluate at least {min_threats_per_asset} most relevant threats."
        elif max_threats_per_asset is not None:
            threats_constraint = f"Evaluate up to {max_threats_per_asset} most relevant threats."
        else:
            threats_constraint = "Evaluate all relevant threats from the catalog."
        # Load threat catalog for language
        all_threat_names = get_threat_names(language)
        allowed_names_set = set(all_threat_names)

        # Load AssetType→Threat priors (optional)
        priors: Dict[str, List[str]] = {}
        if use_priors:
            try:
                priors = load_asset_type_threat_priors()
                logger.info(f"Loaded priors for {len(priors)} asset types")
            except Exception as e:
                logger.warning(f"Failed to load priors, using all threats: {e}")

        # Build shared prompt sections
        sections = self._build_prompt_sections(
            website_content=website_content,
            organization_profile=organization_profile,
            services=services,
            asset_impacts=asset_impacts,
            user_instructions=user_instructions,
        )

        # Create LangChain prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", COMBINED_RISK_SYSTEM_PROMPT),
            ("human", COMBINED_RISK_HUMAN_PROMPT_TEMPLATE),
        ])

        results: List[AssetRisk] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        last_response_text = ""
        last_prompt_text = ""

        # Process each asset individually
        for asset in assets:
            try:
                # Select relevant threats for this asset
                relevant_threats = self._select_relevant_threats(
                    asset=asset,
                    all_threat_names=all_threat_names,
                    priors=priors,
                    max_threats=max_threats_per_asset,
                )

                # Single-asset JSON
                assets_json = json.dumps([asset.model_dump()], ensure_ascii=False)

                # Format prompt for debug preview
                messages = prompt.format_messages(
                    assets_json=assets_json,
                    ordered_catalog_names_json=json.dumps(relevant_threats, ensure_ascii=False),
                    threats_constraint=threats_constraint,
                    language=language,
                    **sections
                )
                last_prompt_text = str(messages)

                # Call LLM with retry logic (using filtered threat list)
                response_text, usage_metrics = await self._call_llm_with_retry(
                    prompt=prompt,
                    assets_json=assets_json,
                    ordered_catalog_names_json=json.dumps(relevant_threats, ensure_ascii=False),
                    threats_constraint=threats_constraint,
                    language=language,
                    **sections
                )

                # Accumulate usage
                total_prompt_tokens += usage_metrics.prompt_tokens
                total_completion_tokens += usage_metrics.completion_tokens
                last_response_text = response_text

                # Parse and validate (using filtered threat list)
                evaluations = self._parse_and_validate_risks(
                    response_text=response_text,
                    allowed_threat_names=relevant_threats,
                    ordered_catalog_names=relevant_threats,
                )

                # Apply max limit if specified
                if max_threats_per_asset is not None:
                    evaluations = evaluations[:max_threats_per_asset]

                # Create AssetRisk
                results.append(AssetRisk(
                    assetId=asset.id,
                    assetName=asset.name,
                    risks=evaluations
                ))

                logger.info(
                    f"Risk evaluation completed for asset '{asset.name}'",
                    extra={
                        "asset_id": str(asset.id) if asset.id else None,
                        "risks_count": len(evaluations),
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
                    f"Risk evaluation failed for asset '{asset.name}'",
                    extra={
                        "asset_id": str(asset.id) if asset.id else None,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                # For other unexpected errors, continue with other assets
                results.append(AssetRisk(
                    assetId=asset.id,
                    assetName=asset.name,
                    risks=[]
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
            "catalog_threats_total": len(all_threat_names),
            "min_threats_per_asset": min_threats_per_asset,
            "max_threats_per_asset": max_threats_per_asset,
            "threats_constraint": threats_constraint,
            "priors_available": len(priors) > 0,
            "priors_count": len(priors),
            "last_prompt_preview": last_prompt_text[:500],
            "last_response_preview": last_response_text[:500],
        }

        logger.info(
            "Risk evaluations generated successfully",
            extra={
                "assets_count": len(assets),
                "total_tokens": aggregated_metrics.total_tokens,
                "cost_usd": aggregated_metrics.cost_usd,
            }
        )

        return results, aggregated_metrics, debug_info
