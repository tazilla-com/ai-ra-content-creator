# workshops_ai/wizard/control_generator.py
"""
Control recommendation generator using LangChain and OpenAI.

Generates control recommendations for each asset based on:
- Official Control catalog from CSV
- Asset risks (threat probabilities)
- Control → Threat coverage mappings

Uses hybrid approach:
1. Pre-filter relevant controls (where control covers at least one asset threat)
2. AI selects optimal controls with cost estimates and coverage adjustments
"""
from __future__ import annotations

import json
import logging
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from decimal import Decimal
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
from workshops_ai.api.schema.threat import AssetRisk
from workshops_ai.api.schema.control import PlannedControl, ThreatCoverage, AssetControls
from workshops_ai.wizard.prompts import (
    CONTROLS_SYSTEM_PROMPT,
    CONTROLS_HUMAN_PROMPT_TEMPLATE,
)
from workshops_ai.wizard.catalog_loader import (
    load_control_catalog,
    load_control_threat_coverage,
)
from workshops_ai.wizard.cost_calculator import create_usage_metrics, UsageMetrics
from workshops_ai.wizard.exceptions import (
    AIResponseError,
    AITimeoutError,
    ValidationRetriesExhausted,
    ModelNotAllowedError,
)

logger = logging.getLogger(__name__)


class ControlGenerator:
    """
    Generates control recommendations for assets based on Control catalog.

    Uses hybrid catalog + AI approach:
    - Pre-filters relevant controls based on threat coverage
    - LLM selects optimal controls with cost estimates
    - Per-asset processing
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initialize ControlGenerator with specified model.

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
            "ControlGenerator initialized",
            extra={
                "model": self.model_name,
                "temperature": settings.llm.temperature,
                "timeout_sec": settings.wizard.llm_timeout_sec,
            }
        )

    def _build_prompt_sections(
        self,
        website_content: str,
        user_instructions: Optional[str],
    ) -> Dict[str, str]:
        """
        Build prompt sections from inputs.

        Args:
            website_content: Website text content
            user_instructions: Optional user refinement instructions

        Returns:
            Dict of section names to content
        """
        sections = {}

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
                f"Additional user instructions for control selection: {user_instructions}\n"
            )
        else:
            sections["user_instructions_section"] = ""

        return sections

    def _filter_relevant_controls(
        self,
        asset_risks: AssetRisk,
        control_catalog: List[PlannedControl],
        threat_coverage: Dict[str, Dict[str, int]],
    ) -> List[Dict[str, Any]]:
        """
        Pre-filter relevant controls for an asset based on its threats.

        A control is relevant if it covers at least ONE threat from asset_risks
        (with probability > 0).

        Args:
            asset_risks: Asset risks with threat probabilities
            control_catalog: Full control catalog
            threat_coverage: Control → {Threat → Coverage %} mapping

        Returns:
            List of relevant controls with catalog coverage hints
            Format: [{"name": "...", "description": "...", "category": "...",
                     "catalog_coverage": {"threat1": 50, "threat2": 30}}]
        """
        # Extract applicable threats (probability > 0)
        applicable_threats = {
            risk.threatName
            for risk in asset_risks.risks
            if risk.probability > 0
        }

        if not applicable_threats:
            logger.warning(
                f"No applicable threats for asset '{asset_risks.assetName}' (all probability 0)"
            )
            return []

        relevant = []

        for control in control_catalog:
            control_name = control.name

            # Get coverage mapping for this control
            coverage = threat_coverage.get(control_name, {})

            # Check if control covers any applicable threat
            covered_threats = {}
            for threat_name in applicable_threats:
                if threat_name in coverage and coverage[threat_name] > 0:
                    covered_threats[threat_name] = coverage[threat_name]

            if covered_threats:
                relevant.append({
                    "name": control_name,
                    "description": control.description or "",
                    "category": control.category,
                    "catalog_coverage": covered_threats,
                })

        logger.info(
            f"Filtered {len(relevant)} relevant controls for asset '{asset_risks.assetName}'",
            extra={
                "asset_name": asset_risks.assetName,
                "applicable_threats": len(applicable_threats),
                "relevant_controls": len(relevant),
            }
        )

        return relevant

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

    def _parse_and_validate_controls(
        self,
        response_text: str,
        allowed_control_names: List[str],
        asset_risk_threats: List[str],
    ) -> List[PlannedControl]:
        """
        Parse JSON response and validate control recommendations.

        Args:
            response_text: Raw LLM response
            allowed_control_names: List of allowed control names from pre-filtered list
            asset_risk_threats: List of threat names from asset risks

        Returns:
            List of validated PlannedControl objects

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
            controls_data = json.loads(json_text)

            if not isinstance(controls_data, list):
                raise AIResponseError(
                    "Response is not a JSON array",
                    model=self.model_name
                )

            # Validate each control
            validated: List[PlannedControl] = []
            allowed_names_set = set(allowed_control_names)
            asset_threats_set = set(asset_risk_threats)

            for idx, ctrl_dict in enumerate(controls_data):
                try:
                    if not isinstance(ctrl_dict, dict):
                        continue

                    name = str(ctrl_dict.get("name", "")).strip()

                    # Validate control name
                    if name not in allowed_names_set:
                        logger.warning(
                            f"Invalid control name '{name}' at index {idx}, skipping"
                        )
                        continue

                    description = ctrl_dict.get("description")
                    category = str(ctrl_dict.get("category", "Other"))

                    # Parse costs (handle both float and Decimal)
                    impl_cost_raw = ctrl_dict.get("implementationCost")
                    annual_cost_raw = ctrl_dict.get("estimatedAnnualCost")

                    impl_cost = None
                    if impl_cost_raw is not None:
                        try:
                            impl_cost = Decimal(str(impl_cost_raw))
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Invalid implementationCost for control '{name}', using None"
                            )

                    annual_cost = None
                    if annual_cost_raw is not None:
                        try:
                            annual_cost = Decimal(str(annual_cost_raw))
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Invalid estimatedAnnualCost for control '{name}', using None"
                            )

                    # Parse threats array
                    threats_data = ctrl_dict.get("threats", [])
                    if not isinstance(threats_data, list):
                        logger.warning(
                            f"Invalid threats array for control '{name}', using empty list"
                        )
                        threats_data = []

                    threats: List[ThreatCoverage] = []
                    for t_idx, threat_dict in enumerate(threats_data):
                        try:
                            threat_name = str(threat_dict.get("threatName", "")).strip()
                            risk_coverage_raw = threat_dict.get("riskCoverage", 0)

                            # Validate threat name
                            if threat_name not in asset_threats_set:
                                logger.warning(
                                    f"Invalid threatName '{threat_name}' in control '{name}', skipping"
                                )
                                continue

                            # Parse coverage (handle both float and Decimal)
                            try:
                                risk_coverage = Decimal(str(risk_coverage_raw))
                            except (ValueError, TypeError):
                                logger.warning(
                                    f"Invalid riskCoverage for threat '{threat_name}' in control '{name}', using 0"
                                )
                                risk_coverage = Decimal("0")

                            # Clamp coverage to 0-100
                            risk_coverage = max(Decimal("0"), min(Decimal("100"), risk_coverage))

                            threats.append(ThreatCoverage(
                                threatName=threat_name,
                                riskCoverage=risk_coverage
                            ))

                        except (ValueError, ValidationError) as e:
                            logger.error(
                                f"Threat coverage validation failed at index {t_idx} for control '{name}'",
                                extra={"error": str(e), "data": threat_dict}
                            )
                            continue

                    # Only include controls with at least one valid threat
                    if threats:
                        validated.append(PlannedControl(
                            id=None,
                            name=name,
                            description=description,
                            category=category,
                            implementationCost=impl_cost,
                            estimatedAnnualCost=annual_cost,
                            threats=threats
                        ))
                    else:
                        logger.warning(
                            f"Control '{name}' has no valid threats, skipping"
                        )

                except (ValueError, ValidationError) as e:
                    logger.error(
                        f"Control validation failed at index {idx}",
                        extra={"error": str(e), "data": ctrl_dict}
                    )
                    continue

            return validated

        except json.JSONDecodeError as e:
            raise AIResponseError(
                f"Failed to parse JSON: {str(e)}",
                model=self.model_name,
                details={"response_preview": response_text[:200]}
            ) from e

    async def generate_asset_controls(
        self,
        assets: List[Asset],
        asset_risks: List[AssetRisk],
        language: str = "en",
        min_controls_per_asset: Optional[int] = None,
        max_controls_per_asset: Optional[int] = None,
        website_content: str = "",
        user_instructions: Optional[str] = None,
    ) -> Tuple[List[AssetControls], UsageMetrics, Dict[str, Any]]:
        """
        Generate control recommendations for each asset.

        Processes one asset at a time using hybrid approach:
        1. Pre-filter relevant controls based on threat coverage
        2. AI selects optimal controls with cost estimates

        Args:
            assets: List of assets to protect
            asset_risks: List of asset risks (threat probabilities)
            language: Output language (en/sk/cs)
            min_controls_per_asset: Minimum controls per asset (optional, if not provided, no minimum enforced)
            max_controls_per_asset: Maximum controls per asset (optional, if not provided, no maximum enforced)
            website_content: Website text content (optional)
            user_instructions: Optional user refinement instructions

        Returns:
            Tuple of (asset controls list, aggregated usage metrics, debug info dict)

        Raises:
            ValidationRetriesExhausted: If validation fails after all retries
            AITimeoutError: If LLM request times out
            AIResponseError: If response cannot be parsed
        """

        # Build controls count constraint text for prompt
        if min_controls_per_asset is not None and max_controls_per_asset is not None:
            controls_constraint = f"Select between {min_controls_per_asset} and {max_controls_per_asset} controls."
        elif min_controls_per_asset is not None:
            controls_constraint = f"Select at least {min_controls_per_asset} controls."
        elif max_controls_per_asset is not None:
            controls_constraint = f"Select up to {max_controls_per_asset} controls."
        else:
            controls_constraint = "Select the optimal number of controls for this asset."
        # Load control catalog for language
        control_catalog = load_control_catalog(language)
        threat_coverage = load_control_threat_coverage()

        # Build asset-risk lookup
        risk_lookup: Dict[str, AssetRisk] = {}
        for ar in asset_risks:
            # Try matching by assetId first, fallback to assetName
            if ar.assetId:
                risk_lookup[str(ar.assetId)] = ar
            risk_lookup[ar.assetName] = ar

        # Build shared prompt sections
        sections = self._build_prompt_sections(
            website_content=website_content,
            user_instructions=user_instructions,
        )

        # Create LangChain prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", CONTROLS_SYSTEM_PROMPT),
            ("human", CONTROLS_HUMAN_PROMPT_TEMPLATE),
        ])

        results: List[AssetControls] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        last_response_text = ""
        last_prompt_text = ""
        skipped_assets = 0

        # Process each asset individually
        for asset in assets:
            try:
                # Find asset risks
                asset_risk = None
                if asset.id:
                    asset_risk = risk_lookup.get(str(asset.id))
                if not asset_risk:
                    asset_risk = risk_lookup.get(asset.name)

                if not asset_risk:
                    logger.warning(
                        f"No risks found for asset '{asset.name}', skipping control selection"
                    )
                    results.append(AssetControls(
                        assetId=asset.id,
                        assetName=asset.name,
                        controls=[]
                    ))
                    skipped_assets += 1
                    continue

                # Check if asset has any applicable threats (probability > 0)
                applicable_threats = [
                    risk for risk in asset_risk.risks
                    if risk.probability > 0
                ]

                if not applicable_threats:
                    logger.warning(
                        f"No applicable threats for asset '{asset.name}' (all probability 0), skipping"
                    )
                    results.append(AssetControls(
                        assetId=asset.id,
                        assetName=asset.name,
                        controls=[]
                    ))
                    skipped_assets += 1
                    continue

                # Pre-filter relevant controls
                relevant_controls = self._filter_relevant_controls(
                    asset_risks=asset_risk,
                    control_catalog=control_catalog,
                    threat_coverage=threat_coverage,
                )

                if not relevant_controls:
                    logger.warning(
                        f"No relevant controls found for asset '{asset.name}', skipping"
                    )
                    results.append(AssetControls(
                        assetId=asset.id,
                        assetName=asset.name,
                        controls=[]
                    ))
                    skipped_assets += 1
                    continue

                # Prepare LLM inputs
                asset_json = json.dumps(asset.model_dump(), ensure_ascii=False)
                asset_risks_json = json.dumps(
                    [{"threatName": r.threatName, "probability": r.probability}
                     for r in asset_risk.risks if r.probability > 0],
                    ensure_ascii=False
                )
                relevant_controls_json = json.dumps(relevant_controls, ensure_ascii=False)

                # Format prompt for debug preview
                messages = prompt.format_messages(
                    asset_json=asset_json,
                    asset_risks_json=asset_risks_json,
                    relevant_controls_json=relevant_controls_json,
                    controls_constraint=controls_constraint,
                    **sections
                )
                last_prompt_text = str(messages)

                # Call LLM with retry logic
                response_text, usage_metrics = await self._call_llm_with_retry(
                    prompt=prompt,
                    asset_json=asset_json,
                    asset_risks_json=asset_risks_json,
                    relevant_controls_json=relevant_controls_json,
                    controls_constraint=controls_constraint,
                    **sections
                )

                # Accumulate usage
                total_prompt_tokens += usage_metrics.prompt_tokens
                total_completion_tokens += usage_metrics.completion_tokens
                last_response_text = response_text

                # Parse and validate
                allowed_control_names = [c["name"] for c in relevant_controls]
                asset_threat_names = [r.threatName for r in asset_risk.risks if r.probability > 0]

                controls = self._parse_and_validate_controls(
                    response_text=response_text,
                    allowed_control_names=allowed_control_names,
                    asset_risk_threats=asset_threat_names,
                )

                # Apply max limit if specified
                if max_controls_per_asset is not None:
                    controls = controls[:max_controls_per_asset]

                # Create AssetControls
                results.append(AssetControls(
                    assetId=asset.id,
                    assetName=asset.name,
                    controls=controls
                ))

                logger.info(
                    f"Control selection completed for asset '{asset.name}'",
                    extra={
                        "asset_id": str(asset.id) if asset.id else None,
                        "controls_count": len(controls),
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
                    f"Control selection failed for asset '{asset.name}'",
                    extra={
                        "asset_id": str(asset.id) if asset.id else None,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                # For other unexpected errors, continue with other assets
                results.append(AssetControls(
                    assetId=asset.id,
                    assetName=asset.name,
                    controls=[]
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
            "assets_skipped": skipped_assets,
            "catalog_controls_total": len(control_catalog),
            "min_controls_per_asset": min_controls_per_asset,
            "max_controls_per_asset": max_controls_per_asset,
            "controls_constraint": controls_constraint,
            "last_prompt_preview": last_prompt_text[:500],
            "last_response_preview": last_response_text[:500],
        }

        logger.info(
            "Control recommendations generated successfully",
            extra={
                "assets_count": len(assets),
                "assets_skipped": skipped_assets,
                "total_tokens": aggregated_metrics.total_tokens,
                "cost_usd": aggregated_metrics.cost_usd,
            }
        )

        return results, aggregated_metrics, debug_info
