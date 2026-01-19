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
from workshops_ai.wizard.schemas import ControlsResponse, BatchControlsResponse
from workshops_ai.wizard.prompts import (
    CONTROLS_SYSTEM_PROMPT,
    CONTROLS_HUMAN_PROMPT_TEMPLATE,
    CONTROLS_JSON_SCHEMA,
    BATCH_CONTROLS_JSON_SCHEMA,
)
from workshops_ai.wizard.catalog_loader import load_control_threat_coverage
from workshops_ai.wizard.catalog_provider import get_catalog_provider
from workshops_ai.wizard.prompt_context import PromptContextBuilder
from workshops_ai.wizard.cost_calculator import create_usage_metrics, UsageMetrics
from workshops_ai.wizard.exceptions import (
    AIResponseError,
    AITimeoutError,
    ValidationRetriesExhausted,
    ModelNotAllowedError,
)
from workshops_ai.llm.langfuse_helper import (
    get_langfuse_client,
    get_langfuse_callback_handler,
    get_langchain_prompt_from_langfuse,
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

        self.llm = create_wizard_llm_client(
            model=model,
            response_format=ControlsResponse
        )
        self.model_name = self.llm.model_name

        logger.info(
            "ControlGenerator initialized",
            extra={
                "model": self.model_name,
                "temperature": settings.llm.temperature,
                "timeout_sec": settings.wizard.llm_timeout_sec,
            }
        )

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
        session_id: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, UsageMetrics]:
        """
        Call LLM with automatic retry on validation errors.

        Args:
            prompt: LangChain ChatPromptTemplate
            session_id: Optional session ID for Langfuse tracing
            **kwargs: Prompt variables

        Returns:
            Tuple of (response text, usage metrics)

        Raises:
            ValidationError: After retries exhausted
            AITimeoutError: If request times out
            AIResponseError: If response cannot be parsed
        """
        # Get Langfuse callback handler for tracing
        langfuse_handler = get_langfuse_callback_handler()
        langfuse_client = get_langfuse_client()

        try:
            # Format messages
            messages = prompt.format_messages(**kwargs)

            # Prepare invocation kwargs with callback handler
            invoke_kwargs = {}
            if langfuse_handler:
                invoke_kwargs["config"] = {"callbacks": [langfuse_handler]}

            # LLM invocation with optional Langfuse span
            if langfuse_client:
                with langfuse_client.start_as_current_span(
                    name="llm-invoke",
                    input={
                        "model": self.model_name,
                        "messages_count": len(messages),
                        "temperature": self.settings.llm.temperature,
                        "timeout_sec": self.settings.wizard.llm_timeout_sec,
                        "attempt_number": getattr(self, '_current_retry_attempt', 1),
                    },
                    metadata={
                        "max_attempts": 3,  # From @retry decorator
                    }
                ) as llm_span:
                    try:
                        # Invoke LLM with callback handler (timeout already configured in ChatOpenAI constructor)
                        response = await self.llm.ainvoke(messages, **invoke_kwargs)

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

                        # Update span with output
                        llm_span.update(
                            output={
                                "response_length": len(response_text),
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": metrics.total_tokens,
                                "cost_usd": metrics.cost_usd,
                            }
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
                        llm_span.update(level="ERROR", status_message=str(e))
                        raise
            else:
                # Fallback without Langfuse tracing (but still use callback handler if available)
                response = await self.llm.ainvoke(messages, **invoke_kwargs)

                response_text = response.content
                usage = response.response_metadata.get("token_usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

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
        relevant_controls: List[Dict[str, Any]],
    ) -> List[PlannedControl]:
        """
        Parse JSON response and validate control recommendations.
        Auto-populates threats and category from catalog for each control.

        Args:
            response_text: Raw LLM response (expects only control names)
            allowed_control_names: List of allowed control names from pre-filtered list
            asset_risk_threats: List of threat names from asset risks
            relevant_controls: Pre-filtered controls with catalog_coverage mapping

        Returns:
            List of validated PlannedControl objects with auto-populated threats

        Raises:
            AIResponseError: If JSON cannot be parsed
        """
        try:
            # With structured outputs, response is guaranteed valid JSON
            response_data = json.loads(response_text.strip())

            # Extract controls array
            if not isinstance(response_data, dict) or "controls" not in response_data:
                raise AIResponseError(
                    "Response missing 'controls' key",
                    model=self.model_name,
                    details={"response_preview": response_text[:200]}
                )

            controls_data = response_data["controls"]

            if not isinstance(controls_data, list):
                raise AIResponseError(
                    "Controls field is not an array",
                    model=self.model_name
                )

            # Build lookup dict from relevant_controls for fast access
            controls_lookup = {ctrl["name"]: ctrl for ctrl in relevant_controls}

            # Validate each control
            validated: List[PlannedControl] = []
            allowed_names_set = set(allowed_control_names)

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

                    # Get catalog data for this control
                    catalog_control = controls_lookup.get(name)
                    if not catalog_control:
                        logger.warning(
                            f"Control '{name}' not found in relevant_controls catalog, skipping"
                        )
                        continue

                    # Get category from catalog (fallback to LLM if provided, but catalog is preferred)
                    category = ctrl_dict.get("category") or catalog_control.get("category")
                    description = ctrl_dict.get("description") or catalog_control.get("description")

                    # Parse costs from LLM response if provided (handle both float and Decimal)
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

                    # Auto-populate threats from catalog_coverage (not from LLM response)
                    catalog_coverage = catalog_control.get("catalog_coverage", {})
                    threats: List[ThreatCoverage] = []

                    for threat_name, coverage_pct in catalog_coverage.items():
                        try:
                            # Convert coverage to Decimal
                            risk_coverage = Decimal(str(coverage_pct))
                            # Clamp coverage to 0-100
                            risk_coverage = max(Decimal("0"), min(Decimal("100"), risk_coverage))

                            threats.append(ThreatCoverage(
                                threatName=threat_name,
                                riskCoverage=risk_coverage
                            ))
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Invalid coverage value for threat '{threat_name}' in control '{name}': {e}"
                            )
                            continue

                    # Create PlannedControl (threats can be empty, it's now optional)
                    validated.append(PlannedControl(
                        id=None,
                        name=name,
                        description=description,
                        category=category,
                        implementationCost=impl_cost,
                        estimatedAnnualCost=annual_cost,
                        threats=threats if threats else None
                    ))

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

    async def _process_assets_batch(
        self,
        assets: List[Asset],
        prompt: ChatPromptTemplate,
        asset_risks_dict: Dict[str, AssetRisk],
        control_catalog: List[PlannedControl],
        control_threat_coverage: Dict[str, Dict[str, int]],
        controls_constraint: str,
        language: str,
        sections: Dict[str, Any],
        max_controls_per_asset: Optional[int],
        session_id: Optional[str],
    ) -> Tuple[List[AssetControls], int, int, str]:
        """
        Process multiple assets in a single batch LLM call using OpenAI Structured Outputs.

        Args:
            assets: List of assets to process in this batch
            prompt: ChatPromptTemplate to use
            asset_risks_dict: Asset ID/name to AssetRisk lookup
            control_catalog: Full control catalog
            control_threat_coverage: Control -> {Threat -> Coverage %} mapping
            controls_constraint: Controls count constraint text
            language: Output language
            sections: Prompt sections dict
            max_controls_per_asset: Max controls per asset (optional)
            session_id: Session ID for Langfuse tracing

        Returns:
            Tuple of (asset_controls_list, prompt_tokens, completion_tokens, response_text)

        Raises:
            AITimeoutError: If LLM request times out
            AIResponseError: If response cannot be parsed or validated
        """
        from workshops_ai.llm.wizard_client import create_wizard_llm_client

        logger.info(
            f"Processing batch of {len(assets)} assets for controls",
            extra={"batch_size": len(assets)}
        )

        # Create batch LLM client with BatchControlsResponse format
        batch_llm = create_wizard_llm_client(
            model=self.model_name,
            response_format=BatchControlsResponse
        )

        # Build batch data structures
        batch_assets_json_list = []
        batch_asset_risks_json_list = []
        batch_relevant_controls_json_list = []
        skipped_count = 0

        for asset in assets:
            # Find asset risks
            asset_risk = None
            if asset.id:
                asset_risk = asset_risks_dict.get(str(asset.id))
            if not asset_risk:
                asset_risk = asset_risks_dict.get(asset.name)

            if not asset_risk or not asset_risk.risks:
                logger.warning(
                    f"No risks found for asset '{asset.name}' in batch, skipping"
                )
                skipped_count += 1
                continue

            # Check if asset has any applicable threats (probability > 0)
            applicable_threats = [
                risk for risk in asset_risk.risks
                if risk.probability > 0
            ]

            if not applicable_threats:
                logger.warning(
                    f"No applicable threats for asset '{asset.name}' in batch, skipping"
                )
                skipped_count += 1
                continue

            # Pre-filter relevant controls
            relevant_controls = self._filter_relevant_controls(
                asset_risks=asset_risk,
                control_catalog=control_catalog,
                threat_coverage=control_threat_coverage,
            )

            if not relevant_controls:
                logger.warning(
                    f"No relevant controls found for asset '{asset.name}' in batch, skipping"
                )
                skipped_count += 1
                continue

            # Add to batch
            batch_assets_json_list.append(asset.model_dump())
            batch_asset_risks_json_list.append({
                "assetName": asset.name,
                "risks": [
                    {"threatName": r.threatName, "probability": r.probability}
                    for r in applicable_threats
                ]
            })
            batch_relevant_controls_json_list.append({
                "assetName": asset.name,
                "controls": relevant_controls
            })

        if skipped_count == len(assets):
            logger.warning("All assets in batch skipped, returning empty results")
            return [], 0, 0, ""

        logger.info(
            f"Prepared batch with {len(batch_assets_json_list)} assets ({skipped_count} skipped)",
            extra={
                "batch_size": len(batch_assets_json_list),
                "skipped": skipped_count,
            }
        )

        # Create JSON for batch
        assets_json = json.dumps(batch_assets_json_list, ensure_ascii=False)
        asset_risks_json = json.dumps(batch_asset_risks_json_list, ensure_ascii=False)
        relevant_controls_json = json.dumps(batch_relevant_controls_json_list, ensure_ascii=False)

        # Format messages with batch schema
        messages = prompt.format_messages(
            assets_json=assets_json,
            asset_risks_json=asset_risks_json,
            relevant_controls_json=relevant_controls_json,
            controls_constraint=controls_constraint,
            controls_json_schema=BATCH_CONTROLS_JSON_SCHEMA,  # Use batch schema
            language=language,
            **sections
        )

        # Prepare invocation kwargs with Langfuse tracing
        invoke_kwargs = {}
        langfuse_handler = get_langfuse_callback_handler()
        if langfuse_handler:
            invoke_kwargs["config"] = {"callbacks": [langfuse_handler]}

        try:
            # Call LLM with batch format
            logger.debug(
                "Calling LLM for batch controls processing",
                extra={
                    "batch_size": len(batch_assets_json_list),
                }
            )

            response = await batch_llm.ainvoke(messages, **invoke_kwargs)
            response_text = response.content

            # Extract usage from response metadata
            usage = response.response_metadata.get("token_usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            logger.info(
                "Batch LLM call successful",
                extra={
                    "batch_size": len(batch_assets_json_list),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )

        except Exception as e:
            # Check for timeout
            error_msg = str(e).lower()
            if "timeout" in error_msg or "timed out" in error_msg:
                logger.error(
                    "Batch LLM timeout",
                    extra={
                        "batch_size": len(batch_assets_json_list),
                        "timeout_sec": self.settings.wizard.llm_timeout_sec,
                        "error": str(e),
                    }
                )
                raise AITimeoutError(
                    timeout_sec=self.settings.wizard.llm_timeout_sec,
                    model=self.model_name
                ) from e

            logger.error(
                "Batch LLM call failed",
                extra={
                    "batch_size": len(batch_assets_json_list),
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            raise

        # Parse response using Pydantic model (already validated by OpenAI Structured Outputs)
        try:
            # Extract JSON from response
            response_text_clean = response_text.strip()

            # OpenAI Structured Outputs should return valid JSON directly
            response_data = json.loads(response_text_clean)

            # Validate with Pydantic
            batch_response = BatchControlsResponse.model_validate(response_data)
            asset_controls_data = batch_response.asset_controls

            logger.info(
                "Batch response parsed successfully",
                extra={
                    "requested_assets": len(batch_assets_json_list),
                    "returned_assets": len(asset_controls_data),
                }
            )

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse batch response JSON",
                extra={
                    "error": str(e),
                    "response_preview": response_text[:200],
                }
            )
            raise AIResponseError(
                f"Failed to parse batch controls response: {str(e)}",
                model=self.model_name,
                details={"response_preview": response_text[:200]}
            ) from e
        except ValidationError as e:
            logger.error(
                "Batch response validation failed",
                extra={
                    "error": str(e),
                    "response_preview": response_text[:200],
                }
            )
            raise AIResponseError(
                f"Batch controls response validation failed: {str(e)}",
                model=self.model_name,
                details={"validation_errors": e.errors()}
            ) from e

        # Validate asset count (WARNING if mismatch, not error)
        if len(asset_controls_data) != len(batch_assets_json_list):
            warning_msg = f"Batch response has {len(asset_controls_data)} assets, expected {len(batch_assets_json_list)} - continuing with partial results"
            logger.warning(
                "batch_controls_count_mismatch",
                extra={
                    "expected": len(batch_assets_json_list),
                    "received": len(asset_controls_data),
                }
            )

        # Apply max_controls_per_asset limit if specified
        if max_controls_per_asset is not None:
            for asset_control in asset_controls_data:
                asset_control.controls = asset_control.controls[:max_controls_per_asset]

        # Enrich controls with catalog data (description, category, threats)
        # Build catalog lookup dict for fast access
        catalog_lookup = {ctrl.name: ctrl for ctrl in control_catalog}

        for asset_control in asset_controls_data:
            enriched_controls = []

            # Get asset risks for this asset
            asset_risk = asset_risks_dict.get(str(asset_control.assetId)) if asset_control.assetId else None
            if not asset_risk:
                asset_risk = asset_risks_dict.get(asset_control.assetName)

            for ctrl in asset_control.controls:
                # Find control in catalog
                catalog_ctrl = catalog_lookup.get(ctrl.name)
                if not catalog_ctrl:
                    logger.warning(f"Control '{ctrl.name}' not found in catalog, using raw LLM data")
                    enriched_controls.append(ctrl)
                    continue

                # Get threat coverage from control_threat_coverage mapping
                coverage_mapping = control_threat_coverage.get(ctrl.name, {})
                threats: List[ThreatCoverage] = []

                if asset_risk:
                    # Only include threats that are relevant to this asset
                    asset_threat_names = {risk.threatName for risk in asset_risk.risks if risk.probability > 0}

                    for threat_name, coverage_pct in coverage_mapping.items():
                        # Only add threats that are applicable to this asset
                        if threat_name in asset_threat_names:
                            try:
                                risk_coverage = Decimal(str(coverage_pct))
                                risk_coverage = max(Decimal("0"), min(Decimal("100"), risk_coverage))
                                threats.append(ThreatCoverage(
                                    threatName=threat_name,
                                    riskCoverage=risk_coverage
                                ))
                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    f"Invalid coverage value for threat '{threat_name}' in control '{ctrl.name}': {e}"
                                )
                                continue

                # Enrich with catalog data and computed threats
                enriched_ctrl = PlannedControl(
                    id=catalog_ctrl.id,
                    name=ctrl.name,
                    description=catalog_ctrl.description,
                    category=catalog_ctrl.category,
                    implementationCost=catalog_ctrl.implementationCost,
                    estimatedAnnualCost=catalog_ctrl.estimatedAnnualCost,
                    threats=threats if threats else None  # Use threats from coverage mapping
                )
                enriched_controls.append(enriched_ctrl)

            # Replace with enriched controls
            asset_control.controls = enriched_controls

        logger.info(
            "Batch controls processing completed successfully",
            extra={
                "batch_size": len(batch_assets_json_list),
                "results_count": len(asset_controls_data),
                "total_controls": sum(len(ac.controls) for ac in asset_controls_data),
            }
        )

        return asset_controls_data, prompt_tokens, completion_tokens, response_text

    async def _process_batch_with_recursive_splitting(
        self,
        assets: List[Asset],
        prompt: ChatPromptTemplate,
        asset_risks_dict: Dict[str, AssetRisk],
        control_catalog: List[PlannedControl],
        control_threat_coverage: Dict[str, Dict[str, int]],
        controls_constraint: str,
        language: str,
        sections: Dict[str, Any],
        max_controls_per_asset: Optional[int],
        session_id: Optional[str],
        recursion_level: int = 0,
        max_recursion_level: int = 3,
    ) -> Tuple[List[AssetControls], int, int, str]:
        """
        Process batch with recursive splitting on context/limit errors.

        Strategy:
        - Try processing full batch
        - On context/limit error: split batch in half and recursively process each half
        - Max 3 levels of recursive splitting (batch → half → quarter → sequential)
        - Only fall back to sequential processing after max recursion level reached

        Args:
            assets: Assets to process in this batch
            prompt: Chat prompt template
            asset_risks_dict: Lookup dict for asset risks
            control_catalog: Full control catalog
            control_threat_coverage: Control to threat coverage mapping
            controls_constraint: Controls constraint text
            language: Output language
            sections: Prompt sections
            max_controls_per_asset: Max controls per asset
            session_id: Session ID for tracing
            recursion_level: Current recursion depth (0 = first attempt)
            max_recursion_level: Maximum recursion depth before sequential fallback

        Returns:
            Tuple of (results list, total prompt tokens, total completion tokens, last response text)
        """
        langfuse_client = get_langfuse_client()

        try:
            # Try batch processing
            batch_results, prompt_tokens, completion_tokens, response_text = await self._process_assets_batch(
                assets=assets,
                prompt=prompt,
                asset_risks_dict=asset_risks_dict,
                control_catalog=control_catalog,
                control_threat_coverage=control_threat_coverage,
                controls_constraint=controls_constraint,
                language=language,
                sections=sections,
                max_controls_per_asset=max_controls_per_asset,
                session_id=session_id,
            )

            return batch_results, prompt_tokens, completion_tokens, response_text

        except AITimeoutError as e:
            # Timeout - check if we can split or need sequential fallback
            if recursion_level < max_recursion_level and len(assets) > 1:
                # Split batch in half
                mid = len(assets) // 2
                first_half = assets[:mid]
                second_half = assets[mid:]

                logger.warning(
                    f"Batch timeout at recursion level {recursion_level}, splitting {len(assets)} assets into {len(first_half)} + {len(second_half)}",
                    extra={
                        "recursion_level": recursion_level,
                        "total_assets": len(assets),
                        "first_half_size": len(first_half),
                        "second_half_size": len(second_half),
                        "reason": "timeout",
                    }
                )

                # Recursively process each half
                results1, tokens1_p, tokens1_c, text1 = await self._process_batch_with_recursive_splitting(
                    assets=first_half,
                    prompt=prompt,
                    asset_risks_dict=asset_risks_dict,
                    control_catalog=control_catalog,
                    control_threat_coverage=control_threat_coverage,
                    controls_constraint=controls_constraint,
                    language=language,
                    sections=sections,
                    max_controls_per_asset=max_controls_per_asset,
                    session_id=session_id,
                    recursion_level=recursion_level + 1,
                    max_recursion_level=max_recursion_level,
                )

                results2, tokens2_p, tokens2_c, text2 = await self._process_batch_with_recursive_splitting(
                    assets=second_half,
                    prompt=prompt,
                    asset_risks_dict=asset_risks_dict,
                    control_catalog=control_catalog,
                    control_threat_coverage=control_threat_coverage,
                    controls_constraint=controls_constraint,
                    language=language,
                    sections=sections,
                    max_controls_per_asset=max_controls_per_asset,
                    session_id=session_id,
                    recursion_level=recursion_level + 1,
                    max_recursion_level=max_recursion_level,
                )

                # Combine results
                combined_results = results1 + results2
                combined_prompt_tokens = tokens1_p + tokens2_p
                combined_completion_tokens = tokens1_c + tokens2_c
                last_text = text2  # Use last response text

                return combined_results, combined_prompt_tokens, combined_completion_tokens, last_text
            else:
                # Max recursion level reached or single asset - fallback to sequential
                logger.warning(
                    f"Max recursion level reached or single asset, falling back to sequential processing",
                    extra={
                        "recursion_level": recursion_level,
                        "max_recursion_level": max_recursion_level,
                        "assets_count": len(assets),
                        "reason": "max_recursion_or_single_asset",
                    }
                )
                # Re-raise to trigger sequential fallback in parent
                raise

        except AIResponseError as e:
            # Check if error is context/limit related
            error_msg = str(e).lower()

            if any(keyword in error_msg for keyword in ["context", "limit", "token"]):
                # Context/limit error - check if we can split
                if recursion_level < max_recursion_level and len(assets) > 1:
                    # Split batch in half
                    mid = len(assets) // 2
                    first_half = assets[:mid]
                    second_half = assets[mid:]

                    logger.warning(
                        f"Batch context/limit error at recursion level {recursion_level}, splitting {len(assets)} assets into {len(first_half)} + {len(second_half)}",
                        extra={
                            "recursion_level": recursion_level,
                            "total_assets": len(assets),
                            "first_half_size": len(first_half),
                            "second_half_size": len(second_half),
                            "error": str(e),
                            "reason": "context_limit",
                        }
                    )

                    # Recursively process each half
                    results1, tokens1_p, tokens1_c, text1 = await self._process_batch_with_recursive_splitting(
                        assets=first_half,
                        prompt=prompt,
                        asset_risks_dict=asset_risks_dict,
                        control_catalog=control_catalog,
                        control_threat_coverage=control_threat_coverage,
                        controls_constraint=controls_constraint,
                        language=language,
                        sections=sections,
                        max_controls_per_asset=max_controls_per_asset,
                        session_id=session_id,
                        recursion_level=recursion_level + 1,
                        max_recursion_level=max_recursion_level,
                    )

                    results2, tokens2_p, tokens2_c, text2 = await self._process_batch_with_recursive_splitting(
                        assets=second_half,
                        prompt=prompt,
                        asset_risks_dict=asset_risks_dict,
                        control_catalog=control_catalog,
                        control_threat_coverage=control_threat_coverage,
                        controls_constraint=controls_constraint,
                        language=language,
                        sections=sections,
                        max_controls_per_asset=max_controls_per_asset,
                        session_id=session_id,
                        recursion_level=recursion_level + 1,
                        max_recursion_level=max_recursion_level,
                    )

                    # Combine results
                    combined_results = results1 + results2
                    combined_prompt_tokens = tokens1_p + tokens2_p
                    combined_completion_tokens = tokens1_c + tokens2_c
                    last_text = text2  # Use last response text

                    return combined_results, combined_prompt_tokens, combined_completion_tokens, last_text
                else:
                    # Max recursion level reached or single asset - fallback to sequential
                    logger.warning(
                        f"Max recursion level reached or single asset, falling back to sequential processing",
                        extra={
                            "recursion_level": recursion_level,
                            "max_recursion_level": max_recursion_level,
                            "assets_count": len(assets),
                            "error": str(e),
                            "reason": "max_recursion_or_single_asset",
                        }
                    )
                    # Re-raise to trigger sequential fallback in parent
                    raise
            else:
                # Non-recoverable error - re-raise immediately
                logger.error(
                    f"Non-recoverable error in batch processing",
                    extra={
                        "recursion_level": recursion_level,
                        "assets_count": len(assets),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                raise

    async def generate_asset_controls(
        self,
        assets: List[Asset],
        asset_risks: List[AssetRisk],
        language: str = "en",
        min_controls_per_asset: Optional[int] = None,
        max_controls_per_asset: Optional[int] = None,
        website_content: str = "",
        user_instructions: Optional[str] = None,
        session_id: Optional[str] = None,
        include_catalogs: Optional[List[str]] = None,
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
            session_id: Optional session ID for Langfuse tracing
            include_catalogs: Optional list of catalogs to include in prompt (e.g., ["sectors", "impacts", "threats"])

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

        # Initialize prompt context builder with session-based caching
        builder = PromptContextBuilder(session_id=session_id)

        # Load control catalog (already localized by Tazilla backend if using API)
        provider = get_catalog_provider()
        control_catalog = await provider.load_control_catalog(session_id=session_id)

        # Load control→threat coverage mapping (from old catalog_loader for now)
        threat_coverage = load_control_threat_coverage()

        # Build asset-risk lookup
        risk_lookup: Dict[str, AssetRisk] = {}
        for ar in asset_risks:
            # Try matching by assetId first, fallback to assetName
            if ar.assetId:
                risk_lookup[str(ar.assetId)] = ar
            risk_lookup[ar.assetName] = ar

        # Build shared prompt sections using centralized builder
        sections = await builder.build_sections(
            website_content=website_content,
            user_instructions=user_instructions,
            include_catalogs=include_catalogs,  # Optional catalogs to include in prompt
        )

        results: List[AssetControls] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        last_response_text = ""
        last_prompt_text = ""
        skipped_assets = 0

        # Get Langfuse client for tracing
        langfuse_client = get_langfuse_client()

        # Prepare span input
        span_input = {
            "assets_count": len(assets),
            "language": language,
            "min_controls_per_asset": min_controls_per_asset,
            "max_controls_per_asset": max_controls_per_asset,
        }

        # Main span: generate-controls
        if langfuse_client and session_id:
            from langfuse import propagate_attributes

            with langfuse_client.start_as_current_span(
                name="generate-controls",
                input=span_input,
                metadata={"endpoint": "controls", "model": self.model_name},
            ) as main_span:
                with propagate_attributes(session_id=session_id):
                    # Fetch prompt with child span
                    requested_label = self.settings.langfuse.prompt_label
                    with langfuse_client.start_as_current_span(
                        name="get-controls-prompts",
                        input={
                            "prompt_name": "wizard/controls",
                            "requested_label": requested_label,
                        },
                    ) as prompt_span:
                        # Try to fetch unified prompt from Langfuse
                        prompt = get_langchain_prompt_from_langfuse(
                            prompt_name="wizard/controls",
                            fallback_template=None,
                            fallback_variables=None,
                        )

                        prompt_source = "langfuse"
                        prompt_version = None
                        prompt_type = "chat"
                        prompt_tags = []
                        prompt_labels = []

                        # If Langfuse fetch failed, build fallback from local templates
                        if prompt is None:
                            logger.info("Building fallback prompt from local templates")
                            prompt = ChatPromptTemplate.from_messages([
                                ("system", CONTROLS_SYSTEM_PROMPT),
                                ("human", CONTROLS_HUMAN_PROMPT_TEMPLATE),
                            ])
                            prompt_source = "local"
                        else:
                            # Extract details from metadata
                            if hasattr(prompt, 'metadata') and prompt.metadata:
                                langfuse_prompt_obj = prompt.metadata.get('langfuse_prompt')
                                if langfuse_prompt_obj:
                                    if hasattr(langfuse_prompt_obj, 'version'):
                                        prompt_version = langfuse_prompt_obj.version
                                    if hasattr(langfuse_prompt_obj, 'tags'):
                                        prompt_tags = langfuse_prompt_obj.tags or []
                                    if hasattr(langfuse_prompt_obj, 'labels'):
                                        prompt_labels = langfuse_prompt_obj.labels or []
                                    if hasattr(langfuse_prompt_obj, 'type'):
                                        prompt_type = langfuse_prompt_obj.type

                        # Extract prompt messages/content
                        prompt_content = []
                        if hasattr(prompt, 'messages'):
                            for msg in prompt.messages:
                                if hasattr(msg, 'prompt') and hasattr(msg.prompt, 'template'):
                                    class_name = msg.__class__.__name__
                                    if 'System' in class_name:
                                        role = 'system'
                                    elif 'Human' in class_name:
                                        role = 'user'
                                    elif 'AI' in class_name or 'Assistant' in class_name:
                                        role = 'assistant'
                                    else:
                                        role = 'unknown'

                                    prompt_content.append({
                                        "role": role,
                                        "content": msg.prompt.template
                                    })

                        prompt_span.update(
                            output={
                                "name": "wizard/controls",
                                "isFallback": prompt_source == "local",
                                "version": prompt_version,
                                "type": prompt_type,
                                "tags": prompt_tags,
                                "labels": prompt_labels,
                                "prompt": prompt_content,
                            }
                        )

                    # Split assets into batches based on max_batch_size setting
                    batch_size = self.settings.wizard.max_batch_size
                    total_batches = (len(assets) + batch_size - 1) // batch_size  # Ceiling division

                    logger.info(
                        f"Processing {len(assets)} assets in {total_batches} batch(es) of up to {batch_size} assets",
                        extra={
                            "total_assets": len(assets),
                            "batch_size": batch_size,
                            "total_batches": total_batches,
                        }
                    )

                    # Process batches
                    for batch_idx in range(total_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, len(assets))
                        batch_assets = assets[start_idx:end_idx]

                        # Child span: evaluate-batch-controls
                        with langfuse_client.start_as_current_span(
                            name="evaluate-batch-controls",
                            input={
                                "batch_index": batch_idx + 1,
                                "total_batches": total_batches,
                                "batch_size": len(batch_assets),
                                "asset_names": [a.name for a in batch_assets],
                            },
                            metadata={"catalog_controls_count": len(control_catalog)},
                        ) as batch_span:
                            try:
                                # Process batch with recursive splitting on context/limit errors
                                batch_results, batch_prompt_tokens, batch_completion_tokens, batch_response_text = await self._process_batch_with_recursive_splitting(
                                    assets=batch_assets,
                                    prompt=prompt,
                                    asset_risks_dict=risk_lookup,
                                    control_catalog=control_catalog,
                                    control_threat_coverage=threat_coverage,
                                    controls_constraint=controls_constraint,
                                    language=language,
                                    sections=sections,
                                    max_controls_per_asset=max_controls_per_asset,
                                    session_id=session_id,
                                    recursion_level=0,
                                    max_recursion_level=3,
                                )

                                # Accumulate results and usage
                                results.extend(batch_results)
                                total_prompt_tokens += batch_prompt_tokens
                                total_completion_tokens += batch_completion_tokens
                                last_response_text = batch_response_text

                                # Update batch span with success
                                batch_span.update(
                                    output={
                                        "assets_processed": len(batch_results),
                                        "total_controls_generated": sum(len(r.controls) for r in batch_results),
                                        "controls_preview": [
                                            {
                                                "asset": r.assetName,
                                                "controls_count": len(r.controls)
                                            } for r in batch_results[:5]
                                        ],
                                    }
                                )

                                logger.info(
                                    f"Batch {batch_idx + 1}/{total_batches} completed successfully",
                                    extra={
                                        "batch_index": batch_idx + 1,
                                        "assets_processed": len(batch_results),
                                        "total_controls": sum(len(r.controls) for r in batch_results),
                                    }
                                )

                            except AITimeoutError as e:
                                # Timeout = legitimate failure, fallback is appropriate
                                batch_span.update(
                                    level="WARNING",
                                    status_message=f"Batch timeout, falling back to sequential: {str(e)}",
                                )

                                logger.warning(
                                    f"Batch {batch_idx + 1}/{total_batches} timeout, falling back to sequential processing",
                                    extra={
                                        "batch_index": batch_idx + 1,
                                        "batch_size": len(batch_assets),
                                        "error": str(e),
                                        "fallback_reason": "timeout",
                                    }
                                )

                                # Fallback: Process assets sequentially
                                for asset in batch_assets:
                                    # Child span: evaluate-asset-controls (fallback)
                                    with langfuse_client.start_as_current_span(
                                        name="evaluate-asset-controls",
                                        input={
                                            "asset_id": str(asset.id) if asset.id else None,
                                            "asset_name": asset.name,
                                            "asset_type": asset.assetType if hasattr(asset, 'assetType') else None,
                                        },
                                    ) as asset_span:
                                        try:
                                            # Find asset risks
                                            asset_risk = None
                                            if asset.id:
                                                asset_risk = risk_lookup.get(str(asset.id))
                                            if not asset_risk:
                                                asset_risk = risk_lookup.get(asset.name)

                                            if not asset_risk:
                                                logger.warning(
                                                    f"No risks found for asset '{asset.name}', skipping control selection (fallback)"
                                                )
                                                results.append(AssetControls(
                                                    assetId=asset.id,
                                                    assetName=asset.name,
                                                    controls=[]
                                                ))
                                                skipped_assets += 1
                                                asset_span.update(
                                                    output={
                                                        "status": "skipped",
                                                        "reason": "no_risks_found",
                                                        "controls_count": 0,
                                                    }
                                                )
                                                continue

                                            # Check if asset has any applicable threats (probability > 0)
                                            applicable_threats = [
                                                risk for risk in asset_risk.risks
                                                if risk.probability > 0
                                            ]

                                            if not applicable_threats:
                                                logger.warning(
                                                    f"No applicable threats for asset '{asset.name}' (fallback)"
                                                )
                                                results.append(AssetControls(
                                                    assetId=asset.id,
                                                    assetName=asset.name,
                                                    controls=[]
                                                ))
                                                skipped_assets += 1
                                                asset_span.update(
                                                    output={
                                                        "status": "skipped",
                                                        "reason": "no_applicable_threats",
                                                        "controls_count": 0,
                                                    }
                                                )
                                                continue

                                            # Pre-filter relevant controls
                                            relevant_controls = self._filter_relevant_controls(
                                                asset_risks=asset_risk,
                                                control_catalog=control_catalog,
                                                threat_coverage=threat_coverage,
                                            )

                                            if not relevant_controls:
                                                logger.warning(
                                                    f"No relevant controls found for asset '{asset.name}' (fallback)"
                                                )
                                                results.append(AssetControls(
                                                    assetId=asset.id,
                                                    assetName=asset.name,
                                                    controls=[]
                                                ))
                                                skipped_assets += 1
                                                asset_span.update(
                                                    output={
                                                        "status": "skipped",
                                                        "reason": "no_relevant_controls",
                                                        "controls_count": 0,
                                                    }
                                                )
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
                                                controls_json_schema=CONTROLS_JSON_SCHEMA,
                                                language=language,
                                                **sections
                                            )
                                            last_prompt_text = str(messages)

                                            # Call LLM with retry logic
                                            response_text, usage_metrics = await self._call_llm_with_retry(
                                                prompt=prompt,
                                                session_id=session_id,
                                                asset_json=asset_json,
                                                asset_risks_json=asset_risks_json,
                                                relevant_controls_json=relevant_controls_json,
                                                controls_constraint=controls_constraint,
                                                controls_json_schema=CONTROLS_JSON_SCHEMA,
                                                language=language,
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
                                                relevant_controls=relevant_controls,
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

                                            # Update span with success
                                            asset_span.update(
                                                output={
                                                    "status": "success",
                                                    "controls_count": len(controls),
                                                    "controls_preview": [{"name": c.name} for c in controls[:5]],
                                                }
                                            )

                                            logger.info(
                                                f"Control selection completed for asset '{asset.name}' (fallback)",
                                                extra={
                                                    "asset_id": str(asset.id) if asset.id else None,
                                                    "controls_count": len(controls),
                                                }
                                            )

                                        except AITimeoutError:
                                            # Update span with error
                                            asset_span.update(
                                                level="ERROR",
                                                status_message="LLM timeout",
                                            )
                                            # Re-raise timeout errors immediately
                                            raise
                                        except (AIResponseError, ValidationRetriesExhausted):
                                            # Update span with error
                                            asset_span.update(
                                                level="ERROR",
                                                status_message="AI response or validation error",
                                            )
                                            # Re-raise AI-specific errors immediately
                                            raise
                                        except Exception as e:
                                            # Update span with error
                                            asset_span.update(
                                                level="ERROR",
                                                status_message=str(e),
                                            )

                                            logger.error(
                                                f"Control selection failed for asset '{asset.name}' (fallback)",
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

                            except AIResponseError as e:
                                # Batch response error - decide based on error type
                                error_msg = str(e).lower()

                                # Legitimate failures that warrant fallback
                                if any(keyword in error_msg for keyword in ["timeout", "context", "limit", "token"]):
                                    batch_span.update(
                                        level="WARNING",
                                        status_message=f"Batch context/limit issue, falling back to sequential: {str(e)}",
                                    )

                                    logger.warning(
                                        f"Batch {batch_idx + 1}/{total_batches} hit context/limit, falling back to sequential",
                                        extra={
                                            "batch_index": batch_idx + 1,
                                            "batch_size": len(batch_assets),
                                            "error": str(e),
                                            "fallback_reason": "context_limit",
                                        }
                                    )

                                    # Fallback: Process assets sequentially (same code as timeout fallback above)
                                    for asset in batch_assets:
                                        # Child span: evaluate-asset-controls (fallback)
                                        with langfuse_client.start_as_current_span(
                                            name="evaluate-asset-controls",
                                            input={
                                                "asset_id": str(asset.id) if asset.id else None,
                                                "asset_name": asset.name,
                                                "asset_type": asset.assetType if hasattr(asset, 'assetType') else None,
                                            },
                                        ) as asset_span:
                                            try:
                                                # Find asset risks
                                                asset_risk = None
                                                if asset.id:
                                                    asset_risk = risk_lookup.get(str(asset.id))
                                                if not asset_risk:
                                                    asset_risk = risk_lookup.get(asset.name)

                                                if not asset_risk:
                                                    logger.warning(
                                                        f"No risks found for asset '{asset.name}' (fallback context/limit)"
                                                    )
                                                    results.append(AssetControls(
                                                        assetId=asset.id,
                                                        assetName=asset.name,
                                                        controls=[]
                                                    ))
                                                    skipped_assets += 1
                                                    asset_span.update(
                                                        output={"status": "skipped", "reason": "no_risks_found", "controls_count": 0}
                                                    )
                                                    continue

                                                # Check applicable threats
                                                applicable_threats = [
                                                    risk for risk in asset_risk.risks
                                                    if risk.probability > 0
                                                ]

                                                if not applicable_threats:
                                                    logger.warning(
                                                        f"No applicable threats for asset '{asset.name}' (fallback context/limit)"
                                                    )
                                                    results.append(AssetControls(
                                                        assetId=asset.id,
                                                        assetName=asset.name,
                                                        controls=[]
                                                    ))
                                                    skipped_assets += 1
                                                    asset_span.update(
                                                        output={"status": "skipped", "reason": "no_applicable_threats", "controls_count": 0}
                                                    )
                                                    continue

                                                # Pre-filter relevant controls
                                                relevant_controls = self._filter_relevant_controls(
                                                    asset_risks=asset_risk,
                                                    control_catalog=control_catalog,
                                                    threat_coverage=threat_coverage,
                                                )

                                                if not relevant_controls:
                                                    logger.warning(
                                                        f"No relevant controls found for asset '{asset.name}' (fallback context/limit)"
                                                    )
                                                    results.append(AssetControls(
                                                        assetId=asset.id,
                                                        assetName=asset.name,
                                                        controls=[]
                                                    ))
                                                    skipped_assets += 1
                                                    asset_span.update(
                                                        output={"status": "skipped", "reason": "no_relevant_controls", "controls_count": 0}
                                                    )
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
                                                    controls_json_schema=CONTROLS_JSON_SCHEMA,
                                                    language=language,
                                                    **sections
                                                )
                                                last_prompt_text = str(messages)

                                                # Call LLM with retry logic
                                                response_text, usage_metrics = await self._call_llm_with_retry(
                                                    prompt=prompt,
                                                    session_id=session_id,
                                                    asset_json=asset_json,
                                                    asset_risks_json=asset_risks_json,
                                                    relevant_controls_json=relevant_controls_json,
                                                    controls_constraint=controls_constraint,
                                                    controls_json_schema=CONTROLS_JSON_SCHEMA,
                                                    language=language,
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
                                                    relevant_controls=relevant_controls,
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

                                                # Update asset span with success
                                                asset_span.update(
                                                    output={
                                                        "controls_count": len(controls),
                                                        "controls_preview": [{"name": c.name} for c in controls[:5]],
                                                    }
                                                )

                                                logger.info(
                                                    f"Control selection completed for asset '{asset.name}' (fallback context/limit)",
                                                    extra={
                                                        "asset_id": str(asset.id) if asset.id else None,
                                                        "controls_count": len(controls),
                                                    }
                                                )

                                            except AITimeoutError:
                                                # Update span with error
                                                asset_span.update(
                                                    level="ERROR",
                                                    status_message="LLM timeout",
                                                )
                                                # Re-raise timeout errors immediately
                                                raise
                                            except (AIResponseError, ValidationRetriesExhausted):
                                                # Update span with error
                                                asset_span.update(
                                                    level="ERROR",
                                                    status_message="AI response or validation error",
                                                )
                                                # Re-raise AI-specific errors immediately
                                                raise
                                            except Exception as e:
                                                # Update span with error
                                                asset_span.update(
                                                    level="ERROR",
                                                    status_message=str(e),
                                                )

                                                logger.error(
                                                    f"Control selection failed for asset '{asset.name}' (fallback context/limit)",
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

                                else:
                                    # Unexpected error - don't hide bugs, propagate error
                                    batch_span.update(
                                        level="ERROR",
                                        status_message=f"Unexpected batch error (not retrying): {str(e)}",
                                    )

                                    logger.critical(
                                        f"CRITICAL: Batch {batch_idx + 1}/{total_batches} failed with unexpected error - NOT using fallback",
                                        extra={
                                            "batch_index": batch_idx + 1,
                                            "batch_size": len(batch_assets),
                                            "error": str(e),
                                            "error_type": type(e).__name__,
                                            "error_details": getattr(e, 'details', None),
                                            "asset_names": [a.name for a in batch_assets],
                                        }
                                    )

                                    # Re-raise - force investigation, don't hide bugs
                                    raise

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
                        "last_prompt_preview": last_prompt_text[:500] if last_prompt_text else "",
                        "last_response_preview": last_response_text[:500],
                    }

                    # Update main span with final output
                    main_span.update(
                        output={
                            "assets_processed": len(results),
                            "total_controls_generated": sum(len(r.controls) for r in results),
                            "total_tokens": aggregated_metrics.total_tokens,
                            "cost_usd": aggregated_metrics.cost_usd,
                        }
                    )

                    logger.info(
                        "Control recommendations generated successfully",
                        extra={
                            "assets_count": len(assets),
                            "total_tokens": aggregated_metrics.total_tokens,
                            "cost_usd": aggregated_metrics.cost_usd,
                        }
                    )

                    return results, aggregated_metrics, debug_info
        else:
            # Fallback: Process without Langfuse tracing
            # Fetch prompt locally
            logger.info("Processing controls without Langfuse tracing")
            prompt = ChatPromptTemplate.from_messages([
                ("system", CONTROLS_SYSTEM_PROMPT),
                ("human", CONTROLS_HUMAN_PROMPT_TEMPLATE),
            ])

            # Split assets into batches based on max_batch_size setting
            batch_size = self.settings.wizard.max_batch_size
            total_batches = (len(assets) + batch_size - 1) // batch_size

            logger.info(
                f"Processing {len(assets)} assets in {total_batches} batch(es) (no Langfuse)",
                extra={
                    "total_assets": len(assets),
                    "batch_size": batch_size,
                }
            )

            # Process batches
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(assets))
                batch_assets = assets[start_idx:end_idx]

                try:
                    # Process batch
                    batch_results, batch_prompt_tokens, batch_completion_tokens, batch_response_text = await self._process_assets_batch(
                        assets=batch_assets,
                        prompt=prompt,
                        asset_risks_dict=risk_lookup,
                        control_catalog=control_catalog,
                        control_threat_coverage=threat_coverage,
                        controls_constraint=controls_constraint,
                        language=language,
                        sections=sections,
                        max_controls_per_asset=max_controls_per_asset,
                        session_id=None,
                    )

                    # Accumulate results
                    results.extend(batch_results)
                    total_prompt_tokens += batch_prompt_tokens
                    total_completion_tokens += batch_completion_tokens
                    last_response_text = batch_response_text

                    logger.info(
                        f"Batch {batch_idx + 1}/{total_batches} completed (no Langfuse)",
                        extra={
                            "batch_index": batch_idx + 1,
                            "assets_processed": len(batch_results),
                        }
                    )

                except AITimeoutError as e:
                    # Timeout = legitimate failure, fallback to sequential
                    logger.warning(
                        f"Batch {batch_idx + 1}/{total_batches} timed out, falling back to sequential processing (no Langfuse)",
                        extra={
                            "batch_index": batch_idx + 1,
                            "batch_size": len(batch_assets),
                            "error": str(e),
                        }
                    )

                    # Fallback: Sequential processing
                    for asset in batch_assets:
                        try:
                            # Find asset risks
                            asset_risk = None
                            if asset.id:
                                asset_risk = risk_lookup.get(str(asset.id))
                            if not asset_risk:
                                asset_risk = risk_lookup.get(asset.name)

                            if not asset_risk:
                                logger.warning(
                                    f"No risks found for asset '{asset.name}' (fallback, no Langfuse)"
                                )
                                results.append(AssetControls(
                                    assetId=asset.id,
                                    assetName=asset.name,
                                    controls=[]
                                ))
                                skipped_assets += 1
                                continue

                            # Check applicable threats
                            applicable_threats = [
                                risk for risk in asset_risk.risks
                                if risk.probability > 0
                            ]

                            if not applicable_threats:
                                logger.warning(
                                    f"No applicable threats for asset '{asset.name}' (fallback, no Langfuse)"
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
                                    f"No relevant controls found for asset '{asset.name}' (fallback, no Langfuse)"
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
                                controls_json_schema=CONTROLS_JSON_SCHEMA,
                                language=language,
                                **sections
                            )
                            last_prompt_text = str(messages)

                            # Call LLM with retry logic
                            response_text, usage_metrics = await self._call_llm_with_retry(
                                prompt=prompt,
                                session_id=None,
                                asset_json=asset_json,
                                asset_risks_json=asset_risks_json,
                                relevant_controls_json=relevant_controls_json,
                                controls_constraint=controls_constraint,
                                controls_json_schema=CONTROLS_JSON_SCHEMA,
                                language=language,
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
                                relevant_controls=relevant_controls,
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
                                f"Control selection completed for asset '{asset.name}' (fallback, no Langfuse)",
                                extra={
                                    "asset_id": str(asset.id) if asset.id else None,
                                    "controls_count": len(controls),
                                }
                            )

                        except AITimeoutError:
                            raise
                        except (AIResponseError, ValidationRetriesExhausted):
                            raise
                        except Exception as e:
                            logger.error(
                                f"Control selection failed for asset '{asset.name}' (fallback, no Langfuse)",
                                extra={
                                    "asset_id": str(asset.id) if asset.id else None,
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                }
                            )
                            results.append(AssetControls(
                                assetId=asset.id,
                                assetName=asset.name,
                                controls=[]
                            ))

                except AIResponseError as e:
                    # Check if this is a legitimate failure (context/limit) or unexpected error
                    error_msg = str(e).lower()
                    is_legitimate = any(keyword in error_msg for keyword in ["timeout", "context", "limit", "token"])

                    if is_legitimate:
                        # Context/limit error = legitimate failure, fallback to sequential
                        logger.warning(
                            f"Batch {batch_idx + 1}/{total_batches} hit context/limit, falling back to sequential processing (no Langfuse)",
                            extra={
                                "batch_index": batch_idx + 1,
                                "batch_size": len(batch_assets),
                                "error": str(e),
                            }
                        )

                        # Fallback: Sequential processing
                        for asset in batch_assets:
                            try:
                                # Find asset risks
                                asset_risk = None
                                if asset.id:
                                    asset_risk = risk_lookup.get(str(asset.id))
                                if not asset_risk:
                                    asset_risk = risk_lookup.get(asset.name)

                                if not asset_risk:
                                    logger.warning(
                                        f"No risks found for asset '{asset.name}' (fallback, no Langfuse)"
                                    )
                                    results.append(AssetControls(
                                        assetId=asset.id,
                                        assetName=asset.name,
                                        controls=[]
                                    ))
                                    skipped_assets += 1
                                    continue

                                # Check applicable threats
                                applicable_threats = [
                                    risk for risk in asset_risk.risks
                                    if risk.probability > 0
                                ]

                                if not applicable_threats:
                                    logger.warning(
                                        f"No applicable threats for asset '{asset.name}' (fallback, no Langfuse)"
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
                                        f"No relevant controls found for asset '{asset.name}' (fallback, no Langfuse)"
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
                                    controls_json_schema=CONTROLS_JSON_SCHEMA,
                                    language=language,
                                    **sections
                                )
                                last_prompt_text = str(messages)

                                # Call LLM with retry logic
                                response_text, usage_metrics = await self._call_llm_with_retry(
                                    prompt=prompt,
                                    session_id=None,
                                    asset_json=asset_json,
                                    asset_risks_json=asset_risks_json,
                                    relevant_controls_json=relevant_controls_json,
                                    controls_constraint=controls_constraint,
                                    controls_json_schema=CONTROLS_JSON_SCHEMA,
                                    language=language,
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
                                    relevant_controls=relevant_controls,
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
                                    f"Control selection completed for asset '{asset.name}' (fallback, no Langfuse)",
                                    extra={
                                        "asset_id": str(asset.id) if asset.id else None,
                                        "controls_count": len(controls),
                                    }
                                )

                            except AITimeoutError:
                                raise
                            except (AIResponseError, ValidationRetriesExhausted):
                                raise
                            except Exception as e:
                                logger.error(
                                    f"Control selection failed for asset '{asset.name}' (fallback, no Langfuse)",
                                    extra={
                                        "asset_id": str(asset.id) if asset.id else None,
                                        "error": str(e),
                                        "error_type": type(e).__name__,
                                    }
                                )
                                results.append(AssetControls(
                                    assetId=asset.id,
                                    assetName=asset.name,
                                    controls=[]
                                ))

                    else:
                        # Unexpected error - do NOT fallback, force investigation
                        logger.critical(
                            f"CRITICAL: Batch {batch_idx + 1}/{total_batches} failed with UNEXPECTED error (no Langfuse). NOT using fallback to force investigation.",
                            extra={
                                "batch_index": batch_idx + 1,
                                "batch_size": len(batch_assets),
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "asset_ids": [str(a.id) if a.id else None for a in batch_assets],
                            }
                        )
                        # Re-raise to propagate the error and fail fast
                        raise

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
