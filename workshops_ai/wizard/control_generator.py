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
from workshops_ai.wizard.schemas import ControlsResponse
from workshops_ai.wizard.prompts import (
    CONTROLS_SYSTEM_PROMPT,
    CONTROLS_HUMAN_PROMPT_TEMPLATE,
    CONTROLS_JSON_SCHEMA,
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
                    # Process each asset individually
                    for asset in assets:
                        # Child span: evaluate-asset-controls
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
                                        f"No risks found for asset '{asset.name}', skipping control selection"
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
                                        f"No applicable threats for asset '{asset.name}' (all probability 0), skipping"
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
                                        f"No relevant controls found for asset '{asset.name}', skipping"
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
                                    **sections
                                )

                                # Accumulate usage
                                total_prompt_tokens += usage_metrics.prompt_tokens
                                total_completion_tokens += usage_metrics.completion_tokens
                                last_response_text = response_text

                                # Parse, validate and enrich with catalog data
                                allowed_control_names = [c["name"] for c in relevant_controls]
                                asset_threat_names = [r.threatName for r in asset_risk.risks if r.probability > 0]

                                # Add enrichment span (if Langfuse available)
                                if langfuse_client:
                                    with langfuse_client.start_as_current_span(
                                        name="enrich-controls",
                                        input={
                                            "raw_response_preview": response_text[:200],
                                            "allowed_controls_count": len(allowed_control_names),
                                            "asset_threats_count": len(asset_threat_names),
                                        }
                                    ) as enrich_span:
                                        controls = self._parse_and_validate_controls(
                                            response_text=response_text,
                                            allowed_control_names=allowed_control_names,
                                            asset_risk_threats=asset_threat_names,
                                            relevant_controls=relevant_controls,
                                        )

                                        # Apply max limit if specified
                                        if max_controls_per_asset is not None:
                                            controls = controls[:max_controls_per_asset]

                                        # Update enrichment span with output (show actual enriched PlannedControl structure)
                                        enrich_span.update(
                                            output={
                                                "controls_count": len(controls),
                                                "enriched_controls": [c.model_dump() for c in controls[:3]]  # First 3 in full detail
                                            }
                                        )
                                else:
                                    # Without Langfuse, just parse and validate
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
                                    f"Control selection completed for asset '{asset.name}'",
                                    extra={
                                        "asset_id": str(asset.id) if asset.id else None,
                                        "controls_count": len(controls),
                                    }
                                )

                                # Update span with success
                                asset_span.update(
                                    output={
                                        "status": "success",
                                        "controls_count": len(controls),
                                        "controls_preview": [
                                            {
                                                "name": c.name,
                                                "threats": [t.threatName for t in (c.threats or [])][:3],  # First 3 threats
                                            }
                                            for c in controls[:5]  # First 5 controls
                                        ],
                                        "relevant_controls_available": len(relevant_controls),
                                        "threats_count": len(applicable_threats),
                                        "prompt_tokens": usage_metrics.prompt_tokens,
                                        "completion_tokens": usage_metrics.completion_tokens,
                                        "cost_usd": usage_metrics.cost_usd,
                                    }
                                )

                            except AITimeoutError:
                                # Re-raise timeout errors immediately - don't continue with other assets
                                asset_span.update(
                                    output={
                                        "status": "error",
                                        "error_type": "AITimeoutError",
                                    }
                                )
                                raise
                            except (AIResponseError, ValidationRetriesExhausted):
                                # Re-raise AI-specific errors immediately
                                asset_span.update(
                                    output={
                                        "status": "error",
                                        "error_type": type(e).__name__ if 'e' in locals() else "AIError",
                                    }
                                )
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
                                asset_span.update(
                                    output={
                                        "status": "error",
                                        "error": str(e),
                                        "error_type": type(e).__name__,
                                        "controls_count": 0,
                                    }
                                )

                    # Update main span with final results
                    main_span.update(
                        output={
                            "assets_processed": len(assets),
                            "assets_with_controls": len([r for r in results if r.controls]),
                            "assets_skipped": skipped_assets,
                            "total_controls_generated": sum(len(r.controls) for r in results),
                            "total_prompt_tokens": total_prompt_tokens,
                            "total_completion_tokens": total_completion_tokens,
                            "total_tokens": total_prompt_tokens + total_completion_tokens,
                        }
                    )
        else:
            # Fallback: Process without Langfuse tracing
            # Fetch prompt locally
            logger.info("Processing controls without Langfuse tracing")
            prompt = ChatPromptTemplate.from_messages([
                ("system", CONTROLS_SYSTEM_PROMPT),
                ("human", CONTROLS_HUMAN_PROMPT_TEMPLATE),
            ])

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
                        controls_json_schema=CONTROLS_JSON_SCHEMA,
                        **sections
                    )
                    last_prompt_text = str(messages)

                    # Call LLM with retry logic (without session_id for fallback)
                    response_text, usage_metrics = await self._call_llm_with_retry(
                        prompt=prompt,
                        session_id=None,
                        asset_json=asset_json,
                        asset_risks_json=asset_risks_json,
                        relevant_controls_json=relevant_controls_json,
                        controls_constraint=controls_constraint,
                        controls_json_schema=CONTROLS_JSON_SCHEMA,
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
