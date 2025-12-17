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
from workshops_ai.wizard.schemas import Service, RisksResponse
from workshops_ai.wizard.prompts import (
    COMBINED_RISK_SYSTEM_PROMPT,
    COMBINED_RISK_HUMAN_PROMPT_TEMPLATE,
    RISKS_JSON_SCHEMA,
)
from workshops_ai.wizard.catalog_loader import load_asset_type_threat_mapping
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

        self.llm = create_wizard_llm_client(
            model=model,
            response_format=RisksResponse
        )
        self.model_name = self.llm.model_name

        logger.info(
            "RiskGenerator initialized",
            extra={
                "model": self.model_name,
                "temperature": settings.llm.temperature,
                "timeout_sec": settings.wizard.llm_timeout_sec,
            }
        )

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
        Call LLM with automatic retry on validation errors and Langfuse tracing.

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
        try:
            # Get Langfuse callback handler for tracing
            langfuse_handler = get_langfuse_callback_handler()
            langfuse_client = get_langfuse_client()

            # Format messages
            messages = prompt.format_messages(**kwargs)

            # Prepare invocation kwargs
            invoke_kwargs = {}
            if langfuse_handler:
                invoke_kwargs["config"] = {"callbacks": [langfuse_handler]}

            # Invoke LLM with Langfuse tracing
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
                        "max_attempts": 3,  # From @retry decorator stop_after_attempt
                    }
                ) as llm_span:
                    try:
                        # LLM invocation
                        response = await self.llm.ainvoke(messages, **invoke_kwargs)
                        response_text = response.content

                        # Extract usage from response metadata
                        usage = response.response_metadata.get("token_usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)

                        # Create usage metrics
                        metrics = create_usage_metrics(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            model=self.model_name,
                            pricing=self.settings.wizard.openai_pricing
                        )

                        # Update span with success output
                        llm_span.update(
                            output={
                                "response_length": len(response_text),
                                "response_preview": response_text[:200],
                                "usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": metrics.total_tokens,
                                },
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
                # Fallback without Langfuse
                response = await self.llm.ainvoke(messages, **invoke_kwargs)
                response_text = response.content

                # Extract usage from response metadata
                usage = response.response_metadata.get("token_usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # Create usage metrics
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

        LLM should only return relevant threats (probability 1-4). If probability 0 is found,
        it's logged as a warning and filtered out.

        Args:
            response_text: Raw LLM response
            allowed_threat_names: List of allowed threat names
            ordered_catalog_names: List of catalog names (not used for ordering, kept for compatibility)

        Returns:
            List of validated Risk objects (probability 1-4 only, in LLM's order)

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

                    # Validate probability and filter out probability 0
                    if probability == 0:
                        # LLM should not return probability 0, log warning
                        logger.warning(
                            f"LLM returned probability 0 for '{threat_name}' (should only return 1-4), skipping"
                        )
                        continue

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

            # Return threats in LLM's order (already sorted by relevance/probability)
            return refined

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
        threat_mapping: Dict[str, List[str]],
        max_threats: int = 20,
    ) -> List[str]:
        """
        Select relevant threats for an asset using AssetType→Threat mapping.

        If mapping exists for the asset type, use it (limited to max_threats).
        Otherwise, use all threats (but still limit to max_threats for performance).

        Args:
            asset: Asset to evaluate
            all_threat_names: All threat names from catalog
            threat_mapping: AssetType→Threat mapping
            max_threats: Maximum number of threats to evaluate (default 20)

        Returns:
            List of relevant threat names (subset of catalog)
        """
        # Get relevant threats from mapping
        relevant = threat_mapping.get(asset.assetType, [])

        if relevant:
            # Use threat mapping, limit to max_threats
            selected = relevant[:max_threats]
            logger.info(
                f"Using {len(selected)} mapped threats for asset type '{asset.assetType}'",
                extra={
                    "asset_name": asset.name,
                    "asset_type": asset.assetType,
                    "threats_count": len(selected),
                }
            )
        else:
            # No mapping found, use first max_threats from catalog
            selected = all_threat_names[:max_threats]
            logger.warning(
                f"No threat mapping for asset type '{asset.assetType}', using first {len(selected)} catalog threats",
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
        session_id: Optional[str] = None,
        include_catalogs: Optional[List[str]] = None,
    ) -> Tuple[List[AssetRisk], UsageMetrics, Dict[str, Any]]:
        """
        Generate risk evaluations for each asset.

        Processes one asset at a time, evaluating relevant threats from catalog.
        Uses AssetType→Threat mapping for smart threat selection.

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
            session_id: Optional session ID for Langfuse tracing
            include_catalogs: Optional list of catalogs to include in prompt (e.g., ["sectors", "impacts", "threats"])

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

        # Initialize prompt context builder with session-based caching
        builder = PromptContextBuilder(session_id=session_id)

        # Load threat catalog (already localized by Tazilla backend if using API)
        all_threat_names = await builder.get_catalog_names("threats")
        allowed_names_set = set(all_threat_names)

        # Load AssetType→Threat mapping (always required for smart threat selection)
        threat_mapping: Dict[str, List[str]] = {}
        try:
            threat_mapping = load_asset_type_threat_mapping()
            logger.info(f"Loaded threat mapping for {len(threat_mapping)} asset types")
        except Exception as e:
            logger.warning(f"Failed to load threat mapping, using all threats: {e}")

        # Build shared prompt sections using centralized builder
        sections = await builder.build_sections(
            website_content=website_content,
            organization_profile=organization_profile,
            services=services,
            user_instructions=user_instructions,
            include_catalogs=include_catalogs,  # Optional catalogs to include in prompt
        )

        # Add asset_impacts section (specific to risk generator)
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

        results: List[AssetRisk] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        last_response_text = ""
        last_prompt_text = ""

        # Get Langfuse client for tracing
        langfuse_client = get_langfuse_client()

        # Prepare span input
        span_input = {
            "assets_count": len(assets),
            "language": language,
            "min_threats_per_asset": min_threats_per_asset,
            "max_threats_per_asset": max_threats_per_asset,
            "mapped_asset_types_count": len(threat_mapping),
        }

        # Main span: generate-risks
        if langfuse_client and session_id:
            from langfuse import propagate_attributes

            with langfuse_client.start_as_current_span(
                name="generate-risks",
                input=span_input,
                metadata={"endpoint": "risks", "model": self.model_name},
            ) as main_span:
                with propagate_attributes(session_id=session_id):
                    # Fetch prompt with child span
                    requested_label = self.settings.langfuse.prompt_label
                    with langfuse_client.start_as_current_span(
                        name="get-risks-prompts",
                        input={
                            "prompt_name": "wizard/risks",
                            "requested_label": requested_label,
                        },
                    ) as prompt_span:
                        # Try to fetch unified prompt from Langfuse
                        prompt = get_langchain_prompt_from_langfuse(
                            prompt_name="wizard/risks",
                            fallback_template=None,  # Will build fallback manually
                            fallback_variables=None,
                        )

                        prompt_source = "langfuse"
                        prompt_version = None
                        prompt_type = "chat"  # Default to chat
                        prompt_tags = []
                        prompt_labels = []

                        # If Langfuse fetch failed, build fallback from local templates
                        if prompt is None:
                            logger.info("Building fallback prompt from local templates")
                            prompt = ChatPromptTemplate.from_messages([
                                ("system", COMBINED_RISK_SYSTEM_PROMPT),
                                ("human", COMBINED_RISK_HUMAN_PROMPT_TEMPLATE),
                            ])
                            prompt_source = "local"
                        else:
                            # Extract details from metadata (stored by get_langchain_prompt_from_langfuse)
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
                                    # Extract role from class name
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
                                "name": "wizard/risks",
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
                        # Child span: evaluate-asset-risk
                        with langfuse_client.start_as_current_span(
                            name="evaluate-asset-risk",
                            input={
                                "asset_id": str(asset.id) if asset.id else None,
                                "asset_name": asset.name,
                                "asset_type": asset.assetType if hasattr(asset, 'assetType') else None,
                            },
                        ) as asset_span:
                            try:
                                # Select relevant threats for this asset
                                relevant_threats = self._select_relevant_threats(
                                    asset=asset,
                                    all_threat_names=all_threat_names,
                                    threat_mapping=threat_mapping,
                                    max_threats=max_threats_per_asset,
                                )

                                # Single-asset JSON
                                assets_json = json.dumps([asset.model_dump()], ensure_ascii=False)

                                # Format prompt for debug preview
                                messages = prompt.format_messages(
                                    assets_json=assets_json,
                                    ordered_catalog_names_json=json.dumps(relevant_threats, ensure_ascii=False),
                                    threats_constraint=threats_constraint,
                                    risks_json_schema=RISKS_JSON_SCHEMA,
                                    language=language,
                                    **sections
                                )
                                last_prompt_text = str(messages)

                                # Call LLM with retry logic (using filtered threat list)
                                response_text, usage_metrics = await self._call_llm_with_retry(
                                    prompt=prompt,
                                    session_id=session_id,
                                    assets_json=assets_json,
                                    ordered_catalog_names_json=json.dumps(relevant_threats, ensure_ascii=False),
                                    threats_constraint=threats_constraint,
                                    risks_json_schema=RISKS_JSON_SCHEMA,
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

                                # Update asset span with success
                                if langfuse_client:
                                    asset_span.update(
                                        output={
                                            "risks_count": len(evaluations),
                                            "risks_preview": [{"name": r.threatName, "prob": r.probability} for r in evaluations[:5]],
                                        }
                                    )

                                logger.info(
                                    f"Risk evaluation completed for asset '{asset.name}'",
                                    extra={
                                        "asset_id": str(asset.id) if asset.id else None,
                                        "risks_count": len(evaluations),
                                    }
                                )

                            except AITimeoutError:
                                # Update span with error
                                if langfuse_client:
                                    asset_span.update(
                                        level="ERROR",
                                        status_message="LLM timeout",
                                    )
                                # Re-raise timeout errors immediately - don't continue with other assets
                                raise
                            except (AIResponseError, ValidationRetriesExhausted):
                                # Update span with error
                                if langfuse_client:
                                    asset_span.update(
                                        level="ERROR",
                                        status_message="AI response or validation error",
                                    )
                                # Re-raise AI-specific errors immediately
                                raise
                            except Exception as e:
                                # Update span with error
                                if langfuse_client:
                                    asset_span.update(
                                        level="ERROR",
                                        status_message=str(e),
                                    )
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
                        "mapped_asset_types_count": len(threat_mapping),
                        "last_prompt_preview": last_prompt_text[:500],
                        "last_response_preview": last_response_text[:500],
                    }

                    # Update main span with final output
                    main_span.update(
                        output={
                            "assets_processed": len(results),
                            "total_risks_generated": sum(len(r.risks) for r in results),
                            "total_tokens": aggregated_metrics.total_tokens,
                            "cost_usd": aggregated_metrics.cost_usd,
                        }
                    )

                    logger.info(
                        "Risk evaluations generated successfully",
                        extra={
                            "assets_count": len(assets),
                            "total_tokens": aggregated_metrics.total_tokens,
                            "cost_usd": aggregated_metrics.cost_usd,
                        }
                    )

                    return results, aggregated_metrics, debug_info
        else:
            # Fallback without Langfuse tracing
            logger.warning("Langfuse tracing not available for risks generation")

            # Create fallback prompt from local templates
            logger.info("Building fallback prompt from local templates (no Langfuse)")
            prompt = ChatPromptTemplate.from_messages([
                ("system", COMBINED_RISK_SYSTEM_PROMPT),
                ("human", COMBINED_RISK_HUMAN_PROMPT_TEMPLATE),
            ])

            # Process each asset individually (without Langfuse spans)
            for asset in assets:
                try:
                    # Select relevant threats for this asset
                    relevant_threats = self._select_relevant_threats(
                        asset=asset,
                        all_threat_names=all_threat_names,
                        threat_mapping=threat_mapping,
                        max_threats=max_threats_per_asset,
                    )

                    # Single-asset JSON
                    assets_json = json.dumps([asset.model_dump()], ensure_ascii=False)

                    # Format prompt for debug preview
                    messages = prompt.format_messages(
                        assets_json=assets_json,
                        ordered_catalog_names_json=json.dumps(relevant_threats, ensure_ascii=False),
                        threats_constraint=threats_constraint,
                        risks_json_schema=RISKS_JSON_SCHEMA,
                        language=language,
                        **sections
                    )
                    last_prompt_text = str(messages)

                    # Call LLM with retry logic
                    response_text, usage_metrics = await self._call_llm_with_retry(
                        prompt=prompt,
                        session_id=session_id,
                        assets_json=assets_json,
                        ordered_catalog_names_json=json.dumps(relevant_threats, ensure_ascii=False),
                        threats_constraint=threats_constraint,
                        risks_json_schema=RISKS_JSON_SCHEMA,
                        language=language,
                        **sections
                    )

                    # Accumulate usage
                    total_prompt_tokens += usage_metrics.prompt_tokens
                    total_completion_tokens += usage_metrics.completion_tokens
                    last_response_text = response_text

                    # Parse and validate
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
                    raise
                except (AIResponseError, ValidationRetriesExhausted):
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
                "mapped_asset_types_count": len(threat_mapping),
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
