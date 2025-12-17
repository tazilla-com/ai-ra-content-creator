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
from workshops_ai.wizard.schemas import Service, ImpactsResponse
from workshops_ai.wizard.prompts import (
    IMPACTS_SYSTEM_PROMPT,
    IMPACTS_HUMAN_PROMPT_TEMPLATE,
    IMPACTS_JSON_SCHEMA,
)
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
        from workshops_ai.llm.wizard_client import create_wizard_llm_client

        settings = get_settings()
        self.settings = settings

        self.llm = create_wizard_llm_client(
            model=model,
            response_format=ImpactsResponse
        )
        self.model_name = self.llm.model_name

        logger.info(
            "ImpactGenerator initialized",
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
        Call LLM with automatic retry on validation errors.

        Args:
            prompt: LangChain ChatPromptTemplate
            session_id: Session ID for Langfuse tracing
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
                        "max_attempts": self.settings.wizard.llm_max_retries,
                    }
                ) as llm_span:
                    try:
                        # LLM invocation - CallbackHandler will create child generation under current span
                        response = await self.llm.ainvoke(messages, **invoke_kwargs)

                        # Extract response text
                        response_text = response.content

                        # Extract usage from response metadata
                        usage = response.response_metadata.get("token_usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)

                        # Create usage metrics (with pricing override from settings)
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
                                    "cost_usd": metrics.cost_usd,
                                },
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
                        # Update span with error
                        llm_span.update(
                            level="ERROR",
                            status_message=str(e),
                        )
                        raise
            else:
                # Fallback without Langfuse
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

    def _parse_and_validate_impacts(
        self,
        response_text: str,
        allowed_impact_names: List[str],
        ordered_catalog_names: List[str],
    ) -> List[ImpactEvaluation]:
        """
        Parse JSON response and validate impact evaluations.

        LLM should only return relevant impacts (level 1-4). If level 0 is found,
        it's logged as a warning and filtered out.

        Args:
            response_text: Raw LLM response
            allowed_impact_names: List of allowed impact names
            ordered_catalog_names: List of catalog names in order (for fallback sorting)

        Returns:
            List of validated ImpactEvaluation objects (level 1-4 only, in LLM's order)

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

                    # Validate level and filter out level 0
                    if impact_level == 0:
                        # LLM should not return level 0, log warning
                        logger.warning(
                            f"LLM returned level 0 for '{impact_name}' (should only return 1-4), skipping"
                        )
                        continue

                    if 1 <= impact_level <= 4:
                        refined.append(ImpactEvaluation(
                            impactName=impact_name,
                            impactLevel=impact_level
                        ))
                    else:
                        logger.warning(
                            f"Invalid impactLevel {impact_level} for '{impact_name}', skipping"
                        )

                except (ValueError, ValidationError) as e:
                    logger.error(
                        f"Impact evaluation validation failed at index {idx}",
                        extra={"error": str(e), "data": ev_dict}
                    )
                    continue

            # Return impacts in LLM's order (already sorted by relevance)
            return refined

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
        min_impacts_per_asset: Optional[int] = None,
        max_impacts_per_asset: Optional[int] = None,
        website_content: str = "",
        organization_profile: Optional[Dict[str, Any]] = None,
        services: Optional[List[Service]] = None,
        user_instructions: Optional[str] = None,
        session_id: Optional[str] = None,
        include_catalogs: Optional[List[str]] = None,
    ) -> Tuple[List[AssetImpact], UsageMetrics, Dict[str, Any]]:
        """
        Generate impact evaluations for each asset.

        Processes one asset at a time, evaluating impacts from catalog.

        Args:
            assets: List of assets to evaluate
            language: Output language (en/sk/cs)
            min_impacts_per_asset: Minimum impacts per asset (optional, if not provided, no minimum enforced)
            max_impacts_per_asset: Maximum impacts per asset (optional, if not provided, no maximum enforced)
            website_content: Website text content (optional)
            organization_profile: Optional organization metadata
            services: Optional services for context
            user_instructions: Optional user refinement instructions
            session_id: Optional session ID for Langfuse tracing
            include_catalogs: Optional list of catalogs to include in prompt (e.g., ["sectors", "impacts", "threats"])

        Returns:
            Tuple of (asset impacts list, aggregated usage metrics, debug info dict)

        Raises:
            ValidationRetriesExhausted: If validation fails after all retries
            AITimeoutError: If LLM request times out
            AIResponseError: If response cannot be parsed
        """

        # Build impacts count constraint text for prompt
        if min_impacts_per_asset is not None and max_impacts_per_asset is not None:
            impacts_constraint = f"Evaluate between {min_impacts_per_asset} and {max_impacts_per_asset} most relevant impacts."
        elif min_impacts_per_asset is not None:
            impacts_constraint = f"Evaluate at least {min_impacts_per_asset} most relevant impacts."
        elif max_impacts_per_asset is not None:
            impacts_constraint = f"Evaluate up to {max_impacts_per_asset} most relevant impacts."
        else:
            impacts_constraint = "Evaluate all relevant impacts from the catalog."

        # Initialize prompt context builder with session-based caching
        builder = PromptContextBuilder(session_id=session_id)

        # Load impact catalog (already localized by Tazilla backend if using API)
        ordered_catalog_names = await builder.get_catalog_names("impacts")
        allowed_names_set = set(ordered_catalog_names)

        # Build shared prompt sections using centralized builder
        sections = await builder.build_sections(
            website_content=website_content,
            organization_profile=organization_profile,
            services=services,
            user_instructions=user_instructions,
            include_catalogs=include_catalogs,  # Optional catalogs to include in prompt
        )

        # Get Langfuse client for tracing
        langfuse_client = get_langfuse_client()

        # Prepare span input
        span_input = {
            "assets_count": len(assets),
            "language": language,
            "min_impacts_per_asset": min_impacts_per_asset,
            "max_impacts_per_asset": max_impacts_per_asset,
            "has_website_content": bool(website_content),
            "has_organization_profile": bool(organization_profile),
            "has_services": bool(services),
            "has_user_instructions": bool(user_instructions),
        }

        results: List[AssetImpact] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        last_response_text = ""
        last_prompt_text = ""

        # Main span: generate-impacts
        if langfuse_client and session_id:
            from langfuse import propagate_attributes

            with langfuse_client.start_as_current_span(
                name="generate-impacts",
                input=span_input,
                metadata={"endpoint": "impacts", "model": self.model_name},
            ) as main_span:
                with propagate_attributes(session_id=session_id):
                    # Fetch prompt with child span
                    requested_label = self.settings.langfuse.prompt_label
                    with langfuse_client.start_as_current_span(
                        name="get-impacts-prompts",
                        input={
                            "prompt_name": "wizard/impacts",
                            "requested_label": requested_label,
                        },
                    ) as prompt_span:
                        # Try to fetch unified prompt from Langfuse
                        prompt = get_langchain_prompt_from_langfuse(
                            prompt_name="wizard/impacts",
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
                                ("system", IMPACTS_SYSTEM_PROMPT),
                                ("human", IMPACTS_HUMAN_PROMPT_TEMPLATE),
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
                                "name": "wizard/impacts",
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
                        # Child span: evaluate-asset-impact
                        with langfuse_client.start_as_current_span(
                            name="evaluate-asset-impact",
                            input={
                                "asset_id": str(asset.id) if asset.id else None,
                                "asset_name": asset.name,
                                "catalog_impacts_count": len(ordered_catalog_names),
                            },
                            metadata={"asset_type": asset.assetType if hasattr(asset, 'assetType') else None},
                        ) as asset_span:
                            try:
                                # Single-asset JSON
                                assets_json = json.dumps([asset.model_dump()], ensure_ascii=False)

                                # Format prompt for debug preview
                                messages = prompt.format_messages(
                                    assets_json=assets_json,
                                    ordered_catalog_names_json=json.dumps(ordered_catalog_names, ensure_ascii=False),
                                    impacts_constraint=impacts_constraint,
                                    impacts_json_schema=IMPACTS_JSON_SCHEMA,
                                    language=language,
                                    **sections
                                )
                                last_prompt_text = str(messages)

                                # Call LLM with retry logic
                                response_text, usage_metrics = await self._call_llm_with_retry(
                                    prompt=prompt,
                                    session_id=session_id,
                                    assets_json=assets_json,
                                    ordered_catalog_names_json=json.dumps(ordered_catalog_names, ensure_ascii=False),
                                    impacts_constraint=impacts_constraint,
                                    impacts_json_schema=IMPACTS_JSON_SCHEMA,
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

                                # Apply max limit if specified
                                if max_impacts_per_asset is not None:
                                    evaluations = evaluations[:max_impacts_per_asset]

                                # Create AssetImpact
                                results.append(AssetImpact(
                                    assetId=asset.id,
                                    assetName=asset.name,
                                    impacts=evaluations
                                ))

                                # Update asset span with success
                                asset_span.update(
                                    output={
                                        "impacts_count": len(evaluations),
                                        "impacts_preview": [{"name": e.impactName, "level": e.impactLevel} for e in evaluations[:5]],
                                    }
                                )

                                logger.info(
                                    f"Impact evaluation completed for asset '{asset.name}'",
                                    extra={
                                        "asset_id": str(asset.id) if asset.id else None,
                                        "impacts_count": len(evaluations),
                                    }
                                )

                            except AITimeoutError:
                                # Update span with error
                                asset_span.update(
                                    level="ERROR",
                                    status_message="LLM timeout",
                                )
                                # Re-raise timeout errors immediately - don't continue with other assets
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
                        "min_impacts_per_asset": min_impacts_per_asset,
                        "max_impacts_per_asset": max_impacts_per_asset,
                        "impacts_constraint": impacts_constraint,
                        "last_prompt_preview": last_prompt_text[:500],
                        "last_response_preview": last_response_text[:500],
                    }

                    # Update main span with final output
                    main_span.update(
                        output={
                            "assets_processed": len(results),
                            "total_impacts_generated": sum(len(r.impacts) for r in results),
                            "total_tokens": aggregated_metrics.total_tokens,
                            "cost_usd": aggregated_metrics.cost_usd,
                        }
                    )

                    logger.info(
                        "Impact evaluations generated successfully",
                        extra={
                            "assets_count": len(assets),
                            "total_tokens": aggregated_metrics.total_tokens,
                            "cost_usd": aggregated_metrics.cost_usd,
                        }
                    )

                    return results, aggregated_metrics, debug_info
        else:
            # Fallback without Langfuse tracing
            # Create fallback prompt from local templates
            logger.info("Building fallback prompt from local templates (no Langfuse)")
            prompt = ChatPromptTemplate.from_messages([
                ("system", IMPACTS_SYSTEM_PROMPT),
                ("human", IMPACTS_HUMAN_PROMPT_TEMPLATE),
            ])

            for asset in assets:
                try:
                    # Single-asset JSON
                    assets_json = json.dumps([asset.model_dump()], ensure_ascii=False)

                    # Format prompt for debug preview
                    messages = prompt.format_messages(
                        assets_json=assets_json,
                        ordered_catalog_names_json=json.dumps(ordered_catalog_names, ensure_ascii=False),
                        impacts_constraint=impacts_constraint,
                        impacts_json_schema=IMPACTS_JSON_SCHEMA,
                        language=language,
                        **sections
                    )
                    last_prompt_text = str(messages)

                    # Call LLM with retry logic
                    response_text, usage_metrics = await self._call_llm_with_retry(
                        prompt=prompt,
                        session_id=None,
                        assets_json=assets_json,
                        ordered_catalog_names_json=json.dumps(ordered_catalog_names, ensure_ascii=False),
                        impacts_constraint=impacts_constraint,
                        impacts_json_schema=IMPACTS_JSON_SCHEMA,
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

                    # Apply max limit if specified
                    if max_impacts_per_asset is not None:
                        evaluations = evaluations[:max_impacts_per_asset]

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
                    raise
                except (AIResponseError, ValidationRetriesExhausted):
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
                "min_impacts_per_asset": min_impacts_per_asset,
                "max_impacts_per_asset": max_impacts_per_asset,
                "impacts_constraint": impacts_constraint,
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
