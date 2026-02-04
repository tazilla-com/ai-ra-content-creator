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
from workshops_ai.wizard.schemas import Service, RisksResponse, BatchRisksResponse
from workshops_ai.wizard.prompts import (
    COMBINED_RISK_SYSTEM_PROMPT,
    COMBINED_RISK_HUMAN_PROMPT_TEMPLATE,
    RISKS_JSON_SCHEMA,
    BATCH_RISKS_JSON_SCHEMA,
)
from workshops_ai.wizard.catalog_loader import load_asset_type_threat_mapping
from workshops_ai.wizard.prompt_context import PromptContextBuilder
from workshops_ai.wizard.cost_calculator import create_usage_metrics, UsageMetrics
from workshops_ai.wizard.exceptions import (
    AIResponseError,
    AITimeoutError,
    ValidationRetriesExhausted,
    RequestTooLargeError,
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
    ) -> List[str]:
        """
        Select relevant threats for an asset using AssetType→Threat mapping.

        If mapping exists for the asset type, use all mapped threats.
        Otherwise, use all catalog threats.

        Args:
            asset: Asset to evaluate
            all_threat_names: All threat names from catalog
            threat_mapping: AssetType→Threat mapping

        Returns:
            List of relevant threat names
        """
        # Get relevant threats from mapping
        relevant = threat_mapping.get(asset.assetType, [])

        if relevant:
            # Use all mapped threats (no limit)
            selected = relevant
            logger.info(
                f"Using {len(selected)} mapped threats for asset type '{asset.assetType}' (from {len(relevant)} available)",
                extra={
                    "asset_name": asset.name,
                    "asset_type": asset.assetType,
                    "threats_available": len(relevant),
                    "threats_selected": len(selected),
                }
            )
        else:
            # No mapping found, use all catalog threats
            selected = all_threat_names
            logger.warning(
                f"No threat mapping for asset type '{asset.assetType}', using all {len(selected)} catalog threats",
                extra={
                    "asset_name": asset.name,
                    "asset_type": asset.assetType,
                    "threats_available": len(all_threat_names),
                    "threats_selected": len(selected),
                }
            )

        return selected

    async def _process_assets_batch(
        self,
        assets: List[Asset],
        prompt: ChatPromptTemplate,
        all_threat_names: List[str],
        threat_mapping: Dict[str, List[str]],
        threats_constraint: str,
        language: str,
        sections: Dict[str, Any],
        max_threats_per_asset: Optional[int],
        session_id: Optional[str],
    ) -> Tuple[List[AssetRisk], int, int, str]:
        """
        Process multiple assets in a single batch LLM call using OpenAI Structured Outputs.

        Uses union of relevant threats from all assets in the batch for shared context.

        Args:
            assets: List of assets to process in this batch
            prompt: ChatPromptTemplate to use
            all_threat_names: All threat names from catalog
            threat_mapping: AssetType→Threat mapping
            threats_constraint: Threats count constraint text
            language: Output language
            sections: Prompt sections dict
            max_threats_per_asset: Max threats per asset (optional)
            session_id: Session ID for Langfuse tracing

        Returns:
            Tuple of (asset_risks_list, prompt_tokens, completion_tokens, response_text)

        Raises:
            AITimeoutError: If LLM request times out
            AIResponseError: If response cannot be parsed or validated
        """
        from workshops_ai.llm.wizard_client import create_wizard_llm_client

        logger.info(
            f"Processing batch of {len(assets)} assets for risks",
            extra={"batch_size": len(assets)}
        )

        # Create batch LLM client with BatchRisksResponse format
        batch_llm = create_wizard_llm_client(
            model=self.model_name,
            response_format=BatchRisksResponse
        )

        # Select union of relevant threats for all assets in batch
        # This ensures all assets can be evaluated with same threat context
        batch_threats_set = set()
        for asset in assets:
            relevant_threats = self._select_relevant_threats(
                asset=asset,
                all_threat_names=all_threat_names,
                threat_mapping=threat_mapping,
            )
            batch_threats_set.update(relevant_threats)

        # Convert to list and maintain order
        batch_threats = [t for t in all_threat_names if t in batch_threats_set]

        logger.info(
            f"Selected {len(batch_threats)} threats for batch processing",
            extra={
                "batch_size": len(assets),
                "threats_count": len(batch_threats),
            }
        )

        # Create JSON for all assets in batch
        assets_json = json.dumps(
            [a.model_dump(mode='json') for a in assets],
            ensure_ascii=False
        )

        # Format messages with batch schema
        messages = prompt.format_messages(
            assets_json=assets_json,
            ordered_catalog_names_json=json.dumps(batch_threats, ensure_ascii=False),
            threats_constraint=threats_constraint,
            risks_json_schema=BATCH_RISKS_JSON_SCHEMA,  # Use batch schema
            language=language,
            **sections
        )

        # Calculate expected output tokens based on batch size and max threats per asset
        # Each risk = ~88 tokens, plus overhead for JSON structure
        items_per_asset = max_threats_per_asset or 10  # Conservative default if not specified
        expected_output = len(assets) * items_per_asset * 88 + 500  # 500 for JSON wrapper

        # Prepare invocation kwargs with Langfuse tracing
        invoke_kwargs = {
            "batch_size": len(assets),  # Pass batch size for per-request token limit checks
            "expected_output_tokens": expected_output,  # Precise output estimate
        }
        langfuse_handler = get_langfuse_callback_handler()
        if langfuse_handler:
            invoke_kwargs["config"] = {"callbacks": [langfuse_handler]}

        try:
            # Call LLM with batch format
            logger.debug(
                "Calling LLM for batch risks processing",
                extra={
                    "batch_size": len(assets),
                    "threats_count": len(batch_threats),
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
                    "batch_size": len(assets),
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
                        "batch_size": len(assets),
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
                    "batch_size": len(assets),
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
            batch_response = BatchRisksResponse.model_validate(response_data)
            asset_risks_data = batch_response.asset_risks

            # Restore assetId from original assets if LLM returned null
            asset_lookup = {asset.name: asset for asset in assets}
            for asset_risk in asset_risks_data:
                if not asset_risk.assetId:
                    original_asset = asset_lookup.get(asset_risk.assetName)
                    if original_asset and original_asset.id:
                        asset_risk.assetId = original_asset.id

            logger.info(
                "Batch response parsed successfully",
                extra={
                    "requested_assets": len(assets),
                    "returned_assets": len(asset_risks_data),
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
                f"Failed to parse batch risks response: {str(e)}",
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
                f"Batch risks response validation failed: {str(e)}",
                model=self.model_name,
                details={"validation_errors": e.errors()}
            ) from e

        # Validate asset count (WARNING if mismatch, not error)
        if len(asset_risks_data) != len(assets):
            warning_msg = f"Batch response has {len(asset_risks_data)} assets, expected {len(assets)} - continuing with partial results"
            logger.warning(
                "batch_risks_count_mismatch",
                extra={
                    "expected": len(assets),
                    "received": len(asset_risks_data),
                    "requested_asset_names": [a.name for a in assets],
                    "returned_asset_names": [ar.assetName for ar in asset_risks_data],
                }
            )
            # Continue with partial results instead of raising error

        # Apply max_threats_per_asset limit if specified
        if max_threats_per_asset is not None:
            for asset_risk in asset_risks_data:
                asset_risk.risks = asset_risk.risks[:max_threats_per_asset]

        logger.info(
            "Batch risks processing completed successfully",
            extra={
                "batch_size": len(assets),
                "results_count": len(asset_risks_data),
                "total_risks": sum(len(ar.risks) for ar in asset_risks_data),
            }
        )

        return asset_risks_data, prompt_tokens, completion_tokens, response_text

    async def _process_batch_with_recursive_splitting(
        self,
        assets: List[Asset],
        prompt: ChatPromptTemplate,
        all_threat_names: List[str],
        threat_mapping: Dict[str, List[str]],
        threats_constraint: str,
        language: str,
        sections: Dict[str, Any],
        max_threats_per_asset: Optional[int],
        session_id: Optional[str],
        recursion_level: int = 0,
        max_recursion_level: int = 3,
    ) -> Tuple[List[AssetRisk], int, int, str]:
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
            all_threat_names: All available threat names
            threat_mapping: Asset type to threat mapping
            threats_constraint: Threats constraint text
            language: Output language
            sections: Prompt sections
            max_threats_per_asset: Max threats per asset
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
                all_threat_names=all_threat_names,
                threat_mapping=threat_mapping,
                threats_constraint=threats_constraint,
                language=language,
                sections=sections,
                max_threats_per_asset=max_threats_per_asset,
                session_id=session_id,
            )

            return batch_results, prompt_tokens, completion_tokens, response_text

        except RequestTooLargeError as e:
            # Request too large - use suggested batch size or split in half
            if e.suggested_batch_size and e.suggested_batch_size < len(assets):
                # Use suggested batch size from throttle (hybrid approach)
                suggested = e.suggested_batch_size
                logger.warning(
                    f"Request too large, using suggested batch size {suggested} (from {len(assets)} assets)",
                    extra={
                        "recursion_level": recursion_level,
                        "total_assets": len(assets),
                        "suggested_batch_size": suggested,
                        "estimated_tokens": e.estimated_tokens,
                        "model_limit": e.model_limit,
                        "reason": "request_too_large_suggested",
                    }
                )

                # Split into batches of suggested size
                results_list = []
                total_prompt_tokens = 0
                total_completion_tokens = 0
                last_text = ""

                for i in range(0, len(assets), suggested):
                    batch = assets[i:i + suggested]
                    batch_results, p_tokens, c_tokens, text = await self._process_batch_with_recursive_splitting(
                        assets=batch,
                        prompt=prompt,
                        all_threat_names=all_threat_names,
                        threat_mapping=threat_mapping,
                        threats_constraint=threats_constraint,
                        language=language,
                        sections=sections,
                        max_threats_per_asset=max_threats_per_asset,
                        session_id=session_id,
                        recursion_level=recursion_level + 1,
                        max_recursion_level=max_recursion_level,
                    )
                    results_list.extend(batch_results)
                    total_prompt_tokens += p_tokens
                    total_completion_tokens += c_tokens
                    last_text = text

                return results_list, total_prompt_tokens, total_completion_tokens, last_text

            elif recursion_level < max_recursion_level and len(assets) > 1:
                # No suggested size or batch_size=1 - split in half (fallback)
                mid = len(assets) // 2
                first_half = assets[:mid]
                second_half = assets[mid:]

                logger.warning(
                    f"Request too large, splitting {len(assets)} assets into {len(first_half)} + {len(second_half)}",
                    extra={
                        "recursion_level": recursion_level,
                        "total_assets": len(assets),
                        "first_half_size": len(first_half),
                        "second_half_size": len(second_half),
                        "estimated_tokens": e.estimated_tokens,
                        "model_limit": e.model_limit,
                        "reason": "request_too_large_split",
                    }
                )

                # Recursively process each half
                results1, tokens1_p, tokens1_c, text1 = await self._process_batch_with_recursive_splitting(
                    assets=first_half,
                    prompt=prompt,
                    all_threat_names=all_threat_names,
                    threat_mapping=threat_mapping,
                    threats_constraint=threats_constraint,
                    language=language,
                    sections=sections,
                    max_threats_per_asset=max_threats_per_asset,
                    session_id=session_id,
                    recursion_level=recursion_level + 1,
                    max_recursion_level=max_recursion_level,
                )

                results2, tokens2_p, tokens2_c, text2 = await self._process_batch_with_recursive_splitting(
                    assets=second_half,
                    prompt=prompt,
                    all_threat_names=all_threat_names,
                    threat_mapping=threat_mapping,
                    threats_constraint=threats_constraint,
                    language=language,
                    sections=sections,
                    max_threats_per_asset=max_threats_per_asset,
                    session_id=session_id,
                    recursion_level=recursion_level + 1,
                    max_recursion_level=max_recursion_level,
                )

                # Combine results
                combined_results = results1 + results2
                combined_prompt_tokens = tokens1_p + tokens2_p
                combined_completion_tokens = tokens1_c + tokens2_c
                last_text = text2

                return combined_results, combined_prompt_tokens, combined_completion_tokens, last_text
            else:
                # Max recursion level reached or single asset - raise error
                logger.error(
                    f"Request too large and cannot split further",
                    extra={
                        "recursion_level": recursion_level,
                        "max_recursion_level": max_recursion_level,
                        "assets_count": len(assets),
                        "estimated_tokens": e.estimated_tokens,
                        "model_limit": e.model_limit,
                        "reason": "request_too_large_cannot_split",
                    }
                )
                raise

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
                    all_threat_names=all_threat_names,
                    threat_mapping=threat_mapping,
                    threats_constraint=threats_constraint,
                    language=language,
                    sections=sections,
                    max_threats_per_asset=max_threats_per_asset,
                    session_id=session_id,
                    recursion_level=recursion_level + 1,
                    max_recursion_level=max_recursion_level,
                )

                results2, tokens2_p, tokens2_c, text2 = await self._process_batch_with_recursive_splitting(
                    assets=second_half,
                    prompt=prompt,
                    all_threat_names=all_threat_names,
                    threat_mapping=threat_mapping,
                    threats_constraint=threats_constraint,
                    language=language,
                    sections=sections,
                    max_threats_per_asset=max_threats_per_asset,
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
                        all_threat_names=all_threat_names,
                        threat_mapping=threat_mapping,
                        threats_constraint=threats_constraint,
                        language=language,
                        sections=sections,
                        max_threats_per_asset=max_threats_per_asset,
                        session_id=session_id,
                        recursion_level=recursion_level + 1,
                        max_recursion_level=max_recursion_level,
                    )

                    results2, tokens2_p, tokens2_c, text2 = await self._process_batch_with_recursive_splitting(
                        assets=second_half,
                        prompt=prompt,
                        all_threat_names=all_threat_names,
                        threat_mapping=threat_mapping,
                        threats_constraint=threats_constraint,
                        language=language,
                        sections=sections,
                        max_threats_per_asset=max_threats_per_asset,
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
        org_metadata: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[AssetRisk], UsageMetrics, Dict[str, Any]]:
        """
        Generate risk evaluations for each asset.

        Processes assets in batches for better performance (configurable via WORKSHOPS_AI_WIZARD__MAX_BATCH_SIZE).
        Falls back to sequential processing if batch fails.
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
                [ai.model_dump(mode='json') for ai in asset_impacts],
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

            # Merge org metadata into span metadata
            span_metadata = {"endpoint": "risks", "model": self.model_name}
            if org_metadata:
                span_metadata.update(org_metadata)

            with langfuse_client.start_as_current_span(
                name="generate-risks",
                input=span_input,
                metadata=span_metadata,
            ) as main_span:
                with propagate_attributes(session_id=session_id, metadata=org_metadata):
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

                        # Child span: evaluate-batch-risks
                        with langfuse_client.start_as_current_span(
                            name="evaluate-batch-risks",
                            input={
                                "batch_index": batch_idx + 1,
                                "total_batches": total_batches,
                                "batch_size": len(batch_assets),
                                "asset_names": [a.name for a in batch_assets],
                            },
                            metadata={"catalog_threats_count": len(all_threat_names)},
                        ) as batch_span:
                            try:
                                # Process batch with recursive splitting on context/limit errors
                                batch_results, batch_prompt_tokens, batch_completion_tokens, batch_response_text = await self._process_batch_with_recursive_splitting(
                                    assets=batch_assets,
                                    prompt=prompt,
                                    all_threat_names=all_threat_names,
                                    threat_mapping=threat_mapping,
                                    threats_constraint=threats_constraint,
                                    language=language,
                                    sections=sections,
                                    max_threats_per_asset=max_threats_per_asset,
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
                                        "total_risks_generated": sum(len(r.risks) for r in batch_results),
                                        "risks_preview": [
                                            {
                                                "asset": r.assetName,
                                                "risks_count": len(r.risks)
                                            } for r in batch_results[:5]
                                        ],
                                    }
                                )

                                logger.info(
                                    f"Batch {batch_idx + 1}/{total_batches} completed successfully",
                                    extra={
                                        "batch_index": batch_idx + 1,
                                        "assets_processed": len(batch_results),
                                        "total_risks": sum(len(r.risks) for r in batch_results),
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
                                    # Child span: evaluate-asset-risk (fallback)
                                    with langfuse_client.start_as_current_span(
                                        name="evaluate-asset-risk",
                                        input={
                                            "asset_id": str(asset.id) if asset.id else None,
                                            "asset_name": asset.name,
                                            "catalog_threats_count": len(all_threat_names),
                                        },
                                        metadata={"asset_type": asset.assetType if hasattr(asset, 'assetType') else None},
                                    ) as asset_span:
                                        try:
                                            # Select relevant threats for this asset
                                            relevant_threats = self._select_relevant_threats(
                                                asset=asset,
                                                all_threat_names=all_threat_names,
                                                threat_mapping=threat_mapping,
                                            )

                                            # Single-asset JSON
                                            assets_json = json.dumps([asset.model_dump(mode='json')], ensure_ascii=False)

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

                                            # Update asset span with success
                                            asset_span.update(
                                                output={
                                                    "risks_count": len(evaluations),
                                                    "risks_preview": [{"name": r.threatName, "prob": r.probability} for r in evaluations[:5]],
                                                }
                                            )

                                            logger.info(
                                                f"Risk evaluation completed for asset '{asset.name}' (fallback)",
                                                extra={
                                                    "asset_id": str(asset.id) if asset.id else None,
                                                    "risks_count": len(evaluations),
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
                                                f"Risk evaluation failed for asset '{asset.name}' (fallback)",
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

                                    # Fallback: Process assets sequentially
                                    for asset in batch_assets:
                                        # Child span: evaluate-asset-risk (fallback)
                                        with langfuse_client.start_as_current_span(
                                            name="evaluate-asset-risk",
                                            input={
                                                "asset_id": str(asset.id) if asset.id else None,
                                                "asset_name": asset.name,
                                                "catalog_threats_count": len(all_threat_names),
                                            },
                                            metadata={"asset_type": asset.assetType if hasattr(asset, 'assetType') else None},
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
                                                assets_json = json.dumps([asset.model_dump(mode='json')], ensure_ascii=False)

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

                                                # Update asset span with success
                                                asset_span.update(
                                                    output={
                                                        "risks_count": len(evaluations),
                                                        "risks_preview": [{"name": r.threatName, "prob": r.probability} for r in evaluations[:5]],
                                                    }
                                                )

                                                logger.info(
                                                    f"Risk evaluation completed for asset '{asset.name}' (fallback context/limit)",
                                                    extra={
                                                        "asset_id": str(asset.id) if asset.id else None,
                                                        "risks_count": len(evaluations),
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
                                                    f"Risk evaluation failed for asset '{asset.name}' (fallback context/limit)",
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
                        all_threat_names=all_threat_names,
                        threat_mapping=threat_mapping,
                        threats_constraint=threats_constraint,
                        language=language,
                        sections=sections,
                        max_threats_per_asset=max_threats_per_asset,
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
                            # Select relevant threats for this asset
                            relevant_threats = self._select_relevant_threats(
                                asset=asset,
                                all_threat_names=all_threat_names,
                                threat_mapping=threat_mapping,
                            )

                            # Single-asset JSON
                            assets_json = json.dumps([asset.model_dump(mode='json')], ensure_ascii=False)

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
                                session_id=None,
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
                                f"Risk evaluation completed for asset '{asset.name}' (fallback, no Langfuse)",
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
                                f"Risk evaluation failed for asset '{asset.name}' (fallback, no Langfuse)",
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
                                # Select relevant threats for this asset
                                relevant_threats = self._select_relevant_threats(
                                    asset=asset,
                                    all_threat_names=all_threat_names,
                                    threat_mapping=threat_mapping,
                                )

                                # Single-asset JSON
                                assets_json = json.dumps([asset.model_dump(mode='json')], ensure_ascii=False)

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
                                    session_id=None,
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
                                    f"Risk evaluation completed for asset '{asset.name}' (fallback, no Langfuse)",
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
                                    f"Risk evaluation failed for asset '{asset.name}' (fallback, no Langfuse)",
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
