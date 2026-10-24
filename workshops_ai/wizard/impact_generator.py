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
from workshops_ai.wizard.schemas import Service, ImpactsResponse, BatchImpactsResponse
from workshops_ai.wizard.prompts import (
    IMPACTS_SYSTEM_PROMPT,
    IMPACTS_HUMAN_PROMPT_TEMPLATE,
    IMPACTS_JSON_SCHEMA,
    BATCH_IMPACTS_JSON_SCHEMA,
)
from workshops_ai.wizard.prompt_context import PromptContextBuilder
from workshops_ai.wizard.cost_calculator import create_usage_metrics, UsageMetrics
from workshops_ai.wizard.exceptions import (
    AIResponseError,
    AITimeoutError,
    ValidationRetriesExhausted,
    ModelNotAllowedError,
    RequestTooLargeError,
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

    def __init__(self, model: Optional[str] = None, reasoning_effort: Optional[str] = None):
        """
        Initialize ImpactGenerator with specified model.

        Args:
            model: OpenAI model name (defaults to settings if not provided)
            reasoning_effort: Override reasoning effort (cascade: body > env > default)

        Raises:
            ModelNotAllowedError: If model not in allowed list
        """
        from workshops_ai.llm.wizard_client import create_wizard_llm_client

        settings = get_settings()
        self.settings = settings
        self.reasoning_effort = reasoning_effort

        self.llm = create_wizard_llm_client(
            model=model,
            response_format=ImpactsResponse,
            reasoning_effort=reasoning_effort,
        )
        self.model_name = self.llm.model_name

        logger.info(
            "ImpactGenerator initialized",
            extra={
                "model": self.model_name,
                "temperature": settings.wizard.temperature,
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
                        "temperature": self.settings.wizard.temperature,
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

    async def _process_assets_batch(
        self,
        assets: List[Asset],
        prompt: ChatPromptTemplate,
        impacts_with_desc: List[Dict[str, str]],
        ordered_catalog_names: List[str],
        impacts_constraint: str,
        language: str,
        sections: Dict[str, Any],
        max_impacts_per_asset: Optional[int],
        session_id: Optional[str],
    ) -> Tuple[List[AssetImpact], int, int, str]:
        """
        Process a batch of assets in a single LLM call.

        Args:
            assets: List of assets to process (1 to max_batch_size)
            prompt: Prompt template
            impacts_with_desc: Impact catalog with names and descriptions
            ordered_catalog_names: Catalog names for validation
            impacts_constraint: Impacts count constraint text
            language: Output language
            sections: Prompt sections (organization, services, etc.)
            max_impacts_per_asset: Max impacts per asset
            session_id: Session ID for tracing

        Returns:
            Tuple of (asset_impacts list, prompt_tokens, completion_tokens, response_text)

        Raises:
            AIResponseError: If response invalid or missing assets
            AITimeoutError: If LLM times out
        """
        from workshops_ai.llm.wizard_client import create_wizard_llm_client

        # Create batch LLM client with BatchImpactsResponse format
        batch_llm = create_wizard_llm_client(
            model=self.model_name,
            response_format=BatchImpactsResponse,
            reasoning_effort=self.reasoning_effort,
        )

        # Create JSON for all assets in batch
        assets_json = json.dumps([a.model_dump(mode='json') for a in assets], ensure_ascii=False)

        # Format messages with impact descriptions for better context
        messages = prompt.format_messages(
            assets_json=assets_json,
            ordered_catalog_names_json=json.dumps(impacts_with_desc, ensure_ascii=False, indent=2),
            impacts_constraint=impacts_constraint,
            impacts_json_schema=BATCH_IMPACTS_JSON_SCHEMA,
            language=language,
            **sections
        )

        # Get Langfuse callback handler
        langfuse_handler = get_langfuse_callback_handler()

        # Calculate expected output tokens based on batch size and max impacts per asset
        # Each impact = ~52 tokens, plus overhead for JSON structure
        items_per_asset = max_impacts_per_asset or 10  # Conservative default if not specified
        expected_output = len(assets) * items_per_asset * 52 + 500  # 500 for JSON wrapper

        invoke_kwargs = {
            "batch_size": len(assets),  # Pass batch size for per-request token limit checks
            "expected_output_tokens": expected_output,  # Precise output estimate
        }
        if langfuse_handler:
            invoke_kwargs["config"] = {"callbacks": [langfuse_handler]}

        # Call LLM
        try:
            response = await batch_llm.ainvoke(messages, **invoke_kwargs)
        except asyncio.TimeoutError as e:
            raise AITimeoutError(
                f"LLM timeout after {self.settings.wizard.llm_timeout_sec}s",
                model=self.model_name,
                timeout_sec=self.settings.wizard.llm_timeout_sec
            ) from e

        # Extract response text
        response_text = response.content

        # Extract usage
        usage = response.response_metadata.get("token_usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Parse response (BatchImpactsResponse should be already validated by OpenAI)
        try:
            # Extract JSON from response
            response_text_stripped = response_text.strip()
            start_idx = response_text_stripped.find("{")
            end_idx = response_text_stripped.rfind("}")

            if start_idx == -1 or end_idx == -1:
                raise AIResponseError(
                    "No JSON object found in batch response",
                    model=self.model_name,
                    details={"response_preview": response_text[:200]}
                )

            json_text = response_text_stripped[start_idx:end_idx + 1]
            batch_data = json.loads(json_text)

            # Validate structure
            if "asset_impacts" not in batch_data:
                raise AIResponseError(
                    "Missing 'asset_impacts' field in batch response",
                    model=self.model_name,
                    details={"response_preview": response_text[:200]}
                )

            asset_impacts_data = batch_data["asset_impacts"]

            # Validate count matches input (log warning if mismatch, but continue)
            if len(asset_impacts_data) != len(assets):
                warning_msg = f"Batch response has {len(asset_impacts_data)} assets, expected {len(assets)} - continuing with partial results"
                logger.warning(
                    warning_msg,
                    extra={
                        "expected_count": len(assets),
                        "received_count": len(asset_impacts_data),
                        "expected_assets": [a.name for a in assets],
                        "received_assets": [ai.get("assetName") for ai in asset_impacts_data],
                    }
                )

            # Parse each AssetImpact
            results: List[AssetImpact] = []
            allowed_names_set = set(ordered_catalog_names)

            # Build asset lookup for UUID restoration (LLM may return null assetId)
            asset_lookup = {asset.name: asset for asset in assets}

            for ai_data in asset_impacts_data:
                # Validate and filter impacts
                valid_impacts = []
                for imp_data in ai_data.get("impacts", []):
                    impact_name = imp_data.get("impactName")
                    impact_level = imp_data.get("impactLevel")

                    # Validate impact name
                    if impact_name not in allowed_names_set:
                        logger.warning(
                            f"Invalid impact name '{impact_name}' for asset '{ai_data.get('assetName')}', skipping"
                        )
                        continue

                    # Filter out level 0 (not relevant)
                    if impact_level == 0:
                        continue

                    valid_impacts.append(ImpactEvaluation(
                        impactName=impact_name,
                        impactLevel=impact_level
                    ))

                # Apply max limit if specified
                if max_impacts_per_asset is not None and len(valid_impacts) > max_impacts_per_asset:
                    valid_impacts = valid_impacts[:max_impacts_per_asset]

                # Restore assetId from original assets if LLM returned null
                asset_id = ai_data.get("assetId")
                if not asset_id:
                    original_asset = asset_lookup.get(ai_data["assetName"])
                    if original_asset and original_asset.id:
                        asset_id = original_asset.id

                # Create AssetImpact
                results.append(AssetImpact(
                    assetId=asset_id,
                    assetName=ai_data["assetName"],
                    impacts=valid_impacts
                ))

            return results, prompt_tokens, completion_tokens, response_text

        except (json.JSONDecodeError, KeyError, ValidationError) as e:
            raise AIResponseError(
                f"Failed to parse batch response: {str(e)}",
                model=self.model_name,
                details={"response_preview": response_text[:200]}
            ) from e

    async def _process_batch_with_recursive_splitting(
        self,
        assets: List[Asset],
        prompt: ChatPromptTemplate,
        impacts_with_desc: List[Dict[str, str]],
        ordered_catalog_names: List[str],
        impacts_constraint: str,
        language: str,
        sections: Dict[str, Any],
        max_impacts_per_asset: Optional[int],
        session_id: Optional[str],
        recursion_level: int = 0,
        max_recursion_level: int = 3,
    ) -> Tuple[List[AssetImpact], int, int, str]:
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
            impacts_with_desc: Impact catalog with names and descriptions
            ordered_catalog_names: Ordered impact catalog names
            impacts_constraint: Impacts constraint text
            language: Output language
            sections: Prompt sections
            max_impacts_per_asset: Max impacts per asset
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
                impacts_with_desc=impacts_with_desc,
                ordered_catalog_names=ordered_catalog_names,
                impacts_constraint=impacts_constraint,
                language=language,
                sections=sections,
                max_impacts_per_asset=max_impacts_per_asset,
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
                        impacts_with_desc=impacts_with_desc,
                        ordered_catalog_names=ordered_catalog_names,
                        impacts_constraint=impacts_constraint,
                        language=language,
                        sections=sections,
                        max_impacts_per_asset=max_impacts_per_asset,
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
                    impacts_with_desc=impacts_with_desc,
                    ordered_catalog_names=ordered_catalog_names,
                    impacts_constraint=impacts_constraint,
                    language=language,
                    sections=sections,
                    max_impacts_per_asset=max_impacts_per_asset,
                    session_id=session_id,
                    recursion_level=recursion_level + 1,
                    max_recursion_level=max_recursion_level,
                )

                results2, tokens2_p, tokens2_c, text2 = await self._process_batch_with_recursive_splitting(
                    assets=second_half,
                    prompt=prompt,
                    impacts_with_desc=impacts_with_desc,
                    ordered_catalog_names=ordered_catalog_names,
                    impacts_constraint=impacts_constraint,
                    language=language,
                    sections=sections,
                    max_impacts_per_asset=max_impacts_per_asset,
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
                    impacts_with_desc=impacts_with_desc,
                    ordered_catalog_names=ordered_catalog_names,
                    impacts_constraint=impacts_constraint,
                    language=language,
                    sections=sections,
                    max_impacts_per_asset=max_impacts_per_asset,
                    session_id=session_id,
                    recursion_level=recursion_level + 1,
                    max_recursion_level=max_recursion_level,
                )

                results2, tokens2_p, tokens2_c, text2 = await self._process_batch_with_recursive_splitting(
                    assets=second_half,
                    prompt=prompt,
                    impacts_with_desc=impacts_with_desc,
                    ordered_catalog_names=ordered_catalog_names,
                    impacts_constraint=impacts_constraint,
                    language=language,
                    sections=sections,
                    max_impacts_per_asset=max_impacts_per_asset,
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
                        ordered_catalog_names=ordered_catalog_names,
                        impacts_constraint=impacts_constraint,
                        language=language,
                        sections=sections,
                        max_impacts_per_asset=max_impacts_per_asset,
                        session_id=session_id,
                        recursion_level=recursion_level + 1,
                        max_recursion_level=max_recursion_level,
                    )

                    results2, tokens2_p, tokens2_c, text2 = await self._process_batch_with_recursive_splitting(
                        assets=second_half,
                        prompt=prompt,
                        ordered_catalog_names=ordered_catalog_names,
                        impacts_constraint=impacts_constraint,
                        language=language,
                        sections=sections,
                        max_impacts_per_asset=max_impacts_per_asset,
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
        org_metadata: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[AssetImpact], UsageMetrics, Dict[str, Any]]:
        """
        Generate impact evaluations for each asset.

        Processes assets in batches for better performance (configurable via WORKSHOPS_AI_WIZARD__MAX_BATCH_SIZE).
        Falls back to sequential processing if batch fails.

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

        # Load impact catalog with descriptions (already localized by Tazilla backend if using API)
        impacts_with_desc = await builder.get_catalog_with_descriptions("impacts")
        # Extract names for validation
        ordered_catalog_names = [imp["name"] for imp in impacts_with_desc]
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

            # Merge org metadata into span metadata
            span_metadata = {
                "endpoint": "impacts",
                "model": self.model_name,
                "reasoning_effort": self.llm.reasoning_effort,
            }
            if org_metadata:
                span_metadata.update(org_metadata)

            with langfuse_client.start_as_current_span(
                name="generate-impacts",
                input=span_input,
                metadata=span_metadata,
            ) as main_span:
                with propagate_attributes(session_id=session_id, metadata=org_metadata):
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

                        # Child span: evaluate-batch-impacts
                        with langfuse_client.start_as_current_span(
                            name="evaluate-batch-impacts",
                            input={
                                "batch_index": batch_idx + 1,
                                "total_batches": total_batches,
                                "batch_size": len(batch_assets),
                                "asset_names": [a.name for a in batch_assets],
                            },
                            metadata={"catalog_impacts_count": len(ordered_catalog_names)},
                        ) as batch_span:
                            try:
                                # Process batch with recursive splitting on context/limit errors
                                batch_results, batch_prompt_tokens, batch_completion_tokens, batch_response_text = await self._process_batch_with_recursive_splitting(
                                    assets=batch_assets,
                                    prompt=prompt,
                                    impacts_with_desc=impacts_with_desc,
                                    ordered_catalog_names=ordered_catalog_names,
                                    impacts_constraint=impacts_constraint,
                                    language=language,
                                    sections=sections,
                                    max_impacts_per_asset=max_impacts_per_asset,
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
                                        "total_impacts_generated": sum(len(r.impacts) for r in batch_results),
                                        "impacts_preview": [
                                            {
                                                "asset": r.assetName,
                                                "impacts_count": len(r.impacts)
                                            } for r in batch_results[:5]
                                        ],
                                    }
                                )

                                logger.info(
                                    f"Batch {batch_idx + 1}/{total_batches} completed successfully",
                                    extra={
                                        "batch_index": batch_idx + 1,
                                        "assets_processed": len(batch_results),
                                        "total_impacts": sum(len(r.impacts) for r in batch_results),
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
                                    # Child span: evaluate-asset-impact (fallback)
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
                                            assets_json = json.dumps([asset.model_dump(mode='json')], ensure_ascii=False)

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
                                                f"Impact evaluation completed for asset '{asset.name}' (fallback)",
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
                                                f"Impact evaluation failed for asset '{asset.name}' (fallback)",
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
                                        # Child span: evaluate-asset-impact (fallback)
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
                                                assets_json = json.dumps([asset.model_dump(mode='json')], ensure_ascii=False)

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
                                                    f"Impact evaluation completed for asset '{asset.name}' (fallback context/limit)",
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
                                                    f"Impact evaluation failed for asset '{asset.name}' (fallback context/limit)",
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
                        impacts_with_desc=impacts_with_desc,
                        ordered_catalog_names=ordered_catalog_names,
                        impacts_constraint=impacts_constraint,
                        language=language,
                        sections=sections,
                        max_impacts_per_asset=max_impacts_per_asset,
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
                            # Single-asset JSON
                            assets_json = json.dumps([asset.model_dump(mode='json')], ensure_ascii=False)

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
                                f"Impact evaluation completed for asset '{asset.name}' (fallback, no Langfuse)",
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
                                f"Impact evaluation failed for asset '{asset.name}' (fallback, no Langfuse)",
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
                                # Single-asset JSON
                                assets_json = json.dumps([asset.model_dump(mode='json')], ensure_ascii=False)

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
                                    f"Impact evaluation completed for asset '{asset.name}' (fallback, no Langfuse)",
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
                                    f"Impact evaluation failed for asset '{asset.name}' (fallback, no Langfuse)",
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
