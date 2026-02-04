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
    ASSETS_JSON_SCHEMA,
)
from workshops_ai.llm.langfuse_helper import (
    get_langfuse_client,
    get_langfuse_callback_handler,
    get_langchain_prompt_from_langfuse,
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
        from workshops_ai.llm.wizard_client import create_wizard_llm_client
        from workshops_ai.wizard.schemas import AssetsResponse

        settings = get_settings()
        self.settings = settings

        # Create LLM client with structured outputs (Pydantic model)
        self.llm = create_wizard_llm_client(
            model=model,
            response_format=AssetsResponse
        )
        self.model_name = self.llm.model_name
        self._current_retry_attempt = 1  # Track current retry attempt

        logger.info(
            "AssetGenerator initialized",
            extra={
                "model": self.model_name,
                "temperature": settings.llm.temperature,
                "timeout_sec": settings.wizard.llm_timeout_sec,
                "structured_outputs": True,
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
                [a.model_dump(mode='json') for a in previous_assets],
                indent=2,
                ensure_ascii=False
            )
            sections["previous_assets_section"] = (
                f"Previous Assets (refine these):\n{assets_json}\n"
            )
        else:
            sections["previous_assets_section"] = ""

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

        return sections

    def _build_full_prompt_text(
        self,
        language: str,
        min_count: Optional[int],
        max_count: Optional[int],
        asset_types: List[str],
        sections: Dict[str, str]
    ) -> str:
        """
        Build full prompt text for debugging.

        Returns truncated version for logging.
        """
        # Build count constraint text
        if min_count is not None and max_count is not None:
            count_constraint = f"Generate between {min_count} and {max_count} IT and business assets."
        elif min_count is not None:
            count_constraint = f"Generate at least {min_count} IT and business assets."
        elif max_count is not None:
            count_constraint = f"Generate up to {max_count} IT and business assets."
        else:
            count_constraint = "Generate IT and business assets based on the organization's needs."

        human_prompt = ASSETS_HUMAN_PROMPT_TEMPLATE.format(
            count_constraint=count_constraint,
            language=language,
            asset_types=", ".join(asset_types),
            assets_json_schema=ASSETS_JSON_SCHEMA,
            **sections
        )

        full_prompt = f"{ASSETS_SYSTEM_PROMPT}\n\n{human_prompt}"

        # Truncate for logging
        max_log_length = 500
        if len(full_prompt) > max_log_length:
            return full_prompt[:max_log_length] + "... [truncated for logging]"
        return full_prompt

    def _before_retry(self, retry_state: RetryCallState) -> None:
        """Track retry attempt number before each retry."""
        self._current_retry_attempt = retry_state.attempt_number

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ValidationError),
        before=lambda retry_state: retry_state.args[0]._before_retry(retry_state),
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

            # Invoke LLM with Langfuse tracing (timeout already configured in ChatOpenAI constructor)
            invoke_kwargs = {}
            if langfuse_handler is not None:
                invoke_kwargs["config"] = {"callbacks": [langfuse_handler]}

            # Wrap LLM call in span if Langfuse is available
            if langfuse_client:
                with langfuse_client.start_as_current_span(
                    name="llm-invoke",
                    input={
                        "model": self.model_name,
                        "messages_count": len(messages),
                        "temperature": self.settings.llm.temperature,
                        "timeout_sec": self.settings.wizard.llm_timeout_sec,
                        "attempt_number": self._current_retry_attempt,
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
                                    "cost_usd": round(metrics.cost_usd, 8),
                                }
                            },
                            metadata={"status": "success"}
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
                            metadata={
                                "status": "error",
                                "error": str(e),
                                "error_type": type(e).__name__
                            }
                        )
                        raise
            else:
                # No Langfuse - proceed without span
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

        With OpenAI structured outputs, response is guaranteed to be valid JSON
        in format: {"assets": [...]}

        Enforces ISO 27001 rules:
        - Primary assets (isPrimary=true): CIA ratings required
        - Supporting assets (isPrimary=false): CIA must be null

        Args:
            response_text: Raw LLM response (guaranteed valid JSON)
            allowed_asset_types: List of allowed asset type strings

        Returns:
            List of validated Asset objects

        Raises:
            ValidationError: If response invalid or assets don't match schema
            AIResponseError: If JSON cannot be parsed
        """
        try:
            # With structured outputs, response is guaranteed valid JSON
            response_data = json.loads(response_text.strip())

            # Extract assets array
            if not isinstance(response_data, dict) or "assets" not in response_data:
                raise AIResponseError(
                    "Response missing 'assets' key",
                    model=self.model_name,
                    details={"response_preview": response_text[:200]}
                )

            assets_data = response_data["assets"]

            if not isinstance(assets_data, list):
                raise AIResponseError(
                    "Assets field is not an array",
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
        min_assets: Optional[int] = None,
        max_assets: Optional[int] = None,
        services: Optional[List[Service]] = None,
        organization_profile: Optional[Dict[str, Any]] = None,
        user_instructions: Optional[str] = None,
        previous_assets: Optional[List[Asset]] = None,
        session_id: Optional[str] = None,
        org_metadata: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[Asset], UsageMetrics, Dict[str, Any]]:
        """
        Generate assets from website content and services.

        Args:
            website_content: Website text content
            language: Output language (en/sk/cs)
            min_assets: Minimum number of assets (optional, if not provided, no minimum enforced)
            max_assets: Maximum number of assets (optional, if not provided, no maximum enforced)
            services: Optional services from previous step
            organization_profile: Optional organization metadata
            user_instructions: Optional user refinement instructions
            previous_assets: Optional previous assets for refinement
            session_id: Session ID for Langfuse tracing

        Returns:
            Tuple of (assets list, usage metrics, debug info dict)

        Raises:
            ValidationRetriesExhausted: If validation fails after all retries
            AITimeoutError: If LLM request times out
            AIResponseError: If response cannot be parsed
        """

        # Extract allowed asset types from Asset model (for validation)
        allowed_asset_type_names = self._allowed_asset_types_from_model()
        if not allowed_asset_type_names:
            raise RuntimeError("Failed to extract asset types from Asset model")

        # Load asset types with descriptions from catalog (for prompt context)
        from workshops_ai.wizard.prompt_context import PromptContextBuilder
        builder = PromptContextBuilder(session_id=session_id)
        asset_types_with_desc = await builder.get_catalog_with_descriptions("asset_types")

        # Build prompt sections
        sections = self._build_prompt_sections(
            website_content=website_content,
            services=services,
            organization_profile=organization_profile,
            user_instructions=user_instructions,
            previous_assets=previous_assets,
        )

        # Build count constraint text for prompt
        if min_assets is not None and max_assets is not None:
            count_constraint = f"Generate between {min_assets} and {max_assets} IT and business assets."
        elif min_assets is not None:
            count_constraint = f"Generate at least {min_assets} IT and business assets."
        elif max_assets is not None:
            count_constraint = f"Generate up to {max_assets} IT and business assets."
        else:
            count_constraint = "Generate IT and business assets based on the organization's needs."

        # Build full prompt for debugging
        prompt_text = self._build_full_prompt_text(
            language=language,
            min_count=min_assets,
            max_count=max_assets,
            asset_types=allowed_asset_type_names,
            sections=sections
        )

        # Try to load prompts from Langfuse with fallback to local
        langfuse_client = get_langfuse_client()

        # Wrap everything in top-level span if Langfuse is available
        if langfuse_client and session_id:
            from langfuse import propagate_attributes

            # Prepare span input data
            span_input = {
                "language": language,
                "min_assets": min_assets,
                "max_assets": max_assets,
                "allowed_asset_types_count": len(allowed_asset_type_names),
                "organization_profile": organization_profile,
                "user_instructions_preview": user_instructions[:100] + "..." if user_instructions and len(user_instructions) > 100 else user_instructions,
                "has_previous_assets": bool(previous_assets),
                "previous_assets_count": len(previous_assets) if previous_assets else 0,
                "services_count": len(services) if services else 0,
                "website_content_length": len(website_content) if website_content else 0,
            }

            # Merge org metadata into span metadata
            span_metadata = {
                "endpoint": "assets",
                "model": self.model_name,
            }
            if org_metadata:
                span_metadata.update(org_metadata)

            with langfuse_client.start_as_current_span(
                name="generate-assets",
                input=span_input,
                metadata=span_metadata,
            ) as main_span:
                with propagate_attributes(session_id=session_id, metadata=org_metadata):
                    # Fetch prompt with child span
                    with langfuse_client.start_as_current_span(
                        name="get-assets-prompts",
                        input={"prompt_name": "wizard/assets"},
                    ) as prompt_span:
                        # Try to fetch unified prompt from Langfuse
                        prompt = get_langchain_prompt_from_langfuse(
                            prompt_name="wizard/assets",
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
                                ("system", ASSETS_SYSTEM_PROMPT),
                                ("human", ASSETS_HUMAN_PROMPT_TEMPLATE),
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
                                "name": "wizard/assets",
                                "isFallback": prompt_source == "local",
                                "version": prompt_version,
                                "type": prompt_type,
                                "tags": prompt_tags,
                                "labels": prompt_labels,
                                "prompt": prompt_content,
                            }
                        )

                    # Call LLM with retry logic
                    try:
                        # Format asset types with descriptions as JSON for better LLM understanding
                        asset_types_json = json.dumps(asset_types_with_desc, ensure_ascii=False, indent=2)

                        response_text, usage_metrics = await self._call_llm_with_retry(
                            prompt=prompt,
                            session_id=session_id,
                            count_constraint=count_constraint,
                            language=language,
                            asset_types=asset_types_json,
                            assets_json_schema=ASSETS_JSON_SCHEMA,
                            **sections
                        )

                        # Parse and validate with dedicated span
                        # Try to count raw assets before validation
                        raw_assets_count = None
                        try:
                            # With structured outputs, response is {"assets": [...]}
                            response_data = json.loads(response_text.strip())
                            if isinstance(response_data, dict) and "assets" in response_data:
                                assets_data = response_data["assets"]
                                if isinstance(assets_data, list):
                                    raw_assets_count = len(assets_data)
                        except Exception:
                            pass  # If counting fails, proceed without it

                        # Parse JSON for span input
                        try:
                            response_json = json.loads(response_text.strip())
                        except Exception:
                            response_json = {"error": "Failed to parse JSON", "preview": response_text[:200]}

                        with langfuse_client.start_as_current_span(
                            name="parse-and-validate-assets",
                            input={
                                "response_json": response_json if isinstance(response_json, dict) else {"assets_count": len(response_json)},
                                "allowed_asset_types_count": len(allowed_asset_type_names),
                                "raw_assets_count": raw_assets_count,
                            }
                        ) as parse_span:
                            try:
                                assets = self._parse_and_validate_assets(
                                    response_text=response_text,
                                    allowed_asset_types=allowed_asset_type_names
                                )

                                # Apply max limit if specified
                                if max_assets is not None:
                                    assets = assets[:max_assets]

                                parse_span.update(
                                    output={
                                        "assets_count": len(assets),
                                        "assets": [a.model_dump() for a in assets[:5]],  # First 5 for preview
                                    }
                                )
                            except (ValidationError, AIResponseError) as e:
                                parse_span.update(
                                    metadata={
                                        "status": "error",
                                        "error": str(e),
                                        "error_type": type(e).__name__
                                    }
                                )
                                raise

                        # Update main span with enhanced output
                        main_span.update(
                            output={
                                "assets_count": len(assets),
                                "response_preview": response_text[:200],
                                "prompt_source": prompt_source,
                                "prompt_version": prompt_version,
                                "usage": {
                                    "total_tokens": usage_metrics.total_tokens,
                                    "prompt_tokens": usage_metrics.prompt_tokens,
                                    "completion_tokens": usage_metrics.completion_tokens,
                                    "cost_usd": round(usage_metrics.cost_usd, 8),
                                }
                            },
                            metadata={"status": "success"}
                        )

                        # Build debug info
                        debug_info = {
                            "prompt_text": prompt_text,
                            "response_text": response_text[:500],  # Truncate for logging
                            "assets_count": len(assets),
                            "allowed_asset_types": allowed_asset_type_names,
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
                        main_span.update(
                            metadata={"status": "error", "error": str(e)}
                        )
                        raise ValidationRetriesExhausted(
                            retries=self.settings.wizard.llm_max_retries,
                            errors=[str(e)]
                        ) from e

        else:
            # No Langfuse - proceed without tracing
            # Try to fetch unified prompt from Langfuse (even without tracing)
            prompt = get_langchain_prompt_from_langfuse(
                prompt_name="wizard/assets",
                fallback_template=None,
            )

            # If Langfuse fetch failed, build fallback from local templates
            if prompt is None:
                logger.info("Building fallback prompt from local templates (no Langfuse)")
                prompt = ChatPromptTemplate.from_messages([
                    ("system", ASSETS_SYSTEM_PROMPT),
                    ("human", ASSETS_HUMAN_PROMPT_TEMPLATE),
                ])

            # Call LLM with retry logic
            try:
                # Format asset types with descriptions as JSON for better LLM understanding
                asset_types_json = json.dumps(asset_types_with_desc, ensure_ascii=False, indent=2)

                response_text, usage_metrics = await self._call_llm_with_retry(
                    prompt=prompt,
                    session_id=session_id,
                    count_constraint=count_constraint,
                    language=language,
                    asset_types=asset_types_json,
                    assets_json_schema=ASSETS_JSON_SCHEMA,
                    **sections
                )

                # Parse and validate
                assets = self._parse_and_validate_assets(
                    response_text=response_text,
                    allowed_asset_types=allowed_asset_type_names
                )

                # Apply max limit if specified
                if max_assets is not None:
                    assets = assets[:max_assets]

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
