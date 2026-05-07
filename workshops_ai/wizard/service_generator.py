# workshops_ai/wizard/service_generator.py
"""
Service generation using LangChain and OpenAI.
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
    RetryCallState,
)
from pydantic import ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from workshops_ai.settings import get_settings
from workshops_ai.wizard.schemas import Service
from workshops_ai.wizard.prompts import (
    SERVICES_SYSTEM_PROMPT,
    SERVICES_HUMAN_PROMPT_TEMPLATE,
    SERVICES_JSON_SCHEMA,
)
from workshops_ai.llm.langfuse_helper import get_langchain_prompt_from_langfuse
from workshops_ai.wizard.cost_calculator import create_usage_metrics, UsageMetrics
from workshops_ai.wizard.exceptions import (
    AIResponseError,
    AITimeoutError,
    ValidationRetriesExhausted,
    ModelNotAllowedError,
)
from workshops_ai.wizard.prompt_context import PromptContextBuilder

logger = logging.getLogger(__name__)


class ServiceGenerator:
    """
    Generates core business services from website content using LangChain + OpenAI.
    """

    def __init__(self, model: Optional[str] = None, reasoning_effort: Optional[str] = None):
        """
        Initialize ServiceGenerator with specified model.

        Args:
            model: OpenAI model name (defaults to settings if not provided)
            reasoning_effort: Override reasoning effort (cascade: body > env > default)

        Raises:
            ModelNotAllowedError: If model not in allowed list
        """
        from workshops_ai.llm.wizard_client import create_wizard_llm_client
        from workshops_ai.wizard.schemas import ServicesResponse

        settings = get_settings()
        self.settings = settings
        self.reasoning_effort = reasoning_effort

        # Create LLM client with structured outputs (Pydantic model)
        self.llm = create_wizard_llm_client(
            model=model,
            response_format=ServicesResponse,
            reasoning_effort=reasoning_effort,
        )
        self.model_name = self.llm.model_name
        self._current_retry_attempt = 1  # Track current retry attempt

        logger.info(
            "ServiceGenerator initialized",
            extra={
                "model": self.model_name,
                "temperature": settings.wizard.temperature,
                "timeout_sec": settings.wizard.llm_timeout_sec,
                "structured_outputs": True,
            }
        )

    def _build_full_prompt_text(
        self,
        language: str,
        min_count: Optional[int],
        max_count: Optional[int],
        sections: Dict[str, str],
        valid_sectors: List[str]
    ) -> str:
        """
        Build full prompt text for debugging.

        Returns truncated version for logging.
        """
        # Build count constraint text
        if min_count is not None and max_count is not None:
            count_constraint = f"Generate between {min_count} and {max_count} core business services."
        elif min_count is not None:
            count_constraint = f"Generate at least {min_count} core business services."
        elif max_count is not None:
            count_constraint = f"Generate up to {max_count} core business services."
        else:
            count_constraint = "Generate core business services based on the organization's needs."

        human_prompt = SERVICES_HUMAN_PROMPT_TEMPLATE.format(
            count_constraint=count_constraint,
            language=language,
            valid_sectors=", ".join(valid_sectors),
            services_json_schema=SERVICES_JSON_SCHEMA,
            **sections
        )

        full_prompt = f"{SERVICES_SYSTEM_PROMPT}\n\n{human_prompt}"

        # Truncate for logging
        max_log_length = 500
        if len(full_prompt) > max_log_length:
            return full_prompt[:max_log_length] + "... [truncated for logging]"
        return full_prompt

    def _before_retry(self, retry_state: RetryCallState) -> None:
        """Track retry attempt number before each retry."""
        self._current_retry_attempt = retry_state.attempt_number

    @retry(
        stop=stop_after_attempt(3),  # Must match settings.wizard.llm_max_retries default
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
            from workshops_ai.llm.langfuse_helper import (
                get_langfuse_callback_handler,
                get_langfuse_client
            )

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
                        "temperature": self.settings.wizard.temperature,
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

    def _parse_and_validate_services(
        self,
        response_text: str,
        valid_sectors: List[str]
    ) -> List[Service]:
        """
        Parse JSON response and validate services.

        With OpenAI structured outputs, response is guaranteed to be valid JSON
        in format: {"services": [...]}

        Args:
            response_text: Raw LLM response (guaranteed valid JSON)
            valid_sectors: List of valid sector names

        Returns:
            List of validated Service objects

        Raises:
            ValidationError: If response invalid or services don't match schema
            AIResponseError: If JSON cannot be parsed
        """
        try:
            # With structured outputs, response is guaranteed valid JSON
            response_data = json.loads(response_text.strip())

            # Extract services array
            if not isinstance(response_data, dict) or "services" not in response_data:
                raise AIResponseError(
                    "Response missing 'services' key",
                    model=self.model_name,
                    details={"response_preview": response_text[:200]}
                )

            services_data = response_data["services"]

            if not isinstance(services_data, list):
                raise AIResponseError(
                    "Services field is not an array",
                    model=self.model_name
                )

            # Validate each service
            services = []
            for idx, service_dict in enumerate(services_data):
                try:
                    # Validate sector value
                    sector = service_dict.get("sector")
                    if sector and sector not in valid_sectors:
                        logger.warning(
                            f"Invalid sector '{sector}' at index {idx}, setting to None"
                        )
                        service_dict["sector"] = None

                    service = Service(**service_dict)
                    services.append(service)

                except ValidationError as e:
                    logger.error(
                        f"Service validation failed at index {idx}",
                        extra={"error": str(e), "data": service_dict}
                    )
                    raise

            return services

        except json.JSONDecodeError as e:
            raise AIResponseError(
                f"Failed to parse JSON: {str(e)}",
                model=self.model_name,
                details={"response_preview": response_text[:200]}
            ) from e

    async def generate_services(
        self,
        website_content: str,
        language: str = "en",
        min_services: Optional[int] = None,
        max_services: Optional[int] = None,
        organization_profile: Optional[Dict[str, Any]] = None,
        user_instructions: Optional[str] = None,
        previous_services: Optional[List[Service]] = None,
        session_id: Optional[str] = None,
        include_catalogs: Optional[List[str]] = None,
        org_metadata: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[Service], UsageMetrics, Dict[str, Any]]:
        """
        Generate services from website content.

        Args:
            website_content: Website text content
            language: Output language (en/sk/cs)
            min_services: Minimum number of services (optional, if not provided, no minimum enforced)
            max_services: Maximum number of services (optional, if not provided, no maximum enforced)
            organization_profile: Optional organization metadata
            user_instructions: Optional user refinement instructions
            previous_services: Optional previous services for refinement
            session_id: Session ID for client-specific catalogs
            include_catalogs: Optional list of catalogs to include in prompt (e.g., ["sectors", "impacts"])

        Returns:
            Tuple of (services list, usage metrics, debug info dict)

        Raises:
            ValidationRetriesExhausted: If validation fails after all retries
            AITimeoutError: If LLM request times out
            AIResponseError: If response cannot be parsed
        """

        # Initialize prompt context builder with session-based caching
        builder = PromptContextBuilder(session_id=session_id)

        # Load valid sectors from catalog (already localized by Tazilla backend)
        valid_sectors = await builder.get_catalog_names("sectors")

        # Build prompt sections using centralized builder
        sections = await builder.build_sections(
            website_content=website_content,
            organization_profile=organization_profile,
            user_instructions=user_instructions,
            previous_services=previous_services,
            include_catalogs=include_catalogs,  # Optional catalogs to include in prompt
        )

        # Build count constraint text for prompt
        if min_services is not None and max_services is not None:
            count_constraint = f"Generate between {min_services} and {max_services} core business services."
        elif min_services is not None:
            count_constraint = f"Generate at least {min_services} core business services."
        elif max_services is not None:
            count_constraint = f"Generate up to {max_services} core business services."
        else:
            count_constraint = "Generate core business services based on the organization's needs."

        # Build full prompt for debugging
        prompt_text = self._build_full_prompt_text(
            language=language,
            min_count=min_services,
            max_count=max_services,
            sections=sections,
            valid_sectors=valid_sectors
        )

        # Try to load prompts from Langfuse with fallback to local
        from workshops_ai.llm.langfuse_helper import (
            get_langchain_prompt_from_langfuse,
            get_langfuse_client,
        )

        langfuse_client = get_langfuse_client()

        # Wrap everything in top-level span if Langfuse is available
        if langfuse_client and session_id:
            from langfuse import propagate_attributes

            # Prepare span input data
            span_input = {
                "language": language,
                "min_services": min_services,
                "max_services": max_services,
                "valid_sectors_count": len(valid_sectors),
                "organization_profile": organization_profile,
                "user_instructions_preview": user_instructions[:100] + "..." if user_instructions and len(user_instructions) > 100 else user_instructions,
                "has_previous_services": bool(previous_services),
                "previous_services_count": len(previous_services) if previous_services else 0,
                "website_content_length": len(website_content) if website_content else 0,
            }

            # Merge org metadata into span metadata
            span_metadata = {
                "endpoint": "services",
                "model": self.model_name,
            }
            if org_metadata:
                span_metadata.update(org_metadata)

            with langfuse_client.start_as_current_span(
                name="generate-services",
                input=span_input,
                metadata=span_metadata,
            ) as main_span:
                with propagate_attributes(session_id=session_id, metadata=org_metadata):
                    # Fetch prompt with child span
                    requested_label = self.settings.langfuse.prompt_label
                    with langfuse_client.start_as_current_span(
                        name="get-services-prompts",
                        input={
                            "prompt_name": "wizard/services",
                            "requested_label": requested_label,
                        },
                    ) as prompt_span:
                        # Try to fetch unified prompt from Langfuse
                        prompt = get_langchain_prompt_from_langfuse(
                            prompt_name="wizard/services",
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
                                ("system", SERVICES_SYSTEM_PROMPT),
                                ("human", SERVICES_HUMAN_PROMPT_TEMPLATE),
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
                                "name": "wizard/services",
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
                        response_text, usage_metrics = await self._call_llm_with_retry(
                            prompt=prompt,
                            session_id=session_id,
                            count_constraint=count_constraint,
                            language=language,
                            valid_sectors=", ".join(valid_sectors),
                            services_json_schema=SERVICES_JSON_SCHEMA,
                            **sections
                        )

                        # Parse and validate with dedicated span
                        # Try to count raw services before validation
                        raw_services_count = None
                        try:
                            # With structured outputs, response is {"services": [...]}
                            response_data = json.loads(response_text.strip())
                            if isinstance(response_data, dict) and "services" in response_data:
                                services_data = response_data["services"]
                                if isinstance(services_data, list):
                                    raw_services_count = len(services_data)
                        except Exception:
                            pass  # If counting fails, proceed without it

                        # Parse JSON for span input
                        try:
                            response_json = json.loads(response_text.strip())
                        except Exception:
                            response_json = {"error": "Failed to parse JSON", "preview": response_text[:200]}

                        with langfuse_client.start_as_current_span(
                            name="parse-and-validate-services",
                            input={
                                "response_json": response_json,
                                "valid_sectors_count": len(valid_sectors),
                                "raw_services_count": raw_services_count,
                            }
                        ) as parse_span:
                            try:
                                services = self._parse_and_validate_services(
                                    response_text=response_text,
                                    valid_sectors=valid_sectors
                                )

                                parse_span.update(
                                    output={
                                        "services_count": len(services),
                                        "services": [s.model_dump(mode='json') for s in services],
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
                                "services_count": len(services),
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
                            "services_count": len(services),
                        }

                        logger.info(
                            "Services generated successfully",
                            extra={
                                "count": len(services),
                                "model": self.model_name,
                                "tokens": usage_metrics.total_tokens,
                            }
                        )

                        return services, usage_metrics, debug_info

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
                prompt_name="wizard/services",
                fallback_template=None,
            )

            # If Langfuse fetch failed, build fallback from local templates
            if prompt is None:
                logger.info("Building fallback prompt from local templates (no Langfuse)")
                prompt = ChatPromptTemplate.from_messages([
                    ("system", SERVICES_SYSTEM_PROMPT),
                    ("human", SERVICES_HUMAN_PROMPT_TEMPLATE),
                ])

            # Call LLM with retry logic
            try:
                response_text, usage_metrics = await self._call_llm_with_retry(
                    prompt=prompt,
                    session_id=session_id,
                    count_constraint=count_constraint,
                    language=language,
                    valid_sectors=", ".join(valid_sectors),
                    services_json_schema=SERVICES_JSON_SCHEMA,
                    **sections
                )

                # Parse and validate
                services = self._parse_and_validate_services(
                    response_text=response_text,
                    valid_sectors=valid_sectors
                )

                # Build debug info
                debug_info = {
                    "prompt_text": prompt_text,
                    "response_text": response_text[:500],  # Truncate for logging
                    "services_count": len(services),
                }

                logger.info(
                    "Services generated successfully",
                    extra={
                        "count": len(services),
                        "model": self.model_name,
                        "tokens": usage_metrics.total_tokens,
                    }
                )

                return services, usage_metrics, debug_info

            except ValidationError as e:
                # Retries exhausted
                raise ValidationRetriesExhausted(
                    retries=self.settings.wizard.llm_max_retries,
                    errors=[str(e)]
                ) from e
