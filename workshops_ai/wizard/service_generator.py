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
)
from pydantic import ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from workshops_ai.settings import get_settings
from workshops_ai.wizard.schemas import Service, SECTOR_VALUES
from workshops_ai.wizard.prompts import (
    SERVICES_SYSTEM_PROMPT,
    SERVICES_HUMAN_PROMPT_TEMPLATE,
)
from workshops_ai.wizard.cost_calculator import create_usage_metrics, UsageMetrics
from workshops_ai.wizard.exceptions import (
    AIResponseError,
    AITimeoutError,
    ValidationRetriesExhausted,
    ModelNotAllowedError,
)

logger = logging.getLogger(__name__)


class ServiceGenerator:
    """
    Generates core business services from website content using LangChain + OpenAI.
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initialize ServiceGenerator with specified model.

        Args:
            model: OpenAI model name (defaults to settings if not provided)

        Raises:
            ModelNotAllowedError: If model not in allowed list
        """
        settings = get_settings()
        self.settings = settings

        # Resolve model
        self.model_name = model or settings.llm.model

        # Validate model against allowed list
        allowed_models = settings.wizard.allowed_models
        if self.model_name not in allowed_models:
            raise ModelNotAllowedError(self.model_name, allowed_models)

        # Initialize LangChain ChatOpenAI
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=settings.llm.temperature,
            openai_api_key=settings.llm.api_key,
            timeout=settings.wizard.llm_timeout_sec,
            max_retries=0,  # We handle retries ourselves with tenacity
        )

        logger.info(
            "ServiceGenerator initialized",
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
        user_instructions: Optional[str],
        previous_services: Optional[List[Service]],
    ) -> Dict[str, str]:
        """
        Build prompt sections from inputs.

        Args:
            website_content: Website text content
            organization_profile: Optional org metadata
            user_instructions: Optional user refinement instructions
            previous_services: Optional previous services for refinement

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

        # User instructions section
        if user_instructions:
            sections["user_instructions_section"] = (
                f"User Instructions (PRIORITY):\n{user_instructions}\n"
            )
        else:
            sections["user_instructions_section"] = ""

        # Previous services section (for refinement)
        if previous_services:
            services_json = json.dumps(
                [s.model_dump() for s in previous_services],
                indent=2
            )
            sections["previous_services_section"] = (
                f"Previous Services (refine these):\n{services_json}\n"
            )
        else:
            sections["previous_services_section"] = ""

        # Website content section (truncate if needed)
        max_length = self.settings.wizard.max_content_length
        truncated_content = website_content[:max_length]
        if len(website_content) > max_length:
            truncated_content += "\n... [content truncated]"

        sections["website_content_section"] = (
            f"Website Content:\n{truncated_content}\n"
        )

        return sections

    def _build_full_prompt_text(
        self,
        language: str,
        number: int,
        sections: Dict[str, str]
    ) -> str:
        """
        Build full prompt text for debugging.

        Returns truncated version for logging.
        """
        human_prompt = SERVICES_HUMAN_PROMPT_TEMPLATE.format(
            number=number,
            language=language,
            valid_sectors=", ".join(SECTOR_VALUES),
            **sections
        )

        full_prompt = f"{SERVICES_SYSTEM_PROMPT}\n\n{human_prompt}"

        # Truncate for logging
        max_log_length = 500
        if len(full_prompt) > max_log_length:
            return full_prompt[:max_log_length] + "... [truncated for logging]"
        return full_prompt

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
        response_text: str
    ) -> List[Service]:
        """
        Parse JSON response and validate services.

        Args:
            response_text: Raw LLM response

        Returns:
            List of validated Service objects

        Raises:
            ValidationError: If response invalid or services don't match schema
            AIResponseError: If JSON cannot be parsed
        """
        try:
            # Try to extract JSON array from response
            # Handle cases where LLM includes extra text
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
            services_data = json.loads(json_text)

            if not isinstance(services_data, list):
                raise AIResponseError(
                    "Response is not a JSON array",
                    model=self.model_name
                )

            # Validate each service
            services = []
            for idx, service_dict in enumerate(services_data):
                try:
                    # Validate sector value
                    sector = service_dict.get("sector")
                    if sector and sector not in SECTOR_VALUES:
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
        number_of_services: int = 3,
        organization_profile: Optional[Dict[str, Any]] = None,
        user_instructions: Optional[str] = None,
        previous_services: Optional[List[Service]] = None,
    ) -> Tuple[List[Service], UsageMetrics, Dict[str, Any]]:
        """
        Generate services from website content.

        Args:
            website_content: Website text content
            language: Output language (en/sk/cs)
            number_of_services: Number of services to generate (1-3)
            organization_profile: Optional organization metadata
            user_instructions: Optional user refinement instructions
            previous_services: Optional previous services for refinement

        Returns:
            Tuple of (services list, usage metrics, debug info dict)

        Raises:
            ValidationRetriesExhausted: If validation fails after all retries
            AITimeoutError: If LLM request times out
            AIResponseError: If response cannot be parsed
        """
        # Clamp number of services to valid range
        number_of_services = max(1, min(3, number_of_services))

        # Build prompt sections
        sections = self._build_prompt_sections(
            website_content=website_content,
            organization_profile=organization_profile,
            user_instructions=user_instructions,
            previous_services=previous_services,
        )

        # Build full prompt for debugging
        prompt_text = self._build_full_prompt_text(
            language=language,
            number=number_of_services,
            sections=sections
        )

        # Create LangChain prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SERVICES_SYSTEM_PROMPT),
            ("human", SERVICES_HUMAN_PROMPT_TEMPLATE),
        ])

        # Call LLM with retry logic
        try:
            response_text, usage_metrics = await self._call_llm_with_retry(
                prompt=prompt,
                number=number_of_services,
                language=language,
                valid_sectors=", ".join(SECTOR_VALUES),
                **sections
            )

            # Parse and validate
            services = self._parse_and_validate_services(response_text)

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
