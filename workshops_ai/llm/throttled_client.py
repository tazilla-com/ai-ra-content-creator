from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_openai import ChatOpenAI

try:
    from langchain_openai import AzureChatOpenAI
except ImportError:
    AzureChatOpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


class ThrottledChatOpenAI(ChatOpenAI):
    """ChatOpenAI wrapper s OpenAI throttling."""

    async def ainvoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
        from workshops_ai.settings import get_settings

        settings = get_settings()

        if not settings.openai_throttle.enabled:
            return await super().ainvoke(input, config, **kwargs)

        try:
            from workshops_ai.llm.openai_throttle import get_throttle
            from pybreaker import CircuitBreakerError
            throttle = get_throttle()
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Throttle not available ({type(e).__name__}), skipping")
            return await super().ainvoke(input, config, **kwargs)

        messages = self._convert_input_to_messages(input)
        model = kwargs.get("model") or self.model_name
        batch_size = kwargs.pop("batch_size", None)  # Extract batch_size, don't pass to OpenAI
        expected_output_tokens = kwargs.pop("expected_output_tokens", None)  # Extract expected output, don't pass to OpenAI

        try:
            async with throttle.acquire(
                messages, model, kwargs.get("max_tokens"),
                batch_size=batch_size,
                expected_output_tokens=expected_output_tokens
            ):
                logger.info(f"ThrottledChatOpenAI: Before super().ainvoke, model={model}")
                response = await super().ainvoke(input, config, **kwargs)
                logger.info(f"ThrottledChatOpenAI: After super().ainvoke, response type={type(response).__name__}")
                throttle.update_from_response(response)
                logger.info(f"ThrottledChatOpenAI: After update_from_response")
                throttle.record_success()
                return response
        except CircuitBreakerError:
            logger.error("Circuit breaker OPEN - OpenAI unavailable")
            raise
        except Exception as e:
            logger.info(f"ThrottledChatOpenAI: Exception caught: {type(e).__name__}: {str(e)[:100]}")
            throttle.record_failure(e)
            raise

    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
        from workshops_ai.settings import get_settings

        settings = get_settings()

        if not settings.openai_throttle.enabled:
            return super().invoke(input, config, **kwargs)

        try:
            from workshops_ai.llm.openai_throttle import get_throttle
            from pybreaker import CircuitBreakerError
            throttle = get_throttle()
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Throttle not available ({type(e).__name__}), skipping")
            return super().invoke(input, config, **kwargs)

        import asyncio

        messages = self._convert_input_to_messages(input)
        model = kwargs.get("model") or self.model_name
        batch_size = kwargs.pop("batch_size", None)  # Extract batch_size, don't pass to OpenAI
        expected_output_tokens = kwargs.pop("expected_output_tokens", None)  # Extract expected output, don't pass to OpenAI

        async def _throttled_call():
            try:
                async with throttle.acquire(
                    messages, model, kwargs.get("max_tokens"),
                    batch_size=batch_size,
                    expected_output_tokens=expected_output_tokens
                ):
                    response = super(ThrottledChatOpenAI, self).invoke(input, config, **kwargs)
                    throttle.update_from_response(response)
                    throttle.record_success()
                    return response
            except CircuitBreakerError:
                logger.error("Circuit breaker OPEN - OpenAI unavailable")
                raise
            except Exception as e:
                throttle.record_failure(e)
                raise

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(_throttled_call())

    def _convert_input_to_messages(self, input: Any) -> list[dict]:
        if isinstance(input, list):
            messages = []
            for msg in input:
                if hasattr(msg, "content") and hasattr(msg, "type"):
                    messages.append({"role": msg.type, "content": msg.content})
                elif isinstance(msg, dict):
                    messages.append(msg)
            return messages
        elif isinstance(input, str):
            return [{"role": "user", "content": input}]
        return []


# Azure variant with same throttling
if AzureChatOpenAI is not None:
    class ThrottledAzureChatOpenAI(AzureChatOpenAI):
        """AzureChatOpenAI wrapper s OpenAI throttling."""

        def model_post_init(self, __context: Any) -> None:
            super().model_post_init(__context)
            # Azure doesn't set model_name; copy from deployment_name
            if not self.model_name and self.deployment_name:
                self.model_name = self.deployment_name

        async def ainvoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
            from workshops_ai.settings import get_settings
            settings = get_settings()
            if not settings.openai_throttle.enabled:
                return await super().ainvoke(input, config, **kwargs)
            try:
                from workshops_ai.llm.openai_throttle import get_throttle
                from pybreaker import CircuitBreakerError
                throttle = get_throttle()
            except (ImportError, RuntimeError) as e:
                logger.warning(f"Throttle not available ({type(e).__name__}), skipping")
                return await super().ainvoke(input, config, **kwargs)

            messages = ThrottledChatOpenAI._convert_input_to_messages(self, input)
            model = kwargs.get("model") or self.model_name or getattr(self, "deployment_name", None) or "gpt-4o"
            batch_size = kwargs.pop("batch_size", None)
            expected_output_tokens = kwargs.pop("expected_output_tokens", None)

            try:
                async with throttle.acquire(
                    messages, model, kwargs.get("max_tokens"),
                    batch_size=batch_size,
                    expected_output_tokens=expected_output_tokens
                ):
                    logger.info(f"ThrottledAzureChatOpenAI: Before super().ainvoke, model={model}")
                    response = await super().ainvoke(input, config, **kwargs)
                    logger.info(f"ThrottledAzureChatOpenAI: After super().ainvoke, response type={type(response).__name__}")
                    throttle.update_from_response(response)
                    throttle.record_success()
                    return response
            except CircuitBreakerError:
                logger.error("Circuit breaker OPEN - Azure OpenAI unavailable")
                raise
            except Exception as e:
                logger.info(f"ThrottledAzureChatOpenAI: Exception caught: {type(e).__name__}: {str(e)[:100]}")
                throttle.record_failure(e)
                raise

        def invoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
            from workshops_ai.settings import get_settings
            settings = get_settings()
            if not settings.openai_throttle.enabled:
                return super().invoke(input, config, **kwargs)
            try:
                from workshops_ai.llm.openai_throttle import get_throttle
                from pybreaker import CircuitBreakerError
                throttle = get_throttle()
            except (ImportError, RuntimeError) as e:
                logger.warning(f"Throttle not available ({type(e).__name__}), skipping")
                return super().invoke(input, config, **kwargs)

            import asyncio
            messages = ThrottledChatOpenAI._convert_input_to_messages(self, input)
            model = kwargs.get("model") or self.model_name or getattr(self, "deployment_name", None) or "gpt-4o"
            batch_size = kwargs.pop("batch_size", None)
            expected_output_tokens = kwargs.pop("expected_output_tokens", None)

            async def _throttled_call():
                try:
                    async with throttle.acquire(
                        messages, model, kwargs.get("max_tokens"),
                        batch_size=batch_size,
                        expected_output_tokens=expected_output_tokens
                    ):
                        response = super(ThrottledAzureChatOpenAI, self).invoke(input, config, **kwargs)
                        throttle.update_from_response(response)
                        throttle.record_success()
                        return response
                except CircuitBreakerError:
                    logger.error("Circuit breaker OPEN - Azure OpenAI unavailable")
                    raise
                except Exception as e:
                    throttle.record_failure(e)
                    raise

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(_throttled_call())

    ThrottledAzureChatOpenAI.__name__ = "AzureChatOpenAI"
else:
    ThrottledAzureChatOpenAI = None  # type: ignore


# Override class name for better display in Langfuse/tracing tools
ThrottledChatOpenAI.__name__ = "ChatOpenAI"
