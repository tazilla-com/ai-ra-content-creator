from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class ThrottledChatOpenAI(ChatOpenAI):
    """ChatOpenAI wrapper s OpenAI throttling."""

    async def ainvoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
        from workshops_ai.llm.openai_throttle import get_throttle
        from workshops_ai.settings import get_settings
        from pybreaker import CircuitBreakerError

        settings = get_settings()

        if not settings.openai_throttle.enabled:
            return await super().ainvoke(input, config, **kwargs)

        try:
            throttle = get_throttle()
        except RuntimeError:
            logger.warning("Throttle not initialized, skipping")
            return await super().ainvoke(input, config, **kwargs)

        messages = self._convert_input_to_messages(input)
        model = kwargs.get("model") or self.model_name

        try:
            async with throttle.acquire(messages, model, kwargs.get("max_tokens")):
                response = await super().ainvoke(input, config, **kwargs)
                throttle.update_from_response(response)
                throttle.record_success()
                return response
        except CircuitBreakerError:
            logger.error("Circuit breaker OPEN - OpenAI unavailable")
            raise
        except Exception as e:
            throttle.record_failure(e)
            raise

    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
        from workshops_ai.llm.openai_throttle import get_throttle
        from workshops_ai.settings import get_settings
        from pybreaker import CircuitBreakerError

        settings = get_settings()

        if not settings.openai_throttle.enabled:
            return super().invoke(input, config, **kwargs)

        try:
            throttle = get_throttle()
        except RuntimeError:
            logger.warning("Throttle not initialized, skipping")
            return super().invoke(input, config, **kwargs)

        import asyncio

        messages = self._convert_input_to_messages(input)
        model = kwargs.get("model") or self.model_name

        async def _throttled_call():
            try:
                async with throttle.acquire(messages, model, kwargs.get("max_tokens")):
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


# Override class name for better display in Langfuse/tracing tools
ThrottledChatOpenAI.__name__ = "ChatOpenAI"
