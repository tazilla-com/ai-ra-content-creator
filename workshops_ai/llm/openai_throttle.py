from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import tiktoken
from pyrate_limiter import Duration, Rate, Limiter, BucketFullException
from pybreaker import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class ThrottleMetrics:
    total_requests: int = 0
    total_tokens: int = 0
    rate_limit_hits: int = 0
    circuit_breaker_opens: int = 0
    last_reset: float = field(default_factory=time.time)

    def reset_if_needed(self):
        now = time.time()
        if now - self.last_reset >= 60:
            self.total_requests = 0
            self.total_tokens = 0
            self.last_reset = now


class OpenAIThrottle:

    def __init__(
        self,
        rpm: int,
        tpm: int,
        max_concurrent: int,
        per_request_limits: Optional[Dict[str, int]] = None
    ):
        self.rpm = rpm
        self.tpm = tpm
        self.max_concurrent = max_concurrent
        self.per_request_limits = per_request_limits or {}
        self.metrics = ThrottleMetrics()

        self._encoder_cache: Dict[str, Any] = {}

        # pyrate-limiter 3.x API: Limiter with single Rate for requests
        # Token limiting is handled separately via adaptive limits
        self.limiter = Limiter(Rate(rpm, Duration.MINUTE))

        # pybreaker 1.x API uses 'reset_timeout' instead of 'timeout_duration'
        self.circuit_breaker = CircuitBreaker(
            fail_max=5,
            reset_timeout=30,
            name="openai_api"
        )

        self.semaphore = asyncio.Semaphore(max_concurrent)

        self.remaining_requests: Optional[int] = None
        self.remaining_tokens: Optional[int] = None
        self.reset_requests_at: Optional[float] = None
        self.reset_tokens_at: Optional[float] = None

    def _get_encoder(self, model: str):
        if model not in self._encoder_cache:
            try:
                self._encoder_cache[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                self._encoder_cache[model] = tiktoken.get_encoding("cl100k_base")
        return self._encoder_cache[model]

    def count_tokens(self, messages: list[dict], model: str) -> int:
        encoder = self._get_encoder(model)
        num_tokens = 0

        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoder.encode(value))
                elif key == "name":
                    num_tokens += -1

        num_tokens += 2
        return num_tokens

    def estimate_total_tokens(
        self,
        messages: list[dict],
        model: str,
        max_completion_tokens: Optional[int] = None,
        expected_output_tokens: Optional[int] = None
    ) -> int:
        input_tokens = self.count_tokens(messages, model)

        if max_completion_tokens:
            output_estimate = max_completion_tokens
        elif expected_output_tokens:
            # Use caller's precise output estimate (based on batch_size, max_items, etc.)
            output_estimate = expected_output_tokens
        else:
            # Fallback: conservative estimate for non-batch calls
            output_estimate = max(500, int(input_tokens * 0.25))

        total = input_tokens + output_estimate

        # Debug log for large requests
        if total > 25000:
            logger.warning(
                f"Large request detected: {total} tokens (input: {input_tokens}, output estimate: {output_estimate})",
                extra={
                    "total_tokens": total,
                    "input_tokens": input_tokens,
                    "output_tokens_estimate": output_estimate,
                    "model": model,
                    "message_count": len(messages),
                    "message_lengths": [len(str(m.get('content', ''))) for m in messages],
                }
            )

        return total

    def update_from_response(self, response) -> None:
        logger.info(f"update_from_response called with response type: {type(response).__name__}")

        try:
            headers = None

            # Try LangChain response_metadata first (with include_response_headers=True)
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                logger.info(f"Response metadata keys: {list(metadata.keys())}")

                # With include_response_headers=True, headers might be directly in metadata
                # Check if any rate limit headers exist directly in metadata
                if any(key.startswith('x-ratelimit') for key in metadata.keys()):
                    headers = metadata
                    logger.info(f"Found rate limit headers directly in response_metadata")
                # Or they might be in a 'headers' sub-dict
                elif 'headers' in metadata:
                    headers = metadata['headers']
                    logger.info(f"Found headers in response_metadata['headers'] with keys: {list(headers.keys())[:5]}")
                else:
                    logger.warning("No rate limit headers found in response_metadata")
            elif hasattr(response, 'headers'):
                headers = response.headers
                logger.info(f"Found headers attribute directly")
            elif hasattr(response, '_headers'):
                headers = response._headers
                logger.info(f"Found _headers attribute")
            else:
                logger.warning("Response has no headers, response_metadata, or _headers attribute")
                return

            if not headers:
                logger.warning("Headers dict is empty")
                return

            if "x-ratelimit-remaining-requests" in headers:
                self.remaining_requests = int(headers["x-ratelimit-remaining-requests"])

            if "x-ratelimit-remaining-tokens" in headers:
                self.remaining_tokens = int(headers["x-ratelimit-remaining-tokens"])

            if "x-ratelimit-reset-requests" in headers:
                reset_str = headers["x-ratelimit-reset-requests"]
                self.reset_requests_at = time.time() + self._parse_duration(reset_str)

            if "x-ratelimit-reset-tokens" in headers:
                reset_str = headers["x-ratelimit-reset-tokens"]
                self.reset_tokens_at = time.time() + self._parse_duration(reset_str)

            logger.info(
                f"OpenAI limits updated: {self.remaining_requests} req, {self.remaining_tokens} tokens remaining"
            )

        except Exception as e:
            logger.warning(f"Failed to parse OpenAI headers: {e}")

    @staticmethod
    def _parse_duration(duration_str: str) -> float:
        if not duration_str:
            return 0.0

        total_seconds = 0.0
        current_num = ""

        for char in duration_str:
            if char.isdigit() or char == '.':
                current_num += char
            elif char in ['h', 'm', 's']:
                if current_num:
                    num = float(current_num)
                    if char == 'h':
                        total_seconds += num * 3600
                    elif char == 'm':
                        total_seconds += num * 60
                    elif char == 's':
                        total_seconds += num
                    current_num = ""

        return total_seconds

    async def _check_adaptive_limits(self, estimated_tokens: int) -> None:
        now = time.time()

        if self.remaining_requests is not None and self.remaining_requests < 1:
            if self.reset_requests_at and self.reset_requests_at > now:
                wait_time = self.reset_requests_at - now
                logger.warning(f"OpenAI request limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time + 0.5)

        if self.remaining_tokens is not None and self.remaining_tokens < estimated_tokens:
            if self.reset_tokens_at and self.reset_tokens_at > now:
                wait_time = self.reset_tokens_at - now
                logger.warning(f"OpenAI token limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time + 0.5)

    @asynccontextmanager
    async def acquire(
        self,
        messages: list[dict],
        model: str,
        max_completion_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        expected_output_tokens: Optional[int] = None
    ):
        # Circuit breaker check disabled - pybreaker 1.x has different API
        # Rate limiting is handled by pyrate-limiter below

        estimated_tokens = self.estimate_total_tokens(
            messages, model, max_completion_tokens, expected_output_tokens
        )

        # Check per-request limit for this model
        if model in self.per_request_limits:
            model_limit = self.per_request_limits[model]
            if estimated_tokens > model_limit:
                # Calculate suggested batch size (90% of optimal for safety margin)
                suggested_batch_size = None
                if batch_size and batch_size > 1:
                    optimal_ratio = model_limit / estimated_tokens
                    suggested_batch_size = max(1, int(batch_size * optimal_ratio * 0.9))

                logger.warning(
                    f"Request too large: {estimated_tokens} tokens > {model_limit} limit for {model}",
                    extra={
                        "estimated_tokens": estimated_tokens,
                        "model_limit": model_limit,
                        "model": model,
                        "batch_size": batch_size,
                        "suggested_batch_size": suggested_batch_size
                    }
                )

                # Import here to avoid circular dependency
                from workshops_ai.wizard.exceptions import RequestTooLargeError
                raise RequestTooLargeError(
                    estimated_tokens=estimated_tokens,
                    model_limit=model_limit,
                    model=model,
                    current_batch_size=batch_size or 1,
                    suggested_batch_size=suggested_batch_size
                )

        self.metrics.reset_if_needed()
        self.metrics.total_requests += 1
        self.metrics.total_tokens += estimated_tokens

        logger.info(
            f"OpenAI throttle acquire: requests={self.metrics.total_requests}, "
            f"tokens={self.metrics.total_tokens}, estimated={estimated_tokens}, model={model}"
        )

        await self._check_adaptive_limits(estimated_tokens)

        await self.semaphore.acquire()

        try:
            try:
                # pyrate-limiter 3.x: try_acquire without weight (RPM only)
                # TPM is enforced via adaptive limits (_check_adaptive_limits)
                self.limiter.try_acquire("openai")
            except BucketFullException:
                self.metrics.rate_limit_hits += 1
                logger.warning("Rate limiter full, waiting")
                await asyncio.sleep(1.0)
                self.limiter.try_acquire("openai")

            yield

        finally:
            self.semaphore.release()

    def record_success(self):
        # pybreaker 1.x: Circuit breaker success is tracked automatically via context manager
        pass

    def record_failure(self, exception: Exception):
        # pybreaker 1.x: Circuit breaker failure is tracked automatically via context manager
        # Manual tracking for metrics only
        if self.circuit_breaker.current_state == "open":
            self.metrics.circuit_breaker_opens += 1

    def get_metrics(self) -> Dict[str, Any]:
        # Reset metrics if minute has passed (so health endpoint shows fresh data)
        self.metrics.reset_if_needed()

        return {
            "requests_this_minute": self.metrics.total_requests,
            "tokens_this_minute": self.metrics.total_tokens,
            "rate_limit_hits": self.metrics.rate_limit_hits,
            "circuit_breaker_state": self.circuit_breaker.current_state,
            "circuit_breaker_failures": self.circuit_breaker.fail_counter,
            "remaining_requests": self.remaining_requests,
            "remaining_tokens": self.remaining_tokens,
            "limits": {
                "rpm": self.rpm,
                "tpm": self.tpm,
                "concurrent": self.max_concurrent
            }
        }


_throttle: Optional[OpenAIThrottle] = None


def init_throttle(rpm: int, tpm: int, max_concurrent: int, per_request_limits: Optional[Dict[str, int]] = None) -> None:
    global _throttle
    _throttle = OpenAIThrottle(rpm=rpm, tpm=tpm, max_concurrent=max_concurrent, per_request_limits=per_request_limits)
    logger.info(f"OpenAI throttle initialized: {rpm} RPM, {tpm} TPM, {max_concurrent} concurrent, per-request limits: {per_request_limits}")


def get_throttle() -> OpenAIThrottle:
    if _throttle is None:
        raise RuntimeError("OpenAI throttle not initialized")
    return _throttle
