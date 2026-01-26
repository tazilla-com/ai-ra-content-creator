# workshops_ai/wizard/exceptions.py
"""
Custom exceptions for wizard endpoints.
"""
from __future__ import annotations

from typing import Optional, Dict, Any


class WizardException(Exception):
    """Base exception for all wizard-related errors."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class ContentNotFoundError(WizardException):
    """Raised when website content cannot be resolved."""

    def __init__(self, message: str = "No website content available", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            code="content_not_found",
            message=message,
            status_code=400,
            details=details
        )


class ScrapingError(WizardException):
    """Raised when website scraping fails."""

    def __init__(self, url: str, reason: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            code="scraping_failed",
            message=f"Failed to scrape {url}: {reason}",
            status_code=400,
            details=details or {"url": url, "reason": reason}
        )


class AIResponseError(WizardException):
    """Raised when AI response is invalid or cannot be parsed."""

    def __init__(self, message: str, model: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            code="ai_response_error",
            message=message,
            status_code=500,
            details=details or {"model": model}
        )


class AITimeoutError(WizardException):
    """Raised when AI request times out."""

    def __init__(self, timeout_sec: int, model: str):
        super().__init__(
            code="ai_timeout",
            message=f"AI request timed out after {timeout_sec}s",
            status_code=504,
            details={"timeout_sec": timeout_sec, "model": model}
        )


class ValidationRetriesExhausted(WizardException):
    """Raised when LLM validation fails after all retries."""

    def __init__(self, retries: int, errors: list):
        super().__init__(
            code="validation_retries_exhausted",
            message=f"LLM output validation failed after {retries} retries",
            status_code=500,
            details={"retries": retries, "errors": errors}
        )


class ModelNotAllowedError(WizardException):
    """Raised when requested model is not in allowed list."""

    def __init__(self, model: str, allowed_models: list):
        super().__init__(
            code="model_not_allowed",
            message=f"Model '{model}' not allowed. Must be one of: {', '.join(allowed_models)}",
            status_code=400,
            details={"model": model, "allowed_models": allowed_models}
        )


class RequestTooLargeError(WizardException):
    """Raised when request exceeds per-request token limit for the model."""

    def __init__(
        self,
        estimated_tokens: int,
        model_limit: int,
        model: str,
        current_batch_size: int,
        suggested_batch_size: Optional[int] = None
    ):
        super().__init__(
            code="request_too_large",
            message=f"Request estimated at {estimated_tokens} tokens exceeds model limit {model_limit} for {model}",
            status_code=400,
            details={
                "estimated_tokens": estimated_tokens,
                "model_limit": model_limit,
                "model": model,
                "current_batch_size": current_batch_size,
                "suggested_batch_size": suggested_batch_size,
            }
        )
        self.estimated_tokens = estimated_tokens
        self.model_limit = model_limit
        self.model = model
        self.current_batch_size = current_batch_size
        self.suggested_batch_size = suggested_batch_size
