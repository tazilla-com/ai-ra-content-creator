# workshops_ai/wizard/cost_calculator.py
"""
OpenAI cost calculation utilities.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class UsageMetrics(BaseModel):
    """Token usage and cost metrics for an LLM call."""
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cost_usd: float


def calculate_openai_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    pricing: dict[str, dict[str, float]]
) -> float:
    """
    Calculate OpenAI API cost based on token usage.

    Args:
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        model: Model name (e.g., 'gpt-4o-mini')
        pricing: Pricing table from settings (WizardSettings.openai_pricing)

    Returns:
        Total cost in USD

    Note:
        If model not found in pricing table, defaults to gpt-4o-mini pricing
    """
    # Get pricing for model (default to gpt-4o-mini if unknown)
    model_pricing = pricing.get(model, pricing.get("gpt-4o-mini", {"prompt": 0.00015, "completion": 0.0006}))

    # Calculate cost (pricing is per 1M tokens)
    prompt_cost = (prompt_tokens / 1_000_000) * model_pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * model_pricing["completion"]

    return prompt_cost + completion_cost


def create_usage_metrics(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    pricing: dict[str, dict[str, float]]
) -> UsageMetrics:
    """
    Create UsageMetrics object with calculated cost.

    Args:
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        model: Model name
        pricing: Pricing table from settings (WizardSettings.openai_pricing)

    Returns:
        UsageMetrics object with all fields populated
    """
    total_tokens = prompt_tokens + completion_tokens
    cost_usd = calculate_openai_cost(
        prompt_tokens, completion_tokens, model, pricing
    )

    return UsageMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model=model,
        cost_usd=cost_usd
    )
