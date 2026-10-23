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

    # Tokens/cost from prior cancelled calls that we accumulated server-side
    # and rolled into this successful response. recovered_cost_usd is INCLUDED
    # in cost_usd above (i.e. cost_usd is the cumulative total).
    includes_recovered: bool = False
    recovered_cost_usd: float = 0.0


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

    # Calculate cost (pricing is per 1K tokens)
    prompt_cost = (prompt_tokens / 1_000) * model_pricing["prompt"]
    completion_cost = (completion_tokens / 1_000) * model_pricing["completion"]

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


def merge_with_pending(
    current: UsageMetrics,
    pending: dict[str, dict[str, float]] | None,
) -> UsageMetrics:
    """
    Roll a per-model pending-usage ledger into the current call's UsageMetrics.

    Args:
        current: UsageMetrics of the call about to be returned to the client.
        pending: Per-model aggregate of recovered tokens, shape:
            {model_name: {"prompt_tokens": int, "completion_tokens": int, "cost_usd": float}}
            (cost was already computed per-model when accumulated, so we just sum.)

    Returns:
        New UsageMetrics with tokens/cost summed across the current call and all
        pending entries. `model` stays the current call's model (this is the model
        that produced the response payload); `includes_recovered` is set when any
        pending tokens were merged in, and `recovered_cost_usd` reports the total
        recovered cost separately for transparency.
    """
    if not pending:
        return current

    extra_prompt = 0
    extra_completion = 0
    extra_cost = 0.0
    for _model, agg in pending.items():
        extra_prompt += int(agg.get("prompt_tokens", 0))
        extra_completion += int(agg.get("completion_tokens", 0))
        extra_cost += float(agg.get("cost_usd", 0.0))

    if extra_prompt == 0 and extra_completion == 0 and extra_cost == 0.0:
        return current

    return UsageMetrics(
        prompt_tokens=current.prompt_tokens + extra_prompt,
        completion_tokens=current.completion_tokens + extra_completion,
        total_tokens=current.total_tokens + extra_prompt + extra_completion,
        model=current.model,
        cost_usd=current.cost_usd + extra_cost,
        includes_recovered=True,
        recovered_cost_usd=extra_cost,
    )
