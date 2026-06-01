# workshops_ai/wizard/pending_usage.py
"""
Pending-usage ledger for cancelled wizard calls.

When the client closes the HTTP connection before our response is sent (e.g. the
user navigates away), Azure has already consumed tokens but the response never
reaches the client. We persist that usage into the session keyed by model, and
roll it into the next successful response so the client sees cumulative cost.

Storage shape (under `session.wizard_metadata["pending_usage"]`):

    {
        "gpt-5-mini": {"prompt_tokens": 4500, "completion_tokens": 2100, "cost_usd": 0.00532},
        "gpt-4.1":    {"prompt_tokens": 800,  "completion_tokens": 200,  "cost_usd": 0.00320},
    }

Per-model aggregate, so size is bounded by the number of allowed models (~8).
No per-entry history is kept — the client only needs the cumulative sums.
"""
from __future__ import annotations

import logging
from typing import Optional

from workshops_ai.session import get_session, save_session, SessionState
from workshops_ai.wizard.cost_calculator import UsageMetrics

logger = logging.getLogger(__name__)

_LEDGER_KEY = "pending_usage"


async def accumulate_pending_usage(session_id: str, usage: UsageMetrics) -> None:
    """
    Add the given UsageMetrics to the session's pending ledger, keyed by model.

    Called when we detect that the client has disconnected before we could deliver
    the response. The cost was already incurred on Azure; we save it so the next
    successful call can include it.
    """
    if usage.total_tokens == 0 and usage.cost_usd == 0.0:
        return

    session = await get_session(session_id)
    if not session:
        # No session to attach to — log and drop. Without a session we have no
        # way to deliver this to a future call.
        logger.warning(
            "pending_usage.accumulate skipped: session not found",
            extra={"session_id": session_id, "model": usage.model},
        )
        return

    ledger: dict[str, dict[str, float]] = session.wizard_metadata.get(_LEDGER_KEY, {}) or {}
    entry = ledger.get(usage.model, {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
    entry["prompt_tokens"] = int(entry.get("prompt_tokens", 0)) + int(usage.prompt_tokens)
    entry["completion_tokens"] = int(entry.get("completion_tokens", 0)) + int(usage.completion_tokens)
    entry["cost_usd"] = float(entry.get("cost_usd", 0.0)) + float(usage.cost_usd)
    ledger[usage.model] = entry

    session.wizard_metadata[_LEDGER_KEY] = ledger
    await save_session(session)

    logger.info(
        "pending_usage.accumulated",
        extra={
            "session_id": session_id,
            "model": usage.model,
            "added_tokens": usage.total_tokens,
            "added_cost_usd": usage.cost_usd,
            "ledger_model_count": len(ledger),
        },
    )


async def flush_pending_usage(session_id: str) -> Optional[dict[str, dict[str, float]]]:
    """
    Read and clear the pending-usage ledger for this session.

    Returns the per-model aggregate dict (same shape as stored) or None if empty.
    Caller is responsible for merging this into the response's UsageMetrics
    via `cost_calculator.merge_with_pending`.
    """
    session = await get_session(session_id)
    if not session:
        return None

    ledger = session.wizard_metadata.get(_LEDGER_KEY)
    if not ledger:
        return None

    # Clear before returning so a transient failure between flush and response
    # doesn't double-count next time. If the response itself fails to deliver,
    # the disconnect path will re-accumulate the merged usage.
    session.wizard_metadata[_LEDGER_KEY] = {}
    await save_session(session)

    logger.info(
        "pending_usage.flushed",
        extra={
            "session_id": session_id,
            "model_count": len(ledger),
            "total_cost_usd": sum(float(e.get("cost_usd", 0.0)) for e in ledger.values()),
        },
    )

    return ledger
