"""
Immutable context for Asset Agent.

Context is passed to agent at invocation time and accessible via
runtime.context in tools.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetAgentContext:
    """
    Immutable context passed to asset agent.

    Accessible via runtime.context in tools (LangChain 2025 pattern).
    Contains session-level identifiers that don't change during execution.
    """
    session_id: str
    workshop_id: str
