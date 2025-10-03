# File: workshops_ai/graph/policies.py
# Purpose: Centralized policy switches for conversational graphs.
# - Controls preview/apply vs emit-only, clarify-first behavior, history window, and per-tab timeouts.
# - Provides safe defaults and helpers to compute an EffectivePolicy per request/tab.

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr


# ----------------------------- Tab Identification ----------------------------- #
class TabName(StrEnum):
    """Canonical tab names used throughout the app."""
    ASSETS = "assets"
    IMPACTS = "impacts"
    THREATS = "threats"


# ----------------------------- Policy Data Models ----------------------------- #
class GlobalPolicy(BaseModel):
    """
    Global (cross-domain) policy flags. These are merged into an effective
    per-tab policy for each turn. Keep these minimal and strictly typed.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Write behavior:
    emit_changes_enabled: StrictBool = Field(
        default=True,
        description="If true, the agent may apply changes immediately (send-now) once intent is explicit.",
    )
    preview_required: StrictBool = Field(
        default=False,
        description="If true, the agent must show a preview and ask for confirmation before any write.",
    )

    # Conversation behavior:
    clarify_first: StrictBool = Field(
        default=True,
        description="If true, the agent asks one concise clarifying question instead of falling back to 'list'.",
    )
    cross_domain_tools_allowed: StrictBool = Field(
        default=True,
        description="If true, the agent may call tools across domains without requiring a UI tab switch.",
    )

    # Memory window (shared logic; actual storage is handled elsewhere):
    max_history_messages: StrictInt = Field(
        default=6,
        ge=0,
        le=32,
        description="Rolling window size (number of recent messages) supplied to the agent.",
    )
    history_ttl_seconds: StrictInt = Field(
        default=7200,
        ge=60,
        description="TTL for conversation window (seconds).",
    )


class TabPolicy(BaseModel):
    """
    Per-tab execution policy that captures timeouts and small UX knobs.
    Provide conservative defaults; callers can override per request.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Hard timeout for a single graph turn (tooling + LLM decision).
    timeout_seconds: StrictInt = Field(
        default=12,
        ge=1,
        le=120,
        description="Per-turn timeout for this tab (seconds).",
    )

    # Optional display/UX guardrails; used by agent when summarizing lists.
    max_list_items: StrictInt = Field(
        default=5,
        ge=1,
        le=100,
        description="Max items to surface by default; agent can offer 'show more'.",
    )


class EffectivePolicy(BaseModel):
    """
    Fully materialized policy used by a graph turn (tab + global flags merged).
    This is what prompts and tool-routing will actually consume.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Scope
    tab: StrictStr = Field(description="Tab name (assets|impacts|threats).")

    # Global flags (copied from GlobalPolicy; may be overridden)
    emit_changes_enabled: StrictBool
    preview_required: StrictBool
    clarify_first: StrictBool
    cross_domain_tools_allowed: StrictBool

    # Memory
    max_history_messages: StrictInt
    history_ttl_seconds: StrictInt

    # Tab specifics
    timeout_seconds: StrictInt
    max_list_items: StrictInt


class PolicyOverrides(BaseModel):
    """
    Request-scoped overrides. Use sparingly; they modify behavior for the current turn.
    Any field left as None is ignored (no override).
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Writes
    emit_changes_enabled: Optional[StrictBool] = None
    preview_required: Optional[StrictBool] = None

    # Conversation
    clarify_first: Optional[StrictBool] = None
    cross_domain_tools_allowed: Optional[StrictBool] = None

    # Memory
    max_history_messages: Optional[StrictInt] = Field(default=None, ge=0, le=32)
    history_ttl_seconds: Optional[StrictInt] = Field(default=None, ge=60)

    # Tab specifics
    timeout_seconds: Optional[StrictInt] = Field(default=None, ge=1, le=120)
    max_list_items: Optional[StrictInt] = Field(default=None, ge=1, le=100)


# ----------------------------- Default Instances ----------------------------- #
DEFAULT_GLOBAL_POLICY = GlobalPolicy()

DEFAULT_TAB_POLICIES: dict[TabName, TabPolicy] = {
    TabName.ASSETS: TabPolicy(timeout_seconds=10, max_list_items=5),
    TabName.IMPACTS: TabPolicy(timeout_seconds=12, max_list_items=5),
    TabName.THREATS: TabPolicy(timeout_seconds=12, max_list_items=5),
}


# ----------------------------- Merge & Normalize ----------------------------- #
def _coerce_overrides(overrides: PolicyOverrides | Mapping[str, Any] | None) -> PolicyOverrides:
    """Accept typed overrides or plain dict and return a validated PolicyOverrides instance."""
    if overrides is None:
        return PolicyOverrides()
    if isinstance(overrides, PolicyOverrides):
        return overrides
    return PolicyOverrides.model_validate(overrides)


def _coerce_tab_policy(tab_policy: TabPolicy | Mapping[str, Any] | None, tab: TabName) -> TabPolicy:
    """Accept typed tab policy or plain dict and return a validated TabPolicy instance."""
    if tab_policy is None:
        return DEFAULT_TAB_POLICIES[tab]
    if isinstance(tab_policy, TabPolicy):
        return tab_policy
    return TabPolicy.model_validate(tab_policy)


def make_effective_policy(
    *,
    tab: TabName,
    global_policy: GlobalPolicy | None = None,
    tab_policy: TabPolicy | Mapping[str, Any] | None = None,
    overrides: PolicyOverrides | Mapping[str, Any] | None = None,
    staging: Optional[bool] = None,
) -> EffectivePolicy:
    """
    Compute the EffectivePolicy for a single graph turn.
    - Start with defaults (or provided global/tab policies).
    - Apply `staging` flip (if True → preview_required=True & emit_changes_enabled=False).
    - Apply request-scoped `overrides` (last-win).
    """
    g = (global_policy or DEFAULT_GLOBAL_POLICY).model_copy()
    t = _coerce_tab_policy(tab_policy, tab).model_copy()
    o = _coerce_overrides(overrides)

    # Staging flips: force preview, disable direct apply
    if staging is True:
        g.preview_required = True
        g.emit_changes_enabled = False

    # Apply overrides (if provided)
    if o.emit_changes_enabled is not None:
        g.emit_changes_enabled = o.emit_changes_enabled
    if o.preview_required is not None:
        g.preview_required = o.preview_required
    if o.clarify_first is not None:
        g.clarify_first = o.clarify_first
    if o.cross_domain_tools_allowed is not None:
        g.cross_domain_tools_allowed = o.cross_domain_tools_allowed
    if o.max_history_messages is not None:
        g.max_history_messages = o.max_history_messages
    if o.history_ttl_seconds is not None:
        g.history_ttl_seconds = o.history_ttl_seconds
    if o.timeout_seconds is not None:
        t.timeout_seconds = o.timeout_seconds
    if o.max_list_items is not None:
        t.max_list_items = o.max_list_items

    # Materialize effective policy
    return EffectivePolicy(
        tab=tab.value,
        emit_changes_enabled=g.emit_changes_enabled,
        preview_required=g.preview_required,
        clarify_first=g.clarify_first,
        cross_domain_tools_allowed=g.cross_domain_tools_allowed,
        max_history_messages=g.max_history_messages,
        history_ttl_seconds=g.history_ttl_seconds,
        timeout_seconds=t.timeout_seconds,
        max_list_items=t.max_list_items,
    )


# ----------------------------- Prompt Interop ----------------------------- #
def as_prompt_vars(effective: EffectivePolicy) -> dict[str, Any]:
    """
    Minimal variables consumed by the base system prompt. Keep this stable.
    """
    return {
        "policy": {
            "emit_changes_enabled": effective.emit_changes_enabled,
            "preview_required": effective.preview_required,
        }
    }


# ----------------------------- Example (comments) ----------------------------- #
# Example usage in a manager / router (pseudocode):
#
#   eff = make_effective_policy(
#       tab=TabName.ASSETS,
#       staging=request.staging,                        # True → preview only
#       overrides={"timeout_seconds": 15},              # per-request tweak
#   )
#   prompt_vars = as_prompt_vars(eff)
#   # pass `eff.timeout_seconds` to graph invoke timeout
#   # pass `prompt_vars` into system prompt templating
#
# The EffectivePolicy is also safe to serialize into UIResponse.policy if desired.
