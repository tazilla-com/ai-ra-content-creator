# File: workshops_ai/graph/policies.py
# Purpose: Centralized policy switches for conversational graphs.
# - Controls preview/apply vs emit-only, clarify-first behavior, history window, and per-tab timeouts.
# - Provides safe defaults and helpers to compute an EffectivePolicy per request/tab.
# - Adds tiny interop helpers (clamp_list, history_tail, should_preview, attach_policy_to_deps)
#   so graphs can consistently respect policies without duplicating logic.

from __future__ import annotations

from enum import StrEnum
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar

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
    Global (cross-domain) policy flags. Merged into an effective per-tab policy
    for each turn. Keep minimal and strictly typed.
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
        description=(
            "If true, the agent asks a concise clarifying question before defaulting to list. "
            "This is 'one question per turn', not 'only one question in total'."
        ),
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
    NOTE: We intentionally do not put language here; the agent decides based on user input,
    and the graph passes locale/language separately to the LLM prompts.
    """
    return {
        "policy": {
            "emit_changes_enabled": effective.emit_changes_enabled,
            "preview_required": effective.preview_required,
            # clarify_first is a behavior knob for the decision layer, not a template toggle,
            # so we do not surface it here unless your prompt explicitly needs it.
        }
    }


# ----------------------------- Lightweight Helpers ----------------------------- #
T = TypeVar("T")

def clamp_list(items: Sequence[T] | Iterable[T], effective: EffectivePolicy) -> Tuple[List[T], int]:
    """
    Clamp a list/iterable to the tab's default visible size.
    Returns (sliced_items, total_count).
    Use this in read-only list nodes to respect policy.max_list_items while still reporting total.
    """
    if not isinstance(items, Sequence):
        items = list(items)
    total = len(items)  # evaluate once
    limit = max(1, int(effective.max_list_items))
    return list(items[:limit]), total


def history_tail(history: Sequence[dict] | None, effective: EffectivePolicy) -> list[dict]:
    """
    Return the last N conversation messages according to policy.max_history_messages.
    Graphs should prefer this over hardcoded lengths, e.g. history_tail(state.history, eff).
    """
    if not history:
        return []
    n = max(0, int(effective.max_history_messages))
    return list(history[-n:]) if n > 0 else []


def should_preview(effective: EffectivePolicy) -> bool:
    """
    Decide whether a write turn should stage instead of submitting right away.
    - True if preview_required is True.
    - You can also treat (emit_changes_enabled=False) as a strong hint to stage.
    """
    return bool(effective.preview_required or not effective.emit_changes_enabled)


def attach_policy_to_deps(deps: Any, effective: EffectivePolicy) -> None:
    """
    Best-effort helper to make the effective policy visible to graph nodes without
    threading it through every function signature. Manager/runner can call:

        eff = make_effective_policy(tab=TabName.ASSETS, ...)
        attach_policy_to_deps(deps, eff)

    Asset/Impacts graphs then read it via `getattr(deps, "effective_policy", None)`.
    """
    try:
        setattr(deps, "effective_policy", effective)
    except Exception:
        # Keep silent; this is a convenience only.
        pass


# ----------------------------- Example (comments) ----------------------------- #
# Example usage in a manager / router (pseudocode):
#
#   eff = make_effective_policy(
#       tab=TabName.ASSETS,
#       staging=request.staging,                        # True → preview only
#       overrides={"timeout_seconds": 15},              # per-request tweak
#   )
#   attach_policy_to_deps(deps, eff)                   # <- make available to graph nodes
#   prompt_vars = as_prompt_vars(eff)                  # <- pass into system prompt templating
#   # pass `eff.timeout_seconds` to graph invoke timeout
#
# In graphs:
#   effective = getattr(deps, "effective_policy", None)  # may be None if manager omitted
#   if effective and should_preview(effective):
#       # Stage instead of submit; set UIResponse.staging.needs_commit = True, etc.
#
# The EffectivePolicy is also safe to serialize into UIResponse.policy if desired.
