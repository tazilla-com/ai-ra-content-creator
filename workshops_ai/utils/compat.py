# File: workshops_ai/utils/compat.py

# Purpose:
#   Backwards-compat helpers for places where `org_id` is not yet wired
#   end-to-end. We preserve current behavior (silently drop `org_id`), but
#   introduce **cleaner logging** that:
#     - stays silent in single-tenant mode (default),
#     - warns **once** per function in multi-tenant mode if `org_id` is
#       missing/ignored.
#
# Design goals:
#   - Zero behavior change unless you explicitly enable multi-tenant in settings.
#   - No import cycles: settings are read lazily.
#   - Strong typing (Py 3.11), English docstrings, minimal surface.
#
# How to use:
#   1) Decorate legacy functions that don't accept `org_id` with
#      `@accepts_org_id_kw`.
#   2) For call sites where a decorator is not feasible, call
#      `drop_org_id_kw(kwargs, fn_name="...")`.
#
# Example:
#   @accepts_org_id_kw
#   def assets_list(...): ...
#
#   def search_assets(..., **kwargs):
#       drop_org_id_kw(kwargs, fn_name="search_assets")
#       ...
#
# Tenancy behavior:
#   - single-tenant (default): no logs (noise-free).
#   - multi-tenant: one-time WARNING per function if `org_id` is missing or
#     if it was provided but ignored (for visibility during the migration).
# -----------------------------------------------------------------------------

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional, ParamSpec, TypeVar, Dict
import logging

logger = logging.getLogger("workshops_ai.utils.compat")

P = ParamSpec("P")
R = TypeVar("R")

# Keep track of functions we've already warned about to avoid log spam.
_warned_once: set[tuple[str, str]] = set()


def _get_tenancy() -> str:
    """
    Returns the current tenancy mode ("single" | "multi").
    - Defaults to "single" if settings are unavailable.
    - Import is lazy to avoid circular imports during app bootstrap.
    """
    try:
        from workshops_ai.settings import settings  # lazy import
        return str(getattr(settings, "tenancy", "single") or "single").lower()
    except Exception:
        return "single"


def _normalize_org_id(org_id: Optional[str]) -> Optional[str]:
    """Normalize org_id: trim and coerce empty strings to None."""
    if org_id is None:
        return None
    s = str(org_id).strip()
    return s or None


def maybe_warn_org_id_ignored(*, fn: str, org_id: Optional[str]) -> None:
    """
    Emit a one-time WARNING per function **only in multi-tenant mode** when:
      - org_id is missing (None/empty), or
      - org_id was provided but will be ignored by legacy code paths.

    In single-tenant mode this remains completely silent.

    Args:
        fn: Function name for context in logs.
        org_id: The provided org_id (may be None/empty).
    """
    if _get_tenancy() != "multi":
        return  # stay silent in single-tenant

    tag = "missing" if _normalize_org_id(org_id) is None else "ignored"
    key = (fn, tag)
    if key in _warned_once:
        return

    _warned_once.add(key)
    logger.warning(
        {
            "event": f"compat.org_id_{tag}_in_multi_tenant",
            "fn": fn,
        }
    )


def drop_org_id_kw(kwargs: Dict[str, Any], *, fn_name: str) -> None:
    """
    Helper for call sites where a decorator is not applicable.
    - Removes `org_id` from kwargs if present.
    - Logs (once) per function in multi-tenant mode.

    Usage:
        def some_func(..., **kwargs):
            drop_org_id_kw(kwargs, fn_name="some_func")
            ...
    """
    if "org_id" in kwargs:
        org_id = kwargs.pop("org_id", None)
        maybe_warn_org_id_ignored(fn=fn_name, org_id=org_id)
    else:
        # org_id not passed at all; in multi-tenant we warn once to surface it
        maybe_warn_org_id_ignored(fn=fn_name, org_id=None)


def accepts_org_id_kw(fn: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that allows passing `org_id` to legacy functions that don't accept it.
    - Behavior: drops `org_id` and calls the original function unchanged.
    - Logging policy:
        * single-tenant: silent
        * multi-tenant: warn once per function whether `org_id` is missing or ignored
    """
    @wraps(fn)
    def _wrap(*args: P.args, **kwargs: P.kwargs) -> R:
        if "org_id" in kwargs:
            org_id = kwargs.pop("org_id", None)
            maybe_warn_org_id_ignored(fn=fn.__name__, org_id=org_id)
        else:
            maybe_warn_org_id_ignored(fn=fn.__name__, org_id=None)
        return fn(*args, **kwargs)

    return _wrap
