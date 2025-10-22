# File: workshops_ai/graph/_serde.py
# -----------------------------------------------------------------------------
# Purpose
# -------
# Utilities to coerce arbitrary Python objects (Pydantic models, lists/tuples,
# dicts, dataclasses, etc.) into *plain, JSON-serializable structures* with
# protections against:
#   - cyclic references (which otherwise trigger RecursionError),
#   - unbounded depth (depth cap),
#   - non-serializable object types (fallbacks to safe representations).
#
# Why this matters
# ----------------
# LangGraph checkpointers need to serialize the state tree. If we accidentally
# stash Pydantic models or objects with cycles inside state (notably
# `state.actions` or `state.agent_events`), checkpointing can fail or loop.
# These helpers ensure we only store plain dicts/lists/primitives.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Iterable, List, Dict, Set, Mapping, Sequence
import json
import dataclasses
import datetime as _dt
import decimal as _decimal
import uuid as _uuid
import enum as _enum
from collections import deque

try:
    # Optional: faster and more precise Pydantic detection (v2)
    from pydantic import BaseModel  # type: ignore
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore[misc,assignment]


__all__ = [
    "jsonable",
    "as_plain_dicts",
    "safe_json_dumps",
    "clean_event",
]


# ---- helpers ----------------------------------------------------------------


def _is_primitive(obj: Any) -> bool:
    """Return True if `obj` is already JSON-friendly and immutable-ish."""
    return obj is None or isinstance(obj, (str, int, float, bool))


def _is_mapping(obj: Any) -> bool:
    try:
        return isinstance(obj, Mapping)
    except Exception:
        return False


def _is_sequence(obj: Any) -> bool:
    try:
        return isinstance(obj, (list, tuple, set, frozenset, deque))
    except Exception:
        return False


def _is_pydantic_model(obj: Any) -> bool:
    """Return True if obj looks like a Pydantic v2 model instance."""
    if obj is None:
        return False
    # v2 models inherit BaseModel and have .model_dump()
    return isinstance(obj, BaseModel) or hasattr(obj, "model_dump")


def _json_roundtrip_strict(value: Any) -> Any:
    """
    Best-effort 'make it JSON by force':
    Convert `value` to a JSON string with default=str, then load it back to a
    plain Python structure. Safe, but may lose fidelity for exotic objects.
    """
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


# ---- public API --------------------------------------------------------------


def jsonable(obj: Any, *, max_depth: int = 8, _seen: Set[int] | None = None) -> Any:
    """
    Convert `obj` into a JSON-serializable structure.

    Guarantees:
      - No cycles (tracks object ids and emits '<cyclic-ref>' sentinel).
      - Depth limited (emits '<max-depth>' past `max_depth`).
      - Pydantic models are dumped via `.model_dump(mode='json')` if available.
      - Dataclasses are converted via `dataclasses.asdict`.
      - dict/list/tuple/set/deque are converted recursively to plain copies.
      - datetime/UUID/Decimal/Enum are turned into sensible string/primitive
        representations.
      - As a last resort, uses a JSON round-trip with `default=str`.

    Parameters
    ----------
    obj : Any
        The input object to convert.
    max_depth : int
        Maximum recursion depth. Prevents pathological nesting/cycles.
    _seen : set[int] | None
        Internal set of visited object ids (used to cut cycles).

    Returns
    -------
    Any
        A structure composed only of JSON-serializable primitives, lists, and
        dicts (with string keys).
    """
    if _seen is None:
        _seen = set()

    # Fast path for primitives
    if _is_primitive(obj):
        return obj

    # Cycle guard
    oid = id(obj)
    if oid in _seen:
        return "<cyclic-ref>"
    _seen.add(oid)

    # Depth guard
    if max_depth <= 0:
        return "<max-depth>"

    # Common "special" types with canonical JSON representations
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        # Use ISO 8601 consistently
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)

    if isinstance(obj, _dt.timedelta):
        # Represent as total seconds (float); string fallback if needed
        try:
            return obj.total_seconds()
        except Exception:
            return str(obj)

    if isinstance(obj, _decimal.Decimal):
        # Keep precision as string; callers can cast if they really want a float
        return str(obj)

    if isinstance(obj, _uuid.UUID):
        return str(obj)

    if isinstance(obj, _enum.Enum):
        # Prefer value; fall back to name if value isn't JSONable
        val = obj.value
        return val if _is_primitive(val) else str(obj)

    if isinstance(obj, (bytes, bytearray, memoryview)):
        # Avoid b64 wrappers; keep it simple and predictable
        return f"<bytes:{len(obj)}>"

    # Mapping types → plain dict (stringify keys defensively)
    if _is_mapping(obj):
        out: Dict[str, Any] = {}
        for k, v in obj.items():  # type: ignore[assignment]
            ks = str(k)
            out[ks] = jsonable(v, max_depth=max_depth - 1, _seen=_seen)
        return out

    # Sequence types → plain list
    if _is_sequence(obj):
        # Cast to list to avoid iterators/generators being consumed elsewhere
        return [jsonable(v, max_depth=max_depth - 1, _seen=_seen) for v in list(obj)]

    # Dataclass instances
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return jsonable(dataclasses.asdict(obj), max_depth=max_depth - 1, _seen=_seen)
        except Exception:
            # Fall through to round-trip
            pass

    # Pydantic models (v2 preferred)
    if _is_pydantic_model(obj):
        try:
            return obj.model_dump(mode="json")  # type: ignore[attr-defined]
        except Exception:
            # v1 compatibility (dict()) or fall through
            try:
                return obj.dict()  # type: ignore[attr-defined]
            except Exception:
                pass

    # Objects with an explicit JSON hook
    if hasattr(obj, "__json__"):
        try:
            return jsonable(obj.__json__(), max_depth=max_depth - 1, _seen=_seen)  # type: ignore[call-arg]
        except Exception:
            pass

    # Generic Python object with attributes → serialize its __dict__
    if hasattr(obj, "__dict__"):
        try:
            return jsonable(obj.__dict__, max_depth=max_depth - 1, _seen=_seen)
        except Exception:
            pass

    # Last resort: JSON round-trip or final string
    try:
        return _json_roundtrip_strict(obj)
    except Exception:
        return str(obj)


def as_plain_dicts(items: Iterable[Any] | None, *, max_depth: int = 8) -> List[Dict[str, Any]]:
    """
    Convert an iterable of action-like objects into a list of plain dicts.

    This is intentionally strict: callers (e.g., the graph) want to guarantee
    that `state.actions` is ALWAYS a list of *dicts* to avoid checkpointer
    surprises. Each element is passed through `jsonable()`.

    - If an element serializes to a dict → use it as-is.
    - If it serializes to a non-dict (e.g., str/number/list) → wrap it as
      `{"value": ...}` to preserve data shape while maintaining the list-of-dicts
      invariant. (In well-formed flows this path is not taken, because actions
      should already be dict-shaped.)

    Parameters
    ----------
    items : Iterable[Any] | None
        Sequence of objects (Pydantic models, dicts, etc.) to normalize.
    max_depth : int
        Passed to `jsonable()` for defensive depth limiting.

    Returns
    -------
    List[Dict[str, Any]]
        A brand-new list containing only plain dicts.
    """
    out: List[Dict[str, Any]] = []
    if not items:
        return out

    for it in list(items):
        val = jsonable(it, max_depth=max_depth)
        if isinstance(val, dict):
            # Ensure the dict is detached (no shared references, no weird keys)
            out.append(_json_roundtrip_strict(val))
        else:
            # Maintain the "list of dicts" invariant even for odd inputs
            out.append({"value": val})
    return out


def safe_json_dumps(obj: Any, *, max_depth: int = 8, **dumps_kwargs: Any) -> str:
    """
    Convenience helper: `json.dumps(jsonable(obj))` with sane defaults.

    Example:
        payload = safe_json_dumps(state)
        redis.set(key, payload)
    """
    payload = jsonable(obj, max_depth=max_depth)
    if "ensure_ascii" not in dumps_kwargs:
        dumps_kwargs["ensure_ascii"] = False
    if "default" not in dumps_kwargs:
        dumps_kwargs["default"] = str
    # Reasonable size/compat defaults (smaller payloads, still readable)
    dumps_kwargs.setdefault("separators", (",", ":"))
    return json.dumps(payload, **dumps_kwargs)


def clean_event(
    event: Mapping[str, Any] | None,
    *,
    drop_keys: Iterable[str] = ("state", "result", "raw", "response", "events"),
    max_depth: int = 6,
) -> Dict[str, Any]:
    """
    Sanitize a single agent/telemetry event so it is *safe* to embed in state
    (no cycles, no heavy nested blobs).

    - Performs JSON round-trip via `jsonable` to flatten exotic objects.
    - Drops problematic/heavy keys (configurable).
    - Ensures the result is a detached dict.

    Intended usage (example):
        state.agent_events = [clean_event(e) for e in state.agent_events][-50:]

    NOTE: This function lives here (serde) to avoid importing graph internals.
    """
    if not event:
        return {}

    # First, normalize the entire structure
    norm = jsonable(event, max_depth=max_depth)

    # Guarantee dict shape
    if not isinstance(norm, dict):
        norm = {"value": norm}

    # Drop heavy / cyclic-prone fields
    for k in list(norm.keys()):
        if k in drop_keys:
            norm.pop(k, None)

    # Detach to plain primitives
    return _json_roundtrip_strict(norm)
