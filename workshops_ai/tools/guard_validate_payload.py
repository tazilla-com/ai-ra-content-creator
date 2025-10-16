# File: workshops_ai/tools/guard_validate_payload.py
# Purpose:
#   Fail-closed validation helpers for all API/agent edges.
#   - Enforces strict Pydantic v2 parsing (no silent coercion beyond your schema).
#   - Rejects models that are not configured with `extra="forbid"` (opt-out via flag).
#   - Provides list/array validation via TypeAdapter.
#   - Redacts sensitive fields in error logs.
#   - Provides a structured "write blocked" error with a clear confirmation hint
#     (CZ/SK/EN) so the UI/agent can guide the user when writes are gated.
#
# Usage:
#   from workshops_ai.tools.guard_validate_payload import (
#       guard_validate_payload,
#       guard_validate_list,
#       raise_write_blocked,
#       WriteBlockedError,
#   )
#
#   # Validation
#   ui_req = guard_validate_payload(UIRequest, payload_dict)
#   items  = guard_validate_list(ImpactItem, payload_list)  # returns list[ImpactItem]
#
#   # Write-gate (use in manager/graphs when staging==False or policy requires preview)
#   if not can_write:
#       raise_write_blocked("create", {"staging": False, "reason": "preview_required"}, language="sk")
#
# Notes:
#   - Keep schemas strict: use `StrictStr`, `StrictBool`, numeric bounds, and `extra="forbid"`.
#   - This module is dependency-light; do not import app/backends here.

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Type, TypeVar, Union, overload, NoReturn

from pydantic import BaseModel, ValidationError
from pydantic.type_adapter import TypeAdapter

__all__ = [
    "GuardValidationError",
    "WriteBlockedError",
    "guard_validate_payload",
    "guard_validate_list",
    "raise_write_blocked",
    "safe_model_dump",
    "redact_for_log",
]

T = TypeVar("T", bound=BaseModel)


# ------------------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------------------

class GuardValidationError(Exception):
    """Uniform validation error used across the app (wraps pydantic's ValidationError)."""

    def __init__(self, message: str, *, errors: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


class WriteBlockedError(GuardValidationError):
    """
    Specialized error used when a write action is gated (e.g., staging is off, preview required).

    The `.errors` array contains one structured object with fields:
      {
        "kind": "write_blocked",
        "intent": "create" | "update" | "<other>",
        "language": "cs" | "sk" | "en",
        "preview_required": true,
        "confirm_phrase": "Potvrdit vytvoření" | "Potvrdiť vytvorenie" | "Confirm create" | ...,
        "message": "<short localized instruction>",
        "details": {...}   # redacted, best-effort snapshot for telemetry/UI
      }
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__("Write operation is blocked", errors=[payload])
        self.payload = payload


# ------------------------------------------------------------------------------
# Public API (validation)
# ------------------------------------------------------------------------------

@overload
def guard_validate_payload(model: Type[T], payload: T, *, require_forbid_extra: bool = True, allow_instance: bool = True) -> T: ...
@overload
def guard_validate_payload(model: Type[T], payload: Mapping[str, Any], *, require_forbid_extra: bool = True, allow_instance: bool = True) -> T: ...
def guard_validate_payload(
    model: Type[T],
    payload: Union[T, Mapping[str, Any]],
    *,
    require_forbid_extra: bool = True,
    allow_instance: bool = True,
) -> T:
    """
    Validate a single object against a Pydantic v2 model in a fail-closed way.
    - If `payload` is already an instance of `model` and `allow_instance=True`, it is returned
      after revalidation via .model_validate() to ensure config rules apply.
    - If the model does not enforce `extra="forbid"` and `require_forbid_extra=True`,
      a GuardValidationError is raised.
    """
    _assert_model_is_strict(model, require_forbid_extra=require_forbid_extra)

    try:
        if allow_instance and isinstance(payload, model):
            # Re-validate the instance's dict to re-apply 'extra'/'str_strip_whitespace' rules
            data = payload.model_dump(mode="python")
            return model.model_validate(data)
        return model.model_validate(payload)
    except ValidationError as ve:
        raise GuardValidationError(
            f"{model.__name__} validation failed",
            errors=_sanitize_errors(ve.errors()),
        ) from ve


def guard_validate_list(
    model: Type[T],
    payload: Any,
    *,
    require_forbid_extra: bool = True,
) -> list[T]:
    """
    Validate a list/array of objects against a Pydantic v2 model using TypeAdapter.
    The whole collection fails fast if any item is invalid.
    """
    _assert_model_is_strict(model, require_forbid_extra=require_forbid_extra)

    adapter = TypeAdapter(list[model])  # type: ignore[index]
    try:
        return adapter.validate_python(payload)
    except ValidationError as ve:
        raise GuardValidationError(
            f"list[{model.__name__}] validation failed",
            errors=_sanitize_errors(ve.errors()),
        ) from ve


# ------------------------------------------------------------------------------
# Public API (write-gate)
# ------------------------------------------------------------------------------

def raise_write_blocked(
    intent: str,
    details: Mapping[str, Any] | None = None,
    *,
    language: str | None = None,
) -> NoReturn:
    """
    Raise a structured WriteBlockedError with a clear confirmation phrase.
    Localizes to CZ / SK / EN based on `language`.
    """
    lang = _normalize_lang(language)
    phrase = _confirm_phrase(intent, lang)
    msg = _blocked_message(intent, lang, phrase)

    payload = {
        "kind": "write_blocked",
        "intent": str(intent or "").strip().lower(),
        "language": lang,
        "preview_required": True,
        "confirm_phrase": phrase,
        "message": msg,
        "details": redact_for_log(dict(details) if isinstance(details, Mapping) else {}),
    }
    raise WriteBlockedError(payload)


# ------------------------------------------------------------------------------
# Utilities (safe dumps/redaction)
# ------------------------------------------------------------------------------

def safe_model_dump(obj: Any, *, max_len: int = 2048) -> Any:
    """
    Return a serializable snapshot suitable for logs/telemetry:
      - For BaseModel: returns model_dump() (no secrets exposed unless schema marks them).
      - For mappings/sequences: returns a shallow copy with sensitive keys redacted.
      - Caps resulting JSON-ish string length via naive truncation if needed (best-effort).
    """
    try:
        if isinstance(obj, BaseModel):
            data = obj.model_dump(mode="python")
        elif isinstance(obj, Mapping):
            data = dict(obj)  # shallow copy
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            data = list(obj)  # shallow copy
        else:
            data = obj

        redacted = redact_for_log(data)
        s = repr(redacted)
        if len(s) > max_len:
            return s[:max_len] + "…"
        return redacted
    except Exception:
        return "<unserializable>"


def redact_for_log(
    obj: Any,
    *,
    sensitive_keys: Iterable[str] = ("password", "token", "secret", "authorization", "api_key", "apikey", "bearer"),
) -> Any:
    """
    Redact sensitive values by key name (case-insensitive).
    Works for nested dicts/lists. Returns a redacted clone (does not mutate input).
    """
    try:
        if isinstance(obj, Mapping):
            out: dict[str, Any] = {}
            for k, v in obj.items():
                if _is_sensitive_key(k, sensitive_keys):
                    out[str(k)] = "***REDACTED***"
                else:
                    out[str(k)] = redact_for_log(v, sensitive_keys=sensitive_keys)
            return out
        if isinstance(obj, list):
            return [redact_for_log(v, sensitive_keys=sensitive_keys) for v in obj]
        if isinstance(obj, tuple):
            return tuple(redact_for_log(v, sensitive_keys=sensitive_keys) for v in obj)
        return obj
    except Exception:
        return "<redaction-error>"


# ------------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------------

def _assert_model_is_strict(model: Type[T], *, require_forbid_extra: bool) -> None:
    """
    Ensure the model is configured for strict boundary validation.
    By default, requires `extra="forbid"` to avoid silent extra fields.
    """
    if not require_forbid_extra:
        return
    cfg = getattr(model, "model_config", {}) or {}
    extra_cfg = cfg.get("extra")
    if extra_cfg != "forbid":
        raise GuardValidationError(
            f"Model {model.__name__} must set model_config.extra='forbid' for fail-closed validation."
        )


def _sanitize_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize pydantic errors:
      - Convert loc tuples to dot-paths for readability.
      - Redact values from ctx if present and sensitive.
    """
    sanitized: list[dict[str, Any]] = []
    for e in errors:
        loc = e.get("loc")
        if isinstance(loc, (list, tuple)):
            path = ".".join(str(x) for x in loc)
        else:
            path = str(loc)

        entry = {
            "path": path,
            "type": e.get("type"),
            "msg": e.get("msg"),
        }
        ctx = e.get("ctx") or {}
        if isinstance(ctx, dict):
            entry["ctx"] = redact_for_log(ctx)
        sanitized.append(entry)
    return sanitized


def _is_sensitive_key(key: Any, sensitive: Iterable[str]) -> bool:
    if key is None:
        return False
    key_s = str(key).lower()
    return any(s in key_s for s in sensitive)


def _normalize_lang(lang: str | None) -> str:
    """
    Map arbitrary language hints to the primary code used by the app.
    Returns "cs" | "sk" | "en".
    """
    l = (lang or "").strip().lower()
    if l.startswith("cs"):
        return "cs"
    if l.startswith("sk"):
        return "sk"
    return "en"


def _confirm_phrase(intent: str, lang: str) -> str:
    """
    Return the exact confirmation phrase the user should type to allow a one-shot write.
    Keep it SHORT and STABLE so UI tests can assert it.
    """
    it = (intent or "").strip().lower()
    if lang == "cs":
        if it == "create":
            return "Potvrdit vytvoření"
        if it == "update":
            return "Potvrdit změnu"
        return "Potvrdit akci"
    if lang == "sk":
        if it == "create":
            return "Potvrdiť vytvorenie"
        if it == "update":
            return "Potvrdiť zmenu"
        return "Potvrdiť akciu"
    # EN fallback
    if it == "create":
        return "Confirm create"
    if it == "update":
        return "Confirm update"
    return "Confirm action"


def _blocked_message(intent: str, lang: str, phrase: str) -> str:
    """
    Short human-facing instruction explaining why the write is blocked and how to proceed once.
    """
    it = (intent or "").strip().lower()
    if lang == "cs":
        if it == "create":
            return f"Změny jsou zamčené (staging je vypnutý). Napište „{phrase}“ a já aktivum vytvořím."
        return f"Změny jsou zamčené (staging je vypnutý). Napište „{phrase}“ a já akci jednorázově provedu."
    if lang == "sk":
        if it == "create":
            return f"Zmeny sú zamknuté (staging je vypnutý). Napíšte „{phrase}“ a ja aktívum vytvorím."
        return f"Zmeny sú zamknuté (staging je vypnutý). Napíšte „{phrase}“ a ja akciu jednorazovo vykonám."
    # EN
    if it == "create":
        return f"Writes are locked (staging is off). Type “{phrase}” and I will create the asset once."
    return f"Writes are locked (staging is off). Type “{phrase}” and I will execute the action once."
