# workshops_ai/app/services/outbox_writer.py
# -----------------------------------------------------------------------------
# Purpose
#   Submit a ChangeSet to the primary backend if available, otherwise
#   atomically persist the ChangeSet as JSON into the local OUTBOX directory.
#
# Design
#   - Fail-closed: strict typing, explicit exceptions.
#   - Atomic write: write → fsync → rename (same filesystem).
#   - JSON payload: stable, sorted keys, UTF-8 (no ASCII escaping).
#   - Idempotency: filename includes timestamp, kind (if present), and idempotency suffix.
#   - Pluggable submitter: supports async submitter (preferred) and a sync shim.
#
# IMPORTANT
#   OUTBOX path source has changed:
#     ✅ Primary:  get_settings().tazila.outbox_path
#     ⚠️  Deprecated fallback: get_settings().paths.outbox_dir (logs a warning)
#     ⬇️ Final fallback: /app/outbox
#
# Usage (async):
#   from workshops_ai.models.changeset import ChangeSet
#   from workshops_ai.app.services.outbox_writer import submit_or_fallback
#
#   result = await submit_or_fallback(changeset, submitter=maybe_client)
#   if result.fallback_path:
#       log.info("Changeset queued in outbox: %s", result.fallback_path)
#
# Notes
#   - This module is write-only. Readers or drains should be implemented elsewhere.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import ValidationError

from workshops_ai.models.changeset import ChangeSet
from workshops_ai.settings import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "OutboxWriteError",
    "SubmitError",
    "SubmitResult",
    "SubmitOrFallbackResult",
    "ensure_outbox_dir",
    "serialize_changeset",
    "write_changeset_to_outbox",
    "submit_or_fallback",
]


# ----- Exceptions -------------------------------------------------------------


class OutboxWriteError(RuntimeError):
    """Raised when we failed to persist a JSON ChangeSet into the outbox."""


class SubmitError(RuntimeError):
    """Raised when a submitter reports failure (non-2xx or transport error)."""


# ----- Submitter protocol & result -------------------------------------------


@runtime_checkable
class AsyncChangeSetSubmitter(Protocol):
    """Async submitter interface used by API clients."""

    async def submit_changeset(self, changeset: ChangeSet) -> "SubmitResult":
        """
        Submit a ChangeSet to the primary backend.

        Returns:
            SubmitResult: success status including optional HTTP info.
        """
        ...


@runtime_checkable
class SyncChangeSetSubmitter(Protocol):
    """Sync submitter interface (shimmed to async internally)."""

    def submit_changeset(self, changeset: ChangeSet) -> "SubmitResult":
        """
        Submit a ChangeSet to the primary backend (synchronous variant).

        Returns:
            SubmitResult: success status including optional HTTP info.
        """
        ...


@dataclass(frozen=True)
class SubmitResult:
    success: bool
    transport: str = "unknown"  # e.g., "http", "grpc", "local"
    status_code: Optional[int] = None
    response_body: Optional[Any] = None
    error: Optional[str] = None


# ----- Settings / paths -------------------------------------------------------


def _default_outbox_dir() -> Path:
    """
    Resolve the OUTBOX directory with the following precedence:

      1) settings.tazila.outbox_path                 (authoritative)
      2) settings.paths.outbox_dir (DEPRECATED)      (warn + use if set)
      3) '/app/outbox'                               (hard default)
    """
    try:
        settings = get_settings()

        # 1) New canonical location (authoritative)
        tazila = getattr(settings, "tazila", None)
        outbox_path = getattr(tazila, "outbox_path", None) if tazila is not None else None
        if outbox_path:
            return Path(str(outbox_path))

        # 2) Deprecated legacy setting (keep temporarily for compatibility)
        paths = getattr(settings, "paths", None)
        deprecated_outbox = getattr(paths, "outbox_dir", None) if paths is not None else None
        if deprecated_outbox:
            logger.warning(
                "DEPRECATED: settings.paths.outbox_dir is deprecated; "
                "configure settings.tazila.outbox_path instead."
            )
            return Path(str(deprecated_outbox))

    except Exception:
        # Settings may not be fully initialized in some test contexts
        pass

    # 3) Hard default
    return Path("/app/outbox")


def ensure_outbox_dir(path: Optional[Path] = None) -> Path:
    """
    Ensure the OUTBOX directory exists with safe permissions.
    """
    out_dir = Path(path) if path else _default_outbox_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Best-effort perm tightening (does not fail hard if unsupported)
    try:
        os.chmod(out_dir, 0o750)
    except Exception:
        pass
    return out_dir.resolve()


# ----- Serialization helpers --------------------------------------------------


def serialize_changeset(changeset: Any) -> dict:
    """
    Accept ChangeSet | dict and always return a JSON-serializable dict.
    If a dict is passed, validate/normalize via ChangeSet.model_validate().
    """
    if isinstance(changeset, ChangeSet):
        # Already validated
        return changeset.model_dump(
            mode="json",
            by_alias=True,
            serialize_as_any=True,
            exclude_none=True,
        )
    try:
        # Validate unknown mapping into our model (raises on mismatch)
        cs = ChangeSet.model_validate(changeset)
        return cs.model_dump(
            mode="json",
            by_alias=True,
            serialize_as_any=True,
            exclude_none=True,
        )
    except ValidationError:
        # Bubble up pydantic validation details
        raise


def _safe_attr_or_key(obj: Any, key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Helper that tries attribute first, then mapping key, else default.
    """
    val = getattr(obj, key, None)
    if val is not None:
        return str(val)
    if isinstance(obj, dict):
        v = obj.get(key)
        if v is not None:
            return str(v)
        # Common nested location for idempotency in some payloads
        if key == "idempotency_key":
            audit = obj.get("audit") if isinstance(obj.get("audit"), dict) else None
            if audit and "idempotency_key" in audit:
                return str(audit["idempotency_key"])
    return default


def _infer_kind_if_missing(obj: Any) -> str:
    """
    If kind is not provided, try to infer a readable hint from actions[0].op,
    otherwise fall back to 'changeset'.
    """
    kind = _safe_attr_or_key(obj, "kind")
    if kind:
        return kind
    # best-effort inference
    try:
        actions = getattr(obj, "actions", None)
        if actions is None and isinstance(obj, dict):
            actions = obj.get("actions")
        if isinstance(actions, (list, tuple)) and actions:
            op = getattr(actions[0], "op", None)
            if op is None and isinstance(actions[0], dict):
                op = actions[0].get("op")
            if op:
                return f"changeset.{str(op)}"
    except Exception:
        pass
    return "changeset"


def _build_filename(changeset: Any, now: Optional[datetime] = None) -> str:
    """
    Filename pattern: YYYYMMDDTHHMMSSZ_KIND_IDEMP8.json
    Example: 20251002T010203Z_changeset.create_asset_1a2b3c4d.json

    Tolerant to payloads without .kind / .idempotency_key.
    """
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    kind_raw = _infer_kind_if_missing(changeset)
    kind = str(kind_raw).replace("/", ".").replace(" ", "_")
    idem_raw = _safe_attr_or_key(changeset, "idempotency_key", default="noidem")
    idem = str(idem_raw or "noidem").replace("-", "")
    idem8 = idem[:8] if idem else "noidem"
    return f"{ts}_{kind}_{idem8}.json"


# ----- Atomic write -----------------------------------------------------------


def _atomic_write_json(target_path: Path, payload: dict) -> None:
    """
    Atomically write JSON to target_path:
      - write to temporary file in the same directory
      - fsync file, then rename (atomic within same filesystem)
    """
    directory = target_path.parent
    tmp_path = directory / f".{target_path.name}.tmp"

    # Open with O_CREAT|O_EXCL via 'x' to avoid clobbering any existing temp
    with tmp_path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

    # Best-effort permissions
    try:
        os.chmod(tmp_path, 0o640)
    except Exception:
        pass

    # Atomic rename
    os.replace(tmp_path, target_path)


def write_changeset_to_outbox(
    changeset: Any,
    *,
    outbox_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Persist a ChangeSet JSON into the OUTBOX directory atomically.

    Args:
        changeset: validated ChangeSet instance or dict (will be validated).
        outbox_dir: optional directory override; default from settings.
        filename: optional custom filename; otherwise auto-generated.

    Returns:
        Path to the created JSON file.

    Raises:
        OutboxWriteError on any failure to write.
    """
    out_dir = ensure_outbox_dir(outbox_dir)
    try:
        payload = serialize_changeset(changeset)
        name = filename or _build_filename(changeset)
        target = (out_dir / name).resolve()

        # Ensure parent dir is really under outbox root (defense-in-depth)
        if out_dir not in target.parents:
            raise OutboxWriteError(f"Refusing to write outside of OUTBOX: {target}")

        _atomic_write_json(target, payload)
        logger.info("ChangeSet written to OUTBOX: %s", target)
        return target
    except Exception as exc:
        logger.exception("Failed to write ChangeSet to OUTBOX: %s", exc)
        raise OutboxWriteError(str(exc)) from exc


# ----- Submit-or-fallback orchestrator ---------------------------------------


@dataclass(frozen=True)
class SubmitOrFallbackResult:
    submitted: bool
    submit_result: Optional[SubmitResult]
    fallback_path: Optional[Path]


async def _submit_async_via_any(
    changeset: ChangeSet,
    submitter: Any,
) -> SubmitResult:
    """
    Attempt submission using either async or sync submitter implementations.
    """
    # Async submitter (native coroutine method)
    if hasattr(submitter, "submit_changeset") and asyncio.iscoroutinefunction(
        getattr(submitter, "submit_changeset")
    ):
        return await submitter.submit_changeset(changeset)  # type: ignore[arg-type]

    # Sync submitter; run in a worker thread to avoid blocking the event loop
    if hasattr(submitter, "submit_changeset"):
        def _call_sync() -> SubmitResult:
            return submitter.submit_changeset(changeset)  # type: ignore[arg-type]
        return await asyncio.to_thread(_call_sync)

    raise TypeError("submitter does not implement submit_changeset()")


async def submit_or_fallback(
    changeset: ChangeSet,
    *,
    submitter: Optional[Any] = None,
    outbox_dir: Optional[Path] = None,
    require_success: bool = False,
) -> SubmitOrFallbackResult:
    """
    Try to submit the ChangeSet via the provided submitter; on failure (or if
    no submitter is given), persist it into the OUTBOX directory.

    Args:
        changeset: validated ChangeSet.
        submitter: optional async/sync submitter with submit_changeset().
        outbox_dir: override OUTBOX path.
        require_success: if True, raise SubmitError on submission failure
                         (no fallback). If False (default), fallback to outbox.
    """
    # Try online submit if submitter is present
    if submitter is not None:
        try:
            submit_result = await _submit_async_via_any(changeset, submitter)
            if submit_result.success:
                logger.info(
                    "ChangeSet submitted successfully via %s (status=%s)",
                    submit_result.transport,
                    submit_result.status_code,
                )
                return SubmitOrFallbackResult(
                    submitted=True, submit_result=submit_result, fallback_path=None
                )
            # Submission attempted but failed
            err_msg = submit_result.error or f"submit failed (status={submit_result.status_code})"
            logger.warning("ChangeSet submission failed: %s", err_msg)
            if require_success:
                raise SubmitError(err_msg)
            # Fall through to outbox
        except Exception as exc:
            logger.warning("ChangeSet submission error: %s", exc, exc_info=True)
            if require_success:
                raise SubmitError(str(exc)) from exc

    # Fallback to outbox
    path = write_changeset_to_outbox(changeset, outbox_dir=outbox_dir)
    return SubmitOrFallbackResult(
        submitted=False, submit_result=None, fallback_path=path
    )
