# workshops_ai/app/services/outbox_writer.py
# -----------------------------------------------------------------------------
# Persist & submit ChangeSets with a deterministic OUTBOX fallback
#
# - Primary path: submit via an injected submitter (async or sync).
# - Fallback path: atomically write a JSON ChangeSet to OUTBOX.
# - File naming: <UTC>_<kind>_<idem8>.json (UTC timestamp + inferred kind +
#   idempotency key prefix), e.g. 20251002T010203Z_changeset.create_asset_1a2b3c4d.json
#
# This module is intentionally small and dependency-light; it’s imported by the
# graph factory to provide a submit callable that first tries online submit and
# gracefully falls back to the filesystem OUTBOX.
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
    """Raised when we failed to persist a JSON ChangeSet into the OUTBOX."""


class SubmitError(RuntimeError):
    """Raised when a submitter reports failure (non-2xx or transport error)."""


# ----- Submitter protocol & result -------------------------------------------


@runtime_checkable
class AsyncChangeSetSubmitter(Protocol):
    """Async submitter interface (preferred in services/clients)."""

    async def submit_changeset(self, changeset: ChangeSet) -> "SubmitResult":
        ...


@runtime_checkable
class SyncChangeSetSubmitter(Protocol):
    """Sync submitter interface (older clients)."""

    def submit_changeset(self, changeset: ChangeSet) -> "SubmitResult":
        ...


@dataclass(frozen=True)
class SubmitResult:
    """
    Transport-agnostic result for a submission attempt.

    - success: True when the remote accepted the ChangeSet (e.g., 2xx).
    - transport: "http", "grpc", "local", ...
    - status_code / response_body: transport-specific details (optional).
    - error: populated on failure for diagnostic surface.
    """
    success: bool
    transport: str = "unknown"
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

    Note: environment-only overrides are intentionally not used here to keep a
    single source of truth in application settings.
    """
    try:
        settings = get_settings()

        # 1) New canonical location (authoritative)
        tazila = getattr(settings, "tazila", None)
        outbox_path = getattr(tazila, "outbox_path", None) if tazila is not None else None
        if outbox_path:
            return Path(str(outbox_path))

        # 2) Deprecated legacy setting (kept temporarily for compatibility)
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
    """Ensure the OUTBOX directory exists (mkdir -p) and set safe permissions."""
    out_dir = Path(path) if path else _default_outbox_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(out_dir, 0o750)
    except Exception:
        # Best-effort; not fatal on platforms that ignore chmod
        pass
    return out_dir.resolve()


# ----- Serialization helpers --------------------------------------------------


def serialize_changeset(changeset: Any) -> dict:
    """
    Accept ChangeSet | dict and always return a JSON-serializable dict.

    - If given a model instance, dump with aliases and without nulls.
    - If given a dict, validate via ChangeSet.model_validate() to ensure shape
      and then dump the normalized model, so OUTBOX contents are canonical.
    """
    if isinstance(changeset, ChangeSet):
        return changeset.model_dump(
            mode="json",
            by_alias=True,
            serialize_as_any=True,
            exclude_none=True,
        )
    cs = ChangeSet.model_validate(changeset)  # raises ValidationError on mismatch
    return cs.model_dump(
        mode="json",
        by_alias=True,
        serialize_as_any=True,
        exclude_none=True,
    )


def _safe_attr_or_key(obj: Any, key: str, default: Optional[str] = None) -> Optional[str]:
    """Helper that tries attribute first, then mapping key, else default (with audit.idempotency_key fallback)."""
    val = getattr(obj, key, None)
    if val is not None:
        return str(val)
    if isinstance(obj, dict):
        v = obj.get(key)
        if v is not None:
            return str(v)
        if key == "idempotency_key":
            audit = obj.get("audit") if isinstance(obj.get("audit"), dict) else None
            if audit and "idempotency_key" in audit:
                return str(audit["idempotency_key"])
    return default


def _infer_kind_if_missing(obj: Any) -> str:
    """
    If `kind` is missing, infer from first action op (e.g., 'changeset.create_asset').
    This helps produce more informative filenames without relying on upstreams.
    """
    kind = _safe_attr_or_key(obj, "kind")
    if kind:
        return kind
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
    Filename pattern: YYYYMMDDT%H%M%SZ_KIND_IDEMP8.json
    Example: 20251002T010203Z_changeset.create_asset_1a2b3c4d.json

    - UTC timestamp makes duplicates unlikely while preserving an ordering.
    - `kind` is inferred for readability (see _infer_kind_if_missing).
    - `idempotency_key` prefix (first 8 chars, dashes stripped) aids dedupe/audit.
    """
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    kind_raw = _infer_kind_if_missing(changeset)
    kind = str(kind_raw).replace("/", ".").replace(" ", "_")
    idem_raw = _safe_attr_or_key(changeset, "idempotency_key", default="noidem")
    idem = str(idem_raw or "noidem").replace("-", "")
    idem8 = idem[:8] if idem else "noidem"
    return f"{ts}_{kind}_{idem8}.json"


def _get_actions_len(changeset: Any) -> int:
    """Return number of actions, tolerant to dict or model."""
    try:
        acts = getattr(changeset, "actions", None)
        if acts is None and isinstance(changeset, dict):
            acts = changeset.get("actions")
        return len(acts or [])
    except Exception:
        return 0


# ----- Atomic write -----------------------------------------------------------


def _atomic_write_json(target_path: Path, payload: dict) -> None:
    """
    Atomically write JSON to target_path:
      1) write to a temporary file in the same directory
      2) fsync the file
      3) os.replace() to the final name (atomic within same filesystem)
    """
    directory = target_path.parent
    tmp_path = directory / f".{target_path.name}.tmp"

    # "x" ensures we never clobber an existing temp name in unexpected races
    with tmp_path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

    try:
        os.chmod(tmp_path, 0o640)
    except Exception:
        pass

    os.replace(tmp_path, target_path)


def write_changeset_to_outbox(
    changeset: Any,
    *,
    outbox_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Persist a ChangeSet JSON into the OUTBOX directory atomically.

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

        # Guard against path traversal (ensure target is inside out_dir)
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
    """
    Unified outcome used by graphs/manager:

    - submitted: True if online submit succeeded (no OUTBOX write was done).
    - submit_result: transport details on success/failure (None when falling back).
    - fallback_path: filesystem path when queued into OUTBOX (None if submitted).
    - reason: structured code/message for UI logic (e.g., 'empty_changeset').
    """
    submitted: bool
    submit_result: Optional[SubmitResult]
    fallback_path: Optional[Path]
    reason: Optional[dict] = None  # e.g., {"code":"queued_outbox","path":"/app/outbox/..."}


async def _submit_async_via_any(
    changeset: ChangeSet,
    submitter: Any,
) -> SubmitResult:
    """
    Attempt submission using either async or sync submitter implementations.
    We introspect the callable and run sync submitters in a worker thread.
    """
    if hasattr(submitter, "submit_changeset") and asyncio.iscoroutinefunction(
        getattr(submitter, "submit_changeset")
    ):
        return await submitter.submit_changeset(changeset)  # type: ignore[arg-type]

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

    Special case:
      - If the ChangeSet has 0 actions, do not submit or write. Return a
        structured reason for the UI (helps render 'nothing to do' states).
    """
    # Short-circuit: empty changeset → structured reason, no I/O
    if _get_actions_len(changeset) == 0:
        reason = {
            "code": "empty_changeset",
            "message": "No actions to submit.",
            "actions": 0,
        }
        logger.info("Empty ChangeSet: %s", reason)
        return SubmitOrFallbackResult(
            submitted=False,
            submit_result=None,
            fallback_path=None,
            reason=reason,
        )

    # Try online submit when submitter is provided
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
                    submitted=True, submit_result=submit_result, fallback_path=None, reason=None
                )
            # Submission returned a failure result (e.g., non-2xx)
            err_msg = submit_result.error or f"submit failed (status={submit_result.status_code})"
            logger.warning("ChangeSet submission failed: %s", err_msg)
            if require_success:
                # Caller demanded success: escalate to exception
                raise SubmitError(err_msg)
            # Otherwise, fall through to OUTBOX fallback
        except Exception as exc:
            logger.warning("ChangeSet submission error: %s", exc, exc_info=True)
            if require_success:
                raise SubmitError(str(exc)) from exc

    # Fallback to OUTBOX (authoritative path is settings.tazila.outbox_path)
    path = write_changeset_to_outbox(changeset, outbox_dir=outbox_dir)
    return SubmitOrFallbackResult(
        submitted=False,
        submit_result=None,
        fallback_path=path,
        reason={"code": "queued_outbox", "path": str(path)},
    )
