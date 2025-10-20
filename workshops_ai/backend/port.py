# File: workshops_ai/backend/port.py
# Contract-first backend interface for Workshop Agents.
# ------------------------------------------------------------------------------
# Purpose
#   A single, stable protocol (interface) that *all* concrete backends implement
#   (CsvBackend now; ApiClientBackend later) so agents/routers don’t care about
#   the data source. The contract follows the global project spec.
#
#   Catalogs (read-only, EN codes canonical):
#     - list_asset_types()
#     - list_impacts(category=None)                     # legacy alias (kept)
#     - list_threats(apply_asset_type=None)            # legacy alias (kept)
#     - list_controls(category=None, include_coverage=True)
#     - get_control_coverage(control_code)
#
#   State (read):
#     - assets_list(org_id, search=None, type=None, is_primary=None)
#     - asset_get(asset_id)
#     - asset_get_impacts(asset_id)
#     - asset_get_risks(asset_id)
#     - asset_get_controls(asset_id)
#
#   Staging lifecycle (write via batches; some agents may bypass this path):
#     - staging_open_batch(org_id, tags, comment=None) -> batch_id
#     - asset_stage_impacts(batch_id, asset_id, items) -> None
#     - asset_stage_risks(batch_id, asset_id, items) -> None
#     - asset_stage_controls(batch_id, asset_id, items) -> None
#     - staging_diff(batch_id) -> dict
#     - staging_commit(batch_id, signed_off_by, message=None) -> dict
#     - staging_discard(batch_id) -> None
#
#   NEW minimal direct write API (send-now path; dual-write friendly):
#     - create_asset(name, *, asset_type=None, **extra) -> Mapping
#     - update_asset(asset_id, **fields) -> Mapping
#     - list_impact_defs(category=None) -> Sequence[Mapping]
#     - create_impact_def(code, name=None, category=None, description=None) -> Mapping
#     - list_threat_defs(apply_asset_type=None) -> Sequence[Mapping]
#     - create_threat_def(code, name=None, category=None, description=None) -> Mapping
#     - set_asset_impact_level(asset_id, code, level:int, note=None) -> Mapping
#     - set_asset_threat_level(asset_id, code, level:int, note=None) -> Mapping
#
#   REQUIRED behavior for the direct setters:
#     - set_asset_impact_level / set_asset_threat_level MUST enforce “definition
#       exists first” semantics and raise BackendNotFound if the referenced code
#       is unknown (i.e., not present in list_impact_defs/list_threat_defs).
#
# Notes
#   - Keep this module *thin* (no heavy imports, no side effects).
#   - Agents should rely only on this contract, never on backend internals.
#   - EN codes in catalogs are canonical and must not be localized.
#   - Pydantic models are referenced only as type hints (no instantiation here).
# ------------------------------------------------------------------------------
from __future__ import annotations

from typing import Mapping, Protocol, Sequence, TypedDict, runtime_checkable

# Unified DTOs (type hints only)
from workshops_ai.api.schema import Asset

# Lightweight, stdlib-only bits used by the optional cache wrapper below.
from threading import RLock  # thread-safe TTL cache for list_asset_types()
from time import monotonic   # monotonic clock for TTL


# ------------------------------------------------------------------------------
# Lightweight item shapes for staging (used later by Impacts/Threats/Controls)
# ------------------------------------------------------------------------------

class ImpactUpdate(TypedDict, total=False):
    impact_code: str
    level: int              # 0..4
    note: str | None

class RiskUpdate(TypedDict, total=False):
    threat_code: str
    probability: int        # 0..4
    note: str | None

class ControlUpdate(TypedDict, total=False):
    control_code: str
    note: str | None
    rationale: str | None


# ------------------------------------------------------------------------------
# Errors (backend-agnostic)
# ------------------------------------------------------------------------------

class BackendError(Exception):
    """Generic backend failure (I/O, parse, transport, etc.)."""

class BackendNotFound(BackendError):
    """Entity not found (asset / definition / batch / etc.)."""

class BackendConflict(BackendError):
    """Optimistic concurrency/staging conflict."""

class BackendValidationError(BackendError):
    """Payload failed validation at backend boundary."""


# ------------------------------------------------------------------------------
# Contract (Protocol)
# ------------------------------------------------------------------------------

BatchId = str

@runtime_checkable
class BackendPort(Protocol):
    """
    Stable interface for all backends (CSV now, API later).

    Implementations should be strict and raise the Backend* errors from above
    on failures (rather than generic exceptions), keeping agents predictable.
    """

    # ---- Catalogs (read-only) ------------------------------------------------

    def list_asset_types(self) -> list[str]:
        """
        Return canonical asset type codes (EN).
        Example: ["Application", "Server", "Database", ...]
        """

    # Legacy alias kept for backward-compatibility with existing callers.
    def list_impacts(self, category: str | None = None) -> Sequence[Mapping[str, object]]:
        """
        Return impact definitions (catalog).
        Recommended fields:
            [{ "code"|"impact_code": str, "name": str, "category"?: str, "description"?: str }]
        Implementations MAY return additional fields.
        """

    # Legacy alias kept for backward-compatibility with existing callers.
    def list_threats(self, apply_asset_type: str | None = None) -> Sequence[Mapping[str, object]]:
        """
        Return threat definitions (catalog). If `apply_asset_type` is provided,
        the list MAY be filtered by the mapping (Asset type × Threat).
        Recommended fields:
            [{ "code"|"threat_code": str, "name": str, "category"?: str, "description"?: str }]
        Implementations MAY return additional fields.
        """

    def list_controls(self, category: str | None = None, include_coverage: bool = True) -> Sequence[Mapping[str, object]]:
        """
        Return canonical control definitions (EN). If include_coverage, controls may be
        augmented downstream by coverage lookups (not here).
        Recommended shape:
            [{ "control_code": str, "name": str, ... }]
        """

    def get_control_coverage(self, control_code: str) -> int | None:
        """
        Return % coverage (0..100) for a given control code, or None if unknown.
        """

    # ---- State (read) --------------------------------------------------------

    def assets_list(
        self,
        org_id: str | None = None,
        search: str | None = None,
        type: str | None = None,
        is_primary: bool | None = None,
    ) -> Sequence[Asset]:
        """
        List assets for organization with optional filters:
          - search: free text in name/description (backend-defined semantics)
          - type: assetType code (EN)
          - is_primary: True/False to filter primary assets
        Returns a finite, deterministic sequence (order is backend-defined).
        """

    def asset_get(self, asset_id: str) -> Asset | None:
        """Get single asset by ID or return None if not found."""

    def asset_get_impacts(self, asset_id: str) -> list[Mapping[str, object]]:
        """
        Return impact evaluations for the given asset.
        Shape:
          [{ "impact_code": str, "level": int (0..4), "note": str|None }]
        """

    def asset_get_risks(self, asset_id: str) -> list[Mapping[str, object]]:
        """
        Return threat probabilities for the given asset.
        Shape:
          [{ "threat_code": str, "probability": int (0..4), "note": str|None }]
        """

    def asset_get_controls(self, asset_id: str) -> list[Mapping[str, object]]:
        """
        Return applied controls for the given asset.
        Shape:
          [{ "control_code": str, "note": str|None, "rationale": str|None }]
        """

    # ---- Staging lifecycle (write via batch) ---------------------------------

    def staging_open_batch(self, org_id: str, tags: list[str], comment: str | None = None) -> BatchId:
        """
        Start a staging batch. Implementations MUST audit (user_id, ts, tags, source=AI).
        Returns batch_id.
        """

    def asset_stage_impacts(self, batch_id: BatchId, asset_id: str, items: Sequence[ImpactUpdate]) -> None:
        """Stage impact evaluations for an asset (no commit yet)."""

    def asset_stage_risks(self, batch_id: BatchId, asset_id: str, items: Sequence[RiskUpdate]) -> None:
        """Stage threat probabilities for an asset (no commit yet)."""

    def asset_stage_controls(self, batch_id: BatchId, asset_id: str, items: Sequence[ControlUpdate]) -> None:
        """Stage applied controls for an asset (no commit yet)."""

    def staging_diff(self, batch_id: BatchId) -> dict:
        """
        Return a machine- and human-friendly diff of staged mutations.
        The exact dict shape is backend-defined but MUST be stable and serializable.
        """

    def staging_commit(self, batch_id: BatchId, signed_off_by: str, message: str | None = None) -> dict:
        """
        Commit staged changes. Implementations MUST:
          - validate batch state,
          - write audit record (who/when/what),
          - return a receipt dict (ids, counts, timestamps).
        """

    def staging_discard(self, batch_id: BatchId) -> None:
        """Discard staged changes (idempotent)."""

    # ---- NEW: Minimal direct write API (“send-now” path) ---------------------

    # Assets (direct)
    def create_asset(self, name: str, *, asset_type: str | None = None, **extra: object) -> Mapping[str, object]:
        """
        Create a new asset. Backends MAY accept additional keyword fields via **extra
        (e.g., description, CIA, isPrimary, org_id). Must return a mapping (dict/DTO-like).
        """

    def update_asset(self, asset_id: str, **fields: object) -> Mapping[str, object]:
        """
        Update an existing asset with provided fields (kwargs).
          - MUST raise BackendNotFound if the asset_id does not exist.
          - MUST validate supported fields and raise BackendValidationError on bad input.
        Must return a mapping (dict/DTO-like) representing the updated asset.
        """

    # Impact definitions (catalog) — tool-first flow
    def list_impact_defs(self, category: str | None = None) -> Sequence[Mapping[str, object]]:
        """
        Return the impact definition catalog (read-only for CSV backend).
        Recommended fields:
            [{ "code": str, "name": str, "category"?: str, "description"?: str }]
        """

    def create_impact_def(
        self,
        code: str,
        name: str | None = None,
        category: str | None = None,
        description: str | None = None,
    ) -> Mapping[str, object]:
        """
        Create (or upsert) a single impact definition within the org/catalog scope.
        Should be idempotent on `code` when possible.
        Returns a mapping representing the created/updated definition.
        Implementations MAY raise BackendConflict when catalogs are read-only.
        """

    # Threat definitions (catalog) — tool-first flow
    def list_threat_defs(self, apply_asset_type: str | None = None) -> Sequence[Mapping[str, object]]:
        """
        Return the threat definition catalog (optionally filtered by asset type).
        Recommended fields:
            [{ "code": str, "name": str, "category"?: str, "description"?: str }]
        """

    def create_threat_def(
        self,
        code: str,
        name: str | None = None,
        category: str | None = None,
        description: str | None = None,
    ) -> Mapping[str, object]:
        """
        Create (or upsert) a single threat definition within the org/catalog scope.
        Should be idempotent on `code` when possible.
        Returns a mapping representing the created/updated definition.
        Implementations MAY raise BackendConflict when catalogs are read-only.
        """

    # Direct setters (MUST enforce “definition must exist first”)
    def set_asset_impact_level(self, asset_id: str, code: str, level: int, note: str | None = None) -> Mapping[str, object]:
        """
        Set the impact level (0..4) for a single asset & impact code.
        REQUIRED behavior:
          - If `asset_id` does not exist → raise BackendNotFound.
          - If `code` is not present in list_impact_defs() → raise BackendNotFound.
          - If `level` is outside 0..4 → raise BackendValidationError.
        Returns a mapping reflecting the upserted row.
        """

    def set_asset_threat_level(self, asset_id: str, code: str, level: int, note: str | None = None) -> Mapping[str, object]:
        """
        Set the threat probability (0..4) for a single asset & threat code.
        REQUIRED behavior:
          - If `asset_id` does not exist → raise BackendNotFound.
          - If `code` is not present in list_threat_defs() → raise BackendNotFound.
          - If `level` is outside 0..4 → raise BackendValidationError.
        Returns a mapping reflecting the upserted row.
        """


# ------------------------------------------------------------------------------
# Optional wrapper: tiny TTL cache for list_asset_types()
# ------------------------------------------------------------------------------
# Why here?
# - Keeps the cache policy colocated with the contract.
# - Pure stdlib, thread-safe (RLock), zero external deps.
#
# Behavior:
# - Caches the result of `list_asset_types()` for `asset_types_ttl_s` seconds.
# - Single-entry (per backend instance) cache; effectively LRU(1)+TTL.
# - Never caches failures (exceptions are propagated as-is).
# - Returns a shallow copy to avoid callers mutating the cached list.
# - You can call `.invalidate_asset_types_cache()` to drop it eagerly.
# ------------------------------------------------------------------------------

class CachedBackendPort:
    """
    Proxy/wrapper that adds a small TTL cache to `list_asset_types()` and forwards
    everything else to the underlying backend instance.

    Usage:
        backend = CsvBackend(...)
        port = CachedBackendPort(backend, asset_types_ttl_s=600)
        types = port.list_asset_types()  # cached for 10 minutes
    """

    __slots__ = ("_delegate", "_ttl_s", "_lock", "_asset_types", "_asset_types_exp")

    def __init__(self, delegate: BackendPort, *, asset_types_ttl_s: int = 600) -> None:
        self._delegate: BackendPort = delegate
        self._ttl_s: int = max(0, int(asset_types_ttl_s))
        self._lock: RLock = RLock()
        self._asset_types: list[str] | None = None
        self._asset_types_exp: float = 0.0

    # ---- Cached method -------------------------------------------------------

    def list_asset_types(self) -> list[str]:
        now = monotonic()
        with self._lock:
            if self._asset_types is not None and now < self._asset_types_exp:
                # Return a defensive copy so callers can't mutate our cache.
                return list(self._asset_types)

        # Miss or expired → fetch outside the lock to avoid long critical section.
        fresh = self._delegate.list_asset_types()

        # Validate minimal contract: must be a sequence of strings.
        if not isinstance(fresh, list):
            fresh = list(fresh)
        # Defensively cast items to str (lightweight).
        fresh = [str(x) for x in fresh]

        with self._lock:
            self._asset_types = fresh
            self._asset_types_exp = monotonic() + self._ttl_s if self._ttl_s > 0 else 0.0
            return list(self._asset_types)

    # ---- Manual invalidation (eager) ----------------------------------------

    def invalidate_asset_types_cache(self) -> None:
        """Drop the cached asset types immediately (e.g., after a config reload)."""
        with self._lock:
            self._asset_types = None
            self._asset_types_exp = 0.0

    # ---- Transparent forwarding for all other attributes/methods -------------

    def __getattr__(self, name: str):
        # Called only if `name` wasn't found on this wrapper; forward to delegate.
        return getattr(self._delegate, name)


def with_catalog_cache(port: BackendPort, *, asset_types_ttl_s: int = 600) -> BackendPort:
    """
    Convenience helper returning a `BackendPort`-compatible proxy with TTL cache.

    Example:
        real_port: BackendPort = CsvBackend(...)
        cached_port = with_catalog_cache(real_port, asset_types_ttl_s=300)
    """
    return CachedBackendPort(port, asset_types_ttl_s=asset_types_ttl_s)
