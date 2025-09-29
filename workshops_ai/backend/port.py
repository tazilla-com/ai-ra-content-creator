# File: workshops_ai/backend/port.py
# Contract-first backend interface for Workshop Agents.
# ------------------------------------------------------------------------------
# Purpose
#   A single, stable protocol (interface) that *all* concrete backends implement
#   (CsvBackend now; ApiClientBackend later) so agents/routers don’t care about
#   the data source. The contract follows the global project spec:
#
#   Catalogs (read-only, EN codes canonical):
#     - list_asset_types()
#     - list_impacts(category=None)
#     - list_threats(apply_asset_type=None)
#     - list_controls(category=None, include_coverage=True)
#     - get_control_coverage(control_code)
#
#   State (read-only in AssetAgent MVP):
#     - assets_list(org_id, search=None, type=None, is_primary=None)
#     - asset_get(asset_id)
#     - asset_get_impacts(asset_id)
#     - asset_get_risks(asset_id)
#     - asset_get_controls(asset_id)
#
#   Staging lifecycle (used by Impacts/Threats/Controls in later iterations):
#     - staging_open_batch(org_id, tags, comment=None) -> batch_id
#     - asset_stage_impacts(batch_id, asset_id, items) -> None
#     - asset_stage_risks(batch_id, asset_id, items) -> None
#     - asset_stage_controls(batch_id, asset_id, items) -> None
#     - staging_diff(batch_id) -> dict
#     - staging_commit(batch_id, signed_off_by, message=None) -> dict
#     - staging_discard(batch_id) -> None
#
# Notes
#   - Keep this module *thin* (no heavy imports, no side effects).
#   - Agents should rely only on this contract, never on CsvBackend internals.
#   - EN codes in catalogs are canonical and must not be localized.
#   - Pydantic models are referenced only as type hints (no instantiation here).
# ------------------------------------------------------------------------------

from __future__ import annotations

from typing import Iterable, Mapping, Protocol, Sequence, TypedDict, runtime_checkable

# Pydantic domain models (thin type hints; actual definitions live in models/schema/*)
# For AssetAgent MVP we only need Asset. The others are for later agents.
try:  # soft import to avoid circulars during scaffolding
    from workshops_ai.models.schema.asset import Asset  # type: ignore
except Exception:  # pragma: no cover
    # Minimal fallback so type checkers don’t explode if schema isn’t generated yet.
    class Asset:  # type: ignore[deadCode]
        ...

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
    """Entity not found."""

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
    """Stable interface for all backends (CSV now, API later)."""

    # ---- Catalogs (read-only) ------------------------------------------------

    def list_asset_types(self) -> list[str]:
        """
        Return canonical asset type codes (EN).
        Example: ["Application", "Server", "Database", ...]
        """

    def list_impacts(self, category: str | None = None) -> list[str]:
        """
        Return canonical impact codes (EN). Optional category scoping for UI.
        Example: ["Confidentiality", "Integrity", "Availability", "Safety", ...]
        """

    def list_threats(self, apply_asset_type: str | None = None) -> list[str]:
        """
        Return canonical threat codes (EN). When `apply_asset_type` is provided,
        the list MAY be filtered by the mapping (Asset type x Threat).
        """

    def list_controls(self, category: str | None = None, include_coverage: bool = True) -> list[str]:
        """
        Return canonical control codes (EN). If include_coverage, controls may be
        augmented downstream by coverage lookups (not here).
        """

    def get_control_coverage(self, control_code: str) -> int | None:
        """
        Return % coverage (0..100) for a given control code, or None if unknown.
        """

    # ---- State (read) --------------------------------------------------------

    def assets_list(
        self,
        org_id: str,
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
        Shape (later agent depends on this):
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
