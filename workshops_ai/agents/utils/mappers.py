# File: workshops_ai/agents/utils/mappers.py
# -----------------------------------------------------------------------------
# Purpose
# Canonical adapters (mappers) between unified DTOs (api/schema/*) and:
#   - WebSocket events (UI-expected payloads for optimistic updates)
#   - Typed actions from models/changeset.py to be wrapped by build_changeset
#
# Notes
# - Input validation is already done by Pydantic DTOs (AssetImpact, AssetRisk).
# - We keep the 0..4 scale as-is here; any domain mapping belongs elsewhere.
# - Idempotency is handled centrally by build_changeset (models/changeset.py).
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

# OLD:
# from typing import Any, Dict, List, Optional, Literal
# NEW (Python 3.11 native generics used below, no need for Dict/List imports)

from workshops_ai.api.schema import AssetImpact, AssetRisk
# OLD had: ImpactEvaluation (unused) → removed to avoid pyright reportUnusedImport

try:
    # Typed models for building actions going into a ChangeSet
    from workshops_ai.models.changeset import (  # type: ignore
        ImpactItem,
        RiskItem,
        ActionSetImpacts,
        ActionSetRisks,
    )
except Exception:  # pragma: no cover
    # Soft fallback types (only for import-time robustness in dev)
    ImpactItem = RiskItem = ActionSetImpacts = ActionSetRisks = Any  # type: ignore

# OLD:
# __all__ = [
#     "map_asset_impact_to_ws_event",
#     "map_asset_impact_to_action",
#     "map_asset_risk_to_ws_event",
#     "map_asset_risk_to_action",
# ]
# NEW (+ resolvers exported):
__all__ = [
    "map_asset_impact_to_ws_event",
    "map_asset_impact_to_action",
    "map_asset_risk_to_ws_event",
    "map_asset_risk_to_action",
    "resolve_impact_category",   # NEW
    "resolve_threat_code",       # NEW (optional helper for ThreatAgent hardening)
]

# -----------------------------------------------------------------------------
# Lightweight normalization utilities
# -----------------------------------------------------------------------------

def _slug(s: str | None) -> str:
    """
    Lowercase, trim, remove accents, collapse spaces/underscores/hyphens — good
    enough to match human-entered labels to catalog rows.
    """
    s = (s or "").strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = " ".join(s.replace("_", " ").replace("-", " ").split())
    return s


# -----------------------------------------------------------------------------
# Impact catalog loader (CSV) — used by resolve_impact_category
# -----------------------------------------------------------------------------

_DEFAULT_IMPACTS_CSV = Path("workshops_ai/data/catalogs/impacts.csv")

@lru_cache(maxsize=1)
def _load_impact_catalog_rows(csv_path: Path = _DEFAULT_IMPACTS_CSV) -> list[dict[str, str]]:
    """
    Lazy/cached CSV read. Expected headers (case-insensitive): code, name, category.
    Returns an empty list on any error; the graph can then do a clarify prompt.
    """
    rows: list[dict[str, str]] = []
    try:
        if not csv_path.exists():
            return rows
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                r = { (k or "").strip().lower(): (v or "").strip() for k, v in (raw or {}).items() }
                rows.append({
                    "code": r.get("code", ""),
                    "name": r.get("name", ""),
                    "category": r.get("category", ""),
                })
    except Exception:
        # Silent fail → no rows; graph should handle clarify if needed.
        return []
    return rows


def resolve_impact_category(
    *,
    code: Optional[str] = None,
    name: Optional[str] = None,
    catalog_rows: Optional[list[dict[str, str]]] = None,
    csv_path: Optional[str | Path] = None,
) -> Optional[str]:
    """
    Return the canonical impact category (EN string) for a given item.
    Deterministic resolution:
      1) Exact code (case-insensitive) → category
      2) Name slug match → category
      3) Otherwise None (let the graph ask user to choose explicitly)

    Prefer passing `catalog_rows` (preloaded once in the graph) to avoid I/O here.
    """
    rows = catalog_rows
    if rows is None:
        path = Path(csv_path) if csv_path is not None else _DEFAULT_IMPACTS_CSV
        rows = _load_impact_catalog_rows(path)

    if not rows:
        return None

    # 1) Try by code
    if code:
        c_low = code.strip().lower()
        for r in rows:
            if r.get("code", "").strip().lower() == c_low and r.get("category"):
                return r["category"] or None

    # 2) Try by normalized name
    if name:
        name_slug = _slug(name)
        for r in rows:
            if _slug(r.get("name", "")) == name_slug and r.get("category"):
                return r["category"] or None

    return None


# -----------------------------------------------------------------------------
# Threat catalog loader (CSV) — optional hardening for ThreatAgent
# -----------------------------------------------------------------------------

_DEFAULT_THREATS_CSV = Path("workshops_ai/data/catalogs/threats.csv")

@lru_cache(maxsize=1)
def _load_threat_catalog_rows(csv_path: Path = _DEFAULT_THREATS_CSV) -> list[dict[str, str]]:
    """
    Lazy/cached CSV read for threat catalog. Typical headers: code, name.
    """
    rows: list[dict[str, str]] = []
    try:
        if not csv_path.exists():
            return rows
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                r = { (k or "").strip().lower(): (v or "").strip() for k, v in (raw or {}).items() }
                rows.append({
                    "code": r.get("code", ""),
                    "name": r.get("name", ""),
                })
    except Exception:
        return []
    return rows


def resolve_threat_code(
    *,
    code: Optional[str] = None,
    name: Optional[str] = None,
    catalog_rows: Optional[list[dict[str, str]]] = None,
    csv_path: Optional[str | Path] = None,
) -> Optional[str]:
    """
    Resolve canonical threat code. This is handy when the LLM produced only a
    display name. Strategy:
      1) If `code` provided → normalize and return it (optimistic).
      2) Else try name slug match in the catalog and return the catalog `code`.
      3) Else None (graph should clarify).
    """
    # 1) optimistic use of provided code
    if code and code.strip():
        return code.strip()

    rows = catalog_rows
    if rows is None:
        path = Path(csv_path) if csv_path is not None else _DEFAULT_THREATS_CSV
        rows = _load_threat_catalog_rows(path)

    if not rows or not name:
        return None

    name_slug = _slug(name)
    for r in rows:
        if _slug(r.get("name", "")) == name_slug and r.get("code"):
            return r["code"] or None

    return None


# -----------------------------------------------------------------------------
# Impacts (AssetImpact) → WS event
# -----------------------------------------------------------------------------

def map_asset_impact_to_ws_event(dto: AssetImpact) -> dict[str, Any]:
    """
    Convert AssetImpact DTO -> WebSocket payload in the UI-expected shape.
    Keeps the 0..4 scale; no conversion.

    Example:
    {
      "assetId": "uuid|None",
      "assetName": "Customer Database",
      "impacts": [
        {"impactName": "Confidentiality", "impactLevel": 4},
        {"impactName": "Integrity",       "impactLevel": 3},
        {"impactName": "Availability",    "impactLevel": 3}
      ]
    }
    """
    return {
        "assetId": str(dto.assetId) if getattr(dto, "assetId", None) else None,
        "assetName": dto.assetName,
        "impacts": [
            {"impactName": it.impactName, "impactLevel": it.impactLevel}
            for it in (dto.impacts or [])
        ],
    }


# -----------------------------------------------------------------------------
# Impacts (AssetImpact) → typed Action (for ChangeSet)
# -----------------------------------------------------------------------------

def map_asset_impact_to_action(
    dto: AssetImpact,
    *,
    mode: Literal["merge", "replace"] = "merge",
    fallback_asset_ref: Optional[str] = None,
) -> ActionSetImpacts:
    """
    Convert AssetImpact DTO -> ActionSetImpacts (typed action ready for ChangeSet).

    Parameters
    ----------
    dto : AssetImpact
        Source DTO with asset and impact evaluations (0..4).
    mode : "merge" | "replace"
        'merge' updates/inserts provided items; 'replace' overwrites all.
    fallback_asset_ref : Optional[str]
        Used when dto.assetId is missing; e.g., "client:<token>" for correlation.
        If neither assetId nor fallback_asset_ref is available, raises ValueError.

    Returns
    -------
    ActionSetImpacts
    """
    items: list[ImpactItem] = [
        ImpactItem(impact_code=it.impactName, level=it.impactLevel)  # type: ignore
        for it in (dto.impacts or [])
    ]

    asset_id = str(dto.assetId) if getattr(dto, "assetId", None) else None
    asset_ref = None if asset_id else fallback_asset_ref

    if not asset_id and not asset_ref:
        raise ValueError("ActionSetImpacts requires 'asset_id' or 'asset_ref'.")

    return ActionSetImpacts(  # type: ignore[call-arg]
        op="set_impacts",
        mode=mode,
        asset_id=asset_id,
        asset_ref=asset_ref,
        items=items,
    )


# -----------------------------------------------------------------------------
# Risks (AssetRisk) → WS event
# -----------------------------------------------------------------------------

def map_asset_risk_to_ws_event(dto: AssetRisk) -> dict[str, Any]:
    """
    Convert AssetRisk DTO -> WebSocket payload in the UI-expected shape.
    Keeps the 0..4 scale; no conversion.

    Example:
    {
      "assetId": "uuid|None",
      "assetName": "Backend API Server",
      "risks": [
        {"threatName": "Fire", "probability": 2},
        {"threatName": "Water, humidity", "probability": 1}
      ]
    }
    """
    return {
        "assetId": str(dto.assetId) if getattr(dto, "assetId", None) else None,
        "assetName": dto.assetName,
        "risks": [
            {"threatName": r.threatName, "probability": r.probability}
            for r in (dto.risks or [])
        ],
    }


# -----------------------------------------------------------------------------
# Risks (AssetRisk) → typed Action (for ChangeSet)
# -----------------------------------------------------------------------------

def map_asset_risk_to_action(
    dto: AssetRisk,
    *,
    mode: Literal["merge", "replace"] = "merge",
    fallback_asset_ref: Optional[str] = None,
) -> ActionSetRisks:
    """
    Convert AssetRisk DTO -> ActionSetRisks (typed action ready for ChangeSet).

    Parameters
    ----------
    dto : AssetRisk
        Source DTO with asset risks (probability 0..4).
    mode : "merge" | "replace"
        'merge' updates/inserts provided items; 'replace' overwrites all.
    fallback_asset_ref : Optional[str]
        Used when dto.assetId is missing; e.g., "client:<token>" for correlation.
        If neither assetId nor fallback_asset_ref is available, raises ValueError.

    Returns
    -------
    ActionSetRisks
    """
    items: list[RiskItem] = [
        RiskItem(threat_code=r.threatName, probability=r.probability)  # type: ignore
        for r in (dto.risks or [])
    ]

    asset_id = str(dto.assetId) if getattr(dto, "assetId", None) else None
    asset_ref = None if asset_id else fallback_asset_ref

    if not asset_id and not asset_ref:
        raise ValueError("ActionSetRisks requires 'asset_id' or 'asset_ref'.")

    return ActionSetRisks(  # type: ignore[call-arg]
        op="set_risks",
        mode=mode,
        asset_id=asset_id,
        asset_ref=asset_ref,
        items=items,
    )
