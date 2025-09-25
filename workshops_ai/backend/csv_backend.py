# File: workshops_ai/backend/csv_backend.py
# CSV-based backend (MVP) implementing BackendPort for read-only asset workflows.
# - Reads state from <settings.paths.state_dir>/assets.csv
# - Reads catalogs from <settings.paths.catalogs_dir> (validates assetType against "Asset type.csv")
# - Strict, defensive parsing with mtime-based hot reload
# - No external deps (stdlib csv); Pydantic v2 for domain validation
#
# Files (relative to configured paths):
#   catalogs/Asset type.csv
#   catalogs/Threat.csv
#   catalogs/Threat category.csv
#   catalogs/Threat x Threat category.csv
#   catalogs/Asset type x Threat.csv
#   state/assets.csv
#
# Notes:
# - Catalog codes (EN) are canonical (no localization here).
# - All read methods are synchronous under the hood; we wrap them in threads at call sites.
# - Demo-ready enhancement: if canonical Asset requires UUID id and CSV uses string ids,
#   we transparently fall back to a compatible local model with str id (see _build_asset_with_cia_fallback).

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from workshops_ai.backend.port import (
    BackendConflict,
    BackendError,
    BackendPort,
    BackendValidationError,
)
from workshops_ai.settings import get_settings

__all__ = ["CsvBackend"]

# Try to use the canonical Asset model; if not yet present during scaffolding,
# fall back to a compatible local model to keep the app importable.
try:  # pragma: no cover - exercised in runtime, not unit test stubs
    from workshops_ai.models.schema.asset import Asset  # type: ignore
    _HAS_CANONICAL_ASSET = True
except Exception:  # pragma: no cover
    _HAS_CANONICAL_ASSET = False

    class Asset(BaseModel):  # type: ignore[no-redef]
        id: str
        name: str
        description: str | None = None
        isPrimary: bool | None = None
        # In the canonical schema these may be enums. Our builder below maps accordingly.
        confidentiality: int | str | None = Field(default=None)
        integrity: int | str | None = Field(default=None)
        availability: int | str | None = Field(default=None)
        assetType: str
        org_id: str | None = None

        @field_validator("id", "name", "assetType")
        @classmethod
        def _non_empty(cls, v: str) -> str:
            v = (v or "").strip()
            if not v:
                raise ValueError("must not be empty")
            return v


# --------------------------------------------------------------------------------------
# Paths & config (centralized via settings)
# --------------------------------------------------------------------------------------

_s = get_settings()
_DATA_DIR = Path(_s.paths.data_dir).resolve()
_STATE_DIR = Path(_s.paths.state_dir).resolve()
_CAT_DIR = Path(_s.paths.catalogs_dir).resolve()

_ASSETS_STATE_FILE = _STATE_DIR / "assets.csv"
_CAT_ASSET_TYPE_FILE = _CAT_DIR / "Asset type.csv"
_CAT_AT_X_THREAT_FILE = _CAT_DIR / "Asset type x Threat.csv"  # asset-type ↔ threat mapping

# New threat catalogs
_CAT_THREAT_FILE = _CAT_DIR / "Threat.csv"
_CAT_THREAT_CATEGORY_FILE = _CAT_DIR / "Threat category.csv"
_CAT_THREAT_X_CATEGORY_FILE = _CAT_DIR / "Threat x Threat category.csv"


# --------------------------------------------------------------------------------------
# Low-level CSV helpers with mtime-based reload
# --------------------------------------------------------------------------------------

@dataclass
class _CsvCache:
    path: Path
    mtime: float = 0.0
    rows: list[dict[str, str]] | None = None

    def load(self) -> list[dict[str, str]]:
        if not self.path.exists():
            # For catalogs, non-existence means empty; for state, it's a hard error
            if self.path == _ASSETS_STATE_FILE:
                raise BackendError(f"State file not found: {self.path}")
            self.mtime, self.rows = 0.0, []
            return self.rows

        mt = self.path.stat().st_mtime
        if self.rows is not None and self.mtime == mt:
            return self.rows

        self.rows = _read_csv(self.path)
        self.mtime = mt
        return self.rows


def _detect_encoding_and_delimiter(path: Path) -> tuple[str, str]:
    """
    Try to detect encoding (utf-8-sig → utf-8 → cp1250) and delimiter (csv.Sniffer).
    Returns (encoding, delimiter). Defaults to ('utf-8-sig', ',') if unsure.
    """
    sample_bytes = b""
    try:
        with path.open("rb") as fb:
            sample_bytes = fb.read(4096)
    except FileNotFoundError:
        return "utf-8-sig", ","

    for enc in ("utf-8-sig", "utf-8", "cp1250"):
        try:
            sample_text = sample_bytes.decode(enc, errors="strict")
        except UnicodeDecodeError:
            continue
        # Default delimiter
        delim = ","
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample_text, delimiters=",;\t|")
            delim = dialect.delimiter or delim
        except Exception:
            # Heuristic: semicolon is common in EU CSV exports
            if ";" in sample_text and sample_text.count(";") > sample_text.count(","):
                delim = ";"
        return enc, delim

    # Fallback
    return "utf-8-sig", ","


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV into a list of dicts. Robust to BOM, semicolons, CP1250; trims headers/values."""
    encoding, delimiter = _detect_encoding_and_delimiter(path)
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        # Normalize headers: strip whitespace; keep original case for value access
        if reader.fieldnames:
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
        out: list[dict[str, str]] = []
        for row in reader:
            clean = {
                (k.strip() if isinstance(k, str) else k): (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
            }
            out.append(clean)
        return out


def _first_present_column(rows: list[dict[str, str]], candidates: Iterable[str]) -> str | None:
    if not rows:
        return None
    cols = set(rows[0].keys())
    for c in candidates:
        if c in cols:
            return c
    return None


# --------------------------------------------------------------------------------------
# Catalog parsing
# --------------------------------------------------------------------------------------

class _Catalogs:
    """Lazy, cached access to catalog CSVs with resilient column heuristics."""

    def __init__(self) -> None:
        # Asset types + mappings
        self.asset_types = _CsvCache(_CAT_ASSET_TYPE_FILE)
        self.asset_type_x_threat = _CsvCache(_CAT_AT_X_THREAT_FILE)
        # Threats + categories
        self.threats = _CsvCache(_CAT_THREAT_FILE)
        self.threat_categories = _CsvCache(_CAT_THREAT_CATEGORY_FILE)
        self.threat_x_category = _CsvCache(_CAT_THREAT_X_CATEGORY_FILE)

    # ------------------------ Asset type catalog ------------------------

    def list_asset_types(self) -> list[str]:
        rows = self.asset_types.load()
        if not rows:
            return []
        # Try common header candidates – include multilingual/“EN Name” variants
        candidates = (
            "code",
            "asset_type",
            "type",
            "Asset type",
            "EN Name",
            "EN name",
            "Name",
            "name",
        )
        col = _first_present_column(rows, candidates)
        if col is None:
            # No known column – assume first column of the CSV
            col = next(iter(rows[0].keys()))
        # Return unique, non-empty codes in file order
        seen: set[str] = set()
        res: list[str] = []
        for r in rows:
            code = (r.get(col) or "").strip()
            if code and code not in seen:
                seen.add(code)
                res.append(code)
        return res

    # ------------------------ Threat categories ------------------------

    def list_threat_categories(self) -> list[str]:
        """Return canonical category codes found in Threat category.csv."""
        rows = self.threat_categories.load()
        if not rows:
            return []
        col = _first_present_column(
            rows,
            ("category_code", "code", "Category", "Threat category", "threat_category", "Name", "EN Name"),
        )
        if col is None:
            col = next(iter(rows[0].keys()))
        seen: set[str] = set()
        out: list[str] = []
        for r in rows:
            v = (r.get(col) or "").strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    # ------------------------ Threats (detail + list) -------------------

    def _build_threat_index(self) -> dict[str, dict[str, Any]]:
        """
        Build a dict: code -> threat dict with fields:
          code, name, description?, flags{intentional,unintentional,environmental},
          categories: [category_code...], category: first or None,
          asset_types: [asset_type_code...]
        """
        t_rows = self.threats.load()
        if not t_rows:
            return {}

        # Column heuristics
        code_col = _first_present_column(t_rows, ("code", "threat_code", "Code", "Threat", "ID"))
        name_col = _first_present_column(t_rows, ("name", "Name", "EN Name", "Threat name"))
        desc_col = _first_present_column(t_rows, ("description", "Description", "EN Description", "Note"))
        intentional_col = _first_present_column(t_rows, ("intentional", "Intentional"))
        unintentional_col = _first_present_column(t_rows, ("unintentional", "Unintentional", "accidental"))
        environmental_col = _first_present_column(t_rows, ("environmental", "Environmental", "env"))

        # Threat ↔ category mapping
        txc = self.threat_x_category.load()
        txc_threat_col = _first_present_column(txc, ("threat_code", "Threat", "Threat code", "code"))
        txc_cat_col = _first_present_column(
            txc, ("category_code", "Category", "Threat category", "threat_category", "code")
        )
        threat_to_categories: dict[str, list[str]] = {}
        if txc and txc_threat_col and txc_cat_col:
            for r in txc:
                th = (r.get(txc_threat_col) or "").strip()
                cat = (r.get(txc_cat_col) or "").strip()
                if th and cat:
                    threat_to_categories.setdefault(th, [])
                    if cat not in threat_to_categories[th]:
                        threat_to_categories[th].append(cat)

        # Asset type ↔ threat mapping
        att = self.asset_type_x_threat.load()
        att_asset_col = _first_present_column(att, ("asset_type", "Asset type"))
        att_threat_col = _first_present_column(att, ("threat_code", "Threat", "threat"))
        threat_to_asset_types: dict[str, list[str]] = {}
        if att and att_asset_col and att_threat_col:
            for r in att:
                at = (r.get(att_asset_col) or "").strip()
                th = (r.get(att_threat_col) or "").strip()
                if at and th:
                    threat_to_asset_types.setdefault(th, [])
                    if at not in threat_to_asset_types[th]:
                        threat_to_asset_types[th].append(at)

        # Build index
        index: dict[str, dict[str, Any]] = {}
        for r in t_rows:
            code = ((r.get(code_col) or "") if code_col else "").strip()
            if not code:
                continue
            name = ((r.get(name_col) or "") if name_col else "").strip()
            desc = ((r.get(desc_col) or "") if desc_col else "").strip() or None
            flags = {
                "intentional": _parse_bool((r.get(intentional_col) or "") if intentional_col else None) or False,
                "unintentional": _parse_bool((r.get(unintentional_col) or "") if unintentional_col else None) or False,
                "environmental": _parse_bool((r.get(environmental_col) or "") if environmental_col else None) or False,
            }
            cats = list(threat_to_categories.get(code, []))
            cat_primary = cats[0] if cats else None
            ats = list(threat_to_asset_types.get(code, []))
            index[code] = {
                "code": code,
                "name": name or code,
                "description": desc,
                "flags": flags,
                "categories": cats,
                "category": cat_primary,
                "asset_types": ats,
            }
        return index

    def threat_detail(self, code: str) -> dict[str, Any] | None:
        """Return a single threat detail dict or None if not found."""
        idx = self._build_threat_index()
        return idx.get((code or "").strip()) if idx else None

    def list_threats(
        self,
        *,
        q: str | None = None,
        category: str | None = None,
        apply_asset_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return detailed threats with optional filters:
          - q: case-insensitive substring match over code/name/description
          - category: filter by category code (from Threat x Threat category.csv)
          - apply_asset_type: filter by mapped asset type (from Asset type x Threat.csv)
        """
        idx = self._build_threat_index()
        if not idx:
            return []

        items = list(idx.values())

        if category:
            cat_norm = category.strip().casefold()
            items = [t for t in items if any(c.casefold() == cat_norm for c in t.get("categories", []))]

        if apply_asset_type:
            at_norm = apply_asset_type.strip()
            items = [t for t in items if at_norm in (t.get("asset_types") or [])]

        if q:
            qn = q.casefold()
            def _hay(t: dict[str, Any]) -> str:
                return " ".join(
                    [
                        t.get("code") or "",
                        t.get("name") or "",
                        t.get("description") or "",
                        " ".join(t.get("categories") or []),
                        " ".join(t.get("asset_types") or []),
                    ]
                ).casefold()
            items = [t for t in items if qn in _hay(t)]

        # Stable order: by code
        items.sort(key=lambda t: (t.get("code") or "").casefold())
        return items


# --------------------------------------------------------------------------------------
# State (assets.csv) parsing
# --------------------------------------------------------------------------------------

def _parse_bool(s: str | None) -> bool | None:
    if s is None:
        return None
    t = s.strip().lower()
    if t in ("1", "true", "t", "yes", "y", "ano"):
        return True
    if t in ("0", "false", "f", "no", "n", "ne"):
        return False
    return None


def _parse_int_0_4(s: str | None) -> int | None:
    if s is None or s == "":
        return None
    try:
        i = int(s)
    except Exception:
        return None
    return i if 0 <= i <= 4 else None


def _map_conf_int_to_canonical(i: int | None) -> str | None:
    """
    Confidentiality canonical mapping expected by the domain schema:
      1→PUBLIC, 2→INTERNAL, 3→PROTECTED, 4→STRICTLY_PROTECTED; 0/None→None.
    """
    if i in (None, 0):
        return None
    return {1: "PUBLIC", 2: "INTERNAL", 3: "PROTECTED", 4: "STRICTLY_PROTECTED"}.get(i, None)


def _map_ia_int_to_canonical(i: int | None) -> str | None:
    """
    Integrity/Availability canonical mapping expected by the domain schema:
      1→LOW, 2→MEDIUM, 3→HIGH; 4 is clamped to HIGH; 0/None→None.
    """
    if i in (None, 0):
        return None
    return {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "HIGH"}.get(i, None)


def _build_asset_with_cia_fallback(payload: dict[str, Any]) -> Asset:
    """
    Build an Asset instance while being robust to CIA representation *and* ID type:
    - Prefer ints (0..4) if the model accepts them.
    - If the model expects enums/strings, map:
        confidentiality: 1..4 → PUBLIC/INTERNAL/PROTECTED/STRICTLY_PROTECTED
        integrity/availability: 1..3 (4→HIGH) → LOW/MEDIUM/HIGH
      0 or None → None.
    - If the canonical Asset rejects non-UUID id, fall back to a compatible
      local Pydantic model with str id for demo flows.
    """
    last_exc: Exception | None = None

    # 1) optimistic: current payload (likely ints)
    try:
        return Asset(**payload)  # type: ignore[arg-type]
    except Exception as e:
        last_exc = e

    # 2) canonical enum strings (most likely needed by the canonical schema)
    try:
        alt = dict(payload)
        alt["confidentiality"] = _map_conf_int_to_canonical(
            payload.get("confidentiality") if isinstance(payload.get("confidentiality"), int) else None
        ) if payload.get("confidentiality") is not None else None
        alt["integrity"] = _map_ia_int_to_canonical(
            payload.get("integrity") if isinstance(payload.get("integrity"), int) else None
        ) if payload.get("integrity") is not None else None
        alt["availability"] = _map_ia_int_to_canonical(
            payload.get("availability") if isinstance(payload.get("availability"), int) else None
        ) if payload.get("availability") is not None else None
        return Asset(**alt)  # type: ignore[arg-type]
    except Exception as e:
        last_exc = e

    # 3) try generic string-ish mappings for CIA (maximum resilience)
    def _digits(i: int | None) -> str | None:
        return None if i is None else str(i)

    def _level(i: int | None) -> str | None:
        return None if i is None else f"LEVEL_{i}"

    def _severity(i: int | None) -> str | None:
        return {0: "NONE", 1: "LOW", 2: "MODERATE", 3: "HIGH", 4: "CRITICAL"}.get(i, None)

    for mapper in (_digits, _level, _severity):
        try:
            alt = dict(payload)
            for k in ("confidentiality", "integrity", "availability"):
                v = alt.get(k)
                alt[k] = mapper(v) if isinstance(v, int) else None if v is None else v
            return Asset(**alt)  # type: ignore[arg-type]
        except Exception as e:
            last_exc = e
            continue

    # 4) ID compatibility fallback: if canonical Asset refuses non-UUID id,
    #    return a local compatible model that the UI path can serialize and render.
    class _CompatAssetStrId(BaseModel):
        id: str
        name: str
        description: str | None = None
        isPrimary: bool | None = None
        confidentiality: int | str | None = None
        integrity: int | str | None = None
        availability: int | str | None = None
        assetType: str
        org_id: str | None = None

        @field_validator("id", "name", "assetType")
        @classmethod
        def _non_empty(cls, v: str) -> str:
            v = (v or "").strip()
            if not v:
                raise ValueError("must not be empty")
            return v

    try:
        return _CompatAssetStrId(**payload)  # type: ignore[return-value]
    except Exception as e:
        last_exc = e

    # 5) give up – surface as backend validation error
    raise BackendValidationError(f"Invalid Asset row (CIA/ID mapping failed): {last_exc}")


class _AssetsState:
    """Lazy, cached access to assets state CSV with validation against catalogs."""

    def __init__(self, catalogs: _Catalogs) -> None:
        self.catalogs = catalogs
        self.assets = _CsvCache(_ASSETS_STATE_FILE)

    def list(
        self,
        org_id: str,
        search: str | None,
        type_code: str | None,
        is_primary: bool | None,
    ) -> list[Asset]:
        rows = self.assets.load()
        if not rows:
            return []

        # Validate provided type filter early to short-circuit
        if type_code:
            valid_types = set(self.catalogs.list_asset_types())
            if valid_types and type_code not in valid_types:
                # Unknown type filter → empty result (do not raise for user filter)
                return []

        out: list[Asset] = []
        for r in rows:
            try:
                a = _asset_from_row(r, catalogs=self.catalogs)
            except BackendValidationError:
                # Skip invalid row (strictness can be increased later to raise)
                continue

            if org_id and (a.org_id or "") != org_id:
                continue
            if type_code and a.assetType != type_code:
                continue
            if is_primary is not None:
                # If CSV has empty isPrimary, treat as False for filtering
                if bool(a.isPrimary) != bool(is_primary):
                    continue
            if search:
                needle = search.casefold()
                hay_name = (a.name or "").casefold()
                hay_desc = (a.description or "").casefold()
                if needle not in hay_name and needle not in hay_desc:
                    continue

            out.append(a)

        # Deterministic order: primary first, then by name (case-insensitive)
        out.sort(key=lambda x: (0 if bool(x.isPrimary) else 1, (x.name or "").casefold()))
        return out

    def get(self, asset_id: str) -> Asset | None:
        rows = self.assets.load()
        for r in rows:
            rid = (r.get("id") or r.get("ID") or "").strip()
            if rid == asset_id:
                try:
                    return _asset_from_row(r, catalogs=self.catalogs)
                except BackendValidationError:
                    return None
        return None


def _asset_from_row(row: Mapping[str, str], *, catalogs: _Catalogs) -> Asset:
    # Flexible header mapping (supports variations in capitalization and EU CSVs)
    id_ = (row.get("id") or row.get("ID") or "").strip()
    name = (row.get("name") or row.get("Name") or "").strip()
    description_raw = row.get("description") or row.get("Description")
    description = description_raw.strip() if isinstance(description_raw, str) else None

    is_primary = _parse_bool(
        row.get("isPrimary") or row.get("is_primary") or row.get("primary") or row.get("Is primary")
    )

    c = _parse_int_0_4(row.get("confidentiality") or row.get("C") or row.get("Confidentiality"))
    i = _parse_int_0_4(row.get("integrity") or row.get("I") or row.get("Integrity"))
    a = _parse_int_0_4(row.get("availability") or row.get("A") or row.get("Availability"))

    asset_type = (
        row.get("assetType")
        or row.get("Asset type")
        or row.get("type")
        or row.get("EN Name")
        or row.get("Name")
        or ""
    ).strip()

    org_id_raw = row.get("org_id") or row.get("org") or row.get("Org")
    org_id = org_id_raw.strip() if isinstance(org_id_raw, str) else None

    if not id_ or not name or not asset_type:
        raise BackendValidationError("Asset row missing mandatory fields (id, name, assetType).")

    # Validate assetType membership if catalog is available
    valid_types = set(catalogs.list_asset_types())
    if valid_types and asset_type not in valid_types:
        raise BackendValidationError(f"Unknown assetType '{asset_type}' (not in catalog).")

    # CIA rule: if asset is NOT primary, null out CIA (per canonical contract)
    if is_primary is False:
        c = i = a = None

    # Prepare base payload with CIA as ints first (fast path).
    payload: dict[str, Any] = {
        "id": id_,
        "name": name,
        "description": description,
        "isPrimary": is_primary,
        "confidentiality": c,
        "integrity": i,
        "availability": a,
        "assetType": asset_type,
        "org_id": org_id,
    }

    try:
        return _build_asset_with_cia_fallback(payload)
    except BackendValidationError as ve:
        # Normalize pydantic validation into backend error
        raise ve
    except ValidationError as ve:  # safety net if canonical model raised ValidationError directly
        raise BackendValidationError(f"Invalid Asset row: {ve}") from ve


# --------------------------------------------------------------------------------------
# CsvBackend implementation
# --------------------------------------------------------------------------------------

class CsvBackend(BackendPort):
    """CSV-backed implementation of BackendPort. Read-only for AssetAgent MVP."""

    def __init__(self) -> None:
        self._catalogs = _Catalogs()
        self._assets = _AssetsState(self._catalogs)

    # ---- Catalogs ----

    def list_asset_types(self) -> list[str]:
        return self._catalogs.list_asset_types()

    def list_impacts(self, category: str | None = None) -> list[str]:
        # Placeholder – wire to catalogs when available
        return []

    def list_threat_categories(self) -> list[str]:
        """Expose threat categories to callers (codes only)."""
        return self._catalogs.list_threat_categories()

    def list_threats(self, apply_asset_type: str | None = None, **kwargs) -> list[dict[str, Any]]:
        """
        Return detailed threats with optional filters.
        Accepts extra kwargs:
          - q: substring filter (code/name/description)
          - category: category code filter
        """
        q = kwargs.get("q")
        category = kwargs.get("category")
        return self._catalogs.list_threats(q=q, category=category, apply_asset_type=apply_asset_type)

    def threat_detail(self, code: str) -> dict[str, Any] | None:
        """Expose single-threat detail (dict) to callers."""
        return self._catalogs.threat_detail(code)

    def list_controls(self, category: str | None = None, include_coverage: bool = True) -> list[str]:
        # Placeholder – wire to catalogs when available
        return []

    def get_control_coverage(self, control_code: str) -> int | None:
        # Placeholder – wire to catalogs when available
        return None

    # ---- State (read) ----

    def assets_list(
        self,
        org_id: str,
        search: str | None = None,
        type: str | None = None,
        is_primary: bool | None = None,
    ) -> Sequence[Asset]:
        if not org_id:
            raise BackendValidationError("org_id is required")
        # Keep BackendPort signature (param name 'type') but pass as 'type_code'
        return self._assets.list(org_id, search, type, is_primary)

    def asset_get(self, asset_id: str) -> Asset | None:
        if not asset_id:
            raise BackendValidationError("asset_id is required")
        return self._assets.get(asset_id)

    def asset_get_impacts(self, asset_id: str) -> list[Mapping[str, object]]:
        # Not used by AssetAgent MVP; keep a consistent shape for later
        return []

    def asset_get_risks(self, asset_id: str) -> list[Mapping[str, object]]:
        # Not used by AssetAgent MVP
        return []

    def asset_get_controls(self, asset_id: str) -> list[Mapping[str, object]]:
        # Not used by AssetAgent MVP
        return []

    # ---- Staging lifecycle (not implemented in CSV MVP) ----

    def staging_open_batch(self, org_id: str, tags: list[str], comment: str | None = None) -> str:
        raise BackendConflict("CSV backend does not support staging.")

    def asset_stage_impacts(self, batch_id: str, asset_id: str, items: Sequence[Mapping[str, object]]) -> None:
        raise BackendConflict("CSV backend does not support staging.")

    def asset_stage_risks(self, batch_id: str, asset_id: str, items: Sequence[Mapping[str, object]]) -> None:
        raise BackendConflict("CSV backend does not support staging.")

    def asset_stage_controls(self, batch_id: str, asset_id: str, items: Sequence[Mapping[str, object]]) -> None:
        raise BackendConflict("CSV backend does not support staging.")

    def staging_diff(self, batch_id: str) -> dict:
        raise BackendConflict("CSV backend does not support staging.")

    def staging_commit(self, batch_id: str, signed_off_by: str, message: str | None = None) -> dict:
        raise BackendConflict("CSV backend does not support staging.")

    def staging_discard(self, batch_id: str) -> None:
        raise BackendConflict("CSV backend does not support staging.")
