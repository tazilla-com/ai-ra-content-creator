# File: workshops_ai/backend/csv_backend.py
# CSV-based backend implementing BackendPort (read-only for assets/threats/impacts).
# - Reads state from <settings.paths.state_dir>/assets.csv (+ optional asset_impacts.csv)
# - Reads catalogs from <settings.paths.catalogs_dir>
# - Defensive CSV parsing (BOM, ; delimiter, cp1250), mtime-based hot reload
# - Pydantic v2 domain validation; canonical codes in EN
#
# Catalog files (relative to configured paths):
#   catalogs/Asset type.csv
#   catalogs/Threat.csv
#   catalogs/Threat category.csv
#   catalogs/Threat x Threat category.csv
#   catalogs/Asset type x Threat.csv
#   catalogs/Impact.csv
#   catalogs/Impact category.csv
#   catalogs/Impact x Impact category.csv
#
# State files:
#   state/assets.csv
#   state/asset_impacts.csv            # optional: {org_id?, asset_id, impact_code, level (0..4), note?}
#
# Notes:
# - Threat.csv has no explicit "code" column; canonical code = EN Name (fallback Name).
# - Impact.csv same: canonical code = EN Name (fallback Name).
# - "Asset type x Threat.csv" uses column Value (TRUE/FALSE). Only TRUE rows are mapped.

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
from workshops_ai.api.schema import Asset  # canonical DTO

__all__ = ["CsvBackend"]

# --------------------------------------------------------------------------------------
# Paths & config
# --------------------------------------------------------------------------------------

_s = get_settings()
_STATE_DIR = Path(_s.paths.state_dir).resolve()
_CAT_DIR = Path(_s.paths.catalogs_dir).resolve()

_ASSETS_STATE_FILE = _STATE_DIR / "assets.csv"
_ASSET_IMPACTS_FILE = _STATE_DIR / "asset_impacts.csv"  # optional

# Catalogs
_CAT_ASSET_TYPE_FILE = _CAT_DIR / "Asset type.csv"
_CAT_AT_X_THREAT_FILE = _CAT_DIR / "Asset type x Threat.csv"

_CAT_THREAT_FILE = _CAT_DIR / "Threat.csv"
_CAT_THREAT_CATEGORY_FILE = _CAT_DIR / "Threat category.csv"
_CAT_THREAT_X_CATEGORY_FILE = _CAT_DIR / "Threat x Threat category.csv"

_CAT_IMPACT_FILE = _CAT_DIR / "Impact.csv"
_CAT_IMPACT_CATEGORY_FILE = _CAT_DIR / "Impact category.csv"
_CAT_IMPACT_X_CATEGORY_FILE = _CAT_DIR / "Impact x Impact category.csv"

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
            # For state files, non-existence means empty; for assets.csv it's a hard error
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
    Detect encoding (utf-8-sig → utf-8 → cp1250) and delimiter (csv.Sniffer with ,;|\t).
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
        delim = ","
        try:
            dialect = csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
            delim = dialect.delimiter or delim
        except Exception:
            if sample_text.count(";") > sample_text.count(","):
                delim = ";"
        return enc, delim
    return "utf-8-sig", ","


def _read_csv(path: Path) -> list[dict[str, str]]:
    encoding, delimiter = _detect_encoding_and_delimiter(path)
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames:
            reader.fieldnames = [h.strip() if isinstance(h, str) else h for h in reader.fieldnames]
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
# Catalog parsing (asset types, threats, impacts)
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


class _Catalogs:
    """Lazy, cached access to catalog CSVs with resilient column heuristics."""

    def __init__(self) -> None:
        # Asset types + mapping
        self.asset_types = _CsvCache(_CAT_ASSET_TYPE_FILE)
        self.asset_type_x_threat = _CsvCache(_CAT_AT_X_THREAT_FILE)

        # Threats + categories
        self.threats = _CsvCache(_CAT_THREAT_FILE)
        self.threat_categories = _CsvCache(_CAT_THREAT_CATEGORY_FILE)
        self.threat_x_category = _CsvCache(_CAT_THREAT_X_CATEGORY_FILE)

        # Impacts + categories
        self.impacts = _CsvCache(_CAT_IMPACT_FILE)
        self.impact_categories = _CsvCache(_CAT_IMPACT_CATEGORY_FILE)
        self.impact_x_category = _CsvCache(_CAT_IMPACT_X_CATEGORY_FILE)

    # -------- Asset types --------

    def list_asset_types(self) -> list[str]:
        rows = self.asset_types.load()
        if not rows:
            return []
        col = _first_present_column(
            rows,
            ("code", "asset_type", "type", "Asset type", "EN Name", "EN name", "Name", "name"),
        )
        if col is None:
            col = next(iter(rows[0].keys()))
        seen: set[str] = set()
        res: list[str] = []
        for r in rows:
            code = (r.get(col) or "").strip()
            if code and code not in seen:
                seen.add(code)
                res.append(code)
        return res

    # -------- Threat categories --------

    def list_threat_categories(self) -> list[str]:
        rows = self.threat_categories.load()
        if not rows:
            return []
        col = _first_present_column(
            rows,
            ("category_code", "code", "Category", "Threat category", "threat_category", "EN Name", "Name"),
        ) or next(iter(rows[0].keys()))
        seen, out = set(), []
        for r in rows:
            v = (r.get(col) or "").strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    # -------- Threats (detail) --------

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

        # In our CSV, canonical code is EN Name (fallback Name)
        code_col = _first_present_column(t_rows, ("code", "threat_code", "EN Name", "Name"))
        name_col = _first_present_column(t_rows, ("name", "EN Name", "Name"))
        desc_col = _first_present_column(t_rows, ("description", "EN Notes", "Notes"))
        intentional_col = _first_present_column(t_rows, ("intentional", "Intentional"))
        unintentional_col = _first_present_column(t_rows, ("unintentional", "Unintentional", "accidental"))
        environmental_col = _first_present_column(t_rows, ("environmental", "Environmental", "env"))

        # Threat ↔ category
        txc = self.threat_x_category.load()
        txc_threat_col = _first_present_column(txc, ("threat_code", "Threat", "Threat code", "EN Name", "Name", "code"))
        txc_cat_col = _first_present_column(txc, ("category_code", "Category", "Threat category", "threat_category", "EN Name", "Name", "code"))
        threat_to_categories: dict[str, list[str]] = {}
        if txc and txc_threat_col and txc_cat_col:
            for r in txc:
                th = (r.get(txc_threat_col) or "").strip()
                cat = (r.get(txc_cat_col) or "").strip()
                if th and cat:
                    threat_to_categories.setdefault(th, [])
                    if cat not in threat_to_categories[th]:
                        threat_to_categories[th].append(cat)

        # Asset type ↔ threat (respect Value==TRUE)
        att = self.asset_type_x_threat.load()
        att_asset_col = _first_present_column(att, ("asset_type", "Asset type", "EN Name", "Name"))
        att_threat_col = _first_present_column(att, ("threat_code", "Threat", "EN Name", "Name", "threat"))
        att_val_col = _first_present_column(att, ("Value", "value", "Enabled", "enabled"))
        threat_to_asset_types: dict[str, list[str]] = {}
        if att and att_asset_col and att_threat_col:
            for r in att:
                at = (r.get(att_asset_col) or "").strip()
                th = (r.get(att_threat_col) or "").strip()
                ok = True
                if att_val_col:
                    ok = bool(_parse_bool(r.get(att_val_col)) or False)
                if at and th and ok:
                    threat_to_asset_types.setdefault(th, [])
                    if at not in threat_to_asset_types[th]:
                        threat_to_asset_types[th].append(at)

        # Build index
        index: dict[str, dict[str, Any]] = {}
        for r in t_rows:
            code = ((r.get(code_col) or "") if code_col else "").strip()
            if not code:
                continue
            name = ((r.get(name_col) or "") if name_col else "").strip() or code
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
                "name": name,
                "description": desc,
                "flags": flags,
                "categories": cats,
                "category": cat_primary,
                "asset_types": ats,
            }
        return index

    def threat_detail(self, code: str) -> dict[str, Any] | None:
        idx = self._build_threat_index()
        return idx.get((code or "").strip()) if idx else None

    def list_threats(
        self,
        *,
        q: str | None = None,
        category: str | None = None,
        apply_asset_type: str | None = None,
    ) -> list[dict[str, Any]]:
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

        items.sort(key=lambda t: (t.get("code") or "").casefold())
        return items

    # -------- Impacts (catalog) --------

    def _build_impact_index(self) -> dict[str, dict[str, Any]]:
        """
        Build a dict: code -> impact dict with fields:
          code, name, description?, categories[], category
        """
        i_rows = self.impacts.load()
        if not i_rows:
            return {}

        code_col = _first_present_column(i_rows, ("code", "impact_code", "EN Name", "Name"))
        name_col = _first_present_column(i_rows, ("name", "EN Name", "Name"))
        desc_col = _first_present_column(i_rows, ("description", "EN Notes", "Notes"))

        # Impact ↔ category
        ixc = self.impact_x_category.load()
        ixc_imp_col = _first_present_column(ixc, ("impact_code", "Impact", "Impact code", "EN Name", "Name", "code"))
        ixc_cat_col = _first_present_column(ixc, ("category_code", "Category", "Impact category", "impact_category", "EN Name", "Name", "code"))
        impact_to_categories: dict[str, list[str]] = {}
        if ixc and ixc_imp_col and ixc_cat_col:
            for r in ixc:
                im = (r.get(ixc_imp_col) or "").strip()
                cat = (r.get(ixc_cat_col) or "").strip()
                if im and cat:
                    impact_to_categories.setdefault(im, [])
                    if cat not in impact_to_categories[im]:
                        impact_to_categories[im].append(cat)

        index: dict[str, dict[str, Any]] = {}
        for r in i_rows:
            code = ((r.get(code_col) or "") if code_col else "").strip()
            if not code:
                continue
            name = ((r.get(name_col) or "") if name_col else "").strip() or code
            desc = ((r.get(desc_col) or "") if desc_col else "").strip() or None
            cats = list(impact_to_categories.get(code, []))
            cat_primary = cats[0] if cats else None
            index[code] = {
                "code": code,
                "name": name,
                "description": desc,
                "categories": cats,
                "category": cat_primary,
            }
        return index

    def list_impacts(self, *, q: str | None = None, category: str | None = None) -> list[dict[str, Any]]:
        idx = self._build_impact_index()
        if not idx:
            return []
        items = list(idx.values())

        if category:
            cat_norm = category.strip().casefold()
            items = [i for i in items if any(c.casefold() == cat_norm for c in i.get("categories", []))]

        if q:
            qn = q.casefold()
            def _hay(i: dict[str, Any]) -> str:
                return " ".join(
                    [
                        i.get("code") or "",
                        i.get("name") or "",
                        i.get("description") or "",
                        " ".join(i.get("categories") or []),
                    ]
                ).casefold()
            items = [i for i in items if qn in _hay(i)]

        items.sort(key=lambda i: (i.get("code") or "").casefold())
        return items

# --------------------------------------------------------------------------------------
# State (assets.csv) parsing
# --------------------------------------------------------------------------------------

def _parse_int_0_4(s: str | None) -> int | None:
    if s is None or s == "":
        return None
    try:
        i = int(s)
    except Exception:
        return None
    return i if 0 <= i <= 4 else None


def _map_conf_int_to_canonical(i: int | None) -> str | None:
    if i in (None, 0):
        return None
    return {1: "PUBLIC", 2: "INTERNAL", 3: "PROTECTED", 4: "STRICTLY_PROTECTED"}.get(i, None)


def _map_ia_int_to_canonical(i: int | None) -> str | None:
    if i in (None, 0):
        return None
    return {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "HIGH"}.get(i, None)


def _build_asset_with_cia_fallback(payload: dict[str, Any]) -> Asset:
    """
    Build Asset while being robust to CIA representation and ID type.
    Falls back to local str-ID model if canonical UUID is not available.
    """
    last_exc: Exception | None = None

    # 1) optimistic
    try:
        return Asset(**payload)  # type: ignore[arg-type]
    except Exception as e:
        last_exc = e

    # 2) map integers -> canonical enums
    try:
        alt = dict(payload)
        for k in ("confidentiality", "integrity", "availability"):
            v = alt.get(k)
            if k == "confidentiality":
                alt[k] = _map_conf_int_to_canonical(v) if isinstance(v, int) else v
            else:
                alt[k] = _map_ia_int_to_canonical(v) if isinstance(v, int) else v
        return Asset(**alt)  # type: ignore[arg-type]
    except Exception as e:
        last_exc = e

    # 3) fallback mapping attempts
    def _digits(i: int | None) -> str | None: return None if i is None else str(i)
    def _level(i: int | None) -> str | None: return None if i is None else f"LEVEL_{i}"

    for mapper in (_digits, _level):
        try:
            alt = dict(payload)
            for k in ("confidentiality", "integrity", "availability"):
                v = alt.get(k)
                alt[k] = mapper(v) if isinstance(v, int) else None if v is None else v
            return Asset(**alt)  # type: ignore[arg-type]
        except Exception as e:
            last_exc = e
            continue

    # 4) local compatibility model (string id)
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

    raise BackendValidationError(f"Invalid Asset row (CIA/ID mapping failed): {last_exc}")


class _AssetsState:
    """Lazy, cached access to assets state CSV with validation against catalogs."""

    def __init__(self, catalogs: _Catalogs) -> None:
        self.catalogs = catalogs
        self.assets = _CsvCache(_ASSETS_STATE_FILE)
        self.asset_impacts = _CsvCache(_ASSET_IMPACTS_FILE)

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

        # Validate provided type filter early
        if type_code:
            valid_types = set(self.catalogs.list_asset_types())
            if valid_types and type_code not in valid_types:
                return []

        out: list[Asset] = []
        for r in rows:
            try:
                a = _asset_from_row(r, catalogs=self.catalogs)
            except BackendValidationError:
                continue

            if org_id and (a.org_id or "") != org_id:
                continue
            if type_code and a.assetType != type_code:
                continue
            if is_primary is not None and bool(a.isPrimary) != bool(is_primary):
                continue
            if search:
                needle = search.casefold()
                if needle not in (a.name or "").casefold() and needle not in (a.description or "").casefold():
                    continue

            out.append(a)

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

    def get_impacts(self, asset_id: str) -> list[Mapping[str, object]]:
        """
        Return [{impact_code, level, note?}] from optional state/asset_impacts.csv.
        Heuristics for columns: asset_id|Asset ID, impact_code|Impact, level|Level, note|Note.
        """
        rows = self.asset_impacts.load()
        if not rows:
            return []

        # Column heuristics from header of first row
        aid_col = _first_present_column(rows, ("asset_id", "Asset ID", "assetId", "ID"))
        code_col = _first_present_column(rows, ("impact_code", "Impact", "impact", "EN Name", "Name", "code"))
        lvl_col = _first_present_column(rows, ("level", "Level", "impact_level", "impactLevel"))
        note_col = _first_present_column(rows, ("note", "Note", "notes", "Notes"))

        if not aid_col or not code_col or not lvl_col:
            return []

        out: list[dict[str, object]] = []
        for r in rows:
            if (r.get(aid_col) or "").strip() != asset_id:
                continue
            code = (r.get(code_col) or "").strip()
            level = _parse_int_0_4(r.get(lvl_col))
            if not code or level is None:
                continue
            item: dict[str, object] = {"impact_code": code, "level": level}
            note = r.get(note_col) if note_col else None
            if note:
                note = note.strip()
                if note:
                    item["note"] = note
            out.append(item)
        return out


def _asset_from_row(row: Mapping[str, str], *, catalogs: _Catalogs) -> Asset:
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

    # Validate assetType via catalog
    valid_types = set(catalogs.list_asset_types())
    if valid_types and asset_type not in valid_types:
        raise BackendValidationError(f"Unknown assetType '{asset_type}' (not in catalog).")

    # CIA rule for non-primary
    if is_primary is False:
        c = i = a = None

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
    except BackendValidationError:
        raise
    except ValidationError as ve:
        raise BackendValidationError(f"Invalid Asset row: {ve}") from ve

# --------------------------------------------------------------------------------------
# CsvBackend implementation
# --------------------------------------------------------------------------------------

class CsvBackend(BackendPort):
    """CSV-backed implementation of BackendPort (read-only; staging not supported)."""

    def __init__(self) -> None:
        self._catalogs = _Catalogs()
        self._assets = _AssetsState(self._catalogs)

    # ---- Catalogs ----

    def list_asset_types(self) -> list[str]:
        return self._catalogs.list_asset_types()

    def list_threat_categories(self) -> list[str]:
        return self._catalogs.list_threat_categories()

    def list_threats(self, apply_asset_type: str | None = None, **kwargs) -> list[dict[str, Any]]:
        """
        Return detailed threats with optional filters.
        kwargs:
          - q: substring filter (code/name/description/categories/asset_types)
          - category: category code filter
        """
        q = kwargs.get("q")
        category = kwargs.get("category")
        return self._catalogs.list_threats(q=q, category=category, apply_asset_type=apply_asset_type)

    def threat_detail(self, code: str) -> dict[str, Any] | None:
        return self._catalogs.threat_detail(code)

    def list_impacts(self, category: str | None = None) -> list[dict[str, Any]]:
        """Return detailed impacts from catalog with optional category filter."""
        return self._catalogs.list_impacts(q=None, category=category)

    # Controls are intentionally out of scope (placeholders)
    def list_controls(self, category: str | None = None, include_coverage: bool = True) -> list[str]:
        return []

    def get_control_coverage(self, control_code: str) -> int | None:
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
        return self._assets.list(org_id, search, type, is_primary)

    def asset_get(self, asset_id: str) -> Asset | None:
        if not asset_id:
            raise BackendValidationError("asset_id is required")
        return self._assets.get(asset_id)

    def asset_get_impacts(self, asset_id: str) -> list[Mapping[str, object]]:
        if not asset_id:
            raise BackendValidationError("asset_id is required")
        # optional state file; returns [] if missing/empty
        return self._assets.get_impacts(asset_id)

    def asset_get_risks(self, asset_id: str) -> list[Mapping[str, object]]:
        # Not tracked in CSV MVP (keep shape for future)
        return []

    def asset_get_controls(self, asset_id: str) -> list[Mapping[str, object]]:
        return []

    # ---- Staging lifecycle (unsupported in CSV MVP) ----

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
