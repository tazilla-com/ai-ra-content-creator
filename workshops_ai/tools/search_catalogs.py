# File: workshops_ai/tools/search_catalogs.py
# Purpose:
#   Read-only helpers to query CSV catalogs for “explain/help” use-cases.
#   - Zero writes. Dependency-light (stdlib csv).
#   - Hot-reload via mtime cache.
#   - Heuristic column detection (tolerant to header variants).
#   - Safe: never raises for missing catalogs; returns empty results instead.
#
# Catalogs (expected locations; resilient if absent):
#   <settings.paths.catalogs_dir>/Asset type.csv
#   <settings.paths.catalogs_dir>/Asset type x Threat.csv
#   (Placeholders for future: Impacts.csv, Threats.csv, Controls.csv)
#
# API (stable, small):
#   list_asset_types() -> list[str]
#   search_asset_types(q: str|None, *, limit=50) -> list[str]
#   explain_asset_type(code: str) -> dict|None
#   mapped_threats_for_asset_type(asset_type_code: str) -> list[str]
#   generic_search(file_name: str, q: str|None, *, fields: list[str]|None = None, limit=50) -> list[dict]


from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from workshops_ai.settings import get_settings

# Optional telemetry (best-effort)
try:  # pragma: no cover
    from workshops_ai.tools.telemetry import telemetry_log  # type: ignore
except Exception:  # pragma: no cover
    def telemetry_log(*args, **kwargs):  # type: ignore
        return None

# --------------------------------------------------------------------------------------
# Paths (centralized via settings)
# --------------------------------------------------------------------------------------

_SETTINGS = get_settings()
_CAT_DIR = Path(_SETTINGS.paths.catalogs_dir).resolve()

_CAT_ASSET_TYPE_FILE = _CAT_DIR / "Asset type.csv"
_CAT_AT_X_THREAT_FILE = _CAT_DIR / "Asset type x Threat.csv"

# Future reserved (may not exist yet)
_CAT_IMPACTS_FILE = _CAT_DIR / "Impacts.csv"
_CAT_THREATS_FILE = _CAT_DIR / "Threats.csv"
_CAT_CONTROLS_FILE = _CAT_DIR / "Controls.csv"


# --------------------------------------------------------------------------------------
# CSV cache (mtime-aware)
# --------------------------------------------------------------------------------------

@dataclass
class _CsvCache:
    path: Path
    mtime: float = 0.0
    rows: list[dict[str, str]] | None = None

    def load(self) -> list[dict[str, str]]:
        """Load or serve cached rows. Returns [] if missing."""
        try:
            if not self.path.exists():
                self.mtime, self.rows = 0.0, []
                return self.rows
            mt = self.path.stat().st_mtime
            if self.rows is not None and self.mtime == mt:
                return self.rows
            self.rows = _read_csv(self.path)
            self.mtime = mt
            return self.rows
        except Exception as e:  # never blow up caller
            telemetry_log("catalog.load_error", {"file": str(self.path), "error": str(e)}, level="WARN")
            self.mtime, self.rows = 0.0, []
            return self.rows or []


def _read_csv(path: Path) -> list[dict[str, str]]:
    # Try utf-8-sig first (handles BOM); fall back to utf-8 then cp1250 for EU exports
    for enc in ("utf-8-sig", "utf-8", "cp1250"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                # Normalize headers (strip whitespace/BOM remnants)
                if reader.fieldnames:
                    reader.fieldnames = [h.strip() for h in reader.fieldnames]
                out: list[dict[str, str]] = []
                for row in reader:
                    # Trim values; keep strings only (csv gives str or None)
                    clean = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                    out.append(clean)
                return out
        except UnicodeDecodeError:
            continue
    # Last resort: try reading raw and let csv handle whatever it can
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
        return [{k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()} for row in reader]


def _first_present_column(rows: list[dict[str, str]], candidates: Iterable[str]) -> Optional[str]:
    if not rows:
        return None
    cols = set(rows[0].keys())
    for c in candidates:
        if c in cols:
            return c
    return None


# --------------------------------------------------------------------------------------
# Catalog indices
# --------------------------------------------------------------------------------------

_asset_types = _CsvCache(_CAT_ASSET_TYPE_FILE)
_at_x_threat = _CsvCache(_CAT_AT_X_THREAT_FILE)

_impacts = _CsvCache(_CAT_IMPACTS_FILE)
_threats = _CsvCache(_CAT_THREATS_FILE)
_controls = _CsvCache(_CAT_CONTROLS_FILE)


# --------------------------------------------------------------------------------------
# Public API – Asset Types
# --------------------------------------------------------------------------------------

def list_asset_types() -> list[str]:
    """Return unique asset type codes in file order. Empty if catalog missing."""
    rows = _asset_types.load()
    if not rows:
        return []
    # Header heuristics (prefer code-like names)
    col = _first_present_column(rows, ("code", "asset_type", "type", "Asset type", "EN Name", "name")) or next(iter(rows[0].keys()))
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        code = (r.get(col) or "").strip()
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def search_asset_types(q: str | None, *, limit: int = 50) -> list[str]:
    """Simple contains() search across likely columns; returns codes."""
    rows = _asset_types.load()
    if not rows:
        return []
    qn = (q or "").casefold()
    code_col = _first_present_column(rows, ("code", "asset_type", "type", "Asset type", "EN Name", "name")) or next(iter(rows[0].keys()))
    # If there is a separate 'name' column, include it in matching
    name_col = _first_present_column(rows, ("name", "Name", "label", "Label"))

    seen: set[str] = set()
    res: list[str] = []
    for r in rows:
        code = (r.get(code_col) or "").strip()
        name = (r.get(name_col) or "").strip() if name_col else ""
        hay = f"{code} {name}".casefold()
        if not qn or qn in hay:
            if code and code not in seen:
                seen.add(code)
                res.append(code)
        if len(res) >= limit:
            break
    return res


def explain_asset_type(code: str) -> dict[str, str] | None:
    """
    Return the full catalog row for a given asset type code (best-effort match).
    If multiple rows match, the first is returned.
    """
    if not code:
        return None
    rows = _asset_types.load()
    if not rows:
        return None
    col = _first_present_column(rows, ("code", "asset_type", "type", "Asset type", "EN Name", "name")) or next(iter(rows[0].keys()))
    needle = code.strip()
    for r in rows:
        if (r.get(col) or "").strip() == needle:
            return dict(r)  # return a copy
    return None


# --------------------------------------------------------------------------------------
# Public API – Threat mappings (Asset type x Threat)
# --------------------------------------------------------------------------------------

def mapped_threats_for_asset_type(asset_type_code: str) -> list[str]:
    """
    Return unique threat codes mapped to the given asset type using the
    "Asset type x Threat.csv" mapping catalog. Empty if catalog missing.
    """
    if not asset_type_code:
        return []
    rows = _at_x_threat.load()
    if not rows:
        return []

    asset_col = _first_present_column(rows, ("asset_type", "Asset type"))
    threat_col = _first_present_column(rows, ("threat_code", "Threat", "threat"))
    if not threat_col:
        return []

    needle = asset_type_code.strip()
    out: set[str] = set()
    for r in rows:
        if asset_col:
            a = (r.get(asset_col) or "").strip()
            if a != needle:
                continue
        t = (r.get(threat_col) or "").strip()
        if t:
            out.add(t)
    return sorted(out)


# --------------------------------------------------------------------------------------
# Generic search (for future catalogs; safe & reusable)
# --------------------------------------------------------------------------------------

def generic_search(file_name: str, q: str | None, *, fields: list[str] | None = None, limit: int = 50) -> list[dict[str, str]]:
    """
    Generic read-only search over an arbitrary catalog CSV in <settings.paths.catalogs_dir>/.
    - q=None returns first `limit` rows.
    - If `fields` is specified, only those columns are searched (missing are ignored).
    - Otherwise, searches all columns.
    """
    if not file_name:
        return []
    path = (_CAT_DIR / file_name).resolve()
    cache = _CsvCache(path)
    rows = cache.load()
    if not rows:
        return []

    qn = (q or "").casefold()
    cols = list(rows[0].keys())
    use_cols = [c for c in (fields or cols) if c in cols]
    if not use_cols:
        use_cols = cols

    res: list[dict[str, str]] = []
    for r in rows:
        if not qn:
            res.append(dict(r))
        else:
            hay = " ".join((r.get(c) or "") for c in use_cols).casefold()
            if qn in hay:
                res.append(dict(r))
        if len(res) >= limit:
            break

    # Telemetry (best-effort)
    try:
        telemetry_log(
            "catalog.search",
            {"file": file_name, "q": q, "fields": use_cols, "returned": len(res)},
            level="DEBUG",
        )
    except Exception:
        pass

    return res


__all__ = [
    "list_asset_types",
    "search_asset_types",
    "explain_asset_type",
    "mapped_threats_for_asset_type",
    "generic_search",
]
