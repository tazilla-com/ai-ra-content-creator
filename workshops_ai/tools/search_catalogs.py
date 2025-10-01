# File: workshops_ai/tools/search_catalogs.py
# Purpose:
#   Read-only helpers to query CSV catalogs for “explain/help” use-cases.
#   - Zero writes. Dependency-light (stdlib csv).
#   - Hot-reload via mtime cache.
#   - Heuristic column + delimiter/encoding detection.
#   - Safe: never raises for missing catalogs; returns empty results instead.
#
# Catalogs (expected locations; resilient if absent):
#   <settings.paths.catalogs_dir>/...
#   (fallback) workshops_ai/data/catalogs/...
#   (fallback) /app/data/catalogs/...
#
# API:
#   list_asset_types() -> list[str]
#   search_asset_types(q: str|None, *, limit=50) -> list[str]
#   explain_asset_type(code: str) -> dict|None
#   mapped_threats_for_asset_type(asset_type_code: str) -> list[str]
#
#   list_threat_categories() -> list[str]
#   normalize_threat_category(code_or_alias: str) -> str|None
#   explain_threat_category(category_code: str) -> dict|None
#
#   normalize_threat_code(code_or_alias: str) -> str|None
#   search_threats(q: str|None, *, category: str|None = None,
#                  apply_asset_type: str|None = None, limit=200) -> list[dict]
#   list_threats(...)  -> alias of search_threats (same params)
#   explain_threat(code: str) -> dict|None
#
#   list_impact_categories() -> list[str]
#   normalize_impact_code(code_or_alias: str) -> str|None
#   search_impacts(q: str|None, *, category: str|None = None, limit=200) -> list[dict]
#   list_impacts(...) -> alias of search_impacts
#   explain_impact(code: str) -> dict|None
#
#   generic_search(file_name: str, q: str|None, *, fields: list[str]|None = None, limit=50) -> list[dict]
#

from __future__ import annotations

import csv
import unicodedata
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
# Paths (centralized via settings) + robust fallbacks
# --------------------------------------------------------------------------------------

def _resolve_catalog_dir() -> Path:
    """
    Pick the first existing catalogs directory from:
      1) settings.paths.catalogs_dir
      2) package fallback: workshops_ai/data/catalogs
      3) runtime mount fallback: /app/data/catalogs
    """
    s = get_settings()
    candidates: list[Path] = []
    try:
        candidates.append(Path(s.paths.catalogs_dir).resolve())
    except Exception:
        pass
    # package-relative fallback
    pkg_fallback = Path(__file__).resolve().parent.parent / "data" / "catalogs"
    candidates.append(pkg_fallback)
    # runtime bind mount fallback
    candidates.append(Path("/app/data/catalogs"))

    for p in candidates:
        try:
            if p.exists() and p.is_dir():
                return p
        except Exception:
            continue
    # last resort: return the first candidate even if not existing (non-crashing behavior downstream)
    return candidates[0] if candidates else Path.cwd()


_CAT_DIR = _resolve_catalog_dir()

# Asset/Threat catalogs
_CAT_ASSET_TYPE_FILE = _CAT_DIR / "Asset type.csv"
_CAT_AT_X_THREAT_FILE = _CAT_DIR / "Asset type x Threat.csv"
_CAT_THREAT_FILE = _CAT_DIR / "Threat.csv"
_CAT_THREAT_CATEGORY_FILE = _CAT_DIR / "Threat category.csv"
_CAT_THREAT_X_CATEGORY_FILE = _CAT_DIR / "Threat x Threat category.csv"

# Impact catalogs
_CAT_IMPACT_FILE = _CAT_DIR / "Impact.csv"
_CAT_IMPACT_CATEGORY_FILE = _CAT_DIR / "Impact category.csv"
_CAT_IMPACT_X_CATEGORY_FILE = _CAT_DIR / "Impact x Impact category.csv"


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


# ---- file reading (encoding + delimiter detection) -----------------------------------

_DELIMS = [",", ";", "\t", "|"]
_ENCODINGS = ["utf-8-sig", "utf-8", "cp1250", "cp1252", "iso-8859-2"]


def _detect_encoding_and_delimiter(path: Path) -> tuple[str, str]:
    """
    Detect encoding among utf-8-sig → utf-8 → cp1250 → cp1252 → iso-8859-2 and delimiter via csv.Sniffer.
    Falls back to ('utf-8-sig', ',') if unsure. Semicolon is common in EU CSVs.
    """
    sample_bytes = b""
    try:
        with path.open("rb") as fb:
            sample_bytes = fb.read(4096)
    except FileNotFoundError:
        return "utf-8-sig", ","

    # try encodings in order; prefer those that decode without replacement
    for enc in _ENCODINGS:
        try:
            sample_text = sample_bytes.decode(enc, errors="strict")
        except UnicodeDecodeError:
            continue

        # detect delimiter
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample_text, delimiters="".join(_DELIMS))
            delim = dialect.delimiter or ","
        except Exception:
            # fallback: simple frequency count
            counts = {d: sample_text.count(d) for d in _DELIMS}
            # prefer ';' when comparable to ',' (Excel EU export)
            delim = max(counts, key=counts.get) if counts else ","
            if delim == "," and counts.get(";", 0) >= counts.get(",", 0):
                delim = ";"
        return enc, delim

    return "utf-8-sig", ","


def _read_csv(path: Path) -> list[dict[str, str]]:
    """
    Read CSV into list[dict]. Robust to BOM, semicolons, CP1250/CP1252; trims headers/values.
    If the initially-detected encoding fails during full read, retry with other encodings.
    """
    detected_enc, delim = _detect_encoding_and_delimiter(path)

    # Build a preference order: (detected first) + remaining candidates
    enc_try_order = [detected_enc] + [e for e in _ENCODINGS if e != detected_enc]

    last_err: Exception | None = None
    for enc in enc_try_order:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f, delimiter=delim)
                # normalize headers (strip + remove BOM if present)
                if reader.fieldnames:
                    reader.fieldnames = [
                        (h or "").replace("\ufeff", "").strip() for h in reader.fieldnames
                    ]
                out: list[dict[str, str]] = []
                for row in reader:
                    if not row:
                        continue
                    clean = {
                        (k.strip() if isinstance(k, str) else k): (v.strip() if isinstance(v, str) else v)
                        for k, v in row.items()
                    }
                    out.append(clean)
                return out
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            break

    # If all attempts failed, log and return []
    telemetry_log("catalog.load_error", {"file": str(path), "error": str(last_err)}, level="WARN")
    return []


def _first_present_column(rows: list[dict[str, str]], candidates: Iterable[str]) -> Optional[str]:
    if not rows:
        return None
    cols_lc = {(c or "").strip().casefold(): c for c in rows[0].keys()}
    for c in candidates:
        key = (c or "").strip().casefold()
        if key in cols_lc:
            return cols_lc[key]
    return None


# --------------------------------------------------------------------------------------
# Text normalization (diacritics, case, separators)
# --------------------------------------------------------------------------------------

def _slug(s: str | None) -> str:
    """
    Lowercased, ASCII-only slug for tolerant matching:
      - removes diacritics (č->c, ě->e, ľ->l, etc.)
      - strips non-alnum to single spaces
      - trims spaces
    """
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    base = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    out = []
    prev_space = False
    for ch in base.lower():
        if ch.isalnum():
            out.append(ch)
            prev_space = False
        else:
            if not prev_space:
                out.append(" ")
                prev_space = True
    return " ".join("".join(out).split())


# --------------------------------------------------------------------------------------
# Catalog indices (caches)
# --------------------------------------------------------------------------------------

_asset_types = _CsvCache(_CAT_ASSET_TYPE_FILE)
_at_x_threat = _CsvCache(_CAT_AT_X_THREAT_FILE)

_threats = _CsvCache(_CAT_THREAT_FILE)
_threat_categories = _CsvCache(_CAT_THREAT_CATEGORY_FILE)
_threat_x_category = _CsvCache(_CAT_THREAT_X_CATEGORY_FILE)

_impacts = _CsvCache(_CAT_IMPACT_FILE)
_impact_categories = _CsvCache(_CAT_IMPACT_CATEGORY_FILE)
_impact_x_category = _CsvCache(_CAT_IMPACT_X_CATEGORY_FILE)


# --------------------------------------------------------------------------------------
# Public API – Asset Types
# --------------------------------------------------------------------------------------

def list_asset_types() -> list[str]:
    """Return unique asset type codes in file order. Empty if catalog missing."""
    rows = _asset_types.load()
    if not rows:
        return []
    col = _first_present_column(
        rows, ("code", "asset_type", "type", "Asset type", "EN Name", "name", "Name")
    ) or next(iter(rows[0].keys()))
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
    code_col = _first_present_column(
        rows, ("code", "asset_type", "type", "Asset type", "EN Name", "name", "Name")
    ) or next(iter(rows[0].keys()))
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
    col = _first_present_column(
        rows, ("code", "asset_type", "type", "Asset type", "EN Name", "name", "Name")
    ) or next(iter(rows[0].keys()))
    needle = code.strip()
    for r in rows:
        if (r.get(col) or "").strip() == needle:
            return dict(r)
    return None


# --------------------------------------------------------------------------------------
# Public API – Threat categories
# --------------------------------------------------------------------------------------

def list_threat_categories() -> list[str]:
    """Return canonical threat category codes (best-effort header detection)."""
    rows = _threat_categories.load()
    if not rows:
        return []
    col = _first_present_column(
        rows,
        ("category_code", "code", "Category", "Threat category", "threat_category", "EN Name", "Name"),
    ) or next(iter(rows[0].keys()) if rows else [], None)
    if not col:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        v = (r.get(col) or "").strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _category_row_by_any(value: str) -> dict[str, str] | None:
    """Best-effort match of a category by code or human label (CZ/SK/EN tolerant)."""
    if not value:
        return None
    rows = _threat_categories.load()
    if not rows:
        return None
    code_cols = ("category_code", "code")
    name_cols = ("name", "Name", "EN Name", "CZ Name", "SK Name", "Threat category", "Category")
    code_col = _first_present_column(rows, code_cols)
    sl_value = _slug(value)
    if code_col:
        for r in rows:
            if (r.get(code_col) or "").strip() == value.strip():
                return r
    cols = [c for c in (code_cols + name_cols) if _first_present_column(rows, (c,))]
    for r in rows:
        for c in cols:
            v = r.get(c)
            if v and _slug(v) == sl_value:
                return r
    return None


def normalize_threat_category(code_or_alias: str) -> Optional[str]:
    """
    Normalize a user-provided category (EN code or CZ/SK/EN alias) to the canonical EN code.
    Returns None if no match.
    """
    row = _category_row_by_any(code_or_alias)
    if not row:
        return None
    for c in ("category_code", "code"):
        if c in row and row[c]:
            return row[c].strip()
    for c in ("name", "Name", "EN Name", "Threat category", "Category"):
        if c in row and row[c]:
            return row[c].strip()
    return None


def explain_threat_category(category_code: str) -> Optional[dict[str, str]]:
    """Return the full row for a canonical category code (or None)."""
    if not category_code:
        return None
    rows = _threat_categories.load()
    if not rows:
        return None
    col = _first_present_column(rows, ("category_code", "code"))
    needle = category_code.strip()
    if not col:
        return None
    for r in rows:
        if (r.get(col) or "").strip() == needle:
            return dict(r)
    return None


# --------------------------------------------------------------------------------------
# Public API – Threats
# --------------------------------------------------------------------------------------

def _build_threat_index() -> dict[str, dict[str, Any]]:
    """
    Build a minimal, search-friendly index:
      code, name, description?, flags{intentional,unintentional,environmental},
      categories: [code], category: first or None, asset_types: [code]
    Canonical 'code' = 'EN Name' (fallback 'Name') if 'code' is not present.
    """
    t_rows = _threats.load()
    if not t_rows:
        return {}

    # --- columns in Threat.csv ---
    code_col = _first_present_column(t_rows, ("code", "threat_code", "EN Name", "Name", "Code", "Threat", "ID"))
    name_col = _first_present_column(t_rows, ("name", "EN Name", "Name", "Threat name", "Threat"))
    desc_col = _first_present_column(t_rows, ("description", "EN Notes", "Notes", "Description", "EN Description"))
    intentional_col = _first_present_column(t_rows, ("intentional", "Intentional"))
    unintentional_col = _first_present_column(t_rows, ("unintentional", "Unintentional", "accidental"))
    environmental_col = _first_present_column(t_rows, ("environmental", "Environmental", "env"))

    # --- resolver name→code from Threat.csv (handles diacritics) ---
    name_slug_to_code: dict[str, str] = {}
    lower_code_map: dict[str, str] = {}
    for r in t_rows:
        c = ((r.get(code_col) or "") if code_col else "").strip()
        n = ((r.get(name_col) or "") if name_col else "").strip()
        if not c:
            continue
        lower_code_map[c.lower()] = c
        if n:
            name_slug_to_code.setdefault(_slug(n), c)

    def _resolve_threat_handle(raw: str | None) -> Optional[str]:
        """Return canonical code for a code-or-name handle."""
        if not raw:
            return None
        s = raw.strip()
        # exact code
        if s in lower_code_map.values():
            return s
        # case-insensitive code
        if s.lower() in lower_code_map:
            return lower_code_map[s.lower()]
        # name match by slug
        sl = _slug(s)
        return name_slug_to_code.get(sl)

    # --- Threat ↔ category mapping ---
    txc = _threat_x_category.load()
    txc_threat_col = _first_present_column(txc, ("threat_code", "Threat", "Threat code", "code", "EN Name", "Name"))
    txc_cat_col = _first_present_column(txc, ("category_code", "Category", "Threat category", "threat_category", "code", "EN Name", "Name"))
    threat_to_categories: dict[str, list[str]] = {}
    if txc and txc_threat_col and txc_cat_col:
        for r in txc:
            th_raw = (r.get(txc_threat_col) or "").strip()
            cat = (r.get(txc_cat_col) or "").strip()
            if not th_raw or not cat:
                continue
            th_code = _resolve_threat_handle(th_raw)
            if not th_code:
                continue
            threat_to_categories.setdefault(th_code, [])
            if cat not in threat_to_categories[th_code]:
                threat_to_categories[th_code].append(cat)

    # --- Asset type ↔ threat mapping (respect "Value" truthiness when present) ---
    # --- Asset type ↔ Threat mapping (IGNORE Value) ---
    att = _at_x_threat.load()
    att_asset_col  = _first_present_column(att, ("asset_type", "Asset type", "EN Name", "Name"))
    att_threat_col = _first_present_column(att, ("threat_code", "Threat", "threat", "EN Name", "Name"))

    threat_to_asset_types: dict[str, list[str]] = {}
    if att and att_threat_col:
        for r in att:
            at = (r.get(att_asset_col) or "").strip() if att_asset_col else None
            raw_th = (r.get(att_threat_col) or "").strip()
            if not raw_th:
                continue
            # normalize name/alias → canonical code (fallback to raw if unknown)
            th_code = _resolve_threat_handle(raw_th) or raw_th

            threat_to_asset_types.setdefault(th_code, [])
            if at and at not in threat_to_asset_types[th_code]:
                threat_to_asset_types[th_code].append(at)

    # --- build final index (keyed by canonical code) ---
    idx: dict[str, dict[str, Any]] = {}
    for r in t_rows:
        code = ((r.get(code_col) or "") if code_col else "").strip()
        if not code:
            continue
        name = ((r.get(name_col) or "") if name_col else "").strip()
        desc = ((r.get(desc_col) or "") if desc_col else "").strip() or None
        flags = {
            "intentional": _truthy(r.get(intentional_col)) if intentional_col else False,
            "unintentional": _truthy(r.get(unintentional_col)) if unintentional_col else False,
            "environmental": _truthy(r.get(environmental_col)) if environmental_col else False,
        }
        cats = list(threat_to_categories.get(code, []))
        cat_primary = cats[0] if cats else None
        ats = list(threat_to_asset_types.get(code, []))
        idx[code] = {
            "code": code,
            "name": name or code,
            "description": desc,
            "flags": flags,
            "categories": cats,
            "category": cat_primary,
            "asset_types": ats,
        }
    return idx


def _truthy(v: Any) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "ano"}


def normalize_threat_code(code_or_alias: str) -> Optional[str]:
    """
    Normalize a user-provided threat handle (code or human name) to the canonical code.
    - Exact code match first (case-sensitive then case-insensitive).
    - Then slug-match against the 'name' field.
    """
    if not code_or_alias:
        return None
    idx = _build_threat_index()
    if not idx:
        return None
    raw = code_or_alias.strip()
    if raw in idx:
        return raw
    lower_map = {k.lower(): k for k in idx.keys()}
    if raw.lower() in lower_map:
        return lower_map[raw.lower()]
    sl = _slug(raw)
    for th in idx.values():
        if _slug(th.get("name")) == sl:
            return th["code"]
    return None


def search_threats(
    q: str | None,
    *,
    category: str | None = None,
    apply_asset_type: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """
    Search threats with tolerant filters:
      - q: substring (code/name/description/categories/asset_types), diacritics-insensitive
      - category: accepts EN code or CZ/SK/EN alias; normalized to canonical code
      - apply_asset_type: filter threats mapped to given asset-type code (respects Value=TRUE)
    Returns detailed dicts: {code, name, description?, flags, categories, category, asset_types}
    """
    idx = _build_threat_index()
    if not idx:
        return []
    items = list(idx.values())

    if category:
        norm_cat = normalize_threat_category(category)
        if norm_cat:
            items = [t for t in items if norm_cat in (t.get("categories") or [])]
        else:
            items = []  # unknown category → empty

    if apply_asset_type:
        at = apply_asset_type.strip().casefold()

        def _has_at(t: dict[str, Any]) -> bool:
            return any((x or "").strip().casefold() == at for x in (t.get("asset_types") or []))

        items = [t for t in items if _has_at(t)]

    if q:
        qn = _slug(q)

        def _hay(t: dict[str, Any]) -> str:
            return _slug(
                " ".join(
                    [
                        t.get("code") or "",
                        t.get("name") or "",
                        t.get("description") or "",
                        " ".join(t.get("categories") or []),
                        " ".join(t.get("asset_types") or []),
                    ]
                )
            )

        items = [t for t in items if qn in _hay(t)]

    items.sort(key=lambda t: (t.get("code") or "").casefold())
    return items[: max(0, limit)]


def list_threats(
    q: str | None = None,
    *,
    category: str | None = None,
    apply_asset_type: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Alias to search_threats (same parameters)."""
    return search_threats(q, category=category, apply_asset_type=apply_asset_type, limit=limit)


def explain_threat(code: str) -> Optional[dict[str, Any]]:
    """Return the indexed detail dict for a threat code (or None)."""
    if not code:
        return None
    idx = _build_threat_index()
    return dict(idx.get(code)) if (idx and code in idx) else None


# --------------------------------------------------------------------------------------
# Public API – Impacts
# --------------------------------------------------------------------------------------

def _build_impact_index() -> dict[str, dict[str, Any]]:
    """
    Build a minimal index for impacts:
      code, name, description?, categories: [code], category: first or None
    Canonical 'code' = 'EN Name' (fallback 'Name') if 'code' is not present.
    """
    i_rows = _impacts.load()
    if not i_rows:
        return {}

    code_col = _first_present_column(i_rows, ("code", "impact_code", "EN Name", "Name"))
    name_col = _first_present_column(i_rows, ("name", "EN Name", "Name"))
    desc_col = _first_present_column(i_rows, ("description", "EN Notes", "Notes"))

    # Impact ↔ category mapping
    ixc = _impact_x_category.load()
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

    idx: dict[str, dict[str, Any]] = {}
    for r in i_rows:
        code = ((r.get(code_col) or "") if code_col else "").strip()
        if not code:
            continue
        name = ((r.get(name_col) or "") if name_col else "").strip() or code
        desc = ((r.get(desc_col) or "") if desc_col else "").strip() or None
        cats = list(impact_to_categories.get(code, []))
        idx[code] = {
            "code": code,
            "name": name,
            "description": desc,
            "categories": cats,
            "category": cats[0] if cats else None,
        }
    return idx


def list_impact_categories() -> list[str]:
    """Return canonical impact category codes."""
    rows = _impact_categories.load()
    if not rows:
        return []
    col = _first_present_column(
        rows, ("category_code", "code", "Impact category", "Category", "EN Name", "Name")
    ) or next(iter(rows[0].keys()), None)
    if not col:
        return []
    seen, out = set(), []
    for r in rows:
        v = (r.get(col) or "").strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def normalize_impact_code(code_or_alias: str) -> str | None:
    """Normalize a code or human label to canonical impact code."""
    if not code_or_alias:
        return None
    idx = _build_impact_index()
    if not idx:
        return None
    raw = code_or_alias.strip()
    if raw in idx:
        return raw
    low = raw.casefold()
    lowmap = {k.casefold(): k for k in idx.keys()}
    if low in lowmap:
        return lowmap[low]
    # slug by name
    target = _slug(raw)
    for row in idx.values():
        if _slug(row.get("name")) == target:
            return row["code"]
    return None


def search_impacts(q: str | None, *, category: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """
    Search impacts with tolerant filters (q over code/name/description/categories).
    """
    idx = _build_impact_index()
    if not idx:
        return []
    items = list(idx.values())

    if category:
        cat_norm = category.strip().casefold()
        items = [i for i in items if any(c.casefold() == cat_norm for c in i.get("categories", []))]

    if q:
        qn = _slug(q)
        def _hay(i: dict[str, Any]) -> str:
            return _slug(
                " ".join(
                    [
                        i.get("code") or "",
                        i.get("name") or "",
                        i.get("description") or "",
                        " ".join(i.get("categories") or []),
                    ]
                )
            )
        items = [i for i in items if qn in _hay(i)]

    items.sort(key=lambda i: (i.get("code") or "").casefold())
    return items[: max(0, limit)]


def list_impacts(q: str | None = None, *, category: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """Alias to search_impacts."""
    return search_impacts(q, category=category, limit=limit)


def explain_impact(code: str) -> dict[str, Any] | None:
    """Return indexed impact detail by canonical code."""
    if not code:
        return None
    idx = _build_impact_index()
    return dict(idx.get(code)) if (idx and code in idx) else None


# --------------------------------------------------------------------------------------
# Public API – Threat mappings (Asset type x Threat)
# --------------------------------------------------------------------------------------

def mapped_threats_for_asset_type(asset_type_code: str) -> list[str]:
    if not asset_type_code:
        return []
    rows = _at_x_threat.load()
    if not rows:
        return []

    asset_col = _first_present_column(rows, ("asset_type", "Asset type", "EN Name", "Name"))
    threat_col = _first_present_column(rows, ("threat_code", "Threat", "threat", "EN Name", "Name"))

    if not threat_col:
        return []

    needle = asset_type_code.strip()
    out: set[str] = set()
    for r in rows:
        if asset_col:
            a = (r.get(asset_col) or "").strip()
            if a != needle:
                continue
        raw = (r.get(threat_col) or "").strip()
        if not raw:
            continue
        code = normalize_threat_code(raw) or raw
        out.add(code)
    return sorted(out)


# --------------------------------------------------------------------------------------
# Generic search (for arbitrary catalogs; safe & reusable)
# --------------------------------------------------------------------------------------

def generic_search(file_name: str, q: str | None, *, fields: list[str] | None = None, limit: int = 50) -> list[dict[str, str]]:
    """
    Generic read-only search over an arbitrary catalog CSV in <catalogs_dir>.
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

    try:
        telemetry_log(
            "catalog.search",
            {"file": file_name, "q": q, "fields": use_cols, "returned": len(res), "dir": str(_CAT_DIR)},
            level="DEBUG",
        )
    except Exception:
        pass

    return res


__all__ = [
    # Asset types
    "list_asset_types",
    "search_asset_types",
    "explain_asset_type",
    # Threat categories
    "list_threat_categories",
    "normalize_threat_category",
    "explain_threat_category",
    # Threats
    "normalize_threat_code",
    "search_threats",
    "list_threats",
    "explain_threat",
    # Impacts
    "list_impact_categories",
    "normalize_impact_code",
    "search_impacts",
    "list_impacts",
    "explain_impact",
    # Mappings
    "mapped_threats_for_asset_type",
    # Generic
    "generic_search",
]
