# File: workshops_ai/backend/csv_backend.py
# CSV-based backend implementing BackendPort (read-only for assets/impacts/risks).
# - Reads state from <settings.paths.state_dir>:
#       assets.csv (required)
#       asset_impacts.csv (optional)   -> [{asset_id, impact_code, level(0..4), note?}]
#       asset_risks.csv   (optional)   -> [{asset_id, threat_code, probability(0..4), note?}]
# - Reads catalogs from <settings.paths.catalogs_dir> with robust fallbacks
# - Defensive CSV parsing (BOM, ; delimiter, CP1250/CP1252/ISO-8859-2), mtime-based hot reload
# - Pydantic v2 validation of Asset; STATE is the source of truth (fail-open for assetType)
#
# Catalog files (relative to resolved catalogs dir) – read-only helpers:
#   Asset type.csv
#   Threat.csv
#   Threat category.csv
#   Threat x Threat category.csv
#   Asset type x Threat.csv
#   Impact.csv
#   Impact category.csv
#   Impact x Impact category.csv
#
# IMPORTANT DOMAIN POLICY:
#   - STATE is tenant truth. We DO NOT drop assets just because their assetType
#     is not present in catalogs. We only use catalogs for optional filtering
#     (when explicitly requested) and for explain/list operations.

from __future__ import annotations

import csv
import uuid
from uuid import UUID
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Optional

from pydantic import ValidationError

from workshops_ai.backend.port import (
    BackendConflict,
    BackendError,
    BackendPort,
    BackendValidationError,
    BackendNotFound, 
)
from workshops_ai.settings import get_settings
from workshops_ai.api.schema import Asset  # canonical DTO

# Optional telemetry (best-effort, never fail)
try:  # pragma: no cover
    from workshops_ai.tools.telemetry import telemetry_log  # type: ignore
except Exception:  # pragma: no cover
    def telemetry_log(*args, **kwargs):  # type: ignore
        return None

__all__ = ["CsvBackend"]

# --------------------------------------------------------------------------------------
# Paths & config (harmonized with tools/search_catalogs.py)
# --------------------------------------------------------------------------------------

_s = get_settings()


def _resolve_catalog_dir() -> Path:
    """
    Choose the first existing catalogs directory from:
      1) settings.paths.catalogs_dir
      2) package fallback: workshops_ai/data/catalogs
      3) runtime mount fallback: /app/data/catalogs
    """
    candidates: list[Path] = []
    try:
        candidates.append(Path(_s.paths.catalogs_dir).resolve())
    except Exception:
        pass
    pkg_fallback = Path(__file__).resolve().parent.parent / "data" / "catalogs"
    candidates.append(pkg_fallback)
    candidates.append(Path("/app/data/catalogs"))
    for p in candidates:
        try:
            if p.exists() and p.is_dir():
                telemetry_log("catalog.dir_selected", {"dir": str(p)}, level="DEBUG")
                return p
        except Exception:
            continue
    fallback = candidates[0] if candidates else Path.cwd()
    telemetry_log("catalog.dir_fallback", {"dir": str(fallback)}, level="WARN")
    return fallback


def _resolve_state_dir() -> Path:
    """
    Choose the first existing state directory from:
      1) settings.paths.state_dir
      2) package fallback: workshops_ai/data/state
      3) runtime mount fallback: /app/data/state
    """
    candidates: list[Path] = []
    try:
        candidates.append(Path(_s.paths.state_dir).resolve())
    except Exception:
        pass
    pkg_fallback = Path(__file__).resolve().parent.parent / "data" / "state"
    candidates.append(pkg_fallback)
    candidates.append(Path("/app/data/state"))
    for p in candidates:
        try:
            if p.exists() and p.is_dir():
                telemetry_log("state.dir_selected", {"dir": str(p)}, level="DEBUG")
                return p
        except Exception:
            continue
    fallback = candidates[0] if candidates else Path.cwd()
    telemetry_log("state.dir_fallback", {"dir": str(fallback)}, level="WARN")
    return fallback


_STATE_DIR = _resolve_state_dir()
_CAT_DIR = _resolve_catalog_dir()

_ASSETS_STATE_FILE = _STATE_DIR / "assets.csv"
_ASSET_IMPACTS_FILE = _STATE_DIR / "asset_impacts.csv"   # optional
_ASSET_RISKS_FILE = _STATE_DIR / "asset_risks.csv"       # optional

# Catalogs (read-only helpers)
_CAT_ASSET_TYPE_FILE = _CAT_DIR / "Asset type.csv"
_CAT_AT_X_THREAT_FILE = _CAT_DIR / "Asset type x Threat.csv"
_CAT_THREAT_FILE = _CAT_DIR / "Threat.csv"
_CAT_THREAT_CATEGORY_FILE = _CAT_DIR / "Threat category.csv"
_CAT_THREAT_X_CATEGORY_FILE = _CAT_DIR / "Threat x Threat category.csv"
_CAT_IMPACT_FILE = _CAT_DIR / "Impact.csv"
_CAT_IMPACT_CATEGORY_FILE = _CAT_DIR / "Impact category.csv"
_CAT_IMPACT_X_CATEGORY_FILE = _CAT_DIR / "Impact x Impact category.csv"

# --------------------------------------------------------------------------------------
# UUID helpers: map any string asset id -> deterministic UUIDv5 (stable across runs)
# --------------------------------------------------------------------------------------

def _uuid_namespace_from_settings() -> UUID:
    """
    Use settings.rag.uuid5_namespace if valid, otherwise a fixed, private fallback.
    The fallback is constant so IDs remain deterministic across environments.
    """
    try:
        ns = str(get_settings().rag.uuid5_namespace or "").strip()
        if ns:
            return UUID(ns)
    except Exception:
        pass
    return UUID("00000000-0000-5000-a000-000000000000")


_NS_UUID5 = _uuid_namespace_from_settings()


def _uuid5_for_asset_id(raw_id: str) -> UUID:
    """
    Deterministically map any arbitrary asset identifier to UUIDv5.
    If `raw_id` is already a UUID string, return that UUID.
    """
    if not raw_id:
        return uuid.uuid5(_NS_UUID5, "asset:")
    try:
        return UUID(raw_id)
    except Exception:
        return uuid.uuid5(_NS_UUID5, f"asset:{raw_id}")


def _id_matches(row_id: str, query_id: str) -> bool:
    """
    True iff:
      - exact string match, OR
      - UUIDv5(row_id) equals query_id (parsed as UUID), OR
      - query_id parses as UUID and equals row_id parsed as UUID (if row_id itself is UUID text)
    """
    if not row_id or not query_id:
        return False
    if row_id == query_id:
        return True
    try:
        q_uuid = UUID(query_id)
    except Exception:
        q_uuid = None
    try:
        r_uuid = UUID(row_id)
    except Exception:
        r_uuid = None
    if r_uuid and q_uuid and r_uuid == q_uuid:
        return True
    if q_uuid and _uuid5_for_asset_id(row_id) == q_uuid:
        return True
    return False

# --------------------------------------------------------------------------------------
# CSV helpers with mtime-based reload (encoding & delimiter tolerant)
# --------------------------------------------------------------------------------------

@dataclass
class _CsvCache:
    path: Path
    mtime: float = 0.0
    rows: list[dict[str, str]] | None = None

    def load(self) -> list[dict[str, str]]:
        """
        Load or serve cached rows.
        - For non-state files: returns [] if missing or on read error.
        - For assets.csv specifically: missing file is a hard error to surface misconfiguration early.
        """
        if not self.path.exists():
            if self.path == _ASSETS_STATE_FILE:
                raise BackendError(f"State file not found: {self.path}")
            self.mtime, self.rows = 0.0, []
            return self.rows

        mt = self.path.stat().st_mtime
        if self.rows is not None and self.mtime == mt:
            return self.rows

        try:
            self.rows = _read_csv(self.path)
        except Exception as e:
            telemetry_log("csv.read_error", {"file": str(self.path), "error": str(e)}, level="WARN")
            self.rows = []
        self.mtime = mt
        return self.rows


_DELIMS = [",", ";", "\t", "|"]
_ENCODINGS = ["utf-8-sig", "utf-8", "cp1250", "cp1252", "iso-8859-2"]


def _detect_encoding_and_delimiter(path: Path) -> tuple[str, str]:
    """
    Detect encoding among utf-8-sig → utf-8 → cp1250 → cp1252 → iso-8859-2
    and delimiter among , ; \t | via csv.Sniffer (fallback to frequency tally).
    """
    sample_bytes = b""
    try:
        with path.open("rb") as fb:
            sample_bytes = fb.read(4096)
    except FileNotFoundError:
        return "utf-8-sig", ","

    for enc in _ENCODINGS:
        try:
            sample_text = sample_bytes.decode(enc, errors="strict")
        except UnicodeDecodeError:
            continue

        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample_text, delimiters="".join(_DELIMS))
            delim = dialect.delimiter or ","
        except Exception:
            counts = {d: sample_text.count(d) for d in _DELIMS}
            delim = max(counts, key=counts.get) if counts else ","
            if delim == "," and counts.get(";", 0) >= counts.get(",", 0):
                delim = ";"
        return enc, delim

    return "utf-8-sig", ","


def _read_csv(path: Path) -> list[dict[str, str]]:
    detected_enc, delim = _detect_encoding_and_delimiter(path)
    order = [detected_enc] + [e for e in _ENCODINGS if e != detected_enc]
    last_err: Exception | None = None

    for enc in order:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f, delimiter=delim)
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

    telemetry_log("csv.read_error", {"file": str(path), "error": str(last_err)}, level="WARN")
    return []


def _first_present_column(rows: list[dict[str, str]], candidates: Iterable[str]) -> Optional[str]:
    """Case-insensitive header match (aligned with tools/search_catalogs.py)."""
    if not rows:
        return None
    cols_lc = {(c or "").strip().casefold(): c for c in rows[0].keys()}
    for c in candidates:
        key = (c or "").strip().casefold()
        if key in cols_lc:
            return cols_lc[key]
    return None

# --------------------------------------------------------------------------------------
# Catalog helpers (asset types, threats, impacts) 
# --------------------------------------------------------------------------------------

# --- CSV write helpers (atomic replace) --------------------------------------

def _atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # coerce keys; ensure all headers present
            row = {k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames}
            w.writerow(row)
    tmp.replace(path)


def _load_state_rows(path: Path) -> list[dict[str, Any]]:
    """Read CSV or return [] if missing; tolerant just like _read_csv."""
    try:
        return _read_csv(path)
    except Exception:
        return []


_ASSETS_HEADERS = [
    "id", "name", "description", "isPrimary",
    "confidentiality", "integrity", "availability",
    "assetType", "org_id",
]

_ASSET_IMPACTS_HEADERS = ["asset_id", "impact_code", "level", "note"]
_ASSET_RISKS_HEADERS   = ["asset_id", "threat_code", "probability", "note"]


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

    # -------- Asset types (helper) --------

    def list_asset_types(self) -> list[str]:
        rows = self.asset_types.load()
        if not rows:
            return []
        col = _first_present_column(
            rows,
            ("code", "asset_type", "type", "Asset type", "EN Name", "EN name", "Name", "name"),
        ) or next(iter(rows[0].keys()))
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
        code_col = _first_present_column(t_rows, ("code", "threat_code", "EN Name", "Name", "Code", "Threat", "ID"))
        name_col = _first_present_column(t_rows, ("name", "EN Name", "Name", "Threat name", "Threat"))
        desc_col = _first_present_column(t_rows, ("description", "EN Notes", "Notes", "Description", "EN Description"))
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

    # -------- Impacts (catalog helper) --------

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
# State (assets.csv / asset_impacts.csv / asset_risks.csv)
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
    - If 'id' is not a valid UUID, set it to None (graphs/DTO pak nepadnou).
    - Map numeric CIA → canonical enums.
    Always returns `Asset`.
    """
    import uuid

    def _uuid_or_none(v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        try:
            return str(uuid.UUID(s))
        except Exception:
            return None

    # 0) normalize ID up-front
    base = dict(payload)
    base["id"] = _uuid_or_none(base.get("id"))

    # 1) optimistic 
    try:
        return Asset(**base)  # type: ignore[arg-type]
    except Exception:
        pass

    # 2) map integers -> canonical enums
    try:
        alt = dict(base)
        for k in ("confidentiality", "integrity", "availability"):
            v = alt.get(k)
            if k == "confidentiality":
                alt[k] = _map_conf_int_to_canonical(v) if isinstance(v, int) else v
            else:
                alt[k] = _map_ia_int_to_canonical(v) if isinstance(v, int) else v
        return Asset(**alt)  # type: ignore[arg-type]
    except Exception:
        pass

    # 3) last resort: force CIA to strings the DTO 
    try:
        alt = dict(base)
        def _digits(i: int | None) -> str | None: return None if i is None else str(i)
        for k in ("confidentiality", "integrity", "availability"):
            v = alt.get(k)
            if isinstance(v, int):
                alt[k] = _digits(v)
        return Asset(**alt)  # type: ignore[arg-type]
    except Exception as e:
        raise BackendValidationError(f"Invalid Asset row (CIA/ID mapping failed): {e}")



class _AssetsState:
    """
    Lazy, cached access to state CSVs with validation of basic shapes.
    POLICY: state is source of truth (fail-open for assetType).
    """

    def __init__(self, catalogs: _Catalogs) -> None:
        self.catalogs = catalogs
        self.assets = _CsvCache(_ASSETS_STATE_FILE)
        self.asset_impacts = _CsvCache(_ASSET_IMPACTS_FILE)
        self.asset_risks = _CsvCache(_ASSET_RISKS_FILE)

    # ---- Assets --------------------------------------------------------------

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

        out: list[Asset] = []
        for r in rows:
            try:
                a = _asset_from_row(r, catalogs=self.catalogs)
            except BackendValidationError:
                continue

            # org filter
            if org_id and (getattr(a, "org_id", None) or "") != org_id:
                continue

            # explicit type filter (exact string match on the asset's value)
            if type_code and (a.assetType or "").strip() != type_code.strip():
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
            if _id_matches(rid, asset_id):
                try:
                    return _asset_from_row(r, catalogs=self.catalogs)
                except BackendValidationError:
                    return None
        return None

    # ---- Impacts (state) -----------------------------------------------------

    def get_impacts(self, asset_id: str) -> list[Mapping[str, object]]:
        """
        Return [{impact_code, level, note?}] from optional state/asset_impacts.csv.
        Heuristics for columns: asset_id|Asset ID, impact_code|Impact, level|Level, note|Note.
        No catalog validation here (state is truth); only shape/level 0..4 is enforced.
        """
        rows = self.asset_impacts.load()
        if not rows:
            return []

        aid_col = _first_present_column(rows, ("asset_id", "Asset ID", "assetId", "ID"))
        code_col = _first_present_column(rows, ("impact_code", "Impact", "impact", "EN Name", "Name", "code"))
        lvl_col = _first_present_column(rows, ("level", "Level", "impact_level", "impactLevel"))
        note_col = _first_present_column(rows, ("note", "Note", "notes", "Notes"))

        if not aid_col or not code_col or not lvl_col:
            return []

        out: list[dict[str, object]] = []
        for r in rows:
            rid = (r.get(aid_col) or "").strip()
            if not _id_matches(rid, asset_id):
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

    # ---- Risks (state) -------------------------------------------------------

    def get_risks(self, asset_id: str) -> list[Mapping[str, object]]:
        """
        Return [{threat_code, probability, note?}] from optional state/asset_risks.csv.
        Heuristics for columns: asset_id|Asset ID, threat_code|Threat, probability|Probability, note|Note.
        No catalog validation (state is truth); only probability 0..4 is enforced.
        """
        rows = self.asset_risks.load()
        if not rows:
            return []

        aid_col = _first_present_column(rows, ("asset_id", "Asset ID", "assetId", "ID"))
        code_col = _first_present_column(rows, ("threat_code", "Threat", "threat", "EN Name", "Name", "code"))
        prob_col = _first_present_column(rows, ("probability", "Probability", "level", "Level"))
        note_col = _first_present_column(rows, ("note", "Note", "notes", "Notes"))

        if not aid_col or not code_col or not prob_col:
            return []

        out: list[dict[str, object]] = []
        for r in rows:
            rid = (r.get(aid_col) or "").strip()
            if not _id_matches(rid, asset_id):
                continue
            code = (r.get(code_col) or "").strip()
            prob = _parse_int_0_4(r.get(prob_col))
            if not code or prob is None:
                continue
            item: dict[str, object] = {"threat_code": code, "probability": prob}
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

    # FAIL-OPEN POLICY: do NOT drop assets when assetType is unknown in catalogs.
    valid_types = set(catalogs.list_asset_types())
    if valid_types and asset_type not in valid_types:
        telemetry_log(
            "asset.unknown_asset_type",
            {"id": id_, "assetType": asset_type, "known_types_count": len(valid_types)},
            level="WARN",
        )

    # CIA rule for non-primary
    if is_primary is False:
        c = i = a = None

    payload: dict[str, Any] = {
        "id": id_,  # normalized to UUID inside _build_asset_with_cia_fallback
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
    except ValidationError as ve:
        raise BackendValidationError(f"Invalid Asset row: {ve}") from ve

# --------------------------------------------------------------------------------------
# CsvBackend implementation
# --------------------------------------------------------------------------------------

class CsvBackend(BackendPort):
    """
    CSV-backed implementation of BackendPort.

    - Catalogs: READ-ONLY (dropdown sources, no writes).
    - State: READ + minimal WRITE (assets.csv, asset_impacts.csv, asset_risks.csv).
    - Staging: not supported (raises BackendConflict).
    - Pure code lists are available from the objects via extracting 'code'.
      Richer definition objects are available via list_impact_defs / list_threat_defs.
    """

    def __init__(self) -> None:
        self._catalogs = _Catalogs()
        self._assets = _AssetsState(self._catalogs)

    # ---- Catalog helpers ----

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
    def list_controls(self, category: str | None = None, include_coverage: bool = True) -> list[dict[str, Any]]:
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
        if not asset_id:
            raise BackendValidationError("asset_id is required")
        # optional state file; returns [] if missing/empty
        return self._assets.get_risks(asset_id)

    def asset_get_controls(self, asset_id: str) -> list[Mapping[str, object]]:
        # Not tracked in CSV MVP
        return []

     # ---- Minimal WRITE API over state/*.csv ---------------------------------

    def create_asset(self, name: str, type: str | None = None, *, org_id: str | None = None,
                     description: str | None = None, is_primary: bool | None = None) -> dict:
        """
        Append a new asset into data/state/assets.csv.
        - Generates a UUIDv4 for 'id'
        - CIA left empty (None) unless you later update
        - assetType validated only softly (catalogs are advisory; do not block)
        """
        if not name:
            raise BackendValidationError("Asset name is required")

        new_id = str(uuid.uuid4())
        rows = _load_state_rows(_ASSETS_STATE_FILE)

        # soft validation: warn when type not in catalogs
        if type:
            types = set(self._catalogs.list_asset_types())
            if types and type not in types:
                telemetry_log("asset.create.unknown_type", {"assetType": type}, level="WARN")

        def _bool_to_str(v: bool | None) -> str:
            return "true" if v is True else ("false" if v is False else "")

        row = {
            "id": new_id,
            "name": name,
            "description": description or "",
            "isPrimary": _bool_to_str(is_primary),
            "confidentiality": "",
            "integrity": "",
            "availability": "",
            "assetType": (type or "").strip(),
            "org_id": (org_id or "").strip(),
        }
        rows.append(row)
        _atomic_write_csv(_ASSETS_STATE_FILE, rows, _ASSETS_HEADERS)

        # hot-reload cache (very light; next read will pick it up anyway)
        try:
            self._assets.assets.rows = None
        except Exception:
            pass

        # Return a minimal DTO-like dict (not pydantic instance on purpose)
        return dict(row)

    def update_asset(self, asset_id: str, **fields) -> dict:
        """
        Update an existing asset row by 'id'. Accepts any of:
          name, description, isPrimary, confidentiality, integrity, availability, assetType, org_id
        Values are written as-is (CIA expected as 0..4 or canonical enums depending on your importer).
        """
        if not asset_id:
            raise BackendValidationError("asset_id is required")

        rows = _load_state_rows(_ASSETS_STATE_FILE)
        if not rows:
            raise BackendNotFound(f"Asset not found: {asset_id}")

        found = False
        for r in rows:
            rid = (r.get("id") or r.get("ID") or "").strip()
            if _id_matches(rid, asset_id):
                found = True
                # Normalize booleans to "true"/"false"
                if "isPrimary" in fields and isinstance(fields["isPrimary"], bool):
                    fields["isPrimary"] = "true" if fields["isPrimary"] else "false"
                for k in ("name", "description", "isPrimary", "confidentiality",
                          "integrity", "availability", "assetType", "org_id"):
                    if k in fields and fields[k] is not None:
                        r[k] = str(fields[k])
                break

        if not found:
            raise BackendNotFound(f"Asset not found: {asset_id}")

        _atomic_write_csv(_ASSETS_STATE_FILE, rows, _ASSETS_HEADERS)
        try:
            self._assets.assets.rows = None
        except Exception:
            pass

        # return the updated row
        for r in rows:
            if _id_matches(r.get("id", ""), asset_id):
                return dict(r)
        return {}

    def list_impact_defs(self) -> list[dict[str, Any]]:
        """Read-only impact definitions from catalogs (dropdown source)."""
        return self._catalogs.list_impacts(q=None, category=None)

    def list_threat_defs(self, apply_asset_type: str | None = None) -> list[dict[str, Any]]:
        """Read-only threat definitions from catalogs (dropdown source)."""
        return self._catalogs.list_threats(q=None, category=None, apply_asset_type=apply_asset_type)

    def create_impact_def(self, code: str, *, name: str | None = None,
                          category: str | None = None, description: str | None = None) -> dict:
        """
        Catalogs are read-only in CSV backend. We intentionally do not write to catalogs/.
        The graph will still submit a JSON ChangeSet to outbox for backend to ingest.
        """
        raise BackendConflict("CSV catalogs are read-only; impact definitions cannot be created here.")

    def create_threat_def(self, code: str, *, name: str | None = None,
                          category: str | None = None, description: str | None = None) -> dict:
        """Same policy as create_impact_def: no write to catalogs in CSV backend."""
        raise BackendConflict("CSV catalogs are read-only; threat definitions cannot be created here.")

    def set_asset_impact_level(self, asset_id: str, code: str, level: int, note: str | None = None) -> dict:
        """
        Upsert a single (asset_id, impact_code) record in data/state/asset_impacts.csv.
        Validates that `code` exists in catalogs; otherwise raises BackendNotFound.
        """
        if not asset_id or not code:
            raise BackendValidationError("asset_id and impact code are required")
        if level is None or not (0 <= int(level) <= 4):
            raise BackendValidationError("level must be an integer 0..4")

        # 1) asset must exist
        if self._assets.get(asset_id) is None:
            raise BackendNotFound(f"Asset not found: {asset_id}")

        # 2) definition must exist
        defs = self._catalogs.list_impacts(q=None, category=None)
        known = {d.get("code") for d in defs}
        if code not in known:
            raise BackendNotFound(f"Impact code not found in catalogs: {code}")

        rows = _load_state_rows(_ASSET_IMPACTS_FILE)
        updated = False
        for r in rows:
            if _id_matches(r.get("asset_id", ""), asset_id) and (r.get("impact_code") or "").strip() == code:
                r["level"] = str(int(level))
                r["note"] = (note or r.get("note") or "") if note is not None else (r.get("note") or "")
                updated = True
                break
        if not updated:
            rows.append({
                "asset_id": asset_id,
                "impact_code": code,
                "level": str(int(level)),
                "note": note or "",
            })
        _atomic_write_csv(_ASSET_IMPACTS_FILE, rows, _ASSET_IMPACTS_HEADERS)
        try:
            self._assets.asset_impacts.rows = None
        except Exception:
            pass
        return None

    def set_asset_threat_level(self, asset_id: str, code: str, level: int, note: str | None = None) -> dict:
        """
        Upsert a single (asset_id, threat_code) record in data/state/asset_risks.csv.
        Validates that `code` exists in catalogs; otherwise raises BackendNotFound.
        Uses 'probability' column name for level.
        """
        if not asset_id or not code:
            raise BackendValidationError("asset_id and threat code are required")
        if level is None or not (0 <= int(level) <= 4):
            raise BackendValidationError("level must be an integer 0..4")

        # 1) asset must exist
        if self._assets.get(asset_id) is None:
            raise BackendNotFound(f"Asset not found: {asset_id}")

        # 2) definition must exist
        defs = self._catalogs.list_threats(q=None, category=None, apply_asset_type=None)
        known = {d.get("code") for d in defs}
        if code not in known:
            raise BackendNotFound(f"Threat code not found in catalogs: {code}")

        rows = _load_state_rows(_ASSET_RISKS_FILE)
        updated = False
        for r in rows:
            if _id_matches(r.get("asset_id", ""), asset_id) and (r.get("threat_code") or "").strip() == code:
                r["probability"] = str(int(level))
                r["note"] = (note or r.get("note") or "") if note is not None else (r.get("note") or "")
                updated = True
                break
        if not updated:
            rows.append({
                "asset_id": asset_id,
                "threat_code": code,
                "probability": str(int(level)),
                "note": note or "",
            })
        _atomic_write_csv(_ASSET_RISKS_FILE, rows, _ASSET_RISKS_HEADERS)
        try:
            self._assets.asset_risks.rows = None
        except Exception:
            pass
        return None
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
