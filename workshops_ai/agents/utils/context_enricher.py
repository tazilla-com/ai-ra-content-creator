# File: workshops_ai/agents/utils/context_enricher.py
# Purpose: Small, testable helpers that turn UIRequest context into
#          (a) concise prompt snippets and (b) seed asset candidates.
#
# Design:
# - No network calls, no I/O.
# - Deterministic heuristics; safe defaults.
# - Minimal dependencies; Pydantic optional (works with plain dicts too).

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional

# Optional typed imports (work with dicts if models change)
try:  # soft import – do not make this a hard dependency
    from workshops_ai.schemas.ui_contract import ServiceItem, WebsiteContent, ExistingAssetIn  # type: ignore
except Exception:  # pragma: no cover
    ServiceItem = Any  # type: ignore
    WebsiteContent = Any  # type: ignore
    ExistingAssetIn = Any  # type: ignore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_org_profile(profile: Optional[Mapping[str, Any]], *, max_pairs: int = 8, max_chars: int = 300) -> str:
    """
    Produce a short, flat, human-readable summary for LLM context.
    - Flattens nested dicts one level deep (k.sub=v).
    - Drops obviously sensitive keys by name.
    - Trims length.
    """
    if not profile or not isinstance(profile, Mapping):
        return ""

    SENSITIVE_KEYS = {"password", "secret", "token", "api_key", "apikey", "pwd", "ssn", "dob"}
    parts: List[str] = []

    def _flat_items(d: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, str]]:
        for k, v in d.items():
            if not k:
                continue
            key = f"{prefix}{k}".strip(".")
            if isinstance(v, Mapping):
                # only one level deep
                for kk, vv in v.items():
                    if not kk:
                        continue
                    yield (f"{key}.{kk}", _short_val(vv))
            else:
                yield (key, _short_val(v))

    for k, v in _flat_items(profile):
        lk = k.lower()
        if any(s in lk for s in SENSITIVE_KEYS):
            continue
        if not v:
            continue
        parts.append(f"{k}={v}")
        if len(parts) >= max_pairs:
            break

    summary = ", ".join(parts)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1] + "…"
    return summary


def extract_seed_candidates(
    services: Optional[Iterable[ServiceItem | Mapping[str, Any]]] = None,
    website_content: Optional[WebsiteContent | Mapping[str, Any]] = None,
    *,
    max_from_services: int = 8,
    max_from_website: int = 10,
) -> List[dict]:
    """
    Turn (services + website content) into candidate assets:
      [{"name","asset_type_code","rationale","source"}]
    - Uses light keyword mapping.
    - De-duplicates *within* the produced list (final dedupe against existing is separate).
    """
    candidates: List[dict] = []

    # 1) From services
    for svc in (services or []):
        rec = _service_to_candidate(svc)
        if rec:
            candidates.append(rec)
        if len(candidates) >= max_from_services:
            break

    # 2) From website pages content
    if website_content:
        try:
            pages = website_content.get("pages", []) if isinstance(website_content, Mapping) else website_content.pages  # type: ignore
        except Exception:
            pages = []

        # Collect top phrases per page
        seen_local: set[tuple[str, str]] = set()
        for page in pages or []:
            url = _get(page, "url")
            content = _get(page, "content") or ""
            if not content or len(content) < 10:
                continue
            for phrase, code in _extract_keywords(content):
                key = (phrase.lower(), code)
                if key in seen_local:
                    continue
                seen_local.add(key)
                candidates.append(
                    {
                        "name": phrase,
                        "asset_type_code": code,
                        "rationale": f"Found on website page: {url or 'n/a'}",
                        "source": "website",
                    }
                )
                if len(seen_local) >= max_from_website:
                    break

    # 3) Intra-list dedupe (name+type)
    return dedupe_candidates_in_memory(candidates)


def dedupe_against_existing(
    candidates: Iterable[Mapping[str, Any]],
    existing_assets: Optional[Iterable[ExistingAssetIn | Mapping[str, Any]]] = None,
) -> List[dict]:
    """
    Remove candidates that already exist (case-insensitive by normalized name + type).
    """
    if not candidates:
        return []

    existing_idx: set[tuple[str, str]] = set()
    for a in (existing_assets or []):
        nm = _norm_name(_get(a, "name"))
        tp = _norm_type(_get(a, "assetType") or _get(a, "asset_type") or _get(a, "asset_type_code"))
        if nm:
            existing_idx.add((nm, tp))

    out: List[dict] = []
    seen: set[tuple[str, str]] = set()
    for c in candidates:
        nm = _norm_name(_get(c, "name"))
        tp = _norm_type(_get(c, "asset_type_code"))
        key = (nm, tp)
        if not nm:
            continue
        if key in existing_idx:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(c))
    return out


def limit_results(items: Iterable[Any], n: Optional[int]) -> List[Any]:
    """
    Return at most n items (if n is None or invalid, no extra cap).
    """
    try:
        cap = int(n) if n is not None else None
    except Exception:
        cap = None
    if cap is None or cap <= 0:
        return list(items)
    out: List[Any] = []
    for it in items:
        out.append(it)
        if len(out) >= cap:
            break
    return out


# ---------------------------------------------------------------------------
# Heuristics & helpers
# ---------------------------------------------------------------------------

# Very small mapping (extend as needed) – canonical EN asset_type_code values
_SERVICE_KEYWORD_TO_TYPE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(payment|payments|gateway|checkout)\b", re.I), "Information system"),
    (re.compile(r"\b(cloud|hosting|infrastructure)\b", re.I), "IT service"),
    (re.compile(r"\b(monitoring|observability|metrics|logging)\b", re.I), "Monitoring system"),
    (re.compile(r"\b(telemetry|tracing)\b", re.I), "Monitoring system"),
    (re.compile(r"\b(api|platform|portal)\b", re.I), "Information system"),
    (re.compile(r"\b(mobile|android|ios|app)\b", re.I), "Mobile application"),
    (re.compile(r"\b(website|web\s*app|frontend)\b", re.I), "Web application"),
    (re.compile(r"\b(database|db|postgres|mysql|mariadb|mssql)\b", re.I), "Database"),
    (re.compile(r"\b(data\s*warehouse|dwh|bigquery|snowflake)\b", re.I), "Data warehouse"),
    (re.compile(r"\b(ci/cd|pipeline|build|deploy|gitlab|github actions)\b", re.I), "DevOps toolchain"),
]

_WEBSITE_KEYWORD_TO_TYPE: list[tuple[re.Pattern, str, str]] = [
    # (pattern, type_code, label extractor group or fixed)
    (re.compile(r"\b(customer|client)\s+portal\b", re.I), "Information system", "Customer Portal"),
    (re.compile(r"\badmin(istrative)?\s+(portal|console|panel)\b", re.I), "Information system", "Admin Console"),
    (re.compile(r"\bpublic\s+api\b|\brest\s+api\b|\bgraphql\b", re.I), "Information system", "Public API"),
    (re.compile(r"\bmobile\s+app(lication)?\b", re.I), "Mobile application", "Mobile App"),
    (re.compile(r"\bweb\s+app(lication)?\b", re.I), "Web application", "Web App"),
    (re.compile(r"\bdata\b\s*(platform|hub|warehouse)\b", re.I), "Data warehouse", "Data Platform"),
    (re.compile(r"\bmonitoring\b|\bobservability\b", re.I), "Monitoring system", "Monitoring"),
    (re.compile(r"\b(single\s+sign[- ]on|sso)\b", re.I), "Identity provider", "SSO / IdP"),
]

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/_. ]{2,}")

def _service_to_candidate(svc: ServiceItem | Mapping[str, Any]) -> Optional[dict]:
    name = _get(svc, "name") or ""
    desc = _get(svc, "description") or ""
    sector = _get(svc, "sector") or ""

    text = " ".join([name, desc, sector]).strip()
    if not text:
        return None

    for pat, code in _SERVICE_KEYWORD_TO_TYPE:
        if pat.search(text):
            label = _pick_label_from_service(name, desc) or name or "Service asset"
            return {
                "name": label,
                "asset_type_code": code,
                "rationale": f"Derived from service keywords: {pat.pattern}",
                "source": "service",
            }

    # Fallback: assume Information system, keep service name as candidate
    if name:
        return {
            "name": _titlecase(name),
            "asset_type_code": "Information system",
            "rationale": "Default mapping from service name",
            "source": "service",
        }
    return None


def _extract_keywords(text: str) -> List[tuple[str, str]]:
    """
    Extract (label, asset_type_code) pairs from website text using keyword rules.
    """
    out: List[tuple[str, str]] = []
    for pat, code, label in _WEBSITE_KEYWORD_TO_TYPE:
        if pat.search(text):
            out.append((label, code))

    # Additionally, detect generic nouns that often denote assets (API, Portal, Gateway)
    extra_patterns = [
        (re.compile(r"\bapi\b", re.I), "Information system", "API"),
        (re.compile(r"\bgateway\b", re.I), "Information system", "Gateway"),
        (re.compile(r"\bportal\b", re.I), "Information system", "Portal"),
    ]
    for pat, code, label in extra_patterns:
        if pat.search(text):
            out.append((label, code))

    # De-duplicate preserving order
    seen: set[tuple[str, str]] = set()
    uniq: List[tuple[str, str]] = []
    for pair in out:
        if pair in seen:
            continue
        seen.add(pair)
        uniq.append(pair)
    return uniq


def dedupe_candidates_in_memory(candidates: Iterable[Mapping[str, Any]]) -> List[dict]:
    """
    De-duplicate candidates among themselves by (normalized name, type).
    """
    out: List[dict] = []
    seen: set[tuple[str, str]] = set()
    for c in candidates or []:
        nm = _norm_name(_get(c, "name"))
        tp = _norm_type(_get(c, "asset_type_code"))
        if not nm:
            continue
        key = (nm, tp)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(c))
    return out


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str, default: Any = None) -> Any:
    try:
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return getattr(obj, key, default)
    except Exception:
        return default


def _short_val(v: Any, *, max_len: int = 60) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _norm_name(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_type(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    # normalize a few common aliases
    aliases = {
        "webapp": "web application",
        "mobile app": "mobile application",
        "db": "database",
        "information system": "information system",
    }
    return aliases.get(s, s)


def _titlecase(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    # simple title-case without changing acronyms too much
    return re.sub(r"\b([a-z])", lambda m: m.group(1).upper(), s)


def _pick_label_from_service(name: str, desc: str) -> str:
    """
    Prefer a crisp label extracted from service name/desc (e.g., 'Payment API', 'Customer Portal').
    """
    text = " ".join([name or "", desc or ""]).strip()
    # prefer tokens that look like proper asset labels
    for pat in (r"\b[\w\- ]+ (API|Portal|Gateway|Console)\b", r"\bMobile App\b", r"\bWeb App\b"):
        m = re.search(pat, text, re.I)
        if m:
            return _titlecase(m.group(0))
    return _titlecase(name or desc or "Service Asset")


__all__ = [
    "summarize_org_profile",
    "extract_seed_candidates",
    "dedupe_against_existing",
    "limit_results",
    "dedupe_candidates_in_memory",
]
