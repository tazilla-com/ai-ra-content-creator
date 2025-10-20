# File: workshops_ai/utils/normalize.py
# -----------------------------------------------------------------------------
# Central, locale-aware normalizers used by the chat graph and pydantic models.
#
# What’s included:
#   - `canon(text)` – lowercasing + diacritics removal + whitespace collapse
#   - `normalize_is_primary(value)` – robust CZ/EN boolean parsing (backward compat)
#   - `normalize_level(text)` – map LOW/MEDIUM/HIGH (cz/en/abbrev) -> 'LOW'|'MEDIUM'|'HIGH'
#   - `normalize_confidentiality(text)` – map cz/en -> 'PUBLIC'|'INTERNAL'|'PROTECTED'|'STRICTLY_PROTECTED'
#   - `normalize_slot_value(slot, text)` – dispatcher for the common slots
#
# Design:
#   - stdlib-only (no extra deps), fast, side-effect free
#   - tolerant to diacritics, case, and common synonyms/abbreviations
#   - returns canonical UPPERCASE enums for CIA, and bool for isPrimary
#
# Backward compatibility:
#   - Keeps the original public function name `normalize_is_primary(...)`
#     but improves handling (diacritics, synonyms, ints).
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Iterable, Mapping, Optional
import re
import unicodedata

__all__ = [
    "canon",
    "normalize_is_primary",
    "normalize_level",
    "normalize_confidentiality",
    "normalize_slot_value",
]

# -----------------------------------------------------------------------------
# Canonicalization utilities
# -----------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def _nfkd(s: str) -> str:
    """Strip diacritics using NFKD decomposition."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def canon(s: str | None) -> str:
    """
    Canonicalize free text:
      - str() -> strip -> lower
      - replace punctuation with spaces
      - NFKD (remove diacritics)
      - collapse whitespace
    """
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)  # punctuation -> space
    s = _nfkd(s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


# -----------------------------------------------------------------------------
# Normalization dictionaries
# -----------------------------------------------------------------------------

# LOW/MEDIUM/HIGH triad (integrity/availability)
_LEVEL_MAP: Mapping[str, Iterable[str]] = {
    "LOW": {
        "low", "l",
        "nizka", "nizke", "nizky", "nizkou", "niz",
        "mala", "male", "maly", "malo",
        "spodni", "minimalni", "min",
    },
    "MEDIUM": {
        "medium", "m",
        "stredni", "stred", "mezistupen", "prumer", "prumerne", "prumerna",
        "mid",
    },
    "HIGH": {
        "high", "h",
        "vysoka", "vysoke", "vysoky", "vysokou", "vys",
        "velka", "velke", "velky", "max",
    },
}

# PUBLIC/INTERNAL/PROTECTED/STRICTLY_PROTECTED (confidentiality)
_CONF_MAP: Mapping[str, Iterable[str]] = {
    "PUBLIC": {
        "public", "publicni", "verejna", "verejne", "verejny", "otevrena", "open",
    },
    "INTERNAL": {
        "internal", "interni", "vnitrni", "pro vnitrni pouziti", "inside",
        "internal use",
    },
    "PROTECTED": {
        "protected", "chranena", "chranene", "chraneny", "citliva", "citlive",
        "sensitive",
    },
    "STRICTLY_PROTECTED": {
        "strictly protected", "prisne chranena", "prisne chranene", "prisne chraneny",
        "prisne_chranena", "vysoce citliva", "tajna",
    },
}

# truthy / falsy sets for isPrimary — CZ/EN + shorthand
_TRUTHY: set[str] = {
    "true", "t", "1", "yes", "y", "ok", "okay", "confirm",
    "ano", "a", "potvrzuji", "potvrdit",
    "primary", "primarni", "hlavni", "business critical", "core",
    "je primarni", "ano je", "ano je primarni",
}
_FALSY: set[str] = {
    "false", "f", "0", "no", "n",
    "ne", "neni", "není",
    "secondary", "podporne", "podporna", "podporne aktivum", "supporting",
    "ne neni", "ne, neni", "ne, není",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize_from_map(text: str, variants: Mapping[str, Iterable[str]]) -> Optional[str]:
    """Generic helper to normalize a string via a {CANONICAL: {variants...}} map."""
    t = canon(text)
    if not t:
        return None
    # Allow already-canonical tokens
    if t in {"low", "medium", "high"}:
        return t.upper()
    if t in {"public", "internal", "protected", "strictly protected"}:
        return "STRICTLY_PROTECTED" if t == "strictly protected" else t.upper()
    for canonical, alts in variants.items():
        if t in alts:
            return canonical
    return None


# -----------------------------------------------------------------------------
# Public normalizers
# -----------------------------------------------------------------------------

def normalize_is_primary(value) -> Optional[bool]:
    """
    Backward-compatible normalization for 'isPrimary' answers.

    Accepts:
      - bool → returned as-is
      - ints 0/1 → False/True
      - text (cz/en/shorthand) → True/False
      - otherwise → None
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int,)):
        if value == 1:
            return True
        if value == 0:
            return False
    s = canon(value)
    if not s:
        return None
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    return None


def normalize_level(text: str) -> Optional[str]:
    """
    Normalize LOW/MEDIUM/HIGH answers (CZ/EN).
    Returns: 'LOW' | 'MEDIUM' | 'HIGH' | None
    """
    return _normalize_from_map(text, _LEVEL_MAP)


def normalize_confidentiality(text: str) -> Optional[str]:
    """
    Normalize confidentiality answers (CZ/EN).
    Returns: 'PUBLIC' | 'INTERNAL' | 'PROTECTED' | 'STRICTLY_PROTECTED' | None
    """
    return _normalize_from_map(text, _CONF_MAP)


def normalize_slot_value(slot: str, text: str) -> Optional[object]:
    """
    Dispatcher by slot name:
      - 'isPrimary' -> bool
      - 'integrity' | 'availability' -> 'LOW'|'MEDIUM'|'HIGH'
      - 'confidentiality' -> 'PUBLIC'|'INTERNAL'|'PROTECTED'|'STRICTLY_PROTECTED'
    Returns None if undecidable/unknown slot.
    """
    key = (slot or "").strip()
    if key == "isPrimary":
        return normalize_is_primary(text)
    if key in {"integrity", "availability"}:
        return normalize_level(text)
    if key == "confidentiality":
        return normalize_confidentiality(text)
    return None
