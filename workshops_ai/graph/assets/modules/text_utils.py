# workshops_ai/graph/assets/modules/text_utils.py

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

__all__ = [
    "lang",
    "looks_czech",
    "tokenize",
    "parse_list_limit",
    "extract_name_from_text",
    "is_yes_like",
    "is_no_like",
    "looks_like_create",
    "looks_like_explain",
    "mentions_types",
    "looks_like_greeting",
    # new, for safe rendering of lists coming from tools/backends
    "as_label",
    "to_labels",
    "safe_join_labels",
]

# -----------------------------------------------------------------------------
# Locale helpers
# -----------------------------------------------------------------------------

def lang(loc: Optional[str]) -> str:
    """
    Return primary language subtag ('cs', 'en', ...) in lowercase.
    """
    return (loc or "en").split("-")[0].lower()


# -----------------------------------------------------------------------------
# Czech heuristics
# -----------------------------------------------------------------------------

_CZ_DIACRITICS = set("áéíóúýčďěňřšťůžďťľĺôäÁÉÍÓÚÝČĎĚŇŘŠŤŮŽĎŤĽĹÔÄ")
_CZ_TOKENS = {
    "vytvoř", "vytvor", "název", "nazev", "potvrdit", "potvrzuji",
    "schvaluji", "ano", "áno", "ne", "není", "neni", "aktiva", "typ",
    "primární", "primarni",
}

def looks_czech(text: str) -> bool:
    """
    Heuristically decide if text is Czech: presence of diacritics or common tokens.
    """
    s = (text or "").strip()
    if not s:
        return False
    if any(ch in _CZ_DIACRITICS for ch in s):
        return True
    low = s.lower()
    return any(tok in low for tok in _CZ_TOKENS)


# -----------------------------------------------------------------------------
# Tokenization / small NLP utils
# -----------------------------------------------------------------------------

# split on whitespace and strip common punctuation; keep simple and fast
_PUNCT_STRIP = ",.;:!?()[]{}\"'"

def tokenize(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    toks: list[str] = []
    for part in raw.split():
        toks.append(part.strip(_PUNCT_STRIP).casefold())
    return toks


def parse_list_limit(text: str) -> int | None:
    """
    Parse user intent like:
      'top 10', 'first 5', 'limit 3', 'jen 7', 'show 5', 'list 20', 'top-10', 'top10'
    Returns None if no limit is detected.
    """
    s = (text or "").strip().lower()
    if not s:
        return None

    # quick patterns: top-10 / top10
    m = re.search(r"\b(?:top|first|limit|jen|only|zkr[aá]t(?:it|te)?)\s*-?\s*(\d{1,3})\b", s)
    if m:
        return max(1, int(m.group(1)))

    toks = tokenize(s)
    keys = {"top", "first", "limit", "jen", "zkrať", "zkrat", "zkratit", "zkraťte", "only"}
    for i, tok in enumerate(toks):
        if tok in keys:
            if i + 1 < len(toks) and toks[i + 1].isdigit():
                return max(1, int(toks[i + 1]))
            if i + 2 < len(toks) and toks[i + 1] in {"na", "to"} and toks[i + 2].isdigit():
                return max(1, int(toks[i + 2]))
    for i, tok in enumerate(toks):
        if tok in {"show", "list"} and i + 1 < len(toks) and toks[i + 1].isdigit():
            return max(1, int(toks[i + 1]))
    return None


# -----------------------------------------------------------------------------
# Name extraction & confirmations
# -----------------------------------------------------------------------------

_NAME_PATTERNS = [
    r'^(?:název\s+je|nazev\s+je)\s+(?P<name>.+)$',
    r'^(?:name\s+is)\s+(?P<name>.+)$',
    r'^(?P<q1>"[^"]{1,120}")$',
    r"^(?P<q2>'[^']{1,120}')$",
]
NAME_RE_CONFIRM = re.compile(
    r"^(potvrdit(\s+vytvoření|[ ]*vytvoreni)?|potvrzuji|schvaluji|potvrdzujem|schvaľujem|confirm|ok|okay|yes|ano|áno)$",
    re.I,
)
NAME_RE_CANCEL = re.compile(r"^(zrušit|zrušiť|zrusit|cancel|stop|ne|no|nie)$", re.I)

def extract_name_from_text(text: str) -> str | None:
    """
    Try to extract a name from short utterances like:
      - 'název je CRM', 'name is CRM'
      - '"CRM"', "'CRM'"
      - or short 1..5 word raw text → treated as a name candidate.
    """
    if not text:
        return None
    raw = text.strip()
    for pat in _NAME_PATTERNS:
        m = re.match(pat, raw, flags=re.IGNORECASE)
        if m:
            g = m.groupdict()
            for k in ("name", "q1", "q2"):
                val = g.get(k)
                if val:
                    val = val.strip().strip('"').strip("'").strip()
                    return val or None
    if len(raw.split()) <= 5:
        return raw
    return None


def is_yes_like(text: str) -> bool:
    return bool(NAME_RE_CONFIRM.match((text or "").strip().lower()))


def is_no_like(text: str) -> bool:
    return bool(NAME_RE_CANCEL.match((text or "").strip().lower()))


# -----------------------------------------------------------------------------
# Intent heuristics for Asset agent
# -----------------------------------------------------------------------------

def looks_like_create(msg: str) -> bool:
    m = (msg or "").strip().lower()
    tokens = ["vytvoř", "vytvor", "založ", "zaloz", "přidej", "pridej", "add", "create", "nové", "nove", "new"]
    return any(t in m for t in tokens)


def looks_like_explain(msg: str) -> bool:
    m = (msg or "").strip().lower()
    keys = ["explain", "what is", "co je", "vysvětli", "vysvetli", "prosím vysvětli"]
    return any(k in m for k in keys)


def mentions_types(msg: str) -> bool:
    m = (msg or "").strip().lower()
    keys = ["types", "asset types", "typy", "typ", "druhy"]
    return any(k in m for k in keys)


def looks_like_greeting(msg: str) -> bool:
    m = (msg or "").strip().lower()
    if not m:
        return False
    short = len(m.split()) <= 4
    phrases = [
        "ahoj", "čau", "cau", "nazdar", "zdar", "servus",
        "dobrý den", "dobry den", "hezký den", "hezky den",
        "hello", "hi", "hey", "yo", "good morning", "good afternoon", "good evening",
    ]
    return short and any(p in m for p in phrases)


# -----------------------------------------------------------------------------
# NEW: Safe label normalization for lists coming from tools/backends
# -----------------------------------------------------------------------------

def as_label(x: Any) -> str:
    """
    Convert any object (str / dict / pydantic model / dataclass) into a user label.
    Preference order: name, label, code, slug, id → first non-empty string.
    Falls back to str(x). Never returns None; always a string.
    """
    if isinstance(x, str):
        return x.strip()

    if isinstance(x, dict):
        for k in ("name", "label", "code", "slug", "id"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return str(x)

    # pydantic/dataclass/objects with attributes
    for k in ("name", "label", "code", "slug", "id"):
        if hasattr(x, k):
            v = getattr(x, k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return str(x)


def to_labels(xs: Iterable[Any] | None) -> list[str]:
    """
    Map an iterable of heterogeneous items to list[str] using as_label, skipping empties.
    """
    if not xs:
        return []
    out: list[str] = []
    for x in xs:
        v = as_label(x)
        if v:
            out.append(v)
    return out


def safe_join_labels(xs: Iterable[Any] | None, sep: str = ", ") -> str:
    """
    Join labels for rendering. Safe for list[str], list[dict], list[pydantic], etc.
    """
    return sep.join(to_labels(xs))
