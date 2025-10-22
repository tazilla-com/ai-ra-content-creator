# File: workshops_ai/graph/assets/modules/text_utils.py

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

__all__ = [
    "lang",
    "looks_czech",
    "looks_slovak",
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
    # optional helpers
    "looks_like_uuid",
    "looks_like_id",
]

# -----------------------------------------------------------------------------
# Locale helpers
# -----------------------------------------------------------------------------

def lang(loc: Optional[str]) -> str:
    """
    Return primary language subtag ('cs', 'sk', 'en', ...) in lowercase.
    Falls back to 'en' when not provided.
    """
    return (loc or "en").split("-")[0].lower()


# -----------------------------------------------------------------------------
# Czech/Slovak heuristics
# -----------------------------------------------------------------------------

# Contains both CZ and SK diacritics so we can auto-hint cs/sk reliably.
_CZ_SK_DIACRITICS = set(
    "áäčďéěíĺľňóôřŕšťúůýžÁÄČĎÉĚÍĹĽŇÓÔŘŔŠŤÚŮÝŽ"
)

# Tokens that frequently indicate CZ/SK language in short utterances.
_CZ_SK_TOKENS = {
    # intents / verbs
    "vytvoř", "vytvor", "založ", "zaloz", "založiť", "zalozit",
    "přidej", "pridej", "pridaj", "pridať",
    "ukaž", "ukaz", "ukáž", "zobraz", "zobraziť", "zobrazit",
    "vypiš", "vypis", "vypsat", "zoznam",
    # common nouns
    "název", "nazev", "názov", "nazov", "aktiva", "aktíva", "typ", "typy", "druhy",
    # flags / quantifiers
    "všechny", "vsechny", "všetky", "vsetky", "všetko", "vsetko", "jen", "iba", "len",
    # confirmations
    "potvrdit", "potvrzuji", "schvaluji", "potvrdiť", "potvrdzujem", "schvaľujem",
    "ano", "áno", "ne", "nie",
}

def looks_czech(text: str) -> bool:
    """
    Heuristically decide if text is Czech/Slovak: presence of diacritics or common tokens.
    This is purposefully broad: we use it to flip the locale to 'cs' (or 'sk' later),
    which in turn localizes UI questions. Exact language split is handled elsewhere.
    """
    s = (text or "").strip()
    if not s:
        return False
    if any(ch in _CZ_SK_DIACRITICS for ch in s):
        return True
    low = s.lower()
    return any(tok in low for tok in _CZ_SK_TOKENS)

# More precise Slovak-only hints (used by NLU to prefer 'sk' over 'cs' when detectable).
_SK_DIACRITICS_HINTS = set("äĺľŕôÄĹĽŔÔ")
_SK_TOKENS = {
    # typical SK function words and verbs
    "čo", "čo je", "prosim", "prosím", "prosím ťa", "prosím ta",
    "názov", "nazov", "pridať", "pridaj", "ukáž", "ukaz", "zobraziť", "zobrazit", "zoznam",
    "áno", "nie", "potvrdiť", "potvrdzujem", "schvaľujem",
}

def looks_slovak(text: str) -> bool:
    """
    Heuristically decide if text is Slovak: presence of SK-only diacritics
    or common SK tokens. Returns False for empty/neutral text.
    """
    s = (text or "").strip()
    if not s:
        return False
    if any(ch in _SK_DIACRITICS_HINTS for ch in s):
        return True
    low = s.lower()
    return any(tok in low for tok in _SK_TOKENS)


# -----------------------------------------------------------------------------
# Tokenization / small NLP utils
# -----------------------------------------------------------------------------

# Split on whitespace and strip common punctuation; keep simple and fast.
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
    Parse user intent like (CZ/SK/EN):
      'top 10', 'first 5', 'limit 3', 'jen 7', 'iba 7', 'len 7',
      'zobraz 5', 'ukaž 3', 'ukáž 3', 'show 5', 'list 20', 'top-10', 'top10'
    Returns None if no limit is detected.
    """
    s = (text or "").strip().lower()
    if not s:
        return None

    # quick patterns: top-10 / top10 / limit-5 / jen-7 / iba-3 / len-4
    m = re.search(r"\b(?:top|first|limit|jen|iba|len|only)\s*-?\s*(\d{1,3})\b", s)
    if m:
        return max(1, int(m.group(1)))

    toks = tokenize(s)

    # single-key then number (e.g., "top 5", "jen 7", "iba 2", "len 12")
    keys = {"top", "first", "limit", "jen", "iba", "len", "only"}
    for i, tok in enumerate(toks):
        if tok in keys:
            if i + 1 < len(toks) and toks[i + 1].isdigit():
                return max(1, int(toks[i + 1]))
            if i + 2 < len(toks) and toks[i + 1] in {"na", "to"} and toks[i + 2].isdigit():
                return max(1, int(toks[i + 2]))

    # verbs with a number after them (CZ/SK/EN)
    verb_keys = {"show", "list", "zobraz", "zobraziť", "zobrazit", "ukaž", "ukáž", "ukaz", "vypiš", "vypis"}
    for i, tok in enumerate(toks):
        if tok in verb_keys and i + 1 < len(toks) and toks[i + 1].isdigit():
            return max(1, int(toks[i + 1]))

    return None


# -----------------------------------------------------------------------------
# Name extraction & confirmations
# -----------------------------------------------------------------------------

_NAME_PATTERNS = [
    r'^(?:název\s+je|nazev\s+je|názov\s+je|nazov\s+je)\s+(?P<name>.+)$',  # CZ/SK
    r'^(?:name\s+is)\s+(?P<name>.+)$',                                    # EN
    r'^(?P<q1>"[^"]{1,120}")$',
    r"^(?P<q2>'[^']{1,120}')$",
]
NAME_RE_CONFIRM = re.compile(
    r"^(potvrdit(\s+vytvoření|[ ]*vytvoreni)?|potvrzuji|schvaluji|"
    r"potvrdiť|potvrdzujem|schvaľujem|"
    r"confirm|ok|okay|yes|y|sure|ano|áno|jasne|jasné)$",
    re.I,
)
NAME_RE_CANCEL = re.compile(r"^(zrušit|zrušiť|zrusit|cancel|stop|no|nie|ne|n|nope)$", re.I)

def extract_name_from_text(text: str) -> str | None:
    """
    Try to extract a name from short utterances like:
      - 'název je CRM', 'názov je CRM', 'name is CRM'
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
    """Detect short yes-like confirmations across CZ/SK/EN."""
    return bool(NAME_RE_CONFIRM.match((text or "").strip().lower()))


def is_no_like(text: str) -> bool:
    """Detect short no-like rejections across CZ/SK/EN."""
    return bool(NAME_RE_CANCEL.match((text or "").strip().lower()))


# -----------------------------------------------------------------------------
# Intent heuristics for Asset agent
# -----------------------------------------------------------------------------

def looks_like_create(msg: str) -> bool:
    """
    Detect creation intent in CZ/SK/EN texts.
    Examples: 'vytvoř', 'vytvor', 'založ', 'založiť', 'přidej', 'pridaj', 'add', 'create', 'nové', 'nový', 'new'.
    """
    m = (msg or "").strip().lower()
    tokens = [
        # CZ/SK verbs
        "vytvoř", "vytvor", "založ", "zalozit", "založiť",
        "přidej", "pridej", "pridaj", "pridať",
        # adjectives
        "nové", "nove", "nový", "novy",
        # EN
        "add", "create", "new",
    ]
    return any(t in m for t in tokens)


def looks_like_explain(msg: str) -> bool:
    """
    Detect explain intent (CZ/SK/EN).
    Examples: 'explain', 'what is', 'co je', 'čo je', 'vysvětli', 'vysvetli', 'prosím vysvětli'.
    """
    m = (msg or "").strip().lower()
    keys = ["explain", "what is", "co je", "čo je", "vysvětli", "vysvetli", "prosím vysvětli", "prosim vysvetli"]
    return any(k in m for k in keys)


def mentions_types(msg: str) -> bool:
    """
    Detect mentions of asset type listings across languages.
    """
    m = (msg or "").strip().lower()
    keys = ["types", "asset types", "typy", "typ", "druhy", "katalog typů", "katalóg typov"]
    return any(k in m for k in keys)


def looks_like_greeting(msg: str) -> bool:
    """
    Detect short greetings (CZ/SK/EN). Keeps it conservative to avoid false positives.
    """
    m = (msg or "").strip().lower()
    if not m:
        return False
    short = len(m.split()) <= 4
    phrases = [
        # CZ/SK
        "ahoj", "ahojte", "čau", "cau", "čaute", "nazdar", "zdar", "servus",
        "dobrý den", "dobry den", "dobrý deň", "dobry den", "hezký den", "hezky den",
        # EN
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


# -----------------------------------------------------------------------------
# Optional helpers (ID/UUID detection)
# -----------------------------------------------------------------------------

_UUID_RX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5|7][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

def looks_like_uuid(text: str) -> bool:
    """Return True if the text looks like a UUID (v1..v5/v7 heuristic)."""
    return bool(_UUID_RX.match((text or "").strip()))

def looks_like_id(text: str) -> bool:
    """
    Broad ID detector used for free-text 'get' commands.
    Currently considers UUIDs; intentionally conservative.
    """
    s = (text or "").strip()
    if not s:
        return False
    return looks_like_uuid(s)
