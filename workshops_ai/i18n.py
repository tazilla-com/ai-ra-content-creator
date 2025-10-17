# File: workshops_ai/i18n.py
# -----------------------------------------------------------------------------
# Lightweight i18n utilities for WorkshopsAI agents (assets, impacts, threats).
#
# What this module provides
# - `lang_from(...)`: robust language selection from raw Accept-Language / locale.
# - `t(domain, key, ...)`: tiny string lookup with graceful fallbacks + .format(**kwargs).
# - `I18N`: short, agent-friendly message catalog (EN/SK/CS).
#
# Design goals
# - Zero external deps (stdlib only). No network I/O.
# - Safe fallbacks:
#     • Prefer explicit `language`, then `locale`. Both may be raw HTTP headers.
#     • Recognize prefixes "en", "sk", "cs" (and map "cz" → "cs").
#     • If nothing matches, fall back to English ("en").
# - Catalog codes/names (e.g., impact/threat codes) remain in EN; i18n is for surrounding text.
#
# Usage
#   from workshops_ai.i18n import t, lang_from
#
#   lang = lang_from(language=req.language, locale=req.locale)
#   msg = t("impacts", "set_done", lang=lang,
#           impact_code="AVAILABILITY", level=3, asset_id="asset-123")
#
# Domains
#   "common"  – shared phrases
#   "assets"  – AssetAgent
#   "impacts" – ImpactAgent
#   "threats" – ThreatAgent
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Fallback MUST be English so that unknown/unsupported locales don't drift to SK.
DEFAULT_LANG = "en"
SUPPORTED_LANGS: Tuple[str, ...] = ("en", "sk", "cs")


def _parse_accept_language(raw: Optional[str]) -> List[str]:
    """
    Best-effort parser for HTTP Accept-Language-like inputs.

    Examples:
      "cs-CZ, cs;q=0.9, en;q=0.8"  → ["cs-cz", "cs", "en"]
      "EN_us"                      → ["en-us"]
      "sk"                         → ["sk"]
      None / ""                    → []
    """
    if not raw:
        return []
    tokens: List[str] = []
    for part in raw.split(","):
        part = part.strip()
        # Strip optional quality, e.g., ";q=0.9"
        if ";" in part:
            part = part.split(";", 1)[0].strip()
        part = part.replace("_", "-").strip().lower()
        if part:
            tokens.append(part)
    return tokens


def _candidates(language: Optional[str], locale: Optional[str]) -> List[str]:
    """
    Build a prioritized list of language tokens to evaluate:
    - Full tags from `language` first, then from `locale`
    - Also include primary subtags (before '-' if present)
    """
    out: List[str] = []
    for raw in (language, locale):
        for tok in _parse_accept_language(raw):
            out.append(tok)
            # Add primary subtag if available (e.g., "cs" from "cs-cz")
            if "-" in tok:
                out.append(tok.split("-", 1)[0])
    return out


def lang_from(*, language: Optional[str] = None, locale: Optional[str] = None) -> str:
    """
    Decide the output language from potentially raw Accept-Language / locale strings.

    Rules:
      - If any token starts with "en" → "en"
      - If starts with "sk"          → "sk"
      - If starts with "cs" or "cz"  → "cs"  (Czech handled explicitly; never map to Slovak)
      - Fallback: DEFAULT_LANG ("en")
    """
    for token in _candidates(language, locale):
        if token.startswith("en"):
            return "en"
        if token.startswith("sk"):
            return "sk"
        if token.startswith("cs") or token.startswith("cz"):
            return "cs"
    return DEFAULT_LANG


def _lookup(lang: str, domain: str, key: str) -> str:
    """Return translated template for (lang, domain, key) or a safe fallback."""
    payload = I18N.get(lang) or {}
    dom = payload.get(domain) or {}
    if key in dom:
        return dom[key]
    # fallback to English
    dom_en = (I18N.get("en") or {}).get(domain) or {}
    if key in dom_en:
        return dom_en[key]
    # ultimate fallback: a visible breadcrumb for missing strings
    return f"{domain}.{key}"


def t(domain: str, key: str, *, lang: Optional[str] = None, locale: Optional[str] = None, **kwargs: Any) -> str:
    """
    Translate and format a string.

    Args:
      domain: one of "common", "assets", "impacts", "threats"
      key: message key inside the domain dictionary
      lang / locale: language selector (see lang_from). Either may be None or raw headers.
      **kwargs: values for str.format placeholders in the template

    Returns:
      Interpolated message string with graceful fallback behavior.
    """
    selected = lang_from(language=lang, locale=locale)
    template = _lookup(selected, domain, key)
    try:
        return template.format(**kwargs)
    except Exception:
        # If formatting fails (e.g., missing placeholder), return the raw template.
        return template


# -----------------------------------------------------------------------------
# Translations (keep them short and assistant-friendly; do not translate catalog codes)
# -----------------------------------------------------------------------------

I18N: Dict[str, Dict[str, Dict[str, str]]] = {
    "en": {
        "common": {
            "error_generic": "The request couldn't be processed. Please try again.",
            "fallback_generic": "I didn't understand that request.",
            "bulk_capped": "Bulk request capped to {n} items for safety.",
            "ok": "OK.",
            "done": "Done.",
            "confirm_done": "✅ Done.",
            "exists_asset": "ℹ️ Asset “{name}” already exists.",
            "clarify_asset": "Which asset?",
        },
        "assets": {
            "explain_1": "I’m your AssetAgent. Tell me in plain language what you want to do with assets.",
            "explain_2": "• “Create a CRM asset as primary, type Information system …”",
            "explain_3": "• “Update asset asset-db-001: rename to Customer DB and set primary=false.”",
            "explain_4": "• “List primary Application software containing ‘crm’.”",
            "explain_5": "Every change is broadcast via WS and submitted to the backend outbox as a single-action ChangeSet.",
            "hint_generic": "Tell me what you’d like to do with assets (create, update, list, detail).",
            "need_id": "Please provide an asset ID to show its detail.",
            "not_found": "I couldn't find an asset with ID “{id}”.",
            "detail_intro": "Here’s the asset detail for “{name}”.",
            "list_found": "I found {n} item(s){suffix}.",
            "ok": "OK.",
            "done": "Done — {n} change(s) queued to the outbox.",
            "invalid_change": "I couldn't build a valid change from that request. Try: “Create an asset …” or “Update asset …”.",
            "seed_intro": "Here are initial asset candidates based on your services/website.",
        },
        "impacts": {
            "select_asset": "Please select an asset to show its impacts.",
            "need_asset": "Please select an asset to show its impacts.",
            "no_impacts": "No impacts have been set for the selected asset yet.",
            "current_impacts_header": "Current impacts:",
            "impact_line": "- {impact_code}: {level}",
            "set_done": "Set {impact_code}={level} for asset {asset_id} (sent immediately).",
            "bulk_done_header": "Bulk changes (send-now):",
            "bulk_line": "- {code_eq_level} × {count}",
            "unknown_code": "Unknown impact_code: {impact_code}",
            "invalid_level": "Invalid level {level} (allowed 0..4).",
            "fallback": "I didn't understand the request on the Impacts tab.",
            "error_generic": "The request couldn't be processed. Please try again.",
            "bulk_capped": "Bulk request capped to {n} items for safety.",
            "explain_intro": "Rationale for selected impact levels:",
        },
        "threats": {
            "select_asset": "Optionally select an asset to tailor risk probability.",
            "need_asset": "Optionally select an asset to tailor risk probability.",
            "list_header": "Threats:",
            "explain_intro": "Relevance rationale:",
            "unknown_threat": "Unknown threat_code: {threat_code}",
            "invalid_probability": "Invalid probability {probability} (allowed 0..4).",
            "set_done": "Set probability P={probability} for threat {threat_code} on asset {asset_id} (sent immediately).",
            "bulk_done_header": "Bulk probability updates (send-now):",
            "bulk_line": "- {threat_code}=P{probability} × {count}",
            "bulk_capped": "Bulk request capped to {n} items for safety.",
            "error_generic": "The request couldn't be processed. Please try again.",
            "fallback": "I didn't understand the request on the Threats tab.",
        },
    },
    "sk": {
        "common": {
            "error_generic": "Požiadavku sa nepodarilo spracovať. Skúste znova.",
            "fallback_generic": "Nerozumiem tejto požiadavke.",
            "bulk_capped": "Hromadný príkaz bol z bezpečnostných dôvodov skrátený na {n} položiek.",
            "ok": "OK.",
            "done": "Hotovo.",
            "confirm_done": "✅ Hotovo.",
            "exists_asset": "ℹ️ Aktívum „{name}“ už existuje.",
            "clarify_asset": "Pre ktoré aktívum?",
        },
        "assets": {
            "explain_1": "Som tvoj AssetAgent. Povedz mi jednoducho, čo chceš s aktívami urobiť.",
            "explain_2": "• „Vytvor CRM ako primárne, typ Information system …“",
            "explain_3": "• „Aktualizuj asset asset-db-001: premenuj na Customer DB a nastav primary=false.“",
            "explain_4": "• „Vypíš primárne Application software obsahujúce ‘crm’.“",
            "explain_5": "Každá zmena sa posiela cez WS a ukladá do outboxu backendu ako single-action ChangeSet.",
            "hint_generic": "Napíš, čo chceš s aktívami spraviť (vytvoriť, upraviť, vypísať, detail).",
            "need_id": "Uveď prosím ID assetu, aby som zobrazil detail.",
            "not_found": "Nenašiel som asset s ID „{id}“.",
            "detail_intro": "Tu je detail assetu „{name}“.",
            "list_found": "Našiel som {n} položiek{suffix}.",
            "ok": "OK.",
            "done": "Hotovo — {n} zmien odoslaných do outboxu.",
            "invalid_change": "Nepodarilo sa vytvoriť platnú zmenu. Skús: „Vytvor asset …“ alebo „Uprav asset …“.",
            "seed_intro": "Tu sú počiatoční kandidáti aktív podľa vašich služieb/webu.",
        },
        "impacts": {
            "select_asset": "Vyberte, prosím, aktívum, pre ktoré mám zobraziť dopady.",
            "need_asset": "Vyberte, prosím, aktívum, pre ktoré mám zobraziť dopady.",
            "no_impacts": "Pre vybrané aktívum zatiaľ nie sú nastavené žiadne dopady.",
            "current_impacts_header": "Aktuálne dopady:",
            "impact_line": "- {impact_code}: {level}",
            "set_done": "Nastavil som {impact_code}={level} pre aktívum {asset_id} (odoslané okamžite).",
            "bulk_done_header": "Hromadné zmeny (send-now):",
            "bulk_line": "- {code_eq_level} × {count}",
            "unknown_code": "Neznámy impact_code: {impact_code}",
            "invalid_level": "Neplatná úroveň {level} (povolené 0..4).",
            "fallback": "Nerozumiem požiadavke na karte Dopady.",
            "error_generic": "Požiadavku sa nepodarilo spracovať. Skúste znova.",
            "bulk_capped": "Hromadný príkaz bol z bezpečnostných dôvodov skrátený na {n} položiek.",
            "explain_intro": "Odôvodnenie zvolených úrovní dopadov:",
        },
        "threats": {
            "select_asset": "Voliteľne vyberte aktívum na spresnenie pravdepodobnosti rizika.",
            "need_asset": "Voliteľne vyberte aktívum na spresnenie pravdepodobnosti rizika.",
            "list_header": "Hrozby:",
            "explain_intro": "Odôvodnenie relevancie:",
            "unknown_threat": "Neznámy threat_code: {threat_code}",
            "invalid_probability": "Neplatná pravdepodobnosť {probability} (povolené 0..4).",
            "set_done": "Nastavil som pravdepodobnosť P={probability} pre hrozbu {threat_code} pri aktíve {asset_id} (odoslané okamžite).",
            "bulk_done_header": "Hromadné nastavenia pravdepodobnosti (send-now):",
            "bulk_line": "- {threat_code}=P{probability} × {count}",
            "bulk_capped": "Hromadný príkaz bol z bezpečnostných dôvodov skrátený na {n} položiek.",
            "error_generic": "Požiadavku sa nepodarilo spracovať. Skúste znova.",
            "fallback": "Nerozumiem požiadavke na karte Hrozby.",
        },
    },
    "cs": {
        "common": {
            "error_generic": "Požadavek se nepodařilo zpracovat. Zkuste to prosím znovu.",
            "fallback_generic": "Tomuto požadavku nerozumím.",
            "bulk_capped": "Hromadná akce byla z bezpečnostních důvodů zkrácena na {n} položek.",
            "ok": "OK.",
            "done": "Hotovo.",
            "confirm_done": "✅ Hotovo.",
            "exists_asset": "ℹ️ Aktivum „{name}“ už existuje.",
            "clarify_asset": "Pro které aktivum?",
        },
        "assets": {
            "explain_1": "Jsem tvůj AssetAgent. Napiš jednoduše, co chceš dělat s assety.",
            "explain_2": "• „Vytvoř CRM jako primární, typ Information system …“",
            "explain_3": "• „Uprav asset asset-db-001: přejmenuj na Customer DB a nastav primary=false.“",
            "explain_4": "• „Vypiš primární Application software obsahující ‘crm’.“",
            "explain_5": "Každá změna se posílá přes WS a ukládá do outboxu backendu jako ChangeSet s jednou akcí.",
            "hint_generic": "Napiš, co chceš s assety udělat (vytvořit, upravit, vypsat, detail).",
            "need_id": "Uveď prosím ID assetu pro zobrazení detailu.",
            "not_found": "Asset s ID „{id}“ jsem nenašel.",
            "detail_intro": "Tady je detail assetu „{name}“.",
            "list_found": "Našel jsem {n} položek{suffix}.",
            "ok": "OK.",
            "done": "Hotovo — {n} změn odesláno do outboxu.",
            "invalid_change": "Nepodařilo se vytvořit platnou změnu. Zkus: „Vytvoř asset …“ nebo „Uprav asset …“.",
            "seed_intro": "Tady jsou počáteční kandidáti aktiv podle vašich služeb/webu.",
        },
        "impacts": {
            "select_asset": "Vyber prosím aktivum, u kterého mám zobrazit dopady.",
            "need_asset": "Vyber prosím aktivum, u kterého mám zobrazit dopady.",
            "no_impacts": "U vybraného aktiva zatím nejsou nastaveny žádné dopady.",
            "current_impacts_header": "Aktuální dopady:",
            "impact_line": "- {impact_code}: {level}",
            "set_done": "Nastavil jsem {impact_code}={level} pro aktivum {asset_id} (odesláno okamžitě).",
            "bulk_done_header": "Hromadné změny (send-now):",
            "bulk_line": "- {code_eq_level} × {count}",
            "unknown_code": "Neznámý impact_code: {impact_code}",
            "invalid_level": "Neplatná úroveň {level} (povoleno 0..4).",
            "fallback": "Na kartě Dopady jsem nerozuměl požadavku.",
            "error_generic": "Požadavek se nepodařilo zpracovat. Zkuste to prosím znovu.",
            "bulk_capped": "Hromadná akce byla z bezpečnostních důvodů zkrácena na {n} položek.",
            "explain_intro": "Odůvodnění zvolených úrovní dopadů:",
        },
        "threats": {
            "select_asset": "Volitelně vyber aktivum pro upřesnění pravděpodobnosti rizika.",
            "need_asset": "Volitelně vyber aktivum pro upřesnění pravděpodobnosti rizika.",
            "list_header": "Hrozby:",
            "explain_intro": "Odůvodnění relevance:",
            "unknown_threat": "Neznámý threat_code: {threat_code}",
            "invalid_probability": "Neplatná pravděpodobnost {probability} (povoleno 0..4).",
            "set_done": "Nastavil jsem pravděpodobnost P={probability} pro hrozbu {threat_code} u aktiva {asset_id} (odesláno okamžitě).",
            "bulk_done_header": "Hromadná nastavení pravděpodobnosti (send-now):",
            "bulk_line": "- {threat_code}=P{probability} × {count}",
            "bulk_capped": "Hromadná akce byla z bezpečnostních důvodů zkrácena na {n} položek.",
            "error_generic": "Požadavek se nepodařilo zpracovat. Zkuste to prosím znovu.",
            "fallback": "Na kartě Hrozby jsem nerozuměl požadavku.",
        },
    },
}

__all__ = ["t", "lang_from", "I18N", "SUPPORTED_LANGS", "DEFAULT_LANG"]
