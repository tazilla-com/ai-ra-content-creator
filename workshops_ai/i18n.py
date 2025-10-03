# File: workshops_ai/i18n.py
# -----------------------------------------------------------------------------
# Lightweight i18n utilities for WorkshopsAI agents (assets, impacts, threats).
#
# Goals
# - Single source of truth for short UI/assistant strings used by agents.
# - Minimal dependencies (stdlib only). No runtime network I/O.
# - Safe lookups with graceful fallbacks:
#     • prefer explicit `lang` ("en", "sk", "cs"); if not given, derive from `locale`
#     • DO NOT map Czech to Slovak (cs/cz → "cs" explicitly)
#     • domain/key missing → fall back to English; if still missing, return "{domain}.{key}"
# - Keep catalog codes/names in EN (i18n is for surrounding messages only).
#
# Usage
#   from workshops_ai.i18n import t, lang_from
#
#   lang = lang_from(language=req.language, locale=req.locale)
#   msg = t("impacts", "set_done", lang=lang, impact_code="AVAILABILITY", level=3, asset_id="asset-123")
#
# Domains
#   "common"  – shared phrases
#   "assets"  – AssetAgent
#   "impacts" – ImpactAgent
#   "threats" – ThreatAgent
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional

DEFAULT_LANG = "sk"
SUPPORTED_LANGS = ("en", "sk", "cs")


def lang_from(*, language: Optional[str] = None, locale: Optional[str] = None) -> str:
    """
    Decide the output language:
      - If `language`/`locale` starts with "en" → "en"
      - If starts with "sk" → "sk"
      - If starts with "cs" or "cz" → "cs"   (Czech handled explicitly)
      - Fallback: "en"
    """
    def _norm(x: Optional[str]) -> str:
        return (x or "").strip().lower()

    for token in (_norm(language), _norm(locale)):
        if token.startswith("en"):
            return "en"
        if token.startswith("sk"):
            return "sk"
        if token.startswith(("cs", "cz")):
            return "cs"
    return DEFAULT_LANG


def _lookup(lang: str, domain: str, key: str) -> str:
    # best-effort: given lang → fallback to 'en'
    payload = I18N.get(lang) or {}
    dom = payload.get(domain) or {}
    if key in dom:
        return dom[key]
    # fallback to English
    dom_en = (I18N.get("en") or {}).get(domain) or {}
    if key in dom_en:
        return dom_en[key]
    # ultimate fallback
    return f"{domain}.{key}"


def t(domain: str, key: str, *, lang: Optional[str] = None, locale: Optional[str] = None, **kwargs: Any) -> str:
    """
    Translate and format a string.
    - `domain`: one of "common", "assets", "impacts", "threats"
    - `key`: message key inside the domain dictionary
    - `lang` or `locale`: language selector (see lang_from); either is optional
    - `**kwargs`: values for str.format placeholders in the template
    """
    selected = lang_from(language=lang, locale=locale)
    template = _lookup(selected, domain, key)
    try:
        return template.format(**kwargs)
    except Exception:
        # If formatting fails (missing placeholder), return the raw template.
        return template


# -----------------------------------------------------------------------------
# Translations
# NOTE: Keep messages short and agent-friendly. Do not translate catalog codes.
# -----------------------------------------------------------------------------

I18N: Dict[str, Dict[str, Dict[str, str]]] = {
    "en": {
        "common": {
            "error_generic": "The request couldn't be processed. Please try again.",
            "fallback_generic": "I didn't understand that request.",
            "bulk_capped": "Bulk request capped to {n} items for safety.",
            "ok": "OK.",
            "done": "Done.",
            # Unified short confirmations / clarifications:
            "confirm_done": "✅ Done.",
            "exists_asset": "ℹ️ Asset “{name}” already exists.",
            "clarify_asset": "Which asset?",
        },
        # --------------------------- AssetAgent --------------------------------
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
        # --------------------------- ImpactAgent -------------------------------
        "impacts": {
            # both keys provided so older/newer agents never 404 on i18n
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
        # --------------------------- ThreatAgent -------------------------------
        "threats": {
            "select_asset": "Optionally select an asset to tailor risk probability.",
            "need_asset": "Optionally select an asset to tailor risk probability.",  # alias
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
            # Unified short confirmations / clarifications:
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
            "need_asset": "Vyberte, prosím, aktívum, pre ktoré mám zobraziť dopady.",  # alias
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
            "need_asset": "Voliteľne vyberte aktívum na spresnenie pravdepodobnosti rizika.",  # alias
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
            # Unified short confirmations / clarifications:
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
            "need_asset": "Vyber prosím aktivum, u kterého mám zobrazit dopady.",  # alias
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
            "need_asset": "Volitelně vyber aktivum pro upřesnění pravděpodobnosti rizika.",  # alias
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
