# File: workshops_ai/graph/assets/modules/nodes_update.py
from __future__ import annotations
import json
import re
from typing import Any, Optional, Tuple, List

from workshops_ai.graph.factory import GraphDeps
from .state import AssetState
from .text_utils import lang, extract_name_from_text
from .enums_json import CONF_LEVELS, TRI_LEVELS, conf_from_text, tri_from_text
from .ui_helpers import ui_response as _ui_response
from .write_utils import apply_actions_async, validate_write_actions
from .enums_json import jsonify_iter
from .llm_utils import extract_update_asset_intent
from .telemetry_utils import event, finish


# -----------------------------------------------------------------------------
# Localized helpers
# -----------------------------------------------------------------------------
def _L(l: str | None, cs: str, sk: str, en: str) -> str:
    """Return a localized string based on normalized locale ('cs' | 'sk' | other→'en')."""
    return cs if l == "cs" else sk if l == "sk" else en


# -----------------------------------------------------------------------------
# Canonicalizers
# -----------------------------------------------------------------------------
def _boolish(v: Any) -> Optional[bool]:
    """Return True/False for common yes/no-like values across cs/sk/en; otherwise None."""
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    if s in {"1", "true", "yes", "y", "ano", "áno", "primary", "ok", "okay", "sure"}:
        return True
    if s in {"0", "false", "no", "n", "ne", "nie"}:
        return False
    return None


def _canon_conf(v: Any) -> Optional[str]:
    """Canonicalize confidentiality to one of CONF_LEVELS; None if invalid/unknown."""
    if v is None:
        return None
    c = conf_from_text(str(v))
    if c:
        return c
    t = str(v).strip().upper()
    return t if t in set(CONF_LEVELS) else None


def _canon_tri(v: Any) -> Optional[str]:
    """Canonicalize integrity/availability to one of TRI_LEVELS; None if invalid/unknown."""
    if v is None:
        return None
    t = tri_from_text(str(v))
    if t:
        return t
    u = str(v).strip().upper()
    return u if u in set(TRI_LEVELS) else None


# -----------------------------------------------------------------------------
# Asset lookup utilities (STRICTLY over existing assets, not catalog types)
# -----------------------------------------------------------------------------
def _parse_label_ref(s: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse optional "Name (Type)" label into (name, type). Whitespace tolerant.
    Returns (None, None) if the pattern does not match.
    """
    if not s:
        return None, None
    m = re.match(r"^\s*(.+?)\s*\((.+?)\)\s*$", s)
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def _get_asset_by_id(deps: GraphDeps, asset_id: str | None, org_id: str | None) -> Optional[dict]:
    """Fetch asset by ID; fallback to scanning list if direct get is unsupported; None if not found."""
    if not asset_id:
        return None
    try:
        a = deps.backend_port.asset_get(asset_id)
        d = jsonify_iter(a)
        return d if isinstance(d, dict) else None
    except Exception:
        # Fallback to list scan (defensive for backends without direct get)
        try:
            items = deps.backend_port.assets_list(org_id, None, None, None) or []
            for cand in items:
                d = jsonify_iter(cand)
                if isinstance(d, dict) and (d.get("id") == asset_id or d.get("asset_id") == asset_id):
                    return d
        except Exception:
            pass
    return None


def _ambiguous_msg(l: str, matches: List[dict]) -> str:
    """Build a localized ambiguity message with up to 6 ID/name/type bullets."""
    bullets = []
    for d in matches[:6]:
        bullets.append(f"- {d.get('id') or d.get('asset_id')} — {d.get('name')} ({d.get('assetType')})")
    return _L(
        l,
        "Nalezeno více aktiv se stejným kritériem. Uveď prosím ID:\n" + "\n".join(bullets),
        "Našlo sa viac aktív s rovnakým kritériom. Uveď prosím ID:\n" + "\n".join(bullets),
        "Found multiple matching assets. Please specify the ID:\n" + "\n".join(bullets),
    )


def _get_asset_by_name_or_type_unique(
    deps: GraphDeps, ref: str | None, org_id: str | None, locale: Optional[str]
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Resolve an asset among EXISTING assets by:
      1) "Name (Type)" exact match (case-insensitive, both parts matched)
      2) Name exact (case-insensitive)
      3) If none found by name, try Type exact (case-insensitive) **only if unique among existing assets**
      4) If multiple found for the same criterion → return (None, localized_ambiguity_message)
    """
    l = lang(locale)
    if not (ref and ref.strip()):
        return None, _L(l, "Chybí název/odkaz na aktivum.", "Chýba názov/odkaz na aktívum.", "Asset reference is missing.")

    try:
        items = deps.backend_port.assets_list(org_id, None, None, None) or []
        rows: List[dict] = []
        for cand in items:
            d = jsonify_iter(cand) or {}
            if isinstance(d, dict):
                rows.append(d)

        # 1) Name (Type)
        nm, tp = _parse_label_ref(ref)
        if nm and tp:
            for d in rows:
                n = str(d.get("name") or "").strip().lower()
                t = str(d.get("assetType") or "").strip().lower()
                if n == nm.lower() and t == tp.lower():
                    return d, None
            # fall through to other strategies if not found

        # 2) Name exact (case-insensitive)
        name_lc = ref.strip().lower()
        by_name = [d for d in rows if str(d.get("name") or "").strip().lower() == name_lc]
        if len(by_name) == 1:
            return by_name[0], None
        if len(by_name) > 1:
            return None, _ambiguous_msg(l, by_name)

        # 3) Unique Type among existing assets (case-insensitive)
        by_type = [d for d in rows if str(d.get("assetType") or "").strip().lower() == name_lc]
        if len(by_type) == 1:
            return by_type[0], None
        if len(by_type) > 1:
            return None, _ambiguous_msg(l, by_type)

        # 4) Nothing found
        return None, _L(
            l,
            f"Aktivum „{ref}“ nenalezeno.",
            f"Aktívum „{ref}“ nebolo nájdené.",
            f"Asset “{ref}” not found.",
        )
    except Exception:
        return None, _L(l, "Lookup aktiv selhal.", "Lookup aktív zlyhal.", "Asset lookup failed.")


# -----------------------------------------------------------------------------
# Target explicitness helper (UUID or name-like mention)
# -----------------------------------------------------------------------------
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


def _message_mentions_explicit_target(raw: str) -> bool:
    """True if message contains a UUID-like token or an extractable quoted/name-like segment."""
    s = (raw or "").strip()
    if not s:
        return False
    if _UUID_RE.search(s) is not None:
        return True
    name = (extract_name_from_text(s) or "").strip()
    return bool(name)


def _fallback_selection_from_current_asset_if_safe(state: AssetState, raw_message: str) -> None:
    """
    Populate state.selection from state.current_asset ONLY when:
      - there is no explicit UI selection in state.selection, AND
      - the message does not explicitly point to a different target (UUID or name), AND
      - current_asset contains an id.

    This ensures smooth “detail → update” while avoiding stale selection leaks.
    """
    try:
        sel = state.selection or {}
        if sel.get("asset_id") or sel.get("asset_name"):
            return
        if _message_mentions_explicit_target(raw_message):
            return
        cur = getattr(state, "current_asset", None)
        if isinstance(cur, dict):
            cid = cur.get("asset_id") or cur.get("id")
            cname = cur.get("name")
            if cid:
                state.selection = sel or {}
                state.selection["asset_id"] = str(cid)
                if cname:
                    state.selection["asset_name"] = str(cname)
    except Exception:
        # best-effort only
        pass


# -----------------------------------------------------------------------------
# Core update handler
# -----------------------------------------------------------------------------
def _update_or_set_meta_inner(deps: GraphDeps):
    async def _inner(state: AssetState) -> AssetState:
        from .state import AssetUpdateItem  # dataclass for structured updates

        l = lang(state.locale)
        t0 = event(state, kind="tool.call.start", tool="asset.update_meta")

        def _prompt_for_fields(primary: bool) -> str:
            if primary:
                return _L(
                    l,
                    "Co chceš změnit? (povolené: name, description, confidentiality, integrity, availability)",
                    "Čo chceš zmeniť? (povolené: name, description, confidentiality, integrity, availability)",
                    "What would you like to change? (allowed: name, description, confidentiality, integrity, availability)",
                )
            else:
                return _L(
                    l,
                    "Co chceš změnit? (povolené: name, description). Pozn.: CIA se u podpůrných aktiv neupravuje.",
                    "Čo chceš zmeniť? (povolené: name, description). Pozn.: CIA sa u podporných aktív neupravuje.",
                    "What would you like to change? (allowed: name, description). Note: CIA is not applicable for supporting assets.",
                )

        def _ask_for_target() -> AssetState:
            msg = _L(
                l,
                "Které aktivum chcete změnit? Mám vypsat seznam?",
                "Ktoré aktívum chcete zmeniť? Mám vypísať zoznam?",
                "Which asset do you want to change? Should I list assets?",
            )
            state.result = _ui_response(msg, extra={"agent_events": state.agent_events, "proposals": ["🧾 Vypsat aktiva"]})
            return state

        try:
            raw_msg = (state.message or "")

            # 0) If there is no explicit target and no UI selection, try safe fallback from current_asset
            _fallback_selection_from_current_asset_if_safe(state, raw_msg)

            # 1) Prefer explicit structured items (UI / JSON payload)
            items = state.update_items or []
            if not items:
                # Try to parse free JSON list from the message (backwards compat)
                try:
                    arr = json.loads(state.message or "[]")
                    if isinstance(arr, list) and arr:
                        items = [AssetUpdateItem(**x) for x in arr if isinstance(x, dict)]
                except Exception:
                    pass

            # 2) If still nothing structured → use LLM intent extractor
            if not items:
                intent = extract_update_asset_intent(
                    deps=deps, text=(state.message or ""), locale=(state.locale or "cs")
                )

                if (intent.get("intent") or "none") != "update_asset":
                    # If user actually wanted detail → delegate to read node
                    if intent.get("intent") == "get_asset":
                        from .nodes_read import asset_get_or_preview
                        finish(state, t0, tool="asset.update_meta", ok=True, meta={"delegated": "asset.get"})
                        return asset_get_or_preview(deps)(state)

                    # If there is a selected (or safely-fallen-back) asset, ask for fields; else ask for target
                    sel_id = (state.selection or {}).get("asset_id")
                    if sel_id:
                        cur = _get_asset_by_id(deps, sel_id, getattr(state, "org_id", None))
                        if cur:
                            state.result = _ui_response(
                                _prompt_for_fields(bool(_boolish(cur.get("isPrimary")))),
                                extra={"agent_events": state.agent_events},
                            )
                            finish(state, t0, tool="asset.update_meta", ok=True, meta={"prompt": "fields"})
                            return state

                    finish(state, t0, tool="asset.update_meta", ok=True, meta={"prompt": "target"})
                    return _ask_for_target()

                # intent == update_asset → resolve target among existing assets
                target_id = intent.get("target_id")
                target_name = intent.get("target_name")

                cur = _get_asset_by_id(deps, target_id, getattr(state, "org_id", None)) if target_id else None
                if not cur and target_name:
                    cur, err = _get_asset_by_name_or_type_unique(deps, target_name, getattr(state, "org_id", None), state.locale)
                    if err:
                        state.result = _ui_response(err, extra={"agent_events": state.agent_events})
                        finish(state, t0, tool="asset.update_meta", ok=False, meta={"error": "target_resolution", "detail": "ambiguous_or_missing"})
                        return state

                if not cur:
                    # fallback to (possibly injected) selection
                    sel_id = (state.selection or {}).get("asset_id")
                    cur = _get_asset_by_id(deps, sel_id, getattr(state, "org_id", None)) if sel_id else None

                if not cur:
                    finish(state, t0, tool="asset.update_meta", ok=True, meta={"prompt": "target"})
                    return _ask_for_target()

                # Persist current selection for subsequent steps
                try:
                    sid = cur.get("id") or cur.get("asset_id")
                    if sid:
                        state.selection = state.selection or {}
                        state.selection["asset_id"] = sid
                        if cur.get("name"):
                            state.selection["asset_name"] = cur["name"]
                except Exception:
                    pass

                # Build one synthetic update item from LLM 'changes'
                ch = (intent.get("changes") or {})
                # Forbid assetType / isPrimary in this step
                for forbidden in ("assetType", "isPrimary"):
                    if forbidden in ch:
                        state.result = _ui_response(
                            _L(
                                l,
                                f"`{forbidden}` nelze měnit v tomto kroku.",
                                f"`{forbidden}` nie je možné meniť v tomto kroku.",
                                f"`{forbidden}` cannot be changed in this step.",
                            ),
                            extra={"agent_events": state.agent_events},
                        )
                        finish(state, t0, tool="asset.update_meta", ok=False, meta={"error": "forbidden_field", "field": forbidden})
                        return state

                it_dict = {
                    "asset_id": (cur.get("id") or cur.get("asset_id")),
                    "name": (ch.get("name") or None),
                    "description": (ch.get("description") or None),
                    "confidentiality": (ch.get("confidentiality") or None),
                    "integrity": (ch.get("integrity") or None),
                    "availability": (ch.get("availability") or None),
                }
                items = [AssetUpdateItem(**it_dict)]

            # 3) Unified execution path for structured items
            actions: list[dict] = []
            ignored_notes: list[str] = []

            for it in items:
                asset_id = getattr(it, "asset_id", None)
                cur = _get_asset_by_id(deps, asset_id, getattr(state, "org_id", None))
                if not cur:
                    ignored_notes.append(
                        _L(
                            l,
                            f"Aktivum s id '{asset_id}' nenalezeno.",
                            f"Aktívum s id '{asset_id}' nebolo nájdené.",
                            f"Asset with id '{asset_id}' not found.",
                        )
                    )
                    continue

                # assetType/isPrimary are forbidden here (use separate flow)
                for forbidden in ("assetType", "isPrimary"):
                    if hasattr(it, forbidden) and getattr(it, forbidden) is not None:
                        state.result = _ui_response(
                            _L(
                                l,
                                f"`{forbidden}` nelze měnit v tomto kroku.",
                                f"`{forbidden}` nie je možné meniť v tomto kroku.",
                                f"`{forbidden}` cannot be changed in this step.",
                            ),
                            extra={"agent_events": state.agent_events},
                        )
                        finish(state, t0, tool="asset.update_meta", ok=False, meta={"error": "forbidden_field", "field": forbidden})
                        return state
                    if hasattr(it, "fields") and isinstance(getattr(it, "fields"), dict) and forbidden in getattr(it, "fields"):
                        state.result = _ui_response(
                            _L(
                                l,
                                f"`{forbidden}` nelze měnit v tomto kroku.",
                                f"`{forbidden}` nie je možné meniť v tomto kroku.",
                                f"`{forbidden}` cannot be changed in this step.",
                            ),
                            extra={"agent_events": state.agent_events},
                        )
                        finish(state, t0, tool="asset.update_meta", ok=False, meta={"error": "forbidden_field_in_fields", "field": forbidden})
                        return state

                # Base fields
                fields = {
                    "name": (getattr(it, "name", None) or "").strip() or None,
                    "description": (getattr(it, "description", None) or None),
                }

                # CIA only for primary assets
                primary = bool(_boolish(cur.get("isPrimary")))
                if primary:
                    c_raw = getattr(it, "confidentiality", None)
                    i_raw = getattr(it, "integrity", None)
                    a_raw = getattr(it, "availability", None)

                    c = _canon_conf(c_raw)
                    i = _canon_tri(i_raw)
                    a = _canon_tri(a_raw)

                    if c_raw is not None and c is None:
                        state.result = _ui_response(
                            _L(
                                l,
                                f"Neplatná hodnota confidentiality. Povoleno: {', '.join(CONF_LEVELS)}.",
                                f"Neplatná hodnota confidentiality. Povolené: {', '.join(CONF_LEVELS)}.",
                                f"Invalid confidentiality. Allowed: {', '.join(CONF_LEVELS)}.",
                            ),
                            extra={"agent_events": state.agent_events},
                        )
                        finish(state, t0, tool="asset.update_meta", ok=False, meta={"error": "invalid_confidentiality"})
                        return state
                    if i_raw is not None and i is None:
                        state.result = _ui_response(
                            _L(
                                l,
                                f"Neplatná hodnota integrity. Povoleno: {', '.join(TRI_LEVELS)}.",
                                f"Neplatná hodnota integrity. Povolené: {', '.join(TRI_LEVELS)}.",
                                f"Invalid integrity. Allowed: {', '.join(TRI_LEVELS)}.",
                            ),
                            extra={"agent_events": state.agent_events},
                        )
                        finish(state, t0, tool="asset.update_meta", ok=False, meta={"error": "invalid_integrity"})
                        return state
                    if a_raw is not None and a is None:
                        state.result = _ui_response(
                            _L(
                                l,
                                f"Neplatná hodnota availability. Povoleno: {', '.join(TRI_LEVELS)}.",
                                f"Neplatná hodnota availability. Povolené: {', '.join(TRI_LEVELS)}.",
                                f"Invalid availability. Allowed: {', '.join(TRI_LEVELS)}.",
                            ),
                            extra={"agent_events": state.agent_events},
                        )
                        finish(state, t0, tool="asset.update_meta", ok=False, meta={"error": "invalid_availability"})
                        return state

                    if c is not None:
                        fields["confidentiality"] = c
                    if i is not None:
                        fields["integrity"] = i
                    if a is not None:
                        fields["availability"] = a
                else:
                    # Secondary/supporting assets: CIA cannot be set
                    for key in ("confidentiality", "integrity", "availability"):
                        if getattr(it, key, None) is not None:
                            ignored_notes.append(
                                _L(
                                    l,
                                    f"U podpůrného aktiva (id {asset_id}) není možné nastavit {key}.",
                                    f"U podporného aktíva (id {asset_id}) nie je možné nastaviť {key}.",
                                    f"For a supporting asset (id {asset_id}), {key} cannot be set.",
                                )
                            )

                # Only keep non-None fields
                fields = {k: v for k, v in fields.items() if v is not None}
                if fields:
                    actions.append({"op": "set_asset_meta", "asset_id": asset_id, "fields": fields})

            if not actions:
                msg = _L(
                    l,
                    "Žádné použitelné změny (povoleno: name, description; u primárních aktiv navíc confidentiality, integrity, availability).",
                    "Žiadne použiteľné zmeny (povolené: name, description; u primárnych aktív navyše confidentiality, integrity, availability).",
                    "No applicable changes (allowed: name, description; plus CIA only for primary assets).",
                )
                if ignored_notes:
                    msg += "\n\n" + "\n".join(f"- {n}" for n in ignored_notes[:6])
                state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
                finish(state, t0, tool="asset.update_meta", ok=False, meta={"error": "no_applicable_changes"})
                return state

            # Validate & write through outbox path
            validated, err = validate_write_actions(actions)
            if err:
                state.result = _ui_response({"error": "validation", "reason": err}, extra={"agent_events": state.agent_events})
                finish(state, t0, tool="asset.update_meta", ok=False, meta={"error": "validation", "reason": err})
                return state

            result_state = await apply_actions_async(deps=deps, state=state, acts=validated)
            if ignored_notes:
                try:
                    # Best-effort append informational notes into the UI stream
                    result_state.result["messages"].append("\n".join(f"- {n}" for n in ignored_notes[:6]))  # type: ignore[index]
                except Exception:
                    pass
            finish(state, t0, tool="asset.update_meta", ok=True, meta={"writes": len(validated)})
            return result_state

        except Exception as e:
            # Defensive failure envelope
            try:
                state.result = _ui_response(
                    _L(l, "_Chyba při aktualizaci aktiva._", "_Chyba pri aktualizácii aktíva._", "_Failed to update asset._"),
                    extra={"agent_events": state.agent_events, "error": str(e)},
                )
            except Exception:
                state.result = _ui_response({"error": "asset.update_meta failed", "reason": str(e)}, extra={"agent_events": state.agent_events})
            finish(state, t0, tool="asset.update_meta", ok=False, meta={"error": str(e)})
            return state

    return _inner


# -----------------------------------------------------------------------------
# Public entry points (aliases)
# -----------------------------------------------------------------------------
def update_meta(deps: GraphDeps):
    """Unified public entry (kept for backward compatibility)."""
    return _update_or_set_meta_inner(deps)


def set_meta(deps: GraphDeps):
    """Alias to the same unified handler (legacy name)."""
    return _update_or_set_meta_inner(deps)
