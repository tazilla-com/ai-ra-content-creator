# workshops_ai/graph/assets/modules/nodes_create_update.py
from __future__ import annotations
import json, uuid
from typing import Any
from workshops_ai.graph.factory import GraphDeps
from .state import AssetState
from .text_utils import lang, extract_name_from_text, is_yes_like, is_no_like
from .enums_json import CONF_LEVELS, TRI_LEVELS, norm_enum, conf_from_text, tri_from_text
from .llm_utils import build_next_question, extract_create_draft_with_llm
from .ui_helpers import ui_response as _ui_response
from .write_utils import apply_actions_async, validate_write_actions


# -------------------------- locale helpers -----------------------------------
def _L(l: str | None, cs: str, sk: str, en: str) -> str:
    """Locale switcher: 'cs' -> cs, 'sk' -> sk, else -> en."""
    return cs if l == "cs" else sk if l == "sk" else en


# ------------------------ normalizers shim -----------------------------------
# Keep backward-compatible behavior even if normalize utils are not present.
try:
    from workshops_ai.utils.normalize import (
        normalize_asset_create_fields as _normalize_create_fields,
        normalize_asset_name as _normalize_name,
        normalize_asset_type as _normalize_type,
        normalize_is_primary as _normalize_is_primary,
    )
except Exception:  # pragma: no cover
    def _normalize_create_fields(fields: dict, catalog_types: list[str] | None = None) -> dict:
        return fields

    def _normalize_name(name: str | None) -> str | None:
        return (name or "").strip() or None

    def _normalize_type(typ: str | None, allowed: list[str] | None = None) -> str | None:
        t = (typ or "").strip()
        if not t:
            return None
        if not allowed:
            return t
        return next((x for x in allowed if x.lower() == t.lower()), None)

    def _normalize_is_primary(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        s = str(value or "").strip().lower()
        pos = {"ano", "áno", "jo", "jj", "j", "y", "yes", "true", "1", "ok", "okay", "confirm", "confirmed", "sure", "primary"}
        neg = {"ne", "nie", "n", "no", "false", "0", "nope", "neni", "není"}
        if s in pos:
            return True
        if s in neg:
            return False
        return None


# ------------------------------- create --------------------------------------
def create_asset(deps: GraphDeps):
    async def _inner(state: AssetState) -> AssetState:
        # Hard-lock into the create flow; the router reads state.create_flow.
        state.create_flow = True
        state.intent = state.intent or "create"

        l = lang(state.locale)  # 'cs' | 'sk' | else -> 'en'
        last_msg = (state.message or "").strip()

        # Working draft and pending slots
        draft = dict(state.create_draft or {})
        raw_pending = state.pending_fields
        pending: list[str] = list(raw_pending or [])

        # Lazy fetch of catalog types (to reduce UI noise and backend calls)
        _catalog_types_cache: list[str] | None = None

        def catalog_types() -> list[str]:
            nonlocal _catalog_types_cache
            if _catalog_types_cache is None:
                _catalog_types_cache = [str(x) for x in (deps.backend_port.list_asset_types() or [])]
            return _catalog_types_cache

        # If user already supplied a type, keep canonical casing (fetch only if needed)
        asset_type_known = bool((draft.get("assetType") or "").strip())
        if asset_type_known:
            ct = catalog_types()
            for t in ct:
                if t.lower() == str(draft["assetType"]).lower():
                    draft["assetType"] = t
                    break

        # Small loop protection: if the same step keeps repeating, stop politely
        state.once_flags = dict(state.once_flags or {})
        core_sig = (
            tuple(pending),
            draft.get("assetType"), draft.get("name"), draft.get("isPrimary"),
            draft.get("confidentiality"), draft.get("integrity"), draft.get("availability"),
        )
        sig = hash(("create", core_sig))
        if state.once_flags.get("create_step_hash") == sig:
            cnt = int(state.once_flags.get("create_step_repeat", 0)) + 1
            state.once_flags["create_step_repeat"] = cnt
            if cnt >= 2:
                state.create_flow = False
                state.pending_fields = None
                note = _L(
                    l,
                    "Zastaveno kvůli opakující se smyčce. Zkus prosím znovu zadat typ nebo název.",
                    "Zastavené pre opakujúcu sa slučku. Skús, prosím, znova zadať typ alebo názov.",
                    "Stopped due to a repeating loop. Please provide the type or the name again.",
                )
                state.result = _ui_response(note, extra={"agent_events": state.agent_events})
                return state
        else:
            state.once_flags["create_step_hash"] = sig
            state.once_flags["create_step_repeat"] = 0

        # If everything is known and staging is off, require explicit confirmation
        waiting_confirm = (
            state.staging is False
            and bool(draft)
            and pending == []
            and isinstance(draft.get("name"), str) and draft.get("name", "").strip()
            and isinstance(draft.get("assetType"), str) and draft.get("assetType", "").strip()
            and isinstance(draft.get("isPrimary"), bool)
        )
        skip_merge = False

        if waiting_confirm:
            if is_yes_like(last_msg) or (last_msg.lower().strip() in {
                "potvrdit vytvoření", "potvrdit vytvoreni", "potvrdiť vytvorenie", "confirm create", "confirm creation"
            }):
                state.once_flags = dict(state.once_flags or {})
                state.once_flags["one_shot_write"] = True
                skip_merge = True
            elif is_no_like(last_msg):
                state.create_flow = False
                state.pending_fields = None
                state.create_draft = None
                cancel_msg = _L(l, "Zrušeno.", "Zrušené.", "Cancelled.")
                state.result = _ui_response(cancel_msg, extra={"agent_events": state.agent_events})
                return state
            else:
                typ = draft.get("assetType") or "?"
                nm = draft.get("name") or "?"
                prim_txt = _L(l, "primární", "primárny", "primary") if draft.get("isPrimary") else _L(l, "neprimární", "neprimárny", "supporting")
                confirm_msg = _L(
                    l,
                    f"Chcete vytvořit aktivum {typ} s názvem „{nm}“, {prim_txt}? Odpověz „Potvrdit vytvoření“ nebo prostě „OK“.",
                    f"Chcete vytvoriť aktívum {typ} s názvom „{nm}“, {prim_txt}? Odpovedz „Potvrdiť vytvorenie“ alebo jednoducho „OK“.",
                    f"Create asset {typ} named “{nm}”, {prim_txt}? Reply “OK” to confirm.",
                )
                state.result = _ui_response(confirm_msg, extra={"agent_events": state.agent_events})
                return state

        # Merge user input according to pending slot(s)
        if not skip_merge:
            if pending == ["isPrimary"]:
                ip = _normalize_is_primary(last_msg)
                if ip is not None:
                    draft["isPrimary"] = ip
                    pending = []
                else:
                    msg = _L(l, "Je aktivum primární? Odpověz Ano/Ne.", "Je aktívum primárne? Odpovedz Áno/Nie.", "Is the asset primary? Please answer Yes/No.")
                    state.create_draft = draft
                    state.pending_fields = ["isPrimary"]
                    state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
                    return state

            elif len(pending) == 1 and pending[0] in {"confidentiality", "integrity", "availability"}:
                slot = pending[0]
                val = conf_from_text(last_msg) if slot == "confidentiality" else tri_from_text(last_msg)
                if val:
                    draft[slot] = val
                    pending = []
                else:
                    slot_cs = {"confidentiality": "důvěrnost", "integrity": "integritu", "availability": "dostupnost"}[slot]
                    slot_sk = {"confidentiality": "dôvernosť", "integrity": "integritu", "availability": "dostupnosť"}[slot]
                    if slot == "confidentiality":
                        msg = _L(
                            l,
                            f"Zadej {slot_cs}: PUBLIC/INTERNAL/PROTECTED/STRICTLY_PROTECTED.",
                            f"Zadaj {slot_sk}: PUBLIC/INTERNAL/PROTECTED/STRICTLY_PROTECTED.",
                            "Provide confidentiality: PUBLIC/INTERNAL/PROTECTED/STRICTLY_PROTECTED.",
                        )
                    else:
                        msg = _L(
                            l,
                            f"Zadej {slot_cs}: LOW/MEDIUM/HIGH.",
                            f"Zadaj {slot_sk}: LOW/MEDIUM/HIGH.",
                            f"Provide {slot}: LOW/MEDIUM/HIGH.",
                        )
                    state.create_draft = draft
                    state.pending_fields = [slot]
                    state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
                    return state

            elif pending == ["name"]:
                nm = extract_name_from_text(last_msg) or last_msg.strip().strip('"').strip("'")
                if nm:
                    draft["name"] = _normalize_name(nm)
                    pending = []
                else:
                    msg = _L(l, "Zadej název aktiva.", "Zadaj názov aktíva.", "Please provide the asset name.")
                    state.create_draft = draft
                    state.pending_fields = ["name"]
                    state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
                    return state

            elif pending == ["description"]:
                desc = (last_msg or "").strip()
                if desc:
                    draft["description"] = desc
                    pending = []
                else:
                    msg = _L(l, "Stručně popiš aktivum.", "Stručne popíš aktívum.", "Please provide a short description.")
                    state.create_draft = draft
                    state.pending_fields = ["description"]
                    state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
                    return state

            else:
                # Wide parse via LLM only for still-missing fields (esp. type)
                types_for_llm: list[str] = [] if asset_type_known else catalog_types()
                inc = extract_create_draft_with_llm(deps=deps, text=last_msg, asset_types=types_for_llm) if last_msg else {}
                allowed_keys = set(pending) if (raw_pending is not None) else None
                prev_type = draft.get("assetType")

                # Do not override assetType if it was already set earlier
                if prev_type and inc.get("assetType") and inc["assetType"] != prev_type:
                    inc.pop("assetType", None)

                for k, v in (inc or {}).items():
                    if v is None:
                        continue
                    if allowed_keys is not None and k not in allowed_keys:
                        continue
                    draft[k] = v

                # Special case for isPrimary typed as text
                if "isPrimary" in (raw_pending or []) and draft.get("isPrimary") is None:
                    ip = _normalize_is_primary(last_msg)
                    if ip is not None:
                        draft["isPrimary"] = ip

        # Normalize draft values (fetch catalog types only when needed)
        if draft.get("name") is not None:
            draft["name"] = _normalize_name(draft.get("name"))

        if draft.get("assetType") is not None or not asset_type_known:
            draft["assetType"] = _normalize_type(draft.get("assetType"), allowed=catalog_types())

        if draft.get("isPrimary") is not None and not isinstance(draft.get("isPrimary"), bool):
            draft["isPrimary"] = _normalize_is_primary(draft["isPrimary"])

        try:
            draft = _normalize_create_fields(draft, catalog_types=catalog_types())
        except Exception:
            pass

        # If type is still unknown or invalid -> delegate to list_asset_types (stay in create flow)
        if not draft.get("assetType") or draft["assetType"] not in catalog_types():
            state.create_draft = draft
            state.pending_fields = ["assetType"]
            state.create_flow = True
            from .nodes_read import list_asset_types
            return list_asset_types(deps)(state)

        # Compute missing fields in fixed order
        def _miss(k: str) -> bool:
            return (draft.get(k) in (None, "", []) or k not in draft)

        order = ["assetType", "name", "isPrimary", "confidentiality", "integrity", "availability", "description"]
        missing = [k for k in order if _miss(k)]
        if draft.get("isPrimary") is False:
            # Supporting assets do not require CIA
            missing = [k for k in missing if k not in ("confidentiality", "integrity", "availability")]

        if missing:
            target = missing[0]
            q = build_next_question(
                deps=deps,
                draft=draft,
                asset_types=(catalog_types() if _miss("assetType") else []),
                locale=(state.locale or "en"),
            )
            state.create_draft = draft
            state.pending_fields = [target]
            state.create_flow = True
            state.result = _ui_response(
                q or _L(l, "Doplň prosím další údaj.", "Doplň prosím ďalší údaj.", "Provide the next value."),
                extra={"agent_events": state.agent_events},
            )
            return state

        # Finalize CIA for primary assets only
        if draft.get("isPrimary") is False:
            draft["confidentiality"] = None
            draft["integrity"] = None
            draft["availability"] = None
        else:
            draft["confidentiality"] = norm_enum(draft.get("confidentiality"), allowed=CONF_LEVELS)
            draft["integrity"] = norm_enum(draft.get("integrity"), allowed=TRI_LEVELS)
            draft["availability"] = norm_enum(draft.get("availability"), allowed=TRI_LEVELS)

        # Prepare write action
        fields = {
            "id": draft.get("id"),
            "name": draft.get("name"),
            "assetType": draft.get("assetType"),
            "isPrimary": draft.get("isPrimary"),
            "confidentiality": draft.get("confidentiality"),
            "integrity": draft.get("integrity"),
            "availability": draft.get("availability"),
            "description": draft.get("description"),
        }
        fields = {k: v for k, v in fields.items() if v is not None}

        actions = [{"op": "create_asset", "client_token": f"tmp-{uuid.uuid4().hex[:10]}", "fields": fields}]
        validated, err = validate_write_actions(actions)
        if err:
            state.result = _ui_response({"error": "validation", "reason": err}, extra={"agent_events": state.agent_events})
            return state

        # When staging is False, require explicit one-shot confirmation in this node
        write_allowed = True
        if state.staging is False:
            write_allowed = bool((state.once_flags or {}).get("one_shot_write", False))

        if not write_allowed:
            typ = fields.get("assetType") or "?"
            nm = fields.get("name") or "?"
            prim_txt = _L(l, "primární", "primárny", "primary") if fields.get("isPrimary") is True else _L(l, "neprimární", "neprimárny", "supporting")
            confirm_msg = _L(
                l,
                f"Chcete vytvořit aktivum {typ} s názvem „{nm}“, {prim_txt}? Napište: Potvrdit vytvoření.",
                f"Chcete vytvoriť aktívum {typ} s názvom „{nm}“, {prim_txt}? Napíšte: Potvrdiť vytvorenie.",
                f"Create asset {typ} named “{nm}”, {prim_txt}? Type: Confirm creation.",
            )
            state.create_draft = draft
            state.pending_fields = []
            state.create_flow = True
            state.result = _ui_response(confirm_msg, extra={"agent_events": state.agent_events})
            return state

        # Submit write and unlock the flow on success
        result_state = await apply_actions_async(deps=deps, state=state, acts=validated)
        result_state.create_flow = False
        result_state.pending_fields = None
        result_state.create_draft = None
        return result_state

    return _inner


# ------------------------------ update meta ----------------------------------
def update_meta(deps: GraphDeps):
    async def _inner(state: AssetState) -> AssetState:
        from .state import AssetUpdateItem
        l = lang(state.locale)

        items = state.update_items or []
        if not items:
            try:
                arr = json.loads(state.message or "[]")
                if isinstance(arr, list):
                    items = [AssetUpdateItem(**x) for x in arr]
            except Exception:
                pass

        if not items:
            sel_id = (state.selection or {}).get("asset_id")
            if not sel_id:
                from .nodes_read import asset_get_or_preview
                return asset_get_or_preview(deps)(state)
            state.result = _ui_response(
                _L(l, "Uveď, co se má změnit (name/description).", "Uveď, čo sa má zmeniť (name/description).", "Specify what to change (name/description).")
            )
            return state

        actions: list[dict] = []
        for it in items:
            fields = {"name": _normalize_name(it.name), "description": it.description}
            fields = {k: v for k, v in fields.items() if v is not None}
            if fields:
                actions.append({"op": "set_asset_meta", "asset_id": it.asset_id, "fields": fields})

        if not actions:
            state.result = _ui_response(
                _L(l, "Žádné změny (povoleno je jen name/description).", "Žiadne zmeny (povolené je len name/description).", "No changes (only name/description allowed).")
            )
            return state

        validated, err = validate_write_actions(actions)
        if err:
            state.result = _ui_response({"error": "validation", "reason": err}, extra={"agent_events": state.agent_events})
            return state

        return await apply_actions_async(deps=deps, state=state, acts=validated)

    return _inner
