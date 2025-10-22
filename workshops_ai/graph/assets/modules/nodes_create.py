# File: workshops_ai/graph/assets/modules/nodes_create.py
from __future__ import annotations
import uuid
from typing import Any

from workshops_ai.graph.factory import GraphDeps
from .state import AssetState
from .text_utils import lang, extract_name_from_text, is_yes_like, is_no_like
from .enums_json import CONF_LEVELS, TRI_LEVELS, norm_enum, conf_from_text, tri_from_text
from .llm_utils import build_next_question, extract_create_draft_with_llm
from .ui_helpers import ui_response as _ui_response
from .write_utils import apply_actions_async, validate_write_actions
from .catalog_utils import fetch_asset_types_raw, normalize_type_labels  # source of truth for catalog types


# -----------------------------------------------------------------------------
# Locale helper
# -----------------------------------------------------------------------------
def _L(l: str | None, cs: str, sk: str, en: str) -> str:
    """Return localized string based on normalized locale ('cs' | 'sk' | other→'en')."""
    return cs if l == "cs" else sk if l == "sk" else en


# -----------------------------------------------------------------------------
# Normalizers shim (kept local to avoid new hard deps; uses project utilities when present)
# -----------------------------------------------------------------------------
try:
    from workshops_ai.utils.normalize import (
        normalize_asset_create_fields as _normalize_create_fields,
        normalize_asset_name as _normalize_name,
        normalize_asset_type as _normalize_type,
        normalize_is_primary as _normalize_is_primary,
    )
except Exception:  # pragma: no cover
    # Minimal in-module fallbacks (keep behavior predictable without central helpers).

    def _normalize_create_fields(fields: dict, catalog_types: list[str] | None = None) -> dict:
        """Best-effort normalization; identity if central helpers are unavailable."""
        return fields

    def _normalize_name(name: str | None) -> str | None:
        """Trim and collapse empty to None."""
        return (name or "").strip() or None

    def _normalize_type(typ: str | None, allowed: list[str] | None = None) -> str | None:
        """Trim + case-insensitive allow-list match."""
        t = (typ or "").strip()
        if not t:
            return None
        if not allowed:
            return t
        return next((x for x in allowed if x.lower() == t.lower()), None)

    def _normalize_is_primary(value: Any) -> bool | None:
        """Map common affirmations/negations to bool; None if uncertain."""
        if isinstance(value, bool):
            return value
        s = str(value or "").strip().lower()
        pos = {"ano", "áno", "j", "y", "yes", "true", "1", "ok", "okay", "confirm", "confirmed", "sure", "primary"}
        neg = {"ne", "nie", "n", "no", "false", "0", "nope", "neni", "není"}
        if s in pos:
            return True
        if s in neg:
            return False
        return None


# -----------------------------------------------------------------------------
# Node: create_asset (slot-filling with catalog-backed type selection)
# -----------------------------------------------------------------------------
def create_asset(deps: GraphDeps):
    async def _inner(state: AssetState) -> AssetState:
        """
        Conversational create flow (slot-filling):
          1) If `assetType` is missing, we collect it first (list types via catalog).
          2) Then: name → isPrimary → (if primary) CIA → description.
          3) Outside staging, ask for a short explicit confirmation before write,
             unless the user already confirmed with a yes-like message.

        Design notes:
          - Users won't paste IDs here; we don't rely on them.
          - Be robust to multiple fields in one sentence (LLM-backed extraction).
          - Preserve and reuse state.create_draft and state.pending_fields.
        """
        state.create_flow = True
        state.intent = state.intent or "create"

        l = lang(state.locale)
        last_msg = (state.message or "").strip()

        # --- Load current draft and pending fields from state ----------------
        draft = dict(state.create_draft or {})
        raw_pending = state.pending_fields
        pending: list[str] = list(raw_pending or [])

        # --- Lazy catalog types fetch with normalization ---------------------
        _catalog_types_cache: list[str] | None = None

        def catalog_types() -> list[str]:
            """Return normalized catalog type labels; never raises."""
            nonlocal _catalog_types_cache
            if _catalog_types_cache is None:
                try:
                    raw = fetch_asset_types_raw(deps, getattr(state, "org_id", None))
                except Exception:
                    raw = []
                try:
                    _catalog_types_cache = normalize_type_labels(raw)
                except Exception:
                    # String-coercion fallback keeps us safe if catalog helpers drift.
                    _catalog_types_cache = [str(x).strip() for x in (raw or []) if str(x).strip()]
            return _catalog_types_cache

        # If user already provided a type in the draft, align casing with catalog
        asset_type_known = bool((draft.get("assetType") or "").strip())
        if asset_type_known:
            ct = catalog_types()
            for t in ct:
                if t.lower() == str(draft["assetType"]).lower():
                    draft["assetType"] = t
                    break

        # --- Loop guard: prevent infinite ping-pong on the same step ----------
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
                # Fail safe: stop the flow cleanly and reset transient state.
                state.create_flow = False
                state.pending_fields = None
                state.create_draft = None
                note = _L(
                    l,
                    "Zastaveno kvůli opakující se smyčce. Zkus prosím znovu zadat typ nebo název.",
                    "Zastavené pre opakujúcu sa slučku. Skús prosím znova zadať typ alebo názov.",
                    "Stopped due to a repeating loop. Please provide the type or the name again.",
                )
                state.result = _ui_response(note, extra={"agent_events": state.agent_events})
                return state
        else:
            state.once_flags["create_step_hash"] = sig
            state.once_flags["create_step_repeat"] = 0

        # --- Fast-path confirmation when draft looks complete -----------------
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
            # Accept lightweight confirmations cross-language.
            if is_yes_like(last_msg) or (last_msg.lower().strip() in {
                "potvrdit vytvoření", "potvrdit vytvoreni", "potvrdiť vytvorenie",
                "confirm create", "confirm creation", "ok"
            }):
                state.once_flags = dict(state.once_flags or {})
                state.once_flags["one_shot_write"] = True
                skip_merge = True
            elif is_no_like(last_msg):
                # Cancel the flow
                state.create_flow = False
                state.pending_fields = None
                state.create_draft = None
                state.result = _ui_response(
                    _L(l, "Zrušeno.", "Zrušené.", "Cancelled."),
                    extra={"agent_events": state.agent_events},
                )
                return state
            else:
                # Ask for explicit confirmation (keep it short and clear).
                typ = draft.get("assetType") or "?"
                nm = draft.get("name") or "?"
                prim_txt = _L(l, "primární", "primárny", "primary") if draft.get("isPrimary") \
                    else _L(l, "neprimární", "neprimárny", "supporting")
                confirm_msg = _L(
                    l,
                    f"Chcete vytvořit aktivum {typ} s názvem „{nm}“, {prim_txt}? Odpověz „Potvrdit vytvoření“ nebo prostě „OK“.",
                     f"Chcete vytvoriť aktívum {typ} s názvom „{nm}“, {prim_txt}? Odpovedz „Potvrdiť vytvorenie“ alebo jednoducho „OK“.",
                    f"Create asset {typ} named “{nm}”, {prim_txt}? Reply “OK” to confirm.",
                )
                state.result = _ui_response(confirm_msg, extra={"agent_events": state.agent_events})
                return state

        # --- Merge user reply into the draft based on pending field -----------
        if not skip_merge:
            # Early capture: if the user just sent a type, normalize and store it.
            if not asset_type_known and last_msg:
                typ_guess = _normalize_type(last_msg, allowed=catalog_types())
                if typ_guess:
                    draft["assetType"] = typ_guess
                    asset_type_known = True
                    if pending == ["assetType"]:
                        pending = []

            # 1) isPrimary slot (Yes/No)
            if pending == ["isPrimary"]:
                ip = _normalize_is_primary(last_msg)
                if ip is not None:
                    draft["isPrimary"] = ip
                    pending = []
                else:
                    msg = _L(
                        l,
                        "Je aktivum primární? Odpověz Ano/Ne.",
                        "Je aktívum primárne? Odpovedz Áno/Nie.",
                        "Is the asset primary? Please answer Yes/No.",
                    )
                    state.create_draft = draft
                    state.pending_fields = ["isPrimary"]
                    state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
                    return state

            # 2) CIA slots (only for primary assets)
            elif len(pending) == 1 and pending[0] in {"confidentiality", "integrity", "availability"}:
                slot = pending[0]
                val = conf_from_text(last_msg) if slot == "confidentiality" else tri_from_text(last_msg)
                if val:
                    draft[slot] = val
                    pending = []
                else:
                    # Ask again with localized guidance
                    slot_cs = {"confidentiality": "důvěrnost", "integrity": "integritu", "availability": "dostupnost"}[slot]
                    slot_sk = {"confidentiality": "dôvernosť", "integrity": "integritu", "availability": "dostupnosť"}[slot]
                    if slot == "confidentiality":
                        msg = _L(
                            l,
                            "Zadej důvěrnost: PUBLIC/INTERNAL/PROTECTED/STRICTLY_PROTECTED.",
                            "Zadaj dôvernosť: PUBLIC/INTERNAL/PROTECTED/STRICTLY_PROTECTED.",
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

            # 3) name slot
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

            # 4) description slot
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

            # 5) General merge via bounded LLM extraction (best-effort)
            else:
                inc = extract_create_draft_with_llm(
                    deps=deps,
                    text=last_msg,
                    asset_types=[] if asset_type_known else catalog_types(),
                ) if last_msg else {}

                # If router put us on a very specific step, restrict the merge to that key.
                allowed_keys = set(pending) if (raw_pending is not None) else None

                prev_type = draft.get("assetType")
                # Never let the extractor overturn an already chosen type.
                if prev_type and inc.get("assetType") and inc["assetType"] != prev_type:
                    inc.pop("assetType", None)

                for k, v in (inc or {}).items():
                    if v is None:
                        continue
                    if allowed_keys is not None and k not in allowed_keys:
                        continue
                    draft[k] = v

                # If we were waiting for isPrimary and extractor missed it, try a simple yes/no parse.
                if "isPrimary" in (raw_pending or []) and draft.get("isPrimary") is None:
                    ip = _normalize_is_primary(last_msg)
                    if ip is not None:
                        draft["isPrimary"] = ip

        # --- Normalize the draft (name/type casing, booleans, enums) ----------
        if draft.get("name") is not None:
            draft["name"] = _normalize_name(draft.get("name"))

        if draft.get("assetType") is not None or not asset_type_known:
            draft["assetType"] = _normalize_type(draft.get("assetType"), allowed=catalog_types())

        if draft.get("isPrimary") is not None and not isinstance(draft.get("isPrimary"), bool):
            draft["isPrimary"] = _normalize_is_primary(draft["isPrimary"])

        # Best-effort overall normalization (non-fatal on failure)
        try:
            draft = _normalize_create_fields(draft, catalog_types=catalog_types())
        except Exception:
            pass

        # --- Compute missing fields in canonical order ------------------------
        def _miss(k: str) -> bool:
            return (draft.get(k) in (None, "", []) or k not in draft)

        order = ["assetType", "name", "isPrimary", "confidentiality", "integrity", "availability", "description"]
        missing = [k for k in order if _miss(k)]

        # If the asset is supporting (not primary), CIA is not required
        if draft.get("isPrimary") is False:
            missing = [k for k in missing if k not in ("confidentiality", "integrity", "availability")]

        # If anything is missing, ask the next question (with catalog help on type)
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

        # --- Finalize CIA for primary vs supporting ---------------------------
        if draft.get("isPrimary") is False:
            # Ensure CIA is cleared for supporting assets (validation expects them absent).
            draft["confidentiality"] = None
            draft["integrity"] = None
            draft["availability"] = None
        else:
            # Canonicalize enums for primary assets
            draft["confidentiality"] = norm_enum(draft.get("confidentiality"), allowed=CONF_LEVELS)
            draft["integrity"]       = norm_enum(draft.get("integrity"),       allowed=TRI_LEVELS)
            draft["availability"]    = norm_enum(draft.get("availability"),    allowed=TRI_LEVELS)

        # --- Build write action payload --------------------------------------
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
        # Strip None values; keeps payload minimal and validation-friendly.
        fields = {k: v for k, v in fields.items() if v is not None}

        actions = [{
            "op": "create_asset",
            "client_token": f"tmp-{uuid.uuid4().hex[:10]}",
            "fields": fields,
        }]

        # Validate action set before write
        validated, err = validate_write_actions(actions)
        if err:
            state.result = _ui_response({"error": "validation", "reason": err}, extra={"agent_events": state.agent_events})
            return state

        # --- Confirmation gate (when not in staging) --------------------------
        write_allowed = True
        if state.staging is False:
            write_allowed = bool((state.once_flags or {}).get("one_shot_write", False))

        if not write_allowed:
            typ = fields.get("assetType") or "?"
            nm = fields.get("name") or "?"
            prim_txt = _L(l, "primární", "primárny", "primary") if fields.get("isPrimary") is True \
                else _L(l, "neprimární", "neprimárny", "supporting")
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

        # --- Perform the write (inline submit) --------------------------------
        result_state = await apply_actions_async(deps=deps, state=state, acts=validated)

        # Cleanup flow state after success (leave the turn in a clean state)
        result_state.create_flow = False
        result_state.pending_fields = None
        result_state.create_draft = None
        return result_state

    return _inner
