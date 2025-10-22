# workshops_ai/graph/assets/modules/write_utils.py
from __future__ import annotations

"""
Write-path helpers for the Assets graph.

This module centralizes:
- Validation of write action payloads (chat-layer safe surface)
- Building & submitting a ChangeSet (inline submit; no state.actions recursion)
- Best-effort WebSocket notifications after successful submit
- Small helpers: guessing created IDs from backend response, one-line LLM confirmation

All functions are side-effect–free except where noted (WS emits).
"""

import json
from typing import Any

from workshops_ai.graph.factory import GraphDeps

from .state import AssetState
from .ui_helpers import ui_response as _ui_response, now_iso
from .text_utils import lang
from .telemetry_utils import event, finish, schedule


def validate_write_actions(actions: list[dict]) -> tuple[list[dict], str | None]:
    """
    Validate chat-layer write actions into a canonical list, or return a reason string.

    Supported ops:
      - create_asset: requires name, assetType, isPrimary; CIA required iff isPrimary=True
      - set_asset_meta: (asset_id, {name?, description?})
      - set_asset_cia:  (asset_id, {confidentiality[1..4]?, integrity[1..3]?, availability[1..3]?})

    Returns:
      (validated_actions, None) on success
      ([], "reason") on failure
    """
    validated: list[dict] = []

    for a in actions:
        op = (a.get("op") or "").strip()
        try:
            if op == "create_asset":
                payload = {
                    "op": "create_asset",
                    "client_token": a.get("client_token"),
                    "fields": a.get("fields") or {},
                }
                f = payload["fields"]

                # Required fields
                if not (isinstance(f.get("name"), str) and f["name"].strip()):
                    return [], "create_asset requires fields.name"
                if not (isinstance(f.get("assetType"), str) and f["assetType"].strip()):
                    return [], "create_asset requires fields.assetType"
                if not isinstance(f.get("isPrimary"), bool):
                    return [], "create_asset requires fields.isPrimary (bool)"

                # CIA policy gating
                c, i, a_ = f.get("confidentiality"), f.get("integrity"), f.get("availability")
                if f["isPrimary"] is True:
                    if not (c and i and a_):
                        return [], "create_asset with isPrimary=True requires CIA"
                else:
                    # isPrimary=False → CIA must be absent
                    if any(x is not None for x in (c, i, a_)):
                        return [], "create_asset with isPrimary=False must not contain CIA"

                validated.append(payload)

            elif op == "set_asset_meta":
                fields = (a.get("fields") or {})
                illegal = set(fields.keys()) - {"name", "description"}
                if illegal:
                    return [], f"set_asset_meta: illegal fields for chat update: {sorted(illegal)}"

                payload = {
                    "op": "set_asset_meta",
                    "asset_id": a.get("asset_id"),
                    "fields": fields,
                }
                if not payload["asset_id"]:
                    return [], "set_asset_meta requires asset_id"
                if not payload["fields"]:
                    return [], "set_asset_meta requires at least one field (name/description)"

                validated.append(payload)

            elif op == "set_asset_cia":
                asset_id = a.get("asset_id")
                fields = (a.get("fields") or {})

                if not asset_id:
                    return [], "set_asset_cia requires asset_id"

                allowed = {"confidentiality", "integrity", "availability"}
                illegal = set(fields.keys()) - allowed
                if illegal:
                    return [], f"set_asset_cia: illegal fields: {sorted(illegal)}"

                # Numeric policy ranges
                for k, v in fields.items():
                    if v is None:
                        continue
                    if not isinstance(v, int):
                        return [], f"set_asset_cia: {k} must be int or null"
                    if k == "confidentiality":
                        if v < 1 or v > 4:
                            return [], "confidentiality must be 1..4 (or None)"
                    else:
                        if v < 1 or v > 3:
                            return [], f"{k} must be 1..3 (or None)"

                validated.append({"op": "set_asset_cia", "asset_id": asset_id, "fields": fields})

            else:
                return [], f"unsupported operation '{op}'"

        except Exception as e:
            # Defensive: surface a single-line reason instead of bubbling exceptions into the graph
            return [], f"write validation failed for op='{op}': {e}"

    return validated, None


async def apply_actions_async(*, deps: GraphDeps, state: AssetState, acts: list[dict]) -> AssetState:
    """
    Build & submit a validated ChangeSet; return an end-user–friendly UI response.

    Behavior:
      - Validates ops elsewhere (this function assumes `acts` are already normalized)
      - Builds Pydantic ChangeSet (adds audit & idempotency key on backend side)
      - Submits via deps.submit_changeset(...); OUTBOX fallback allowed
      - If queued: marks UI staging.needs_commit=True and explains in one sentence
      - Emits WS events on *submitted* path (not on queued)
      - Never raises into the graph; always sets `state.result`
    """
    loc = (state.locale or "en")

    # No-ops early exit
    if not acts:
        state.result = _ui_response("No actions to apply.", extra={"agent_events": state.agent_events})
        return state

    # Guard: submit function is present
    submit_fn = getattr(deps, "submit_changeset", None)
    if not callable(submit_fn):
        state.result = _ui_response({"error": "submit_changeset not available"}, extra={"agent_events": state.agent_events})
        return state

    # Telemetry start
    t0 = event(state, kind="tool.call.start", tool="submit_changeset", meta={"actions": len(acts)})

    # Back-compat: accept canonical ops and map to writer ops
    def _normalize_legacy_action(a: dict) -> dict:
        op_map = {
            "asset.create": "create_asset",
            "asset.update_meta": "set_asset_meta",
            "asset.set_meta": "set_asset_cia",
        }
        op = (a.get("op") or "").strip()
        return {**a, "op": op_map.get(op, op)}

    try:
        # Lazily import ChangeSet builder to avoid heavy imports at module load
        from workshops_ai.models.changeset import build_changeset
        try:
            # If Action DU (Pydantic v2) is available, validate strictly per-action
            from workshops_ai.models.changeset import Action  # type: ignore
            typed_actions = []
            for a in acts:
                a_norm = _normalize_legacy_action(a or {})
                try:
                    typed_actions.append(Action.model_validate(a_norm))  # type: ignore[attr-defined]
                except Exception:
                    # Be permissive on drift; backend will re-validate
                    typed_actions.append(a_norm)
        except Exception:
            # Fallback: pass normalized dicts into the builder (it will validate)
            typed_actions = [_normalize_legacy_action(a or {}) for a in acts]

        # Build ChangeSet (adds audit + idempotency key)
        cs_model = build_changeset(
            org_id=state.org_id,
            user_id=(state.user_id or "unknown"),
            actions=typed_actions,
            intent="apply",
        )
        # Safe echo for UI
        cs_dump = getattr(cs_model, "model_dump", lambda **k: cs_model)(mode="json")

        # Submit without requiring online success (allows OUTBOX)
        submit_result = await submit_fn(cs_model, require_success=False)

        # Compose one-line chat feedback (prefer central helper if present)
        try:
            from workshops_ai.utils.submit_feedback import apply_submit_outcome_and_message
            chat_msg = apply_submit_outcome_and_message(state=state, changeset=cs_model, result=submit_result)
        except Exception:
            chat_msg = llm_confirm(
                deps=deps,
                loc=loc,
                actions=acts,
                queued=bool(getattr(submit_result, "fallback_path", None)),
            )

        submitted = bool(getattr(submit_result, "submitted", False))
        queued = bool(getattr(submit_result, "fallback_path", None)) and not submitted

        # ---------------------
        # Submitted online
        # ---------------------
        if submitted:
            finish(state, t0, tool="submit_changeset", ok=True, meta={"actions": len(acts)})

            # Best-effort WS events (only on real submit, to avoid duplicates)
            hub = getattr(deps, "ws_hub", None)
            if hub:
                async def _emit_created_and_updated():
                    try:
                        from workshops_ai.app.services.change_events import (
                            aemit_asset_created,
                            aemit_asset_updated,
                        )
                    except Exception:
                        return

                    try:
                        created_ids = guess_created_ids(submit_result)  # ordered IDs from backend
                        created_idx = 0

                        for a in acts:
                            a_norm = _normalize_legacy_action(a or {})
                            op = str(a_norm.get("op") or "").strip()
                            fields = (a_norm.get("fields") or {})
                            asset_id_from_payload = (
                                a_norm.get("asset_id") or fields.get("asset_id") or fields.get("id")
                            )

                            # CREATE
                            if op in ("asset.create", "create_asset"):
                                asset_id = (
                                    created_ids[created_idx]
                                    if created_idx < len(created_ids) and created_ids[created_idx]
                                    else asset_id_from_payload
                                )
                                created_idx += 1
                                await aemit_asset_created(
                                    org_id=state.org_id,
                                    asset_id=asset_id,
                                    name=fields.get("name"),
                                    assetType=fields.get("assetType"),
                                    isPrimary=fields.get("isPrimary"),
                                    description=fields.get("description"),
                                    actor=state.user_id,
                                    ts=now_iso(),
                                )
                                continue

                            # META (name/description)
                            if op in ("asset.update_meta", "set_asset_meta", "update_asset_meta"):
                                updated_fields = [k for k in ("name", "description") if k in fields]
                                await aemit_asset_updated(
                                    org_id=state.org_id,
                                    asset_id=asset_id_from_payload,
                                    name=fields.get("name"),
                                    description=fields.get("description"),
                                    updated_fields=updated_fields,
                                    actor=state.user_id,
                                    ts=now_iso(),
                                )
                                continue

                            # CIA (confidentiality/integrity/availability)
                            if op in ("asset.set_meta", "set_asset_cia"):
                                updated_fields = [k for k in ("confidentiality", "integrity", "availability") if k in fields]
                                await aemit_asset_updated(
                                    org_id=state.org_id,
                                    asset_id=asset_id_from_payload,
                                    updated_fields=updated_fields,
                                    confidentiality=fields.get("confidentiality"),
                                    integrity=fields.get("integrity"),
                                    availability=fields.get("availability"),
                                    actor=state.user_id,
                                    ts=now_iso(),
                                )
                                continue
                    except Exception:
                        # Never break the main path due to notifications
                        pass

                schedule(_emit_created_and_updated)

            # Clean runtime-only fields after success
            state.create_flow = False
            state.pending_fields = None
            state.create_draft = None
            state.once_flags = None

            # UI result + echoes
            result = _ui_response(chat_msg, extra={"changeset_actions": len(acts), "agent_events": state.agent_events})
            result["messages"].append({"kind": "changeset", "changeset": cs_dump, "ts": now_iso()})
            result["messages"].append({"kind": "submit_result", "result": {"submitted": True}, "ts": now_iso()})

            try:
                result["staging"]["needs_commit"] = False
            except Exception:
                pass

            state.result = result
            return state

        # ---------------------
        # Queued to OUTBOX
        # ---------------------
        if queued:
            finish(state, t0, tool="submit_changeset", ok=True, meta={"queued": True})

            # Clean runtime-only fields even if queued
            state.create_flow = False
            state.pending_fields = None
            state.create_draft = None
            state.once_flags = None

            res = _ui_response(chat_msg, extra={"agent_events": state.agent_events})
            res["messages"].append({
                "kind": "changeset.queued",
                "tab": "assets",
                "ts": now_iso(),
                "org_id": state.org_id,
                "actor": state.user_id,
            })
            try:
                res["staging"]["needs_commit"] = True
            except Exception:
                pass

            state.result = res
            return state

        # ---------------------
        # Neither submitted nor queued → treat as failure
        # ---------------------
        finish(state, t0, tool="submit_changeset", ok=False, meta={"error": "unknown submit result"})
        fail_msg = chat_msg or "Submission failed."
        res = _ui_response(fail_msg, extra={"agent_events": state.agent_events})
        try:
            res["staging"]["needs_commit"] = False
        except Exception:
            pass
        state.result = res
        return state

    except Exception as e:
        # Defensive failure path with a clean UI message
        finish(state, t0, tool="submit_changeset", ok=False, meta={"error": str(e)})
        res = _ui_response("Submission failed.", extra={"agent_events": state.agent_events})
        try:
            res["staging"]["needs_commit"] = False
        except Exception:
            pass
        state.result = res
        return state


def guess_created_ids(submit_result: Any) -> list[str]:
    """
    Extract created asset IDs from heterogeneous backend submit_result
    (supports dict or object-shaped responses).
    """
    ids: list[str] = []
    try:
        # Dict shape
        if isinstance(submit_result, dict):
            if isinstance(submit_result.get("created_ids"), list):
                for x in submit_result["created_ids"]:
                    if x:
                        ids.append(str(x))
                return ids

            results = submit_result.get("results")
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict) and (r.get("op") == "create_asset"):
                        res = r.get("result") or {}
                        vid = res.get("id") or res.get("asset_id")
                        if vid:
                            ids.append(str(vid))
                return ids

        # Object shape
        results = getattr(submit_result, "results", None)
        if isinstance(results, list):
            for r in results:
                op = getattr(r, "op", None) or (isinstance(r, dict) and r.get("op"))
                result = getattr(r, "result", None) or (isinstance(r, dict) and r.get("result"))
                if op == "create_asset":
                    vid = getattr(result, "id", None) or (isinstance(result, dict) and result.get("id"))
                    if vid:
                        ids.append(str(vid))
    except Exception:
        pass
    return ids


def llm_confirm(*, deps: GraphDeps, loc: str, actions: list[dict], queued: bool = False) -> str:
    """
    Generate a single short confirmation sentence (≤20 words) after a write.
    If `queued` is True, mention the queueing in the sentence.

    Falls back to a static "Done." (or "Hotovo." for cs/sk) on any error.
    """
    from .llm_utils import llm_text

    l = lang(loc)
    system = (
        "You generate ONE short confirmation sentence after a change in an asset system.\n"
        f"Language: {l}. Keep it under 20 words. Neutral tone. If queued=true, mention it."
    )
    human = {"queued": bool(queued), "actions": actions}

    try:
        txt = llm_text(
            deps=deps,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": json.dumps(human, ensure_ascii=False)}],
            temperature=0.1,
        ).strip()
        if txt and len(txt) <= 200:
            return txt
    except Exception:
        pass

    return "Hotovo." if l in ("cs", "sk") else "Done."
