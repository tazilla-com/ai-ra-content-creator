# File: workshops_ai/graph/assets/modules/nodes_read.py
from __future__ import annotations
from typing import Callable, Any, List, Optional, Mapping
import uuid
import json

from workshops_ai.graph.factory import GraphDeps
from .state import AssetState
from .enums_json import jsonify_iter
from .text_utils import lang, parse_list_limit, as_label, extract_name_from_text  # safe label + simple name grab
from .telemetry_utils import event, finish
from .ui_helpers import title, ui_response as _ui_response
from workshops_ai.ui.formatting import bullet_list, chunk, format_asset_types
from .catalog_utils import fetch_asset_types_raw, normalize_type_labels

# -----------------------------------------------------------------------------
# Module constants
# -----------------------------------------------------------------------------
_CHUNK_SIZE = 1800            # unified maximum for text chunking
_MAX_LIST_ROWS = 200          # hard cap for listings (defense against huge payloads)


# -----------------------------------------------------------------------------
# Local utilities
# -----------------------------------------------------------------------------
def _to_labels_any(seq: Any) -> List[str]:
    """
    Convert an arbitrary sequence (possibly list of dicts/models/strings)
    into a clean List[str] using `as_label` for best-effort coercion.
    Never raises; filters out empty strings and preserves order without duplicates.
    """
    out: List[str] = []
    if not seq:
        return out
    for x in seq:
        try:
            lbl = as_label(x)
        except Exception:
            lbl = str(x)
        lbl = (lbl or "").strip()
        if lbl:
            out.append(lbl)
    # preserve order, remove duplicates
    seen = set()
    uniq: List[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _looks_like_uuid(val: str | None) -> bool:
    """Best-effort check if a string is a UUID."""
    if not val:
        return False
    s = str(val).strip()
    try:
        uuid.UUID(s)
        return True
    except Exception:
        return False


def _coerce_selection(sel: Any) -> dict:
    """
    Normalize selection into a dict:
      - {"asset_id": "...", "asset_name": "..."} stays as-is
      - string UUID → {"asset_id": "..."}
      - other string → {"asset_name": "..."}
      - JSON-encoded mapping in bytes/str → parsed
      - everything else → {}
    """
    try:
        if isinstance(sel, Mapping):
            return {k: v for k, v in sel.items() if k in ("asset_id", "asset_name")}
        if isinstance(sel, str):
            s = sel.strip()
            if not s:
                return {}
            # Try JSON first
            try:
                obj = json.loads(s)
                if isinstance(obj, Mapping):
                    return {k: v for k, v in obj.items() if k in ("asset_id", "asset_name")}
            except Exception:
                pass
            # Fallback heuristics
            if _looks_like_uuid(s):
                return {"asset_id": s}
            return {"asset_name": s}
        if isinstance(sel, (bytes, bytearray)):
            s = sel.decode("utf-8", "ignore").strip()
            if not s:
                return {}
            try:
                obj = json.loads(s)
                if isinstance(obj, Mapping):
                    return {k: v for k, v in obj.items() if k in ("asset_id", "asset_name")}
            except Exception:
                pass
            if _looks_like_uuid(s):
                return {"asset_id": s}
            return {"asset_name": s}
        return {}
    except Exception:
        return {}


def _collect_assets_plain(deps: GraphDeps, org_id: str) -> list[dict]:
    """
    Fetch assets and coerce to plain dicts with at least id/name/assetType
    (defensive towards different backend shapes).
    """
    items = deps.backend_port.assets_list(org_id, None, None, None) or []
    rows: list[dict] = []
    for a in items:
        d = jsonify_iter(a)
        row = d if isinstance(d, dict) else {"value": d}
        # Normalize a few common keys for convenience
        aid = row.get("id") or row.get("asset_id")
        if aid:
            row["id"] = aid
        if "name" in row:
            row["name"] = (as_label(row["name"]) or "").strip() or None
        if "assetType" in row:
            row["assetType"] = (as_label(row["assetType"]) or "").strip() or None
        rows.append(row)
    return rows


def _find_asset_by_name(rows: list[dict], raw_msg: str) -> Optional[dict]:
    """
    Try several progressively weaker matches against asset `name` using the user's message:
    1) Exact (case-insensitive) match
    2) Name is substring of message
    3) Message (extracted short name) equals/contained in asset name
    Returns the first reasonable match, or None.
    """
    msg = (raw_msg or "").strip()
    if not msg:
        return None

    # Try to extract a quoted/short name candidate when present
    extracted = extract_name_from_text(msg)
    needle_exact = (extracted or "").strip().lower()
    msg_low = msg.lower()

    # 1) exact match (case-insensitive)
    if needle_exact:
        for r in rows:
            nm = (r.get("name") or "").strip().lower()
            if nm and nm == needle_exact:
                return r

    # 2) name appears in the full message
    for r in rows:
        nm = (r.get("name") or "").strip()
        if nm and nm.lower() in msg_low:
            return r

    # 3) short extracted candidate contained in an asset name
    if needle_exact:
        for r in rows:
            nm = (r.get("name") or "").strip().lower()
            if nm and needle_exact in nm:
                return r

    return None


# -----------------------------------------------------------------------------
# Node: assets_list  (reads existing assets from CSV backend)
# -----------------------------------------------------------------------------
def assets_list(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        t0 = event(state, kind="tool.call.start", tool="asset.list")
        try:
            if not state.org_id:
                is_cs = lang(state.locale) in ("cs", "sk")
                msg = "_Chybí org_id._" if is_cs else "_Missing org_id._"
                state.result = _ui_response(msg, extra={"agent_events": state.agent_events})
                finish(state, t0, tool="asset.list", ok=False, meta={"error": "missing_org_id"})
                return state

            items = deps.backend_port.assets_list(state.org_id, None, None, None) or []

            # Optional limit parsed from the message (e.g., "show top 5")
            n = parse_list_limit(state.message or "")
            if n is not None:
                items = items[:n]
            else:
                items = items[:_MAX_LIST_ROWS]

            # Render one line per asset (name + type)
            lines: list[str] = []
            for a in items:
                d = jsonify_iter(a)
                if not isinstance(d, dict):
                    d = {"value": d}
                nm = as_label(d.get("name")) if d.get("name") is not None else ""
                tp = as_label(d.get("assetType")) if d.get("assetType") is not None else ""
                nm = (nm or "").strip()
                tp = (tp or "").strip()
                line = f"{nm} ({tp})" if (nm and tp) else (nm or tp or "")
                if line:
                    lines.append(line)

            is_cs = lang(state.locale) in ("cs", "sk")
            heading = "Aktiva" if is_cs else "Assets"
            empty_txt = "_Žádná aktiva nenalezena_" if is_cs else "_No assets found_"
            bullets = bullet_list(lines) if lines else empty_txt
            body = f"{title(heading)}\n\n{bullets}"

            parts = chunk(body, max_chars=_CHUNK_SIZE)
            state.result = _ui_response(parts[0], extra={"agent_events": state.agent_events})
            if len(parts) > 1:
                state.result["messages"].extend(parts[1:])  # type: ignore[index]

            finish(state, t0, tool="asset.list", ok=True, meta={"count": len(items)})
        except Exception as e:
            finish(state, t0, tool="asset.list", ok=False, meta={"error": str(e)})
            try:
                fallback = "_Chyba při formátování výpisu aktiv._"
                state.result = _ui_response(fallback, extra={"agent_events": state.agent_events})
            except Exception:
                state.result = _ui_response(
                    {"error": "asset.list failed", "reason": str(e)},
                    extra={"agent_events": state.agent_events},
                )
        return state
    return _inner


# -----------------------------------------------------------------------------
# Node: asset_get_or_preview  (detail by name primarily; UUID/ID as secondary)
# -----------------------------------------------------------------------------
def asset_get_or_preview(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        # Normalize selection defensively (may come as str/bytes/dict)
        sel = _coerce_selection(getattr(state, "selection", None))
        raw = (state.message or "").strip()

        asset_id = sel.get("asset_id")
        asset_name = sel.get("asset_name")

        t0 = event(
            state,
            kind="tool.call.start",
            tool="asset.get",
            meta={"asset_id": asset_id, "asset_name": asset_name},
        )

        try:
            # --- Resolution strategy ----------------------------------------
            # 1) Prefer explicit selection from session (ID > name)
            # 2) Otherwise, derive from the user message:
            #    - if it "looks like" a UUID, treat it as an ID
            #    - else try to extract a name and resolve by name
            # 3) If we found a single asset, persist it into state.selection,
            #    emit an `asset.selected` event, and return an asset.get payload
            # 4) If no exact hit by name, return a small preview selector
            # -----------------------------------------------------------------

            # Step 1: explicit selection
            target_id: Optional[str] = (str(asset_id).strip() or None) if asset_id else None
            target_name: Optional[str] = (str(asset_name).strip() or None) if asset_name else None

            # Step 2: derive from message if nothing explicit
            if not target_id and not target_name:
                if _looks_like_uuid(raw):
                    target_id = raw
                else:
                    target_name = extract_name_from_text(raw)

            # --- Branch A: we have an ID → fetch directly -------------------
            if target_id:
                a = None
                try:
                    a = deps.backend_port.asset_get(target_id)
                except Exception:
                    a = None

                # Fallback: if direct get failed but we know org_id, scan list
                if a is None and state.org_id:
                    rows = _collect_assets_plain(deps, state.org_id)
                    for r in rows:
                        rid = r.get("id")
                        if rid and str(rid) == target_id:
                            a = r
                            break

                if a is None:
                    is_cs = lang(state.locale) in ("cs", "sk")
                    msg = "_Aktivum nenalezeno._" if is_cs else "_Asset not found._"
                    state.result = _ui_response(
                        msg,
                        extra={
                            "agent_events": state.agent_events,
                            "selection": (state.selection or None),
                        },
                    )
                    finish(state, t0, tool="asset.get", ok=False, meta={"error": "not_found_by_id"})
                    return state

                # Normalize to dict and clean labels
                d = jsonify_iter(a)
                if not isinstance(d, dict):
                    d = {"value": d}
                if "name" in d:
                    d["name"] = (as_label(d["name"]) or "").strip() or None
                if "assetType" in d:
                    d["assetType"] = (as_label(d["assetType"]) or "").strip() or None

                # Persist selection so follow-up updates can target this asset
                try:
                    sid = d.get("id") or d.get("asset_id")
                    if sid:
                        state.selection = state.selection or {}
                        state.selection["asset_id"] = str(sid)
                        if d.get("name"):
                            state.selection["asset_name"] = d["name"]

                        # Emit a lightweight selection event so the manager can persist current_asset
                        event(
                            state,
                            kind="asset.selected",
                            tool="asset.get",
                            meta={"asset_id": str(sid), "name": d.get("name")},
                        )
                except Exception:
                    # Never break the flow on best-effort selection persistence
                    pass

                state.result = _ui_response(
                    {"kind": "asset.get", "asset": d},
                    extra={
                        "agent_events": state.agent_events,
                        "selection": state.selection,
                    },
                )
                finish(state, t0, tool="asset.get", ok=True, meta={"by": "id", "id": target_id})
                return state

            # --- Branch B: name-first resolution (preferred conversational UX) ---
            if state.org_id:
                rows = _collect_assets_plain(deps, state.org_id)

                picked = None
                # If caller didn't provide an explicit name, try a message-based heuristic match
                if not target_name:
                    picked = _find_asset_by_name(rows, raw)

                # If we have an explicit name candidate, try exact (case-insensitive) then contains
                if target_name and not picked:
                    tnl = target_name.strip().lower()
                    picked = next((r for r in rows if (r.get("name") or "").strip().lower() == tnl), None)
                    if not picked:
                        picked = next((r for r in rows if tnl in ((r.get("name") or "").strip().lower())), None)

                # Fallback: one more heuristic try from the raw message
                if not picked:
                    picked = _find_asset_by_name(rows, raw)

                if picked:
                    # Persist selection for subsequent update flow
                    try:
                        sid = picked.get("id") or picked.get("asset_id")
                        if sid:
                            state.selection = state.selection or {}
                            state.selection["asset_id"] = str(sid)
                            if picked.get("name"):
                                state.selection["asset_name"] = picked.get("name")

                            # Emit a selection event for manager session-state update
                            event(
                                state,
                                kind="asset.selected",
                                tool="asset.get",
                                meta={"asset_id": str(sid), "name": picked.get("name")},
                            )
                    except Exception:
                        pass

                    state.result = _ui_response(
                        {"kind": "asset.get", "asset": picked},
                        extra={
                            "agent_events": state.agent_events,
                            "selection": state.selection,
                        },
                    )
                    finish(state, t0, tool="asset.get", ok=True, meta={"by": "name"})
                    return state

                # No single hit → provide a short preview selector for the UI
                preview: list[dict] = []
                for r in rows[:8]:
                    rid = r.get("id")
                    if not rid:
                        continue
                    preview.append({
                        "id": rid,
                        "name": r.get("name"),
                        "assetType": r.get("assetType"),
                    })

                state.result = _ui_response(
                    {"kind": "asset.select.suggest", "preview": preview},
                    extra={
                        "agent_events": state.agent_events,
                        "selection": (state.selection or None),
                    },
                )
                finish(state, t0, tool="asset.get", ok=True, meta={"suggested": len(preview)})
                return state

            # --- No org_id and no usable selector → cannot proceed ------------
            is_cs = lang(state.locale) in ("cs", "sk")
            msg = "_Chybí org_id._" if is_cs else "_Missing org_id._"
            state.result = _ui_response(
                msg,
                extra={
                    "agent_events": state.agent_events,
                    "selection": (state.selection or None),
                },
            )
            finish(state, t0, tool="asset.get", ok=False, meta={"error": "missing_org_id"})
            return state

        except Exception as e:
            # Defensive error reporting; keep UI payload simple and localized
            finish(state, t0, tool="asset.get", ok=False, meta={"error": str(e)})
            try:
                state.result = _ui_response(
                    "_Chyba při načítání aktiva._",
                    extra={
                        "agent_events": state.agent_events,
                        "selection": (state.selection or None),
                    },
                )
            except Exception:
                state.result = _ui_response(
                    {"error": "asset.get failed", "reason": str(e)},
                    extra={
                        "agent_events": state.agent_events,
                        "selection": (state.selection or None),
                    },
                )
            return state

    return _inner



# -----------------------------------------------------------------------------
# Node: list_asset_types  (catalog helper; must not block list/get/update flows)
# -----------------------------------------------------------------------------
def list_asset_types(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        t0 = event(state, kind="tool.call.start", tool="asset.list_types")
        try:
            # 1) Fetch raw types via tolerant helper
            raw_types = fetch_asset_types_raw(deps, state.org_id)

            # 2) Prepare localized strings and formatted plain-text list
            is_cs = lang(state.locale) in ("cs", "sk")
            formatted_text = str(format_asset_types(raw_types or []))  # ensure it's a string
            labels = normalize_type_labels(raw_types or [])

            # 3) Build a purely string-based response (no dicts in join)
            parts: list[str] = []
            if formatted_text.strip():
                parts.append(formatted_text.strip())
            else:
                parts.append("_Žádné typy aktiv nenalezeny_" if is_cs else "_No asset types found_")

            if labels:
                parts += [
                    "",
                    "Který typ aktiva chceš vytvořit?" if is_cs else "Which asset type do you want to create?",
                ]

            text = "\n".join(parts)
            chunks = chunk(text, max_chars=_CHUNK_SIZE)

            state.result = _ui_response(chunks[0], extra={"agent_events": state.agent_events})
            if len(chunks) > 1:
                state.result["messages"].extend(chunks[1:])  # type: ignore[index]

            # 4) Seed create-flow slot for assetType when appropriate
            state.create_draft = (state.create_draft or {})
            if state.create_flow:
                state.create_draft.pop("assetType", None)
                state.pending_fields = ["assetType"]
            else:
                state.pending_fields = None

            finish(state, t0, tool="asset.list_types", ok=True, meta={"count": len(labels)})
        except Exception as e:
            # Graceful degradation: send text and record precise reason in telemetry
            finish(state, t0, tool="asset.list_types", ok=False, meta={"error": str(e)})
            try:
                fallback_labels = normalize_type_labels(locals().get("raw_types", []) or [])
                is_cs = lang(state.locale) in ("cs", "sk")
                if fallback_labels:
                    fallback = "\n".join([bullet_list(fallback_labels)])
                else:
                    fallback = "_Chyba při formátování výpisu typů aktiv._" if is_cs else "_Failed to render asset types._"
                state.result = _ui_response(fallback, extra={"agent_events": state.agent_events})
            except Exception:
                state.result = _ui_response(
                    {"error": "asset.list_types failed", "reason": str(e)},
                    extra={"agent_events": state.agent_events},
                )
        return state
    return _inner
