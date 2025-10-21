# workshops_ai/graph/assets/modules/nodes_read.py
from __future__ import annotations
from typing import Callable, Any, List

from workshops_ai.graph.factory import GraphDeps
from .state import AssetState
from .enums_json import jsonify_iter
from .text_utils import lang, parse_list_limit, as_label  # we normalize labels locally
from .telemetry_utils import event, finish
from .ui_helpers import title, ui_response as _ui_response

# -----------------------------------------------------------------------------
# Module constants
# -----------------------------------------------------------------------------
_CHUNK_SIZE = 1800            # unified maximum for text chunking
_MAX_LIST_ROWS = 200          # hard cap for listings (defense against huge payloads)


# -----------------------------------------------------------------------------
# Optional shared formatting helpers (fall back to local minimal ones)
# -----------------------------------------------------------------------------
try:
    # Centralized helpers: predictable Markdown + (possibly) their own signatures
    from workshops_ai.ui.formatting import bullet_list, chunk, format_asset_types  # type: ignore
except Exception:
    # Minimal fallbacks so the agent never crashes if the optional module is missing
    def bullet_list(items: List[str]) -> str:
        """Render a simple bullet list from normalized strings."""
        return "\n".join(f"- {str(lbl).strip()}" for lbl in (items or []))

    def chunk(text: str, max_chars: int = _CHUNK_SIZE) -> list[str]:
        """Split text into fixed-size chunks."""
        s = text or ""
        if not s:
            return [""]
        return [s[i:i + max_chars] for i in range(0, len(s), max_chars)]

    def format_asset_types(types: List[str]) -> str:
        """Format already-normalized type labels."""
        labels = [str(t).strip() for t in (types or []) if str(t).strip()]
        return bullet_list(labels) if labels else "_No types found_"


# -----------------------------------------------------------------------------
# Local safety utilities
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
            lbl = as_label(x)  # imported helper that knows how to stringify common shapes
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


# -----------------------------------------------------------------------------
# Node: assets_list
# -----------------------------------------------------------------------------
def assets_list(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        t0 = event(state, kind="tool.call.start", tool="asset.list")
        try:
            items = deps.backend_port.assets_list(state.org_id, None, None, None) or []
            n = parse_list_limit(state.message or "")
            if n is not None:
                items = items[:n]
            else:
                # Reasonable default limit to keep UI snappy
                items = items[:_MAX_LIST_ROWS]

            # Render one line per asset (name + type), robust to dicts/models
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

            heading = "Assets" if lang(state.locale) not in ("cs", "sk") else "Aktiva"
            empty_txt = "_No assets found_" if lang(state.locale) not in ("cs", "sk") else "_Žádná aktiva nenalezena_"
            bullets = bullet_list(lines) if lines else empty_txt
            body = f"{title(heading)}\n\n{bullets}"

            # Use positional arg to avoid foreign chunk() signature incompatibilities
            parts = chunk(body, _CHUNK_SIZE)

            state.result = _ui_response(parts[0], extra={"agent_events": state.agent_events})
            if len(parts) > 1:
                # UI contract: subsequent strings can be appended as additional messages
                state.result["messages"].extend(parts[1:])  # type: ignore[index]

            finish(state, t0, tool="asset.list", ok=True, meta={"count": len(items)})
        except Exception as e:
            finish(state, t0, tool="asset.list", ok=False, meta={"error": str(e)})
            try:
                # Human-friendly fallback
                fallback = "_Chyba při formátování výpisu aktiv._"
                state.result = _ui_response(fallback, extra={"agent_events": state.agent_events})
            except Exception:
                # Raw error object as a last resort
                state.result = _ui_response(
                    {"error": "asset.list failed", "reason": str(e)},
                    extra={"agent_events": state.agent_events},
                )
        return state
    return _inner


# -----------------------------------------------------------------------------
# Node: asset_get_or_preview
# -----------------------------------------------------------------------------
def asset_get_or_preview(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        asset_id = (state.selection or {}).get("asset_id")
        t0 = event(state, kind="tool.call.start", tool="asset.get", meta={"asset_id": asset_id})
        try:
            if not asset_id:
                # Suggest a small preview list if nothing is selected
                items = deps.backend_port.assets_list(state.org_id, None, None, None) or []
                preview: list[dict] = []
                for a in items[:8]:
                    d = jsonify_iter(a)
                    if not isinstance(d, dict):
                        d = {"value": d}
                    aid = d.get("id") or d.get("asset_id")
                    if not aid:
                        continue  # skip items without id to avoid empty UI entries
                    nm = as_label(d.get("name")) if d.get("name") is not None else None
                    tp = as_label(d.get("assetType")) if d.get("assetType") is not None else None
                    nm = (nm or "").strip() or None
                    tp = (tp or "").strip() or None
                    preview.append({"id": aid, "name": nm, "assetType": tp})

                state.result = _ui_response(
                    {"kind": "asset.select.suggest", "preview": preview},
                    extra={"agent_events": state.agent_events},
                )
                finish(state, t0, tool="asset.get", ok=True, meta={"suggested": len(preview)})
                return state

            a = deps.backend_port.asset_get(asset_id)
            d = jsonify_iter(a)
            if not isinstance(d, dict):
                d = {"value": d}

            if "name" in d:
                nm = as_label(d["name"])
                d["name"] = (nm or "").strip() or None
            if "assetType" in d:
                tp = as_label(d["assetType"])
                d["assetType"] = (tp or "").strip() or None

            state.result = _ui_response(
                {"kind": "asset.get", "asset": d},
                extra={"agent_events": state.agent_events},
            )
            finish(state, t0, tool="asset.get", ok=True, meta={"id": asset_id})
        except Exception as e:
            finish(state, t0, tool="asset.get", ok=False, meta={"error": str(e)})
            try:
                state.result = _ui_response("_Chyba při načítání aktiva._", extra={"agent_events": state.agent_events})
            except Exception:
                state.result = _ui_response(
                    {"error": "asset.get failed", "reason": str(e)},
                    extra={"agent_events": state.agent_events},
                )
        return state
    return _inner


# -----------------------------------------------------------------------------
# Node: list_asset_types
# -----------------------------------------------------------------------------
def list_asset_types(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        t0 = event(state, kind="tool.call.start", tool="asset.list_types")
        try:
            # 1) Fetch raw types (may be dicts / models / strings)
            raw_types = deps.backend_port.list_asset_types() or []

            # Optional: quick diagnostics about shapes
            try:
                from .telemetry_utils import telemetry_log
                telemetry_log(
                    "asset.list_types.sample",
                    {"sample_types": [type(x).__name__ for x in raw_types[:3]]},
                    level="DEBUG",
                )
            except Exception:
                pass

            # 2) Normalize everything to a clean List[str]
            labels = _to_labels_any(raw_types)

            # 3) Localize heading and render through helpers
            heading = "Asset types" if lang(state.locale) not in ("cs", "sk") else "Typy aktiv"

            # Always pass normalized strings to format_asset_types to avoid join/type errors
            formatted = format_asset_types(labels)

            content: list[str] = [
                title(heading),
                "",
                formatted,
            ]

            if labels:
                content += [
                    "",
                    "Který typ aktiva chceš vytvořit?"
                    if lang(state.locale) in ("cs", "sk")
                    else "Which asset type do you want to create?",
                ]

            text = "\n".join(content)

            # Use positional arg to avoid foreign chunk() signature incompatibilities
            parts = chunk(text, _CHUNK_SIZE)

            state.result = _ui_response(parts[0], extra={"agent_events": state.agent_events})
            if len(parts) > 1:
                state.result["messages"].extend(parts[1:])  # type: ignore[index]

            # 4) Seed/create flow slot for assetType when appropriate
            state.create_draft = (state.create_draft or {})
            if state.create_flow:
                state.create_draft.pop("assetType", None)
                state.pending_fields = ["assetType"]
            else:
                state.pending_fields = None

            finish(state, t0, tool="asset.list_types", ok=True, meta={"count": len(labels)})
        except Exception as e:
            # Graceful degradation: human message + precise reason in telemetry
            finish(state, t0, tool="asset.list_types", ok=False, meta={"error": str(e)})
            try:
                # Try to salvage readable bullets from whatever we fetched
                fallback_labels = _to_labels_any(locals().get("raw_types", []))
                err_hdr = "_Chyba při formátování výpisu typů aktiv._"
                fallback = err_hdr + ("\n" + bullet_list(fallback_labels) if fallback_labels else "")
                state.result = _ui_response(fallback, extra={"agent_events": state.agent_events})
            except Exception:
                state.result = _ui_response(
                    {"error": "asset.list_types failed", "reason": str(e)},
                    extra={"agent_events": state.agent_events},
                )
        return state
    return _inner
