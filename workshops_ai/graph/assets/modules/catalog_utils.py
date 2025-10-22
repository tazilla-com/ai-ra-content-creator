# File: workshops_ai/graph/assets/modules/catalog_utils.py
from __future__ import annotations

from typing import Any, Iterable, List, Optional

from workshops_ai.graph.factory import GraphDeps
from .text_utils import as_label


# -----------------------------------------------------------------------------
# Public: text → catalog type matching
# -----------------------------------------------------------------------------
def extract_catalog_type_from_text(text: str, asset_types: list[str]) -> Optional[str]:
    """
    Very simple detector that returns a single matching asset type if (and only if)
    exactly one type label appears (as case-insensitive substring) in the input text.

    This is intentionally conservative to avoid false positives. If multiple labels
    match, we return None and let the calling flow ask a clarifying question.
    """
    m = (text or "").strip().lower()
    if not m or not asset_types:
        return None

    hits: list[str] = []
    for t in asset_types:
        tl = (t or "").strip().lower()
        if tl and tl in m:
            hits.append(t)

    # Preserve order, remove duplicates
    uniq = list(dict.fromkeys(hits))
    return uniq[0] if len(uniq) == 1 else None


# -----------------------------------------------------------------------------
# Public: tolerant catalog access (centralized fallback logic)
# -----------------------------------------------------------------------------
def fetch_asset_types_raw(deps: GraphDeps, org_id: Optional[str]) -> List[Any]:
    """
    Tolerant fetch of asset types from backend. Works with either:
      - backend_port.asset_types(org_id)   (preferred, multi-tenant aware)
      - backend_port.list_asset_types()    (legacy/global)
    Returns an empty list on any failure. Never raises.
    """
    try:
        bp = deps.backend_port
        if hasattr(bp, "asset_types"):
            return list(bp.asset_types(org_id) or [])
        if hasattr(bp, "list_asset_types"):
            return list(bp.list_asset_types() or [])
    except Exception:
        pass
    return []


def normalize_type_labels(items: Iterable[Any]) -> List[str]:
    """
    Normalize an iterable of arbitrary items (dicts/models/strings) to a
    de-duplicated list of non-empty human labels. Preserves input order.
    Never raises.
    """
    out: List[str] = []
    seen = set()
    if not items:
        return out

    for it in items:
        try:
            lbl = (as_label(it) or "").strip()
        except Exception:
            lbl = (str(it) or "").strip()

        if lbl and lbl not in seen:
            seen.add(lbl)
            out.append(lbl)

    return out


def get_asset_type_labels(deps: GraphDeps, org_id: Optional[str]) -> List[str]:
    """
    One-shot helper used by all modules that need type names:
    tolerant fetch → robust normalization.
    """
    return normalize_type_labels(fetch_asset_types_raw(deps, org_id))
