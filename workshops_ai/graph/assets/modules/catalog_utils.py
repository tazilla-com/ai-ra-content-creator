# workshops_ai/graph/assets/modules/catalog_utils.py

from __future__ import annotations
from typing import Optional

def extract_catalog_type_from_text(text: str, asset_types: list[str]) -> Optional[str]:
    m = (text or "").strip().lower()
    hits: list[str] = []
    for t in asset_types:
        tl = (t or "").strip().lower()
        if tl and tl in m:
            hits.append(t)
    uniq = list(dict.fromkeys(hits))
    return uniq[0] if len(uniq) == 1 else None
