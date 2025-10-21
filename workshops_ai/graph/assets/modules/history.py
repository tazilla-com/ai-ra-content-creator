# workshops_ai/graph/assets/modules/history.py

from __future__ import annotations
from .state import AssetState

def history_tail(state: AssetState, k: int) -> list[dict]:
    try:
        return (state.history or [])[-k:]
    except Exception:
        return []
