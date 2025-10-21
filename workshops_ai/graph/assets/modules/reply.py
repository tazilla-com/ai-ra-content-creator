# workshops_ai/graph/assets/modules/reply.py

from __future__ import annotations
from typing import Callable
from workshops_ai.graph.factory import GraphDeps
from .state import AssetState
from .ui_helpers import ui_response as _ui_response
from .text_utils import lang
from .llm_utils import llm_text

def agent_reply(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        text = (state.message or "").strip()
        l = lang(state.locale)
        system = (
            "You are a helpful assistant for an asset management UI.\n"
            f"Language: {l}. Respond briefly (≤80 words). "
            "Mention you can list assets, show details, create assets, update metadata, or explain concepts if needed. "
            "Do not invent data; do not claim to have taken actions."
        )
        try:
            ans = llm_text(deps=deps, messages=[{"role":"system","content": system}, {"role":"user","content": text}], temperature=0.3)
        except Exception:
            ans = "S čím mám v aktivech pomoci?" if l in ("cs","sk") else "How can I help with your assets?"
        state.result = _ui_response(ans, extra={"agent_events": state.agent_events})
        return state
    return _inner
