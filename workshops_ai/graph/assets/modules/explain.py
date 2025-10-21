# workshops_ai/graph/assets/modules/explain.py

from __future__ import annotations
import json
from typing import Optional
from workshops_ai.graph.factory import GraphDeps
from .state import AssetState
from .ui_helpers import ui_response as _ui_response
from .telemetry_utils import event, finish
from .llm_utils import llm_text
from .text_utils import lang

async def explain_assets_async(deps: GraphDeps, state: AssetState) -> AssetState:
    q = (state.question or state.message or "").strip()
    if not q:
        state.result = _ui_response("Please provide a question about assets.")
        return state

    retriever = None
    try:
        if hasattr(deps, "make_retriever") and callable(deps.make_retriever):
            retriever = deps.make_retriever(state.org_id, "assets")
    except Exception:
        retriever = None

    snippets: list[dict] = []
    if retriever:
        t0 = event(state, kind="tool.call.start", tool="retriever.search", meta={"tab":"assets"})
        try:
            docs = await retriever.asearch(q, k=5) if hasattr(retriever, "asearch") else retriever.search(q, k=5)  # type: ignore
            for d in (docs or [])[:5]:
                if isinstance(d, dict):
                    txt = (d.get("text") or d.get("page_content") or "").strip()
                    src = d.get("source") or (d.get("metadata") or {}).get("source")
                    if txt:
                        snippets.append({"text": txt[:800], "source": src})
                else:
                    snippets.append({"text": str(d)})
            finish(state, t0, tool="retriever.search", ok=True, meta={"docs": len(snippets)})
        except Exception as e:
            finish(state, t0, tool="retriever.search", ok=False, meta={"error": str(e)})

    client = getattr(deps, "llm_client", None)
    if snippets and client is not None:
        tagged = [{"id": f"S{i+1}", "text": s["text"], "source": s.get("source")} for i, s in enumerate(snippets[:5])]
        system = (
            "You are a precise cybersecurity assistant.\n"
            "- Answer briefly (≤120 words) in the user's language; ground claims in snippets with [S#]."
        )
        human = {"question": q, "snippets": tagged, "locale": (state.locale or "en")}
        t0 = event(state, kind="tool.call.start", tool="llm.synthesize", meta={"snippets": len(tagged)})
        try:
            txt = llm_text(deps=deps, messages=[{"role":"system","content": system}, {"role":"user","content": json.dumps(human, ensure_ascii=False)}], temperature=0.2)
            if txt:
                finish(state, t0, tool="llm.synthesize", ok=True, meta={})
                state.result = _ui_response(txt, extra={"snippets": tagged, "agent_events": state.agent_events})
                _append_pending_reminder(state)
                return state
        except Exception as e:
            finish(state, t0, tool="llm.synthesize", ok=False, meta={"error": str(e)})

    if client is not None:
        system = "You are a cybersecurity assistant. Provide a concise (≤120 words) answer without hallucinations."
        human = {"question": q, "locale": (state.locale or "en")}
        t0 = event(state, kind="tool.call.start", tool="llm.explain")
        try:
            txt = llm_text(deps=deps, messages=[{"role":"system","content": system}, {"role":"user","content": json.dumps(human, ensure_ascii=False)}], temperature=0.2)
            if txt:
                finish(state, t0, tool="llm.explain", ok=True, meta={})
                state.result = _ui_response(txt, extra={"agent_events": state.agent_events})
                _append_pending_reminder(state)
                return state
        except Exception as e:
            finish(state, t0, tool="llm.explain", ok=False, meta={"error": str(e)})

    state.result = _ui_response("No evidence available and fallback did not answer.", extra={"agent_events": state.agent_events})
    _append_pending_reminder(state)
    return state


def _append_pending_reminder(state: AssetState) -> None:
    if not (state.create_flow or state.pending_fields or state.create_draft):
        return
    draft = state.create_draft or {}
    order = ["assetType","name","isPrimary","confidentiality","integrity","availability","description"]
    missing = [k for k in order if (draft.get(k) in (None,"",[]) or k not in draft)]
    if draft.get("isPrimary") is False:
        missing = [k for k in missing if k not in ("confidentiality","integrity","availability")]
    if not missing:
        return
    note = (f"Chceš pokračovat? Aktuálně čekám na: {', '.join(missing[:3])}."
            if lang(state.locale) in ('cs','sk')
            else f"Do you want to continue? I’m waiting for: {', '.join(missing[:3])}.")
    try:
        state.result["messages"].append(note)  # type: ignore[index]
    except Exception:
        pass
