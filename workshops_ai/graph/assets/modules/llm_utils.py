# workshops_ai/graph/assets/modules/llm_utils.py
from __future__ import annotations

import json
from typing import Optional, Any, Dict, List

from workshops_ai.graph.factory import GraphDeps
from .text_utils import lang
from .enums_json import norm_enum, CONF_LEVELS, TRI_LEVELS


def _to_lc_messages(messages: List[Dict[str, str]]) -> List[object]:
    """
    Best-effort adapter to LangChain message objects.
    If LC is unavailable, raises ImportError to let caller try other bridges.
    """
    try:
        # Prefer langchain_core on newer stacks, fallback to legacy schema.
        try:
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # type: ignore
        except Exception:  # pragma: no cover
            from langchain.schema import SystemMessage, HumanMessage, AIMessage  # type: ignore

        lc_msgs: List[object] = []
        for m in messages:
            role = (m.get("role") or "").lower()
            content = m.get("content") or ""
            if role == "system":
                lc_msgs.append(SystemMessage(content=content))
            elif role in ("assistant", "ai"):
                lc_msgs.append(AIMessage(content=content))
            else:
                lc_msgs.append(HumanMessage(content=content))
        return lc_msgs
    except Exception as e:  # pragma: no cover
        raise ImportError("LangChain messages not available") from e


def _extract_text_from_obj(obj: Any) -> str:
    """
    Normalize various client return shapes into a plain string.
    """
    if obj is None:
        return ""

    # Direct string
    if isinstance(obj, str):
        return obj.strip()

    # Common attributes on SDK/LC objects
    for attr in ("content", "text", "message", "output_text"):
        val = getattr(obj, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # OpenAI ChatCompletions-like
    # obj.choices[0].message.content OR obj.choices[0].text
    try:
        choices = getattr(obj, "choices", None) or obj.get("choices")  # type: ignore
        if choices and len(choices) > 0:
            choice0 = choices[0]
            # message.content
            msg = getattr(choice0, "message", None) or (choice0.get("message") if isinstance(choice0, dict) else None)  # type: ignore
            if msg:
                c = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)  # type: ignore
                if isinstance(c, str) and c.strip():
                    return c.strip()
            # text
            t = getattr(choice0, "text", None) or (choice0.get("text") if isinstance(choice0, dict) else None)  # type: ignore
            if isinstance(t, str) and t.strip():
                return t.strip()
    except Exception:
        pass

    # As a last resort, stringify
    try:
        s = str(obj).strip()
        return s
    except Exception:
        return ""


def llm_text(
    *,
    deps: GraphDeps,
    messages: List[Dict[str, str]],
    temperature: float | None = 0.2,
    model: Optional[str] = None,
) -> str:
    """
    Portable LLM bridge:
    1) Tries LangChain-style client.invoke(messages)
    2) Tries OpenAI-like client.chat.completions.create(...)
    3) Tries a generic callable client(messages=..., temperature=..., model=...)
    Returns plain text or raises if all strategies fail.
    """
    client = getattr(deps, "llm_client", None) or getattr(deps, "llm", None)
    if client is None:
        raise RuntimeError("LLM client is not configured.")

    temp = temperature if temperature is not None else getattr(deps, "llm_temperature", None) or 0.2

    # Strategy 1: LangChain invoke
    try:
        invoke = getattr(client, "invoke", None)
        if callable(invoke):
            lc_messages = _to_lc_messages(messages)
            try:
                resp = client.invoke(lc_messages, temperature=temp)  # type: ignore
            except TypeError:
                resp = client.invoke(lc_messages)  # type: ignore
            txt = _extract_text_from_obj(resp)
            if txt:
                return txt
    except Exception:
        # Fall through to next strategy
        pass

    # Strategy 2: OpenAI ChatCompletions-like
    try:
        chat = getattr(client, "chat", None)
        completions = getattr(chat, "completions", None) if chat else None
        create = getattr(completions, "create", None) if completions else None
        if callable(create):
            # Map messages to OpenAI format
            oa_msgs = [{"role": m.get("role") or "user", "content": m.get("content") or ""} for m in messages]
            resp = create(model=model or getattr(deps, "llm_model", None) or "gpt-4o-mini",
                          messages=oa_msgs, temperature=temp)  # type: ignore
            txt = _extract_text_from_obj(resp)
            if txt:
                return txt
    except Exception:
        # Fall through to next strategy
        pass

    # Strategy 3: Generic callable / SDK wrapper
    try:
        # Try kwargs signature
        obj = client(messages=messages, temperature=temp, model=model)  # type: ignore
        txt = _extract_text_from_obj(obj)
        if txt:
            return txt
    except TypeError:
        # Try positional signature
        try:
            obj = client(messages)  # type: ignore
            txt = _extract_text_from_obj(obj)
            if txt:
                return txt
        except Exception:
            pass
    except Exception:
        pass

    raise RuntimeError("LLM bridge failed: unsupported client shape or empty content.")


def build_next_question(*, deps: GraphDeps, draft: Dict[str, Any], asset_types: List[str], locale: str) -> str:
    """
    Ask ONE concise follow-up question for the next missing field in the create flow.
    Respects field ordering and CIA dependencies. Output is the question text only.
    """
    l = lang(locale)
    order = ["assetType", "name", "isPrimary", "confidentiality", "integrity", "availability", "description"]

    def _missing(k: str) -> bool:
        return (draft.get(k) in (None, "", []) or k not in draft)

    missing = [k for k in order if _missing(k)]

    # If not primary, skip CIA
    if draft.get("isPrimary") is False:
        missing = [k for k in missing if k not in ("confidentiality", "integrity", "availability")]

    # Ensure we ask isPrimary before CIA
    if "isPrimary" in missing:
        missing = [x for x in missing if x not in ("confidentiality", "integrity", "availability")]

    target = missing[0] if missing else None

    system = (
        "You help a user create an asset by asking ONE concise follow-up question.\n"
        f"Language: {l}.\n"
        "- Ask for the most important missing field next, following the given order.\n"
        "- Preserve catalog labels EXACTLY as provided (English).\n"
        "- CIA enums: confidentiality=[PUBLIC,INTERNAL,PROTECTED,STRICTLY_PROTECTED], integrity/availability=[LOW,MEDIUM,HIGH].\n"
        "- Ask 'isPrimary' before CIA; if isPrimary=false, do not ask CIA.\n"
        "- Output only the question text."
    )
    payload = {
        "known_draft": {k: draft.get(k, None) for k in order},
        "next_slot": target,
        "catalog_asset_types": asset_types,
    }
    q = llm_text(
        deps=deps,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=getattr(deps, "llm_temperature", None) or 0.2,
    )
    return (q or "").strip()


def _strip_code_fences(raw: str) -> str:
    """
    Extract JSON from a code-fenced block if present.
    """
    s = (raw or "").strip()
    if s.startswith("```"):
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j != -1 and j > i:
            return s[i : j + 1]
    return s


def extract_create_draft_with_llm(*, deps: GraphDeps, text: str, asset_types: List[str]) -> Dict[str, Any]:
    """
    Convert a short free-text user request into a strict JSON draft for asset creation.
    Keys: name, assetType, isPrimary, confidentiality, integrity, availability, description.
    Enums are canonicalized to known sets (CONF_LEVELS, TRI_LEVELS).
    """
    system = (
        "Transform a short asset request into STRICT JSON.\n"
        "Allowed keys: name, assetType, isPrimary, confidentiality, integrity, availability, description.\n"
        "CIA enums: confidentiality=[PUBLIC,INTERNAL,PROTECTED,STRICTLY_PROTECTED]; integrity/availability=[LOW,MEDIUM,HIGH].\n"
        "- If user_text contains exactly one catalog label, use it as assetType; otherwise null.\n"
        "- Never set name equal to a catalog label unless explicitly stated.\n"
        "Output JSON only."
    )
    human = {"user_text": text, "asset_types": asset_types}

    raw = llm_text(
        deps=deps,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(human, ensure_ascii=False)},
        ],
        temperature=getattr(deps, "llm_temperature", None) or 0.0,
    ).strip()

    raw = _strip_code_fences(raw)

    try:
        obj: Dict[str, Any] = json.loads(raw) if raw else {}
        if not isinstance(obj, dict):
            obj = {}
    except Exception:
        obj = {}

    # Canonicalize CIA enums (None if unknown)
    obj["confidentiality"] = norm_enum(obj.get("confidentiality"), allowed=CONF_LEVELS)
    obj["integrity"] = norm_enum(obj.get("integrity"), allowed=TRI_LEVELS)
    obj["availability"] = norm_enum(obj.get("availability"), allowed=TRI_LEVELS)

    return obj
