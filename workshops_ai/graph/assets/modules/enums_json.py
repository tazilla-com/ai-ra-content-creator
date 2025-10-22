# workshops_ai/graph/assets/modules/enums_json.py
from __future__ import annotations
from collections import deque
from collections.abc import Mapping
from typing import Any
from pydantic import BaseModel

# CIA enums (canonical)
CONF_LEVELS = {"PUBLIC", "INTERNAL", "PROTECTED", "STRICTLY_PROTECTED"}
TRI_LEVELS  = {"LOW", "MEDIUM", "HIGH"}


def norm_enum(value: str | None, *, allowed: set[str]) -> str | None:
    if value is None:
        return None
    v = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if v == "STRICTLYPROTECTED":
        v = "STRICTLY_PROTECTED"
    return v if v in allowed else None


def tri_from_text(text: str) -> str | None:
    s = (text or "").strip().lower()
    toks = {t.strip(",.;:!?") for t in s.split()}
    low = {"low", "lo", "nízká", "nizka", "nízke", "nizke", "nízky", "nizky"}
    med = {"med", "medium", "střední", "stredni", "stredna", "stredné", "středne"}
    hi  = {"high", "hi", "vysoká", "vysoka", "vysoké", "vysoke"}
    if toks & low: return "LOW"
    if toks & med: return "MEDIUM"
    if toks & hi:  return "HIGH"
    if s in {"low","medium","high"}: return s.upper()
    return None


def conf_from_text(text: str) -> str | None:
    s = (text or "").strip().lower().replace("-", " ")
    if any(tok in s.split() for tok in ["public","veřejné","verejne","publicní","publicne"]):
        return "PUBLIC"
    if any(tok in s.split() for tok in ["internal","interní","interni"]):
        return "INTERNAL"
    if any(tok in s.split() for tok in ["protected","chráněné","chranene","chránené","chranene"]):
        if "strict" in s or "přísně" in s or "prisne" in s:
            return "STRICTLY_PROTECTED"
        return "PROTECTED"
    if "strict" in s or "přísně" in s or "prisne" in s:
        return "STRICTLY_PROTECTED"
    if s in {"public"}: return "PUBLIC"
    if s in {"internal"}: return "INTERNAL"
    if s in {"protected"}: return "PROTECTED"
    if s in {"strictly protected","strictly_protected"}: return "STRICTLY_PROTECTED"
    return None


# -------- JSON-safe, recursion-free converters --------

_PRIMITIVES = (str, int, float, bool, type(None))

def jsonify_iter(root: Any, *, max_depth: int = 6) -> Any:
    seen: set[int] = set()
    def _set_root(v): nonlocal out; out = v
    out: Any = None
    q = deque([(root, 0, _set_root)])
    while q:
        obj, depth, setter = q.popleft()
        try:
            if isinstance(obj, _PRIMITIVES):
                setter(obj); continue
            oid = id(obj)
            if oid in seen:
                setter("<cycle>"); continue
            seen.add(oid)
            if depth >= max_depth:
                setter("<max_depth>"); continue
            if isinstance(obj, Mapping):
                container: dict[str, Any] = {}
                setter(container)
                for k, v in obj.items():
                    kk = str(k)
                    q.append((v, depth+1, lambda val, c=container, key=kk: c.__setitem__(key, val)))
                continue
            if isinstance(obj, (list, tuple, set)):
                it = list(obj)
                container: list[Any] = [None] * len(it)
                setter(container)
                for i, v in enumerate(it):
                    q.append((v, depth+1, lambda val, c=container, idx=i: c.__setitem__(idx, val)))
                continue
            if isinstance(obj, BaseModel):
                try:
                    fields = {k: v for k, v in obj.__dict__.items() if not str(k).startswith("_")}
                except Exception:
                    setter({"type": type(obj).__name__, "value": str(obj)}); continue
                container: dict[str, Any] = {}
                setter(container)
                for k, v in fields.items():
                    q.append((v, depth+1, lambda val, c=container, key=str(k): c.__setitem__(key, val)))
                continue
            d = getattr(obj, "__dict__", None)
            if isinstance(d, dict):
                container: dict[str, Any] = {}
                setter(container)
                for k, v in d.items():
                    if str(k).startswith("_"): continue
                    q.append((v, depth+1, lambda val, c=container, key=str(k): c.__setitem__(key, val)))
                continue
            setter(str(obj))
        except Exception:
            setter(f"<unserializable:{type(obj).__name__}>")
    return out


def as_plain_dicts(seq: Any) -> list[dict]:
    out: list[dict] = []
    for x in list(seq or []):
        val = jsonify_iter(x)
        out.append(val if isinstance(val, dict) else {"value": val})
    return out


def to_plain_dict(obj: Any, fallback: dict) -> dict:
    try:
        val = jsonify_iter(obj)
        return val if isinstance(val, dict) else {"value": val}
    except Exception:
        return fallback
