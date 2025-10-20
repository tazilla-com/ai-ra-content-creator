# -*- coding: utf-8 -*-
from __future__ import annotations
import json, logging, types, inspect, traceback, typing as t
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from typing import get_origin, get_args

log = logging.getLogger("workshops_ai.diagnostics")

class ProbeResult(t.TypedDict, total=False):
    summary: str
    first_cycle_path: t.Optional[str]
    field_mismatches: t.List[str]
    union_fields: t.List[str]
    aliases: t.Dict[str, str]
    model: str

def _typename(x) -> str:
    try:
        return type(x).__name__
    except Exception:
        return repr(type(x))

def _is_mapping(x) -> bool:
    return isinstance(x, dict)

def _is_sequence(x) -> bool:
    return isinstance(x, (list, tuple))

def _field_expected_types(fi: FieldInfo) -> t.Tuple[t.Type, ...]:
    ann = fi.annotation
    if ann is None:
        return tuple()
    origin = get_origin(ann)
    if origin is t.Union or origin is types.UnionType:  # py3.10/3.11 | union
        return tuple(a for a in get_args(ann))
    if isinstance(ann, type):
        return (ann,)
    return tuple()

def _walk(obj, path="root", seen: dict[int, str]|None=None, depth=0, max_depth=128):
    """
    Depth-first průchod s detekcí cyklů a limitací hloubky.
    Vrací (first_cycle_path, notes:list[str])
    """
    if seen is None:
        seen = {}
    obj_id = id(obj)
    if obj_id in seen:
        return seen[obj_id], [f"{path}: CYCLE → {seen[obj_id]} -> {_typename(obj)}"]
    seen[obj_id] = path
    notes: list[str] = []
    if depth >= max_depth:
        notes.append(f"{path}: MAX_DEPTH {_typename(obj)}")
        return None, notes

    # Pydantic BaseModel
    if isinstance(obj, BaseModel):
        # uniony, aliasy, validators (jen jako metadata; nic neopravujeme)
        union_fields: list[str] = []
        field_mismatches: list[str] = []
        aliases: dict[str, str] = {}
        for name, fi in obj.model_fields.items():
            aliases[name] = (fi.alias or name)
            exp = _field_expected_types(fi)
            try:
                val = getattr(obj, name)
            except Exception as e:
                notes.append(f"{path}.{name}: getattr error {e!r}")
                continue
            if exp:
                if not any(isinstance(val, et) for et in exp if isinstance(et, type)):
                    # pro Uniony si jen poznačíme
                    union_fields.append(f"{path}.{name}: expected {exp}, got {_typename(val)}")
            # hlubší průchod
            cpath, subnotes = _walk(val, f"{path}.{name}", seen, depth+1, max_depth)
            notes.extend(subnotes)
            if cpath and not any(n.startswith("FIRST_CYCLE=") for n in notes):
                notes.append(f"FIRST_CYCLE={cpath}")
        # Shrnutí metadat (přilepíme na konec)
        obj.__dict__.setdefault("_probe_meta_", {
            "union_fields": union_fields,
            "aliases": aliases,
        })
        return None, notes

    # dict
    if _is_mapping(obj):
        for k, v in list(obj.items()):
            cpath, subnotes = _walk(v, f"{path}[{k!r}]", seen, depth+1, max_depth)
            notes.extend(subnotes)
            if cpath and not any(n.startswith("FIRST_CYCLE=") for n in notes):
                notes.append(f"FIRST_CYCLE={cpath}")
        return None, notes

    # list/tuple
    if _is_sequence(obj):
        for i, v in enumerate(list(obj)):
            cpath, subnotes = _walk(v, f"{path}[{i}]", seen, depth+1, max_depth)
            notes.extend(subnotes)
            if cpath and not any(n.startswith("FIRST_CYCLE=") for n in notes):
                notes.append(f"FIRST_CYCLE={cpath}")
        return None, notes

    # scalars
    return None, notes

def probe_dump(obj) -> ProbeResult:
    first_cycle_path, notes = _walk(obj)
    meta = {}
    if isinstance(obj, BaseModel):
        meta = obj.__dict__.get("_probe_meta_", {})
    # vyzobání první FIRST_CYCLE=...
    fcp = None
    for n in notes:
        if n.startswith("FIRST_CYCLE="):
            fcp = n.split("=", 1)[1]
            break
    union_fields = meta.get("union_fields", [])
    aliases = meta.get("aliases", {})
    # pole s jasnými nesoulady typů (jen ty uniony vypíšeme celé)
    mism = [n for n in notes if "CYCLE" in n or "MAX_DEPTH" in n]
    return t.cast(ProbeResult, {
        "summary": f"type={_typename(obj)} notes={len(notes)}",
        "first_cycle_path": fcp,
        "field_mismatches": union_fields + mism,
        "union_fields": union_fields,
        "aliases": aliases,
        "model": getattr(obj, "__class__", type(obj)).__name__,
    })

def dump_crash_packet(request_id: str, payload_preview: dict, guarded_obj, exc: BaseException) -> str:
    tb = traceback.TracebackException.from_exception(exc, capture_locals=True)
    diag: ProbeResult = {}
    try:
        diag = probe_dump(guarded_obj)
    except Exception as e2:
        diag = {"summary": f"probe_failed: {e2!r}"}
    packet = {
        "request_id": request_id,
        "error": repr(exc),
        "traceback": "".join(tb.format()),
        "payload_preview": payload_preview,
        "guarded_type": _typename(guarded_obj),
        "diagnostics": diag,
    }
    path = f"/app/logs/crashpack-{request_id}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(packet, f, ensure_ascii=False, indent=2)
    except Exception as e3:
        log.exception("CRASHPACK_WRITE_FAILED", extra={"err": repr(e3), "request_id": request_id})
    return path
