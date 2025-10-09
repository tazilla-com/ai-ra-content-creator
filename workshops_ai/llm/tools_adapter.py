# workshops_ai/llm/tools_adapter.py
import re
from typing import Any, Dict, List, Tuple

def sanitize_tools(functions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    name_map: Dict[str, str] = {}
    out: List[Dict[str, Any]] = []
    for f in functions:
        fn = dict(f.get("function") or {})
        name = fn.get("name", "")
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", name)
        name_map[safe] = name
        fn["name"] = safe
        sf = dict(f); sf["function"] = fn
        out.append(sf)
    return out, name_map

def canonical_name(safe: str, name_map: Dict[str, str]) -> str:
    return name_map.get(safe, safe)
