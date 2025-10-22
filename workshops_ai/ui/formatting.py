# File: workshops_ai/ui/formatting.py
# -*- coding: utf-8 -*-
"""
Tiny, dependency-free helpers to render human-friendly Markdown for chat bubbles.

Public helpers:
- bullet_list(items)        → Markdown bullet list
- kv_table(rows)            → Two-column Markdown table (label → value)
- code_block(text, lang)    → Triple backtick fenced code block
- chunk(text, max_len|max_chars)
                           → Split long text into readable chunks
- format_asset_types(types) → Safe, human-friendly list of asset type names

Design goals:
- Predictable output (no surprises for renderers)
- Gentle handling of edge cases (empty inputs, long lines, mixed item types)
- Backwards compatibility for callers (e.g., chunk(..., max_chars=...))
- Never crash: all helpers are defensive and avoid raising in normal use
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Any

__all__ = [
    "bullet_list",
    "kv_table",
    "code_block",
    "chunk",
    "format_asset_types",
]

# Default chunk size used when callers omit both max_len and max_chars
_DEFAULT_CHUNK = 1800


# -----------------------------------------------------------------------------
# Normalization helpers (internal)
# -----------------------------------------------------------------------------
def _as_label(x: Any) -> str:
    """
    Best-effort stringify for common shapes (dicts, pydantic-like models, plain strings).
    Never raises; returns '' on failure.
    """
    try:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            for k in ("name", "label", "title", "id", "type"):
                v = x.get(k)
                if isinstance(v, str) and v.strip():
                    return v
                if v is not None:
                    return str(v)
            return str(x)
        # Try common attribute names on objects
        for attr in ("name", "label", "title", "id", "type"):
            if hasattr(x, attr):
                v = getattr(x, attr)
                if isinstance(v, str) and v.strip():
                    return v
                if v is not None:
                    return str(v)
        return str(x)
    except Exception:
        return ""


def _to_labels_any(items: Iterable[Any]) -> List[str]:
    """
    Normalize an iterable of arbitrary items to a de-duplicated list of non-empty strings.
    Preserves input order.
    """
    out: List[str] = []
    if not items:
        return out
    seen = set()
    for it in items:
        lbl = (_as_label(it) or "").strip()
        if not lbl or lbl in seen:
            continue
        seen.add(lbl)
        out.append(lbl)
    return out


# -----------------------------------------------------------------------------
# Markdown helpers
# -----------------------------------------------------------------------------
def bullet_list(
    items: Iterable[Any],
    bullet: str = "-",
    indent: int = 0,
) -> str:
    """
    Render a simple Markdown bullet list.

    - Accepts any item types; everything is normalized to strings.
    - Empty/whitespace-only items are ignored.
    - Multiline items are indented on subsequent lines for readability.
    - Never raises.
    """
    prefix = " " * max(indent, 0) + f"{bullet} "
    lines: List[str] = []

    for text in _to_labels_any(items):
        split = text.splitlines() or [text]
        first = split[0]
        lines.append(prefix + first)

        if len(split) > 1:
            # Align continuation lines under the text (not under the dash).
            cont_prefix = " " * (len(prefix))
            for cont in split[1:]:
                lines.append(cont_prefix + cont)

    return "\n".join(lines)


def kv_table(
    rows: Sequence[Tuple[Any, Any]],
    header_left: str = "Field",
    header_right: str = "Value",
) -> str:
    """
    Render a two-column Markdown table from (label, value) pairs.

    - Pipes '|' inside cells are escaped.
    - None values are rendered as an en dash '–'.
    - Minimal header row is added for correct rendering.
    - Never raises.
    """
    def esc(cell: str) -> str:
        return cell.replace("|", r"\|")

    out: List[str] = []
    out.append(f"| {esc(header_left)} | {esc(header_right)} |")
    out.append("| --- | --- |")

    for label, value in (rows or []):
        l = esc(str(label) if label is not None else "")
        v = "–" if value is None else esc(str(value))
        out.append(f"| {l} | {v} |")

    return "\n".join(out)


def code_block(text: str, lang: str | None = None) -> str:
    """
    Wrap text in a fenced code block using triple backticks.

    - If `lang` is provided (e.g., 'json'), it's appended to the opening fence.
    - Triple backticks inside the payload are protected with a ZWJ.
    - Always ends with a newline for proper rendering.
    - Never raises.
    """
    safe = (text or "").replace("```", "`\u200d``")  # protect against fence collision
    fence = "```" + (lang or "")
    if not safe.endswith("\n"):
        safe += "\n"
    return f"{fence}\n{safe}```"


# -----------------------------------------------------------------------------
# Chunking helper (backwards compatible and defensive)
# -----------------------------------------------------------------------------
def chunk(
    text: str,
    max_len: int | None = None,   # legacy name
    *,
    max_chars: int | None = None  # preferred alias used by newer code
) -> List[str]:
    """
    Split a long string into chunks not exceeding the limit, preferring natural
    boundaries (paragraphs, lines, sentences, words), and keeping fenced code
    blocks intact whenever reasonable.

    Back-compat: accepts either `max_len` or `max_chars`. If neither is provided,
    a sensible default of 1800 is used. Negative/zero limits are coerced to 1.

    Never raises in normal use.
    """
    # --- Compat: accept both parameter names
    limit = max_chars if (max_chars is not None) else max_len
    try:
        if limit is None:
            limit = _DEFAULT_CHUNK
        limit = int(limit)
        if limit <= 0:
            limit = 1
    except Exception:
        limit = _DEFAULT_CHUNK

    s = (text or "")
    if not s:
        return [""]

    # Fast path
    if len(s) <= limit:
        return [s]

    # --- Step 0: split into alternating segments (normal text vs fenced code)
    segments: List[tuple[str, bool]] = []  # (segment_text, is_code)
    lines = s.splitlines(keepends=True)

    in_code = False
    buf: List[str] = []
    fence = "```"

    def flush_buffer(is_code: bool) -> None:
        if buf:
            segments.append(("".join(buf), is_code))
            buf.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(fence):
            if in_code:
                # Closing fence: include it.
                buf.append(line)
                flush_buffer(is_code=True)
                in_code = False
            else:
                # Opening fence: flush text before starting code block.
                flush_buffer(is_code=False)
                in_code = True
                buf.append(line)
        else:
            buf.append(line)
    flush_buffer(is_code=in_code)

    chunks: List[str] = []

    # Pack a list of units (strings) into <= limit chunks.
    def pack_units(units: List[str]) -> None:
        cur: List[str] = []
        cur_len = 0

        def flush_cur() -> None:
            nonlocal cur, cur_len
            if cur:
                chunks.append("".join(cur))
                cur = []
                cur_len = 0

        for u in units:
            u_len = len(u)
            if not cur:
                if u_len <= limit:
                    cur.append(u)
                    cur_len = u_len
                else:
                    for part in _subdivide(u, limit):
                        if len(part) <= limit:
                            if cur and cur_len + len(part) > limit:
                                flush_cur()
                            cur.append(part)
                            cur_len += len(part)
                        else:
                            flush_cur()
                            chunks.append(part)
            else:
                if cur_len + u_len <= limit:
                    cur.append(u)
                    cur_len += u_len
                else:
                    flush_cur()
                    if u_len <= limit:
                        cur.append(u)
                        cur_len = u_len
                    else:
                        for part in _subdivide(u, limit):
                            if len(part) <= limit:
                                if cur and cur_len + len(part) > limit:
                                    flush_cur()
                                cur.append(part)
                                cur_len += len(part)
                            else:
                                flush_cur()
                                chunks.append(part)
        flush_cur()

    # Subdivide non-code unit by natural boundaries.
    def _subdivide(unit: str, lim: int) -> List[str]:
        if len(unit) <= lim:
            return [unit]

        # 1) Paragraphs
        parts = _greedy_pack(unit, sep="\n\n", limit=lim)
        if all(len(p) <= lim for p in parts):
            return parts

        # 2) Lines
        parts = _greedy_pack(unit, sep="\n", limit=lim)
        if all(len(p) <= lim for p in parts):
            return parts

        # 3) Sentences (naive)
        try:
            import re
            sent_parts = re.split(r"(?<=[.!?])\s+", unit)
            if len(sent_parts) > 1:
                parts = _greedy_pack_from_list(sent_parts, joiner=" ", limit=lim)
                if all(len(p) <= lim for p in parts):
                    return parts
        except Exception:
            pass

        # 4) Words
        words = unit.split(" ")
        if len(words) > 1:
            parts = _greedy_pack_from_list(words, joiner=" ", limit=lim)
            if all(len(p) <= lim for p in parts):
                return parts

        # 5) Hard split as last resort
        return [unit[i : i + lim] for i in range(0, len(unit), lim)]

    def _greedy_pack(text_block: str, sep: str, limit: int) -> List[str]:
        items = text_block.split(sep)
        return _greedy_pack_from_list(items, joiner=sep, limit=limit)

    def _greedy_pack_from_list(items: List[str], joiner: str, limit: int) -> List[str]:
        out: List[str] = []
        cur: List[str] = []
        cur_len = 0
        join_len = len(joiner)

        for piece in items:
            piece = piece if piece is not None else ""
            piece_len = len(piece)

            if not cur:
                cur.append(piece)
                cur_len = piece_len
                continue

            projected = cur_len + join_len + piece_len
            if projected <= limit:
                cur.append(piece)
                cur_len = projected
            else:
                out.append(joiner.join(cur))
                cur = [piece]
                cur_len = piece_len

        if cur:
            out.append(joiner.join(cur))
        return out

    # Keep code blocks intact; subdivide normal text.
    for seg_text, is_code in segments:
        if not seg_text:
            continue
        if is_code:
            pack_units([seg_text])
        else:
            units = _subdivide(seg_text, limit)
            pack_units(units)

    # Ensure no empty chunks leak out
    return [c for c in chunks if c]


# -----------------------------------------------------------------------------
# Asset-type helpers
# -----------------------------------------------------------------------------
def format_asset_types(types: Iterable[Any]) -> str:
    """
    Render a safe, human-friendly bullet list of asset types.
    Accepts dicts, pydantic models or strings. Never raises.
    """
    labels = _to_labels_any(types)
    if not labels:
        return "_No types found_"
    return bullet_list(labels)
