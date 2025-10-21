# workshops_ai/ui/formatting.py
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


# -----------------------------------------------------------------------------
# Markdown helpers
# -----------------------------------------------------------------------------

def bullet_list(
    items: Iterable[str],
    bullet: str = "-",
    indent: int = 0,
) -> str:
    """
    Render a simple Markdown bullet list.

    - Each item becomes a single bullet line.
    - Empty/whitespace-only items are ignored.
    - Multiline items are indented on subsequent lines for readability.
    """
    prefix = " " * max(indent, 0) + f"{bullet} "
    lines: List[str] = []

    for raw in items:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue

        # Indent subsequent lines so the bullet stands out.
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
    rows: Sequence[Tuple[str, str | None]],
    header_left: str = "Field",
    header_right: str = "Value",
) -> str:
    """
    Render a two-column Markdown table from (label, value) pairs.

    - Pipes '|' inside cells are escaped.
    - None values are rendered as an en dash '–'.
    - Minimal header row is added for correct rendering.
    """
    def esc(cell: str) -> str:
        return cell.replace("|", r"\|")

    out: List[str] = []
    out.append(f"| {esc(header_left)} | {esc(header_right)} |")
    out.append("| --- | --- |")

    for label, value in rows:
        l = esc(str(label) if label is not None else "")
        v = "–" if value is None else esc(str(value))
        out.append(f"| {l} | {v} |")

    return "\n".join(out)


def code_block(text: str, lang: str | None = None) -> str:
    """
    Wrap text in a fenced code block using triple backticks.

    - If `lang` is provided (e.g., 'json'), it's appended to the opening fence.
    - Triple backticks inside the payload are protected with a ZWJ.
    """
    safe = (text or "").replace("```", "`\u200d``")  # protect against fence collision
    fence = "```" + (lang or "")
    if not safe.endswith("\n"):
        safe += "\n"
    return f"{fence}\n{safe}```"


# -----------------------------------------------------------------------------
# Chunking helper
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

    Args:
        text: Input string.
        max_len: Historical parameter name for the limit.
        max_chars: Alias used by some callers; if provided, it overrides max_len.
    """
    # --- Compat: accept both parameter names
    limit = max_chars if (max_chars is not None) else max_len
    if limit is None:
        raise ValueError("Provide max_len or max_chars")
    if limit <= 0:
        raise ValueError("max_len/max_chars must be >= 1")

    # Use a single variable name below to avoid mistakes
    max_len = int(limit)

    s = (text or "").strip()
    if len(s) <= max_len:
        return [s] if s else []

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
        if line.strip().startswith(fence):
            if in_code:
                # Closing fence: include it, then close the block.
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

    # Pack a list of units (strings) into <= max_len chunks.
    def pack_units(units: List[str]) -> None:
        cur: List[str] = []
        cur_len = 0

        def flush_cur() -> None:
            nonlocal cur, cur_len
            if cur:
                chunks.append("".join(cur).strip())
                cur = []
                cur_len = 0

        for u in units:
            u_len = len(u)
            if not cur:
                if u_len <= max_len:
                    cur.append(u)
                    cur_len = u_len
                else:
                    subdivided = _subdivide(u, max_len)
                    if len(subdivided) == 1 and len(subdivided[0]) > max_len:
                        # Undividable (e.g., huge single word/line)
                        chunks.append(subdivided[0].strip())
                    else:
                        for part in subdivided:
                            if len(part) <= max_len:
                                if cur and cur_len + len(part) > max_len:
                                    flush_cur()
                                cur.append(part)
                                cur_len += len(part)
                            else:
                                flush_cur()
                                chunks.append(part.strip())
            else:
                if cur_len + u_len <= max_len:
                    cur.append(u)
                    cur_len += u_len
                else:
                    flush_cur()
                    if u_len <= max_len:
                        cur.append(u)
                        cur_len = u_len
                    else:
                        subdivided = _subdivide(u, max_len)
                        for part in subdivided:
                            if len(part) <= max_len:
                                if cur and cur_len + len(part) > max_len:
                                    flush_cur()
                                cur.append(part)
                                cur_len += len(part)
                            else:
                                flush_cur()
                                chunks.append(part.strip())
        flush_cur()

    # Subdivide non-code unit by natural boundaries.
    def _subdivide(unit: str, limit: int) -> List[str]:
        if len(unit) <= limit:
            return [unit]

        # 1) Paragraphs
        parts = _greedy_pack(unit, sep="\n\n", limit=limit)
        if all(len(p) <= limit for p in parts):
            return parts

        # 2) Lines
        parts = _greedy_pack(unit, sep="\n", limit=limit)
        if all(len(p) <= limit for p in parts):
            return parts

        # 3) Sentences (naive)
        import re
        sent_parts = re.split(r"(?<=[.!?])\s+", unit)
        if len(sent_parts) > 1:
            parts = _greedy_pack_from_list(sent_parts, joiner=" ", limit=limit)
            if all(len(p) <= limit for p in parts):
                return parts

        # 4) Words
        words = unit.split(" ")
        if len(words) > 1:
            parts = _greedy_pack_from_list(words, joiner=" ", limit=limit)
            if all(len(p) <= limit for p in parts):
                return parts

        # 5) Hard split
        return [unit[i: i + limit] for i in range(0, len(unit), limit)]

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
            units = _subdivide(seg_text, max_len)
            pack_units(units)

    return [c for c in (ch.strip() for ch in chunks) if c]


# -----------------------------------------------------------------------------
# Asset-type helpers
# -----------------------------------------------------------------------------

def _to_name(x: Any) -> str:
    """Extract a display name from dicts, models or plain strings."""
    if isinstance(x, dict):
        return (x.get("name") or x.get("title") or str(x)).strip()
    name = getattr(x, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return str(x).strip()


def format_asset_types(types: Iterable[Any]) -> str:
    """
    Render a safe, human-friendly bullet list of asset types.
    Accepts dicts, pydantic models or strings.
    """
    items = [_to_name(t) for t in (types or []) if t is not None]
    items = [i for i in items if i]  # drop empties
    if not items:
        return "_No types found_"
    return bullet_list(items)
