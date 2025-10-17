# workshops_ai/ui/formatting.py
# -*- coding: utf-8 -*-
"""
Utilities for formatting human-facing Markdown text in chat bubbles.

We keep JSON payloads for the right-side audit panel, while LEFT chat bubbles
should be readable, concise Markdown. These helpers centralize common patterns.

Public helpers:
- bullet_list(items)      -> Markdown bullet list
- kv_table(rows)          -> Two-column Markdown table (label → value)
- code_block(text, lang)  -> Triple backtick fenced code block
- chunk(text, max_len)    -> Split long text into reasonable chunks

Design goals:
- Zero external dependencies
- Predictable output (no surprises for downstream renderers)
- Gentle handling of edge cases (empty inputs, long lines, etc.)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


__all__ = [
    "bullet_list",
    "kv_table",
    "code_block",
    "chunk",
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

    Args:
        items: Strings to render as bullets.
        bullet: Bullet prefix (usually "-" or "*").
        indent: Number of spaces to prepend to each bullet.

    Returns:
        Markdown string with one bullet per (non-empty) item.

    Example:
        bullet_list(["Alpha", "Bravo", "Charlie"])
        ->
        "- Alpha\n- Bravo\n- Charlie"
    """
    prefix = " " * max(indent, 0) + f"{bullet} "
    lines: List[str] = []

    for raw in items:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue

        # For multiline items, indent subsequent lines so the bullet stands out.
        split = text.splitlines() or [text]
        first = split[0]
        lines.append(prefix + first)

        if len(split) > 1:
            # Indent continuation lines to align under the text (not under the dash).
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

    - Pipes '|' inside cells are escaped to avoid breaking the table.
    - None values are rendered as an en dash '–' to signal "no value".
    - A minimal header row is added for proper Markdown rendering.

    Args:
        rows: Sequence of (label, value) pairs.
        header_left: Left column header text (localized upstream if needed).
        header_right: Right column header text.

    Returns:
        Markdown table as a string.

    Example:
        kv_table([("Type", "Information system"), ("Owner", None)])
        ->
        | Field | Value |
        | --- | --- |
        | Type | Information system |
        | Owner | – |
    """
    def esc(cell: str) -> str:
        # Escape pipes; other characters are generally safe in GitHub/Std MD.
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

    - If `lang` is provided (e.g., 'json', 'yaml', 'python'), it is appended
      to the opening fence to enable syntax highlighting in many renderers.
    - If the payload contains triple backticks, we insert a zero-width
      joiner to prevent accidental fence termination.

    Args:
        text: The code or preformatted text to wrap.
        lang: Optional language hint for syntax highlighting.

    Returns:
        Markdown code block string.

    Example:
        code_block('{"ok": true}', "json")
        ->
        ```json
        {"ok": true}
        ```
    """
    safe = (text or "").replace("```", "`\u200d``")  # protect against fence collision
    fence = "```" + (lang or "")
    # Ensure a trailing newline before closing the fence for correct rendering.
    if not safe.endswith("\n"):
        safe += "\n"
    return f"{fence}\n{safe}```"


# -----------------------------------------------------------------------------
# Chunking helper
# -----------------------------------------------------------------------------

def chunk(text: str, max_len: int) -> List[str]:
    """
    Split a long string into chunks not exceeding `max_len` characters
    *whenever reasonable*, favoring natural boundaries:

    1) Keep fenced code blocks intact (never split them if avoidable).
    2) Prefer splitting on double newlines (paragraphs).
    3) Then single newlines (lines).
    4) Then sentence boundaries.
    5) Then words.
    6) As a last resort, hard-split.

    Notes:
    - If a single fenced code block exceeds `max_len`, it is emitted as a
      single chunk that may be longer than `max_len` to preserve integrity.
    - This function is intentionally conservative; it aims for readable chunks,
      not perfect linguistic segmentation.

    Args:
        text: Input string.
        max_len: Target maximum length per chunk (must be >= 1).

    Returns:
        List of chunk strings.

    Example:
        chunk("A long text...", 800)
        -> ["A long ... first part", "second part", ...]
    """
    if max_len <= 0:
        raise ValueError("max_len must be >= 1")

    s = (text or "").strip()
    if len(s) <= max_len:
        return [s] if s else []

    # --- Step 0: Split into alternating segments: normal-text vs fenced code.
    # We treat fenced code blocks (``` ... ```) as atomic units for splitting.
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
            # Fence line goes with whichever side it belongs to.
            if in_code:
                # Closing fence: include it and close the block.
                buf.append(line)
                flush_buffer(is_code=True)
                in_code = False
            else:
                # Opening fence: flush any accumulated normal text first.
                flush_buffer(is_code=False)
                in_code = True
                buf.append(line)
        else:
            buf.append(line)

    # Flush remainder.
    flush_buffer(is_code=in_code)

    chunks: List[str] = []

    # Helper to pack a list of units (strings) into chunks <= max_len.
    def pack_units(units: List[str]) -> None:
        cur: List[str] = []
        cur_len = 0

        def flush_cur():
            nonlocal cur, cur_len
            if cur:
                chunks.append("".join(cur).strip())
                cur = []
                cur_len = 0

        for u in units:
            u_len = len(u)
            if not cur:
                # Start a new chunk with the unit, even if it is large.
                if u_len <= max_len:
                    cur.append(u)
                    cur_len = u_len
                else:
                    # If a single unit is too large, try to subdivide it further.
                    subdivided = _subdivide(u, max_len)
                    if len(subdivided) == 1 and len(subdivided[0]) > max_len:
                        # Non-divisible (e.g., extremely long word/code line):
                        chunks.append(subdivided[0].strip())
                    else:
                        for part in subdivided:
                            if len(part) <= max_len:
                                if cur and cur_len + len(part) > max_len:
                                    flush_cur()
                                cur.append(part)
                                cur_len += len(part)
                            else:
                                # As last resort, append as-is.
                                flush_cur()
                                chunks.append(part.strip())
                                cur_len = 0
                                cur = []
            else:
                # Try to append to current chunk.
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

    # Subdivision strategy for normal text (no code fences inside `unit`).
    def _subdivide(unit: str, limit: int) -> List[str]:
        if len(unit) <= limit:
            return [unit]

        # 1) Paragraphs (double newline)
        parts = _greedy_pack(unit, sep="\n\n", limit=limit)
        if all(len(p) <= limit for p in parts):
            return parts

        # 2) Lines (single newline)
        parts = _greedy_pack(unit, sep="\n", limit=limit)
        if all(len(p) <= limit for p in parts):
            return parts

        # 3) Sentences (very naive split on punctuation + space/newline)
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

        # 5) Last resort: hard split
        return [unit[i : i + limit] for i in range(0, len(unit), limit)]

    # Greedy packing using a separator (keeps separators).
    def _greedy_pack(text_block: str, sep: str, limit: int) -> List[str]:
        items = text_block.split(sep)
        return _greedy_pack_from_list(items, joiner=sep, limit=limit)

    # Greedy packing from pre-split items; joins them with `joiner` respecting limit.
    def _greedy_pack_from_list(items: List[str], joiner: str, limit: int) -> List[str]:
        out: List[str] = []
        cur: List[str] = []
        cur_len = 0
        join_len = len(joiner)

        for i, piece in enumerate(items):
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

    # Process segments: keep code segments intact; subdivide normal ones.
    for seg_text, is_code in segments:
        if not seg_text:
            continue
        if is_code:
            # Keep fenced blocks intact; if one block is huge, keep it as a single (possibly > max_len) chunk.
            # Users typically prefer intact code over hard splits.
            pack_units([seg_text])
        else:
            # Subdivide normal text into manageable units and pack.
            units = _subdivide(seg_text, max_len)
            pack_units(units)

    # Final cleanup: remove empty chunks and trim whitespace.
    return [c for c in (ch.strip() for ch in chunks) if c]
