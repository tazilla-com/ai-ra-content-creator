# workshops_ai/wizard/scraper/text_processing.py
"""
Text preprocessing for scraped content.

Handles encoding fixes, Unicode normalization, whitespace cleanup.
"""
from __future__ import annotations

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# Try to import ftfy (optional dependency for encoding fixes)
try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False
    logger.debug(
        "ftfy not available, skipping encoding fixes. "
        "Install with: pip install ftfy"
    )


def preprocess_text(text: str) -> str:
    """
    Preprocess scraped text for LLM consumption.

    Steps:
    1. Fix encoding issues (mojibake)
    2. Normalize Unicode characters
    3. Replace special whitespace characters
    4. Normalize typography (quotes, dashes, ellipsis)
    5. Collapse whitespace

    Args:
        text: Raw text from HTML

    Returns:
        Cleaned and normalized text
    """
    if not text:
        return text

    # 1. Fix encoding issues (mojibake, garbled text)
    if HAS_FTFY:
        text = ftfy.fix_text(text)

    # 2. Normalize Unicode (canonical form)
    text = unicodedata.normalize("NFKC", text)

    # 3. Replace odd whitespace characters
    text = (
        text.replace("\xa0", " ")       # Non-breaking space
        .replace("\u202f", " ")          # Narrow no-break space
        .replace("\u200b", "")           # Zero-width space
        .replace("\t", " ")              # Tab
        .replace("\u2003", " ")          # Em space
        .replace("\u2002", " ")          # En space
    )

    # 4. Normalize typography (smart quotes, dashes)
    text = (
        text.replace(""", '"').replace(""", '"')  # Smart quotes
        .replace("'", "'").replace("'", "'")      # Smart apostrophes
        .replace("…", "...")                      # Ellipsis
        .replace("–", "-").replace("—", "-")      # En/em dashes
    )

    # 5. Collapse whitespace
    text = re.sub(r"[\t ]+", " ", text.strip())   # Multiple spaces → single
    text = re.sub(r"\n\s*", "\n", text)           # Trim line whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)        # Max 2 consecutive newlines

    return text


def limit_text(text: str, max_chars: int) -> str:
    """
    Limit text to maximum characters, preferring sentence boundaries.

    Args:
        text: Text to limit
        max_chars: Maximum characters allowed

    Returns:
        Limited text (truncated at sentence boundary if possible)
    """
    if len(text) <= max_chars:
        return text

    # Try to find last sentence boundary before limit
    truncated = text[:max_chars]

    # Look for sentence endings: . ! ? followed by space or newline
    sentence_ends = [
        truncated.rfind('. '),
        truncated.rfind('! '),
        truncated.rfind('? '),
        truncated.rfind('.\n'),
    ]

    last_sentence = max(sentence_ends)

    # If we found a sentence boundary in last 20% of text, use it
    if last_sentence > max_chars * 0.8:
        return truncated[:last_sentence + 1].strip()

    # Otherwise, hard truncate
    return truncated.strip()
