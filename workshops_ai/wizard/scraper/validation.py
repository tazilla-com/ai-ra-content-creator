# workshops_ai/wizard/scraper/validation.py
"""
Content validation for scraped text.

Checks for minimum length, quality, spam detection.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def is_valid_content(content: str, min_chars: int = 100) -> bool:
    """
    Check if content is substantial enough to be useful.

    Validation criteria:
    1. Minimum length (chars)
    2. Sufficient alphanumeric content (not just punctuation/whitespace)
    3. Sufficient word count
    4. Sufficient unique words (spam detection)

    Args:
        content: Text content to validate
        min_chars: Minimum character count required

    Returns:
        True if content passes all validation checks
    """
    if not content:
        logger.debug("validation_failed", extra={"reason": "empty_content"})
        return False

    cleaned = content.strip()

    # 1. Too short
    if len(cleaned) < min_chars:
        logger.debug(
            "validation_failed",
            extra={"reason": "too_short", "length": len(cleaned), "min_required": min_chars}
        )
        return False

    # 2. Check for mostly whitespace/punctuation
    alphanumeric = sum(c.isalnum() for c in cleaned)
    if alphanumeric < min_chars * 0.5:  # At least 50% should be actual content
        logger.debug(
            "validation_failed",
            extra={
                "reason": "low_alphanumeric_ratio",
                "alphanumeric": alphanumeric,
                "total": len(cleaned)
            }
        )
        return False

    # 3. Check for repetitive content (spam detection)
    words = cleaned.split()
    if len(words) < 20:  # Less than 20 words = too short
        logger.debug(
            "validation_failed",
            extra={"reason": "too_few_words", "word_count": len(words)}
        )
        return False

    # 4. Check unique word ratio (spam detection)
    unique_words = len(set(words))
    uniqueness_ratio = unique_words / len(words)

    if uniqueness_ratio < 0.3:  # Less than 30% unique = likely spam
        logger.debug(
            "validation_failed",
            extra={
                "reason": "low_uniqueness",
                "unique_words": unique_words,
                "total_words": len(words),
                "ratio": uniqueness_ratio
            }
        )
        return False

    logger.debug(
        "validation_passed",
        extra={
            "length": len(cleaned),
            "word_count": len(words),
            "unique_words": unique_words,
            "uniqueness_ratio": uniqueness_ratio
        }
    )

    return True
