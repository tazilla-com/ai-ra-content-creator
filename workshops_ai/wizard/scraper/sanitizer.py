# workshops_ai/wizard/scraper/sanitizer.py
"""
Content sanitization to prevent prompt injection attacks.

SECURITY: This module protects against malicious content in scraped websites
that might attempt to manipulate the LLM.
"""
from __future__ import annotations

import re
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class ContentSanitizer:
    """
    Sanitize scraped content to prevent prompt injection attacks.

    Detection patterns for:
    - Direct instruction attempts
    - Role manipulation
    - Prompt leaking attempts
    - Delimiter injection
    - Excessive repetition (spam/DoS)
    """

    # Patterns that might indicate prompt injection attempts
    SUSPICIOUS_PATTERNS = [
        # Direct instruction attempts
        r"(?i)(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"(?i)you\s+are\s+(now|a)\s+",
        r"(?i)(system|assistant|user)\s*:\s*",
        r"(?i)new\s+(instructions?|task|role)",

        # Role manipulation
        r"(?i)act\s+as\s+(a\s+)?(?!customer|client|user)",  # Allow normal business terms
        r"(?i)pretend\s+(to\s+be|you\s+are)",
        r"(?i)simulate\s+(a\s+)?",

        # Prompt leaking attempts
        r"(?i)(show|reveal|display|print)\s+(your\s+)?(system\s+prompt|instructions|rules)",
        r"(?i)what\s+(are\s+)?(your\s+)?(instructions|rules|prompt)",

        # Delimiter injection
        r"```\s*(system|user|assistant)",
        r"<\|?(system|user|assistant)\|?>",
        r"\[SYSTEM\]|\[USER\]|\[ASSISTANT\]",
    ]

    # Suspicious repeated patterns
    MAX_REPEATED_CHARS = 50  # "AAAAA..." repeated
    MAX_REPEATED_WORDS = 10  # "test test test..."

    @staticmethod
    def sanitize(content: str, strict: bool = True) -> Tuple[str, List[str]]:
        """
        Sanitize content and return (cleaned_content, warnings).

        Args:
            content: Raw scraped content
            strict: If True, remove suspicious content. If False, just warn.

        Returns:
            Tuple of (sanitized_content, list_of_warnings)
        """
        if not content:
            return content, []

        warnings: List[str] = []
        sanitized = content

        # 1. Check for prompt injection patterns
        for pattern in ContentSanitizer.SUSPICIOUS_PATTERNS:
            matches = re.findall(pattern, sanitized, re.MULTILINE)
            if matches:
                warnings.append(f"Suspicious pattern detected: {pattern[:50]}...")
                logger.warning(
                    "prompt_injection_detected",
                    extra={"pattern": pattern[:100], "matches_count": len(matches)}
                )

                if strict:
                    # Replace with placeholder
                    sanitized = re.sub(
                        pattern,
                        "[CONTENT REMOVED]",
                        sanitized,
                        flags=re.IGNORECASE | re.MULTILINE
                    )

        # 2. Check for excessive repetition (spam/attack)
        if ContentSanitizer._has_excessive_repetition(sanitized):
            warnings.append("Excessive character/word repetition detected")
            logger.warning("excessive_repetition_detected")

            if strict:
                sanitized = ContentSanitizer._remove_repetitions(sanitized)

        # 3. Check for unusual character patterns
        non_printable = sum(
            1 for c in sanitized
            if not c.isprintable() and c not in '\n\r\t'
        )

        if non_printable > len(sanitized) * 0.05:  # More than 5% non-printable
            warnings.append(f"High non-printable character count: {non_printable}")
            logger.warning(
                "high_non_printable_chars",
                extra={"count": non_printable, "total": len(sanitized)}
            )

            if strict:
                sanitized = ''.join(
                    c for c in sanitized
                    if c.isprintable() or c in '\n\r\t'
                )

        # 4. Check for excessive special characters (obfuscation attempt)
        special_chars = sum(
            1 for c in sanitized
            if not c.isalnum() and not c.isspace()
        )

        if special_chars > len(sanitized) * 0.3:  # More than 30% special chars
            warnings.append(
                f"Unusually high special character ratio: {special_chars}/{len(sanitized)}"
            )
            logger.warning(
                "high_special_char_ratio",
                extra={"special": special_chars, "total": len(sanitized)}
            )

        # 5. Normalize whitespace (prevent whitespace-based attacks)
        sanitized = re.sub(r'\s+', ' ', sanitized)
        sanitized = re.sub(r'\n\s*\n\s*\n+', '\n\n', sanitized)  # Max 2 consecutive newlines

        return sanitized.strip(), warnings

    @staticmethod
    def _has_excessive_repetition(text: str) -> bool:
        """
        Detect spam-like repetition.

        Returns:
            True if excessive repetition detected
        """
        # Check character repetition
        char_runs = re.findall(
            r'(.)\1{' + str(ContentSanitizer.MAX_REPEATED_CHARS) + r',}',
            text
        )
        if char_runs:
            return True

        # Check word repetition
        words = text.split()
        if len(words) < 20:
            return False

        for i in range(len(words) - ContentSanitizer.MAX_REPEATED_WORDS):
            word = words[i]
            if len(word) < 3:  # Skip short words like "a", "is"
                continue

            count = sum(
                1 for w in words[i:i + ContentSanitizer.MAX_REPEATED_WORDS]
                if w == word
            )

            if count > ContentSanitizer.MAX_REPEATED_WORDS * 0.7:
                return True

        return False

    @staticmethod
    def _remove_repetitions(text: str) -> str:
        """
        Remove excessive repetitions.

        Returns:
            Text with repetitions removed
        """
        # Remove character runs (keep max 3)
        text = re.sub(r'(.)\1{20,}', r'\1\1\1', text)

        # Remove word repetitions (keep max 3)
        words = text.split()
        cleaned = []
        prev_word = None
        repeat_count = 0

        for word in words:
            if word == prev_word:
                repeat_count += 1
                if repeat_count < 3:  # Allow up to 3 repetitions
                    cleaned.append(word)
            else:
                cleaned.append(word)
                prev_word = word
                repeat_count = 0

        return ' '.join(cleaned)
