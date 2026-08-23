"""Keyword and regex based crisis detection for Serenity."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final


CRISIS_RESOURCE_MESSAGE: Final[str] = (
    "I'm really concerned about what you've shared. Please reach out to "
    "iCall India at 9152987821 or Vandrevala Foundation at 1860-2662-345 "
    "immediately. You are not alone."
)

# Every phrase is matched across its inflected forms. The original patterns were
# fixed strings, so "I have been thinking about ending my life" went undetected
# because only "end my life" was listed - a miss observed against the live stack.
# Recall is deliberately favoured over precision here: an unnecessary helpline
# message is a small cost, a missed disclosure is not.
CRISIS_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bsuicid(?:e|es|al|ally|ing)?\b", re.IGNORECASE),
    re.compile(r"\bself[\s-]?harm(?:ing|ed|s)?\b", re.IGNORECASE),
    # "cut myself a slice" / "cut myself some slack" are benign, so a following
    # determiner rules the match out. Verbs are listed separately because
    # "cutting" doubles its consonant.
    re.compile(
        r"\b(?:hurt(?:ing|s)?|harm(?:ing|s|ed)?|injur(?:ing|es|ed)?|cut(?:ting|s)?)"
        r"\s+myself\b(?!\s+(?:a|an|the|some|another)\b)",
        re.IGNORECASE,
    ),
    re.compile(r"\bend(?:ing|s|ed)?\s+my\s+(?:own\s+)?life\b", re.IGNORECASE),
    re.compile(r"\bend(?:ing|s|ed)?\s+it\s+all\b", re.IGNORECASE),
    re.compile(r"\bkill(?:ing|s|ed)?\s+myself\b", re.IGNORECASE),
    re.compile(r"\btak(?:e|es|ing|en)\s+my\s+own\s+life\b", re.IGNORECASE),
    re.compile(r"\bwant(?:ed|ing|s)?\s+to\s+die\b", re.IGNORECASE),
    re.compile(r"\bdon'?t\s+want\s+to\s+(?:live|be\s+here|wake\s+up)\b", re.IGNORECASE),
    re.compile(r"\bbetter\s+off\s+dead\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:reason|point)\s+(?:in\s+)?(?:to\s+)?liv(?:e|ing)\b", re.IGNORECASE),
    re.compile(r"\bcan(?:not|'?t)\s+go\s+on\b", re.IGNORECASE),
    re.compile(r"\boverdos(?:e|es|ed|ing)\b", re.IGNORECASE),
)


# iOS, macOS, and Word substitute typographic apostrophes automatically, so a
# pasted "I can't go on" often arrives with U+2019 and would slip past the
# ASCII-only patterns above.
_APOSTROPHE_TRANSLATION: Final[dict[int, str]] = {
    0x2018: "'",  # left single quotation mark
    0x2019: "'",  # right single quotation mark
    0x02BC: "'",  # modifier letter apostrophe
    0x2032: "'",  # prime
}


@dataclass(frozen=True)
class CrisisDetectionResult:
    """Structured crisis detection outcome."""

    crisis_detected: bool
    matched_patterns: tuple[str, ...]
    resource_message: str | None


def normalize_for_matching(text: str) -> str:
    """Fold typographic apostrophe variants down to the ASCII form."""

    return text.translate(_APOSTROPHE_TRANSLATION)


def detect_crisis(text: str) -> CrisisDetectionResult:
    """Detect whether user input suggests immediate crisis risk."""

    normalized = normalize_for_matching(text)
    matches = tuple(pattern.pattern for pattern in CRISIS_PATTERNS if pattern.search(normalized))
    return CrisisDetectionResult(
        crisis_detected=bool(matches),
        matched_patterns=matches,
        resource_message=CRISIS_RESOURCE_MESSAGE if matches else None,
    )
