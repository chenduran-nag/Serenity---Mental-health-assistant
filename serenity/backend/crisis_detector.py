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

CRISIS_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bsuicid(?:e|al|ing)?\b", re.IGNORECASE),
    re.compile(r"\bself[\s-]?harm\b", re.IGNORECASE),
    re.compile(r"\bhurt myself\b", re.IGNORECASE),
    re.compile(r"\bend my life\b", re.IGNORECASE),
    re.compile(r"\bkill myself\b", re.IGNORECASE),
    re.compile(r"\bwant to die\b", re.IGNORECASE),
    re.compile(r"\bcan(?:not|'t) go on\b", re.IGNORECASE),
    re.compile(r"\boverdos(?:e|ing)\b", re.IGNORECASE),
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
