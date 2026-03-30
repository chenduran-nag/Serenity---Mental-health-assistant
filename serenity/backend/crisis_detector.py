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


@dataclass(frozen=True)
class CrisisDetectionResult:
    """Structured crisis detection outcome."""

    crisis_detected: bool
    matched_patterns: tuple[str, ...]
    resource_message: str | None


def detect_crisis(text: str) -> CrisisDetectionResult:
    """Detect whether user input suggests immediate crisis risk."""

    matches = tuple(pattern.pattern for pattern in CRISIS_PATTERNS if pattern.search(text))
    return CrisisDetectionResult(
        crisis_detected=bool(matches),
        matched_patterns=matches,
        resource_message=CRISIS_RESOURCE_MESSAGE if matches else None,
    )
