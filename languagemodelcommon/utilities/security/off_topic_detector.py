"""Detection of off-topic and encoding-based content manipulation attempts."""

import re
from typing import Optional

from languagemodelcommon.utilities.security.normalize import (
    REFUSAL_MESSAGE,
    normalize_for_detection,
)


ENCODING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(decode|encode|convert|translate|interpret)\b.*\b(ascii|base64|hex|binary|unicode|rot13|utf|latin)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(ascii|base64|hex|binary|unicode|rot13|utf|latin)\b.*\b(decode|encode|convert|translate|interpret)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(what(\'s| is| does))\b.*\b(in ascii|in base64|in hex|in binary)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\b\d{2,3}\b[\s,]+){5,}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(respond|answer|reply|output)\b.*\b(in ascii|in base64|in hex|in binary|encoded)\b",
        re.IGNORECASE,
    ),
]

OFFENSIVE_CONTENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(is\s+(it|this|that)\s+)?(a\s+)?(derogatory|offensive|slur|racial|racist|sexist|hate)\s*(term|word|phrase|language)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(what|why)\b.*\b(derogatory|offensive|slur|racist|sexist)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(explain|define|tell me about)\b.*\b(slur|derogatory|hate speech|offensive term)\b",
        re.IGNORECASE,
    ),
]


class OffTopicAttempt(Exception):
    """Raised when an off-topic manipulation attempt is detected."""

    def __init__(self, *, category: str) -> None:
        self.category = category
        super().__init__(REFUSAL_MESSAGE)


def detect_encoding_manipulation(*, text: str) -> Optional[str]:
    """Check if user input attempts encoding-based content manipulation.

    Returns the matched pattern description if detected, None otherwise.
    """
    for pattern in ENCODING_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def detect_offensive_content_request(*, text: str) -> Optional[str]:
    """Check if user input requests discussion of offensive language.

    Returns the matched pattern description if detected, None otherwise.
    """
    for pattern in OFFENSIVE_CONTENT_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def detect_off_topic_manipulation(*, text: str) -> Optional[str]:
    """Check for any off-topic manipulation attempt (encoding or offensive content).

    Returns the category of detection if found, None otherwise.
    """
    cleaned = normalize_for_detection(text=text)

    result = detect_encoding_manipulation(text=cleaned)
    if result:
        return "encoding_manipulation"

    result = detect_offensive_content_request(text=cleaned)
    if result:
        return "offensive_content_request"

    return None
