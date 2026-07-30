"""Detection of prompt extraction and system instruction disclosure attempts."""

import re
from typing import Optional

from languagemodelcommon.utilities.security.normalize import (
    REFUSAL_MESSAGE,
    normalize_for_detection,
)


class PromptExtractionAttempt(Exception):
    """Raised when a prompt extraction attempt is detected in user input."""

    def __init__(self) -> None:
        super().__init__(REFUSAL_MESSAGE)


PROMPT_EXTRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(repeat|output|print|show|display|reveal|disclose|give me|write out|put)\b"
        r".*\b(system prompt|system instructions?|your instructions?|your prompt|the prompt|system message|your configuration|your rules)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(what are|tell me|share)\b.*\b(your system instructions?|your system prompt|your prompt|your rules|your system message)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(text|everything|all)\s+(above|before this)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcode\s*block\b.*\b(above|instructions?|prompt|system)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(above|instructions?|prompt|system)\b.*\bcode\s*block\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(ignore|forget|disregard)\b.*\b(previous|above|prior|instructions?|rules)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(pretend|act as if|imagine|you are now)\b.*\b(no rules|no restrictions|unrestricted|jailbreak)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\binitial\s+(instructions?|prompt|message)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(the|your)\s+system\s*(:|message|prompt|instructions?)\b",
        re.IGNORECASE,
    ),
]


def detect_prompt_extraction(*, text: str) -> Optional[str]:
    """Check if user input attempts to extract system instructions.

    Returns the matched pattern description if detected, None otherwise.
    """
    normalized = normalize_for_detection(text=text)
    for pattern in PROMPT_EXTRACTION_PATTERNS:
        if pattern.search(normalized):
            return pattern.pattern
    return None
