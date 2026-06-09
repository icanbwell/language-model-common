"""Input normalization for security detection.

Centralizes Unicode normalization, zero-width stripping, and homoglyph
mapping so both prompt_extraction_detector and off_topic_detector apply
consistent pre-processing before regex matching.
"""

import re
import unicodedata

REFUSAL_MESSAGE = (
    "I'm here to help you with your medical records. What can I help you with?"
)


_ZERO_WIDTH_RE = re.compile(
    "[​‌‍‎‏⁠⁡⁢⁣⁤﻿­]"  # nosec B613
)

# Cyrillic/Greek characters that visually resemble Latin letters.
# Covers the practical attack surface for trigger words in our patterns.
_HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic -> Latin
    "А": "A",  # А
    "В": "B",  # В
    "С": "C",  # С
    "Е": "E",  # Е
    "Н": "H",  # Н
    "К": "K",  # К
    "М": "M",  # М
    "О": "O",  # О
    "Р": "P",  # Р
    "Т": "T",  # Т
    "Х": "X",  # Х
    "а": "a",  # а
    "е": "e",  # е
    "о": "o",  # о
    "р": "p",  # р
    "с": "c",  # с
    "у": "y",  # у
    "х": "x",  # х
    "ѕ": "s",  # ѕ
    "і": "i",  # і
    "ј": "j",  # ј
    "һ": "h",  # һ
    "ӏ": "l",  # ӏ
    # Greek -> Latin
    "Α": "A",  # Α
    "Β": "B",  # Β
    "Ε": "E",  # Ε
    "Η": "H",  # Η
    "Ι": "I",  # Ι
    "Κ": "K",  # Κ
    "Μ": "M",  # Μ
    "Ν": "N",  # Ν
    "Ο": "O",  # Ο
    "Ρ": "P",  # Ρ
    "Τ": "T",  # Τ
    "Υ": "Y",  # Υ
    "Χ": "X",  # Χ
    "α": "a",  # α (borderline but used in attacks)
    "ο": "o",  # ο
    "ρ": "p",  # ρ (lowercase rho)
    # Fullwidth Latin -> ASCII Latin
    # NFKC handles most of these, but belt-and-suspenders for the critical ones
    "ａ": "a",
    "ｂ": "b",
    "ｃ": "c",
    "ｄ": "d",
    "ｅ": "e",
    "ｉ": "i",
    "ｌ": "l",
    "ｍ": "m",
    "ｎ": "n",
    "ｏ": "o",
    "ｐ": "p",
    "ｒ": "r",
    "ｓ": "s",
    "ｔ": "t",
    "ｕ": "u",
    "ｙ": "y",
}

_HOMOGLYPH_TRANS = str.maketrans(_HOMOGLYPH_MAP)


def normalize_for_detection(*, text: str) -> str:
    """Normalize user input for security pattern matching.

    Applies in order:
    1. NFKC Unicode normalization (collapses compatibility equivalents)
    2. Zero-width character removal
    3. Homoglyph transliteration (Cyrillic/Greek -> Latin)
    4. Whitespace collapse
    """
    result = unicodedata.normalize("NFKC", text)
    result = _ZERO_WIDTH_RE.sub("", result)
    result = result.translate(_HOMOGLYPH_TRANS)
    result = re.sub(r"\s+", " ", result)
    return result
