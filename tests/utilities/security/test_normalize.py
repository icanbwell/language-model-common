from __future__ import annotations

import pytest

from languagemodelcommon.utilities.security.normalize import normalize_for_detection


class TestNormalizeForDetection:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            pytest.param("hello world", "hello world", id="plain_text_unchanged"),
            pytest.param("hello   world", "hello world", id="whitespace_collapsed"),
            pytest.param(
                "h​e​l​l​o",
                "hello",
                id="zero_width_chars_removed",
            ),
            pytest.param(
                "АВС",  # Cyrillic А В С
                "ABC",
                id="cyrillic_homoglyphs_transliterated",
            ),
            pytest.param(
                "ΑΒΕ",  # Greek Α Β Ε
                "ABE",
                id="greek_homoglyphs_transliterated",
            ),
            pytest.param(
                "ａｂｃ",  # Fullwidth ａｂｃ
                "abc",
                id="fullwidth_latin_normalized",
            ),
            pytest.param(
                "résumé",  # résumé — NFKC keeps composed forms
                "résumé",
                id="nfkc_preserves_composed_accents",
            ),
        ],
    )
    def test_normalization_cases(self, input_text: str, expected: str) -> None:
        assert normalize_for_detection(text=input_text) == expected

    def test_combined_attack_vector(self) -> None:
        # Cyrillic 'а' + zero-width + fullwidth 'ｂ' + extra spaces
        attack = "а​ｂ  test"
        result = normalize_for_detection(text=attack)
        assert result == "ab test"
