"""Tests for the shared redaction-safe logging summarizer.

BAI-955: this is the canonical contract primitive shared across
language-model-common, mcp-fhir-agent, baileyai, and
baileyai-skills-service. It must never surface a raw value or a
non-identifier-shaped key - only a flat shape describing type, length,
and (for dict-shaped values) the safe key names.
"""

import pytest

from languagemodelcommon.utilities.logger.summarization import summarize_for_logging


class TestSummarizeForLoggingScalar:
    def test_scalar_string_is_redacted_to_type_and_length(self) -> None:
        result = summarize_for_logging(value="a-secret-patient-identifier")

        assert result == {"type": "str", "length": len("a-secret-patient-identifier")}

    def test_scalar_value_never_appears_in_output(self) -> None:
        marker = "MARKER-DO-NOT-LEAK-12345"

        result = summarize_for_logging(value=marker)

        assert marker not in str(result)


class TestSummarizeForLoggingDict:
    def test_dict_is_flat_with_no_recursion_into_nested_values(self) -> None:
        nested = {
            "patient_id": "abc-123",
            "nested": {"note": "some free text PHI"},
        }

        result = summarize_for_logging(value=nested)

        assert result == {"type": "dict", "length": 2, "keys": ["nested", "patient_id"]}

    def test_dict_no_raw_value_anywhere_in_output(self) -> None:
        marker = "MARKER-DO-NOT-LEAK-98765"
        nested = {"top": marker}

        result = summarize_for_logging(value=nested)

        assert marker not in str(result)

    def test_unsafe_keys_are_counted_but_never_named(self) -> None:
        value = {"patient_id": "SECRET", "12345": 1, "John Smith": 2}

        result = summarize_for_logging(value=value)

        assert result == {
            "type": "dict",
            "length": 3,
            "keys": ["patient_id"],
            "redacted_keys": 2,
        }

    def test_redacted_keys_omitted_when_all_keys_are_safe(self) -> None:
        result = summarize_for_logging(value={"a": {"b": "SECRET"}})

        assert result == {"type": "dict", "length": 1, "keys": ["a"]}


class TestSummarizeForLoggingListLike:
    @pytest.mark.parametrize(
        ("value", "expected_type"),
        [
            ([1, 2, 3], "list"),
            ((1, 2, 3), "tuple"),
            ({1, 2, 3}, "set"),
            (frozenset({1, 2, 3}), "frozenset"),
        ],
    )
    def test_list_like_reports_element_count_only(
        self, value: object, expected_type: str
    ) -> None:
        result = summarize_for_logging(value=value)

        assert result == {"type": expected_type, "length": 3}


class TestSummarizeForLoggingBytes:
    def test_bytes_length_is_byte_count_not_repr_length(self) -> None:
        result = summarize_for_logging(value=b"hi")

        assert result == {"type": "bytes", "length": 2}

    def test_non_ascii_bytes_length_is_byte_count_not_repr_length(self) -> None:
        # "héllo".encode() is 6 bytes (UTF-8 for "é" is 2 bytes), but its
        # repr is far longer than 6 characters.
        value = "héllo".encode()

        result = summarize_for_logging(value=value)

        assert result == {"type": "bytes", "length": 6}

    def test_bytearray_length_is_byte_count(self) -> None:
        result = summarize_for_logging(value=bytearray(b"hi"))

        assert result == {"type": "bytearray", "length": 2}

    def test_memoryview_length_is_nbytes(self) -> None:
        result = summarize_for_logging(value=memoryview(b"hi"))

        assert result == {"type": "memoryview", "length": 2}


class TestSummarizeForLoggingJsonContent:
    def test_str_containing_json_object_adds_keys(self) -> None:
        value = '{"name": "SECRET", "John Smith": 1}'

        result = summarize_for_logging(value=value)

        assert result == {
            "type": "str",
            "length": len(value),
            "keys": ["name"],
            "redacted_keys": 1,
        }

    def test_str_containing_json_array_adds_nothing_beyond_type_and_length(
        self,
    ) -> None:
        result = summarize_for_logging(value="[1,2]")

        assert result == {"type": "str", "length": 5}

    def test_bytes_containing_json_object_adds_keys(self) -> None:
        value = b'{"name": "SECRET"}'

        result = summarize_for_logging(value=value)

        assert result == {
            "type": "bytes",
            "length": len(value),
            "keys": ["name"],
        }

    def test_str_that_fails_json_parse_adds_nothing_and_never_raises(self) -> None:
        result = summarize_for_logging(value="not json at all")

        assert result == {"type": "str", "length": len("not json at all")}

    def test_bytes_that_fail_utf8_and_json_parse_never_raise(self) -> None:
        # Invalid UTF-8 continuation byte; must decode with errors="replace"
        # for the parse attempt only, never raise, and never add keys.
        result = summarize_for_logging(value=b"\xff\xfe not valid json")

        assert result == {"type": "bytes", "length": len(b"\xff\xfe not valid json")}


class TestSummarizeForLoggingOtherScalars:
    def test_int_uses_str_length(self) -> None:
        result = summarize_for_logging(value=42)

        assert result == {"type": "int", "length": 2}


class TestSummarizeForLoggingKeywordOnly:
    def test_positional_call_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            summarize_for_logging("a-secret-patient-identifier")  # type: ignore[misc]


class TestSummarizeForLoggingNeverLeaksSecret:
    @pytest.mark.parametrize(
        "value",
        [
            "SECRET",
            b"SECRET",
            bytearray(b"SECRET"),
            memoryview(b"SECRET"),
            {"patient_id": "SECRET"},
            ["SECRET", "SECRET"],
            ("SECRET",),
            {"SECRET"},
            '{"name": "SECRET"}',
            b'{"name": "SECRET"}',
        ],
    )
    def test_secret_never_appears_in_rendered_result(self, value: object) -> None:
        result = summarize_for_logging(value=value)

        assert "SECRET" not in str(result)
