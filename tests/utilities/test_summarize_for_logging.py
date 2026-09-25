"""Tests for the shared redaction-safe logging summarizer.

BAI-955: this is the contract primitive three adopter repos (mcp-fhir-agent,
baileyai, baileyai-skills-service) depend on. It must never surface a raw
value, only its type and length.
"""

from languagemodelcommon.utilities.logger.summarization import summarize_for_logging


class TestSummarizeForLoggingScalar:
    def test_scalar_string_is_redacted_to_type_and_length(self) -> None:
        # Positional call, matching the contract signature exactly:
        # def summarize_for_logging(value: Any) -> dict[str, Any]
        result = summarize_for_logging("a-secret-patient-identifier")

        assert result == {"type": "str", "length": len("a-secret-patient-identifier")}

    def test_scalar_value_never_appears_in_output(self) -> None:
        marker = "MARKER-DO-NOT-LEAK-12345"

        result = summarize_for_logging(marker)

        assert marker not in str(result)


class TestSummarizeForLoggingDict:
    def test_dict_value_redacted_per_key_recursively(self) -> None:
        nested = {
            "patient_id": "abc-123",
            "nested": {"note": "some free text PHI"},
        }

        result = summarize_for_logging(nested)

        assert result == {
            "patient_id": {"type": "str", "length": len("abc-123")},
            "nested": {"note": {"type": "str", "length": len("some free text PHI")}},
        }

    def test_dict_no_raw_value_anywhere_in_output(self) -> None:
        marker = "MARKER-DO-NOT-LEAK-98765"
        nested = {"top": {"inner": marker}}

        result = summarize_for_logging(nested)

        assert marker not in str(result)


class TestSummarizeForLoggingBytes:
    def test_bytes_length_is_byte_count_not_repr_length(self) -> None:
        result = summarize_for_logging(b"hi")

        assert result == {"type": "bytes", "length": 2}

    def test_non_ascii_bytes_length_is_byte_count_not_repr_length(self) -> None:
        # b"\xc3\xa9" is 2 bytes (UTF-8 for "é"), but its repr
        # ("b'\\xc3\\xa9'") is far longer than 2 characters.
        value = b"\xc3\xa9"

        result = summarize_for_logging(value)

        assert result == {"type": "bytes", "length": 2}

    def test_bytearray_length_is_byte_count(self) -> None:
        result = summarize_for_logging(bytearray(b"hi"))

        assert result == {"type": "bytearray", "length": 2}

    def test_memoryview_length_is_byte_count(self) -> None:
        result = summarize_for_logging(memoryview(b"hi"))

        assert result == {"type": "memoryview", "length": 2}
