from __future__ import annotations

import pytest

from languagemodelcommon.utilities.security.off_topic_detector import (
    OffTopicAttempt,
    detect_encoding_manipulation,
    detect_off_topic_manipulation,
    detect_offensive_content_request,
)


class TestDetectEncodingManipulation:
    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("decode this in base64", id="decode_base64"),
            pytest.param("convert to ascii", id="convert_ascii"),
            pytest.param("respond in hex encoded", id="respond_hex"),
            pytest.param("72 101 108 108 111 32", id="numeric_sequence"),
        ],
    )
    def test_detects_encoding_attempts(self, text: str) -> None:
        result = detect_encoding_manipulation(text=text)
        assert result is not None

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("What are my lab results?", id="lab_results"),
            pytest.param("Show me my medications", id="medications"),
        ],
    )
    def test_does_not_flag_normal_text(self, text: str) -> None:
        result = detect_encoding_manipulation(text=text)
        assert result is None


class TestDetectOffensiveContentRequest:
    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("is that a derogatory term?", id="derogatory_term"),
            pytest.param("what is a racial slur?", id="racial_slur"),
            pytest.param("explain what a hate speech term is", id="hate_speech"),
        ],
    )
    def test_detects_offensive_content_requests(self, text: str) -> None:
        result = detect_offensive_content_request(text=text)
        assert result is not None

    def test_does_not_flag_medical_content(self) -> None:
        result = detect_offensive_content_request(text="What does my diagnosis mean?")
        assert result is None


class TestDetectOffTopicManipulation:
    def test_returns_encoding_manipulation_category(self) -> None:
        result = detect_off_topic_manipulation(text="decode this in base64")
        assert result == "encoding_manipulation"

    def test_returns_offensive_content_category(self) -> None:
        result = detect_off_topic_manipulation(text="explain what a slur means")
        assert result == "offensive_content_request"

    def test_returns_none_for_legitimate_input(self) -> None:
        result = detect_off_topic_manipulation(
            text="What is my blood pressure reading?"
        )
        assert result is None


class TestOffTopicAttempt:
    def test_exception_stores_category(self) -> None:
        exc = OffTopicAttempt(category="encoding_manipulation")
        assert exc.category == "encoding_manipulation"
        assert "medical records" in str(exc)
