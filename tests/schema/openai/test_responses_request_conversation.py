"""Tests for the OpenAI-style `conversation` field on ResponsesRequest (BAI-1244)."""

from typing import Any

import pytest
from pydantic import ValidationError

from languagemodelcommon.schema.openai.responses import (
    MAX_CONVERSATION_ID_LENGTH,
    ResponsesRequest,
)


def _payload(**extra: Any) -> dict[str, Any]:
    return {"model": "m", "input": "hi", **extra}


class TestConversationAccepted:
    @pytest.mark.parametrize(
        "value",
        ["u1:chat-ab12", {"id": "u1:chat-ab12"}, "x" * MAX_CONVERSATION_ID_LENGTH],
    )
    def test_valid_conversation_is_accepted(self, value: str | dict[str, str]) -> None:
        request = ResponsesRequest.model_validate(_payload(conversation=value))

        expected = value["id"] if isinstance(value, dict) else value
        assert request.requested_conversation_id == expected

    def test_request_without_conversation_is_unchanged(self) -> None:
        request = ResponsesRequest.model_validate(_payload())

        assert request.conversation is None
        assert request.requested_conversation_id is None
        assert request.model_dump(mode="json")["conversation"] is None

    def test_thread_id_and_user_id_are_still_accepted(self) -> None:
        request = ResponsesRequest.model_validate(_payload(thread_id="t", user_id="u"))

        assert request.requested_conversation_id is None


class TestConversationRejected:
    @pytest.mark.parametrize("conversation", ["c1", {"id": "c1"}])
    @pytest.mark.parametrize("previous_response_id", ["resp_1", ""])
    def test_conversation_with_previous_response_id_is_rejected(
        self, conversation: str | dict[str, str], previous_response_id: str
    ) -> None:
        with pytest.raises(ValidationError, match="previous_response_id"):
            ResponsesRequest.model_validate(
                _payload(
                    conversation=conversation,
                    previous_response_id=previous_response_id,
                )
            )

    @pytest.mark.parametrize(
        "conversation",
        [
            "",
            "   ",
            " a",
            "a ",
            "a\nb",
            "a\r\nb",
            "a\x00b",
            "a\x7fb",
            {"id": " a\nb "},
            {"id": ""},
            {"id": "  "},
            "x" * (MAX_CONVERSATION_ID_LENGTH + 1),
            {"id": "x" * (MAX_CONVERSATION_ID_LENGTH + 1)},
            {"id": "a", "extra": 1},
            {},
        ],
    )
    def test_invalid_conversation_values_are_rejected(self, conversation: Any) -> None:
        with pytest.raises(ValidationError):
            ResponsesRequest.model_validate(_payload(conversation=conversation))

    @pytest.mark.parametrize(
        "conversation", [123, True, ["a"], {"id": 123}, {"id": None}]
    )
    def test_wrong_typed_conversation_is_rejected(self, conversation: Any) -> None:
        with pytest.raises(ValidationError):
            ResponsesRequest.model_validate(_payload(conversation=conversation))

    def test_unknown_top_level_field_is_still_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResponsesRequest.model_validate(_payload(conversations="x"))
