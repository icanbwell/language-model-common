"""Tests for adopting a conversation id on a request wrapper (BAI-1264)."""

import logging
from unittest.mock import MagicMock

import pytest

from languagemodelcommon.schema.openai.completions import ChatRequest
from languagemodelcommon.schema.openai.responses import (
    MAX_CONVERSATION_ID_LENGTH,
    ResponsesRequest,
)
from languagemodelcommon.structures.openai.request.chat_completion_api_request_wrapper import (
    ChatCompletionApiRequestWrapper,
)
from languagemodelcommon.structures.openai.request.responses_api_request_wrapper import (
    ResponsesApiRequestWrapper,
)


def _make_env() -> MagicMock:
    env = MagicMock()
    env.debug_prefixes = ("DEBUG:", "/debug ")
    return env


def _responses_wrapper() -> ResponsesApiRequestWrapper:
    request = ResponsesRequest.model_validate(
        {"model": "gpt-4", "input": "hello", "stream": True}
    )
    return ResponsesApiRequestWrapper(
        chat_request=request,
        enable_debug_logging=False,
        environment_variables=_make_env(),
    )


def _chat_completion_wrapper() -> ChatCompletionApiRequestWrapper:
    request = ChatRequest(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-4",
        stream=True,
    )
    return ChatCompletionApiRequestWrapper(
        chat_request=request,
        enable_debug_logging=False,
        environment_variables=_make_env(),
    )


class TestAdoptConversationId:
    def test_adopts_when_unset(self) -> None:
        wrapper = _responses_wrapper()

        wrapper.adopt_conversation_id(conversation_id="u1:chat-ab12")

        assert wrapper.conversation_id == "u1:chat-ab12"

    def test_does_not_overwrite_a_caller_set_id(self) -> None:
        wrapper = _responses_wrapper()
        wrapper.conversation_id = "caller-set"

        wrapper.adopt_conversation_id(conversation_id="u1:chat-ab12")

        assert wrapper.conversation_id == "caller-set"

    @pytest.mark.parametrize(
        "conversation_id",
        [
            None,
            "",
            "   ",
            "x" * (MAX_CONVERSATION_ID_LENGTH + 1),
            " padded",
            "sub:chat\x01x",
        ],
        ids=["none", "empty", "blank", "too-long", "padded", "control-character"],
    )
    def test_skips_missing_or_invalid_ids(self, conversation_id: str | None) -> None:
        wrapper = _responses_wrapper()

        wrapper.adopt_conversation_id(conversation_id=conversation_id)

        assert wrapper.conversation_id is None

    @pytest.mark.parametrize(
        "invalid_id",
        [
            "subject-" + "x" * MAX_CONVERSATION_ID_LENGTH,
            " subject-padded",
            "subject-chat\x01x",
        ],
        ids=["too-long", "padded", "control-character"],
    )
    def test_invalid_id_is_skipped_with_a_value_free_validation_warning(
        self, caplog: pytest.LogCaptureFixture, invalid_id: str
    ) -> None:
        wrapper = _responses_wrapper()

        with caplog.at_level(logging.WARNING):
            wrapper.adopt_conversation_id(conversation_id=invalid_id)

        assert "failed validation" in caplog.text
        assert "subject-" not in caplog.text
        assert "blank or longer" not in caplog.text

    def test_chat_completions_wrapper_ignores_it(self) -> None:
        wrapper = _chat_completion_wrapper()

        wrapper.adopt_conversation_id(conversation_id="u1:chat-ab12")

        assert not hasattr(wrapper, "conversation_id")
