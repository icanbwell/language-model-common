"""Tests for adopting a conversation id on a request wrapper and emitting the
initial stream event (BAI-1256)."""

import json
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
        [None, "", "   ", "x" * (MAX_CONVERSATION_ID_LENGTH + 1)],
    )
    def test_skips_missing_or_invalid_ids(self, conversation_id: str | None) -> None:
        wrapper = _responses_wrapper()

        wrapper.adopt_conversation_id(conversation_id=conversation_id)

        assert wrapper.conversation_id is None

    def test_invalid_id_is_not_written_to_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        wrapper = _responses_wrapper()
        too_long = "subject-" + "x" * MAX_CONVERSATION_ID_LENGTH

        with caplog.at_level(logging.WARNING):
            wrapper.adopt_conversation_id(conversation_id=too_long)

        assert caplog.records
        assert "subject-" not in caplog.text

    def test_chat_completions_wrapper_ignores_it(self) -> None:
        wrapper = _chat_completion_wrapper()

        wrapper.adopt_conversation_id(conversation_id="u1:chat-ab12")


class TestStreamStartSseEvent:
    def test_responses_wrapper_emits_response_created_with_conversation(self) -> None:
        wrapper = _responses_wrapper()
        wrapper.conversation_id = "u1:chat-ab12"

        raw = wrapper.create_stream_start_sse_event(request_id="req-1", source="test")

        assert raw is not None
        assert raw.count("data: ") == 1
        event = json.loads(raw[len("data: ") :])
        assert event["type"] == "response.created"
        assert event["response"]["conversation"] == {"id": "u1:chat-ab12"}

    def test_responses_wrapper_emits_null_conversation_when_unset(self) -> None:
        wrapper = _responses_wrapper()

        raw = wrapper.create_stream_start_sse_event(request_id="req-1", source="test")

        assert raw is not None
        event = json.loads(raw[len("data: ") :])
        assert event["response"]["conversation"] is None

    def test_chat_completions_wrapper_emits_nothing(self) -> None:
        wrapper = _chat_completion_wrapper()

        assert (
            wrapper.create_stream_start_sse_event(request_id="req-1", source="test")
            is None
        )
