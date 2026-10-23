"""Tests for the stream-start SSE event on the request wrappers (BAI-1256)."""

import json
from unittest.mock import MagicMock

from languagemodelcommon.schema.openai.completions import ChatRequest
from languagemodelcommon.schema.openai.responses import ResponsesRequest
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
