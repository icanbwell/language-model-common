"""Tests for ChatRequestWrapper.create_prepare_sse_event (BAI-1440) and its
override in ResponsesApiRequestWrapper. ChatCompletionApiRequestWrapper does
not override it, so it inherits the base no-op."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

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
    env.emit_task_progress_in_chat_completions = False
    env.emit_tool_heartbeat_in_chat_completions = False
    return env


def _make_responses_wrapper() -> ResponsesApiRequestWrapper:
    request = ResponsesRequest(
        model="gpt-4",
        input="hello",
        stream=False,
        instructions=None,
        previous_response_id=None,
        store=False,
        temperature=None,
        top_p=None,
        max_output_tokens=None,
        tools=None,
        parallel_tool_calls=None,
        tool_choice=None,
        metadata=None,
    )
    return ResponsesApiRequestWrapper(
        chat_request=request,
        enable_debug_logging=False,
        environment_variables=_make_env(),
    )


def _make_chat_completion_wrapper() -> ChatCompletionApiRequestWrapper:
    request = ChatRequest(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-4",
        stream=False,
    )
    return ChatCompletionApiRequestWrapper(
        chat_request=request,
        enable_debug_logging=False,
        environment_variables=_make_env(),
    )


def test_chat_completion_wrapper_prepare_is_noop() -> None:
    wrapper = _make_chat_completion_wrapper()
    assert (
        wrapper.create_prepare_sse_event(request_id="req-1", runtime_seconds=1.5)
        is None
    )


def test_responses_api_wrapper_emits_done_only_prepare_item() -> None:
    wrapper = _make_responses_wrapper()
    raw = wrapper.create_prepare_sse_event(request_id="req-1", runtime_seconds=1.5)
    assert raw is not None
    assert raw.startswith("data: ")
    payload: dict[str, Any] = json.loads(raw[len("data: ") :].strip())
    assert payload["type"] == "response.output_item.done"
    assert payload["item"] == {
        "type": "prepare",
        "id": "prepare_req-1",
        "status": "completed",
        "runtime_seconds": 1.5,
    }


@pytest.mark.parametrize("runtime_seconds", [0.0, 0.25, 12.0])
def test_responses_api_wrapper_prepare_carries_runtime(runtime_seconds: float) -> None:
    wrapper = _make_responses_wrapper()
    raw = wrapper.create_prepare_sse_event(
        request_id="req-1", runtime_seconds=runtime_seconds
    )
    assert raw is not None
    item = json.loads(raw[len("data: ") :].strip())["item"]
    assert item["runtime_seconds"] == runtime_seconds
    assert set(item) == {"type", "id", "status", "runtime_seconds"}
