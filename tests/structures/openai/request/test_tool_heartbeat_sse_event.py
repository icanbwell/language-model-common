"""Tests for ChatRequestWrapper.create_tool_heartbeat_sse_event and its
overrides in ResponsesApiRequestWrapper and ChatCompletionApiRequestWrapper."""

import json
from typing import Any
from unittest.mock import MagicMock

from languagemodelcommon.schema.openai.completions import ChatRequest
from languagemodelcommon.schema.openai.responses import ResponsesRequest
from languagemodelcommon.structures.openai.request.chat_completion_api_request_wrapper import (
    ChatCompletionApiRequestWrapper,
)
from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)
from languagemodelcommon.structures.openai.request.responses_api_request_wrapper import (
    ResponsesApiRequestWrapper,
)


def _make_env(*, emit_tool_heartbeat_in_chat_completions: bool = False) -> MagicMock:
    env = MagicMock()
    env.debug_prefixes = ("DEBUG:", "/debug ")
    env.emit_task_progress_in_chat_completions = False
    env.emit_tool_heartbeat_in_chat_completions = (
        emit_tool_heartbeat_in_chat_completions
    )
    return env


def _make_responses_wrapper(
    *, input_: str = "hello", model: str = "gpt-4"
) -> ResponsesApiRequestWrapper:
    request = ResponsesRequest(
        model=model,
        input=input_,
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


def _make_chat_completion_wrapper(
    *,
    model: str = "gpt-4",
    emit_tool_heartbeat_in_chat_completions: bool = False,
) -> ChatCompletionApiRequestWrapper:
    request = ChatRequest(
        messages=[{"role": "user", "content": "hello"}],
        model=model,
        stream=False,
    )
    return ChatCompletionApiRequestWrapper(
        chat_request=request,
        enable_debug_logging=False,
        environment_variables=_make_env(
            emit_tool_heartbeat_in_chat_completions=emit_tool_heartbeat_in_chat_completions
        ),
    )


def test_base_wrapper_tool_heartbeat_is_noop() -> None:
    # ChatRequestWrapper is an ABC, and Python's abstract-method enforcement
    # runs even for object.__new__(ChatRequestWrapper) (it's checked inside
    # object.__new__ itself, not bypassable that way). The base
    # implementation of create_tool_heartbeat_sse_event never touches
    # `self`, so call it as an unbound function with any placeholder object
    # to confirm the no-op default without instantiating the ABC.
    result = ChatRequestWrapper.create_tool_heartbeat_sse_event(
        object(),  # type: ignore[arg-type]
        request_id="req-1",
        tool_name="propose_skill",
        elapsed_seconds=15.0,
    )
    assert result is None


def test_responses_api_wrapper_emits_task_progress_event() -> None:
    wrapper = _make_responses_wrapper()
    result = wrapper.create_tool_heartbeat_sse_event(
        request_id="req-1",
        tool_name="propose_skill",
        elapsed_seconds=15.0,
    )
    assert result is not None
    assert result.startswith("data: ")
    payload: dict[str, Any] = json.loads(result[len("data: ") :].strip())
    assert payload["type"] == "task.progress"
    assert payload["task_id"] == ""
    assert "propose_skill" in payload["message"]
    assert "15" in payload["message"]


def test_chat_completion_wrapper_tool_heartbeat_disabled_by_default() -> None:
    wrapper = _make_chat_completion_wrapper(
        emit_tool_heartbeat_in_chat_completions=False,
    )
    result = wrapper.create_tool_heartbeat_sse_event(
        request_id="req-1",
        tool_name="propose_skill",
        elapsed_seconds=15.0,
    )
    assert result is None


def test_chat_completion_wrapper_tool_heartbeat_enabled_emits_delta() -> None:
    wrapper = _make_chat_completion_wrapper(
        emit_tool_heartbeat_in_chat_completions=True,
    )
    result = wrapper.create_tool_heartbeat_sse_event(
        request_id="req-1",
        tool_name="propose_skill",
        elapsed_seconds=15.0,
    )
    assert result is not None
    assert "propose_skill" in result
