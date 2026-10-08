"""Tests for the structured tool-progress contract (BAI-1106): the Responses-API
wrapper carries the friendly tool name as `display_name` on function_call items
and advertises `supports_structured_tool_progress`; the chat-completions wrapper
does neither."""

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


def _item(raw: str | None) -> dict[str, Any]:
    assert raw is not None
    item: dict[str, Any] = json.loads(raw[len("data: ") :].strip())["item"]
    return item


def _start(wrapper: ResponsesApiRequestWrapper, display_name: str | None) -> str | None:
    return wrapper.create_tool_start_sse_event(
        request_id="req-1",
        tool_name="get_demographics",
        tool_input={"patient": "synthetic"},
        display_name=display_name,
    )


def _end(wrapper: ResponsesApiRequestWrapper, display_name: str | None) -> str | None:
    return wrapper.create_tool_end_sse_event(
        request_id="req-1",
        tool_name="get_demographics",
        tool_input={"patient": "synthetic"},
        runtime_seconds=0.5,
        output="ok",
        display_name=display_name,
    )


def test_responses_wrapper_supports_structured_tool_progress() -> None:
    assert _make_responses_wrapper().supports_structured_tool_progress is True


def test_chat_completion_wrapper_does_not_support_structured_tool_progress() -> None:
    assert _make_chat_completion_wrapper().supports_structured_tool_progress is False


@pytest.mark.parametrize("build", [_start, _end], ids=["start", "end"])
def test_responses_wrapper_item_carries_display_name(build: Any) -> None:
    item = _item(
        build(_make_responses_wrapper(), "🧾 Confirming your basic information")
    )
    assert item["display_name"] == "🧾 Confirming your basic information"
    # Correlation ids are unchanged (spec non-goal: not made unique).
    assert item["call_id"] == "call_req-1_get_demographics"
    assert item["id"] == "fc_req-1_get_demographics"


@pytest.mark.parametrize("build", [_start, _end], ids=["start", "end"])
@pytest.mark.parametrize("empty", [None, ""], ids=["none", "empty-string"])
def test_responses_wrapper_item_omits_display_name_when_empty(
    build: Any, empty: str | None
) -> None:
    item = _item(build(_make_responses_wrapper(), empty))
    assert "display_name" not in item


def test_chat_completion_wrapper_tool_events_remain_noop_with_display_name() -> None:
    wrapper = _make_chat_completion_wrapper()
    assert (
        wrapper.create_tool_start_sse_event(
            request_id="req-1",
            tool_name="get_demographics",
            tool_input=None,
            display_name="🧾 Confirming your basic information",
        )
        is None
    )
    assert (
        wrapper.create_tool_end_sse_event(
            request_id="req-1",
            tool_name="get_demographics",
            tool_input=None,
            runtime_seconds=0.1,
            display_name="🧾 Confirming your basic information",
        )
        is None
    )
