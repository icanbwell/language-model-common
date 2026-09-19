"""Tests for ChatRequestWrapper.create_image_output_sse_event (BAI-806) and its
override in ResponsesApiRequestWrapper. ChatCompletionApiRequestWrapper does
not override it -- same asymmetry as create_tool_start_sse_event/
create_tool_end_sse_event, which it also doesn't override -- so it inherits
the base no-op."""

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

_IMAGE_PART: dict[str, Any] = {
    "type": "output_image",
    "image_url": "https://example.com/chart.png",
    "mime_type": "image/png",
}


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


def test_base_wrapper_image_output_is_noop() -> None:
    # ChatRequestWrapper is an ABC; the base implementation of
    # create_image_output_sse_event never touches `self`, so call it as an
    # unbound function to confirm the no-op default without instantiating
    # the ABC (mirrors test_base_wrapper_tool_heartbeat_is_noop).
    result = ChatRequestWrapper.create_image_output_sse_event(
        object(),  # type: ignore[arg-type]
        request_id="req-1",
        image_part=_IMAGE_PART,
    )
    assert result is None


def test_responses_api_wrapper_emits_output_item_done_with_image() -> None:
    wrapper = _make_responses_wrapper()
    result = wrapper.create_image_output_sse_event(
        request_id="req-1",
        image_part=_IMAGE_PART,
    )
    assert result is not None
    assert result.startswith("data: ")
    payload: dict[str, Any] = json.loads(result[len("data: ") :].strip())
    assert payload["type"] == "response.output_item.done"
    item = payload["item"]
    assert item["type"] == "output_image"
    assert item["status"] == "completed"
    assert item["image_url"] == "https://example.com/chart.png"
    assert item["mime_type"] == "image/png"
    assert item["id"].startswith("img_req-1_")


def test_chat_completion_wrapper_image_output_is_noop() -> None:
    # ChatCompletionApiRequestWrapper has no atomic-item SSE transport at
    # all (same as create_tool_start_sse_event/create_tool_end_sse_event),
    # so it inherits the base no-op rather than overriding it.
    wrapper = _make_chat_completion_wrapper()
    result = wrapper.create_image_output_sse_event(
        request_id="req-1",
        image_part=_IMAGE_PART,
    )
    assert result is None
