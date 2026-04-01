"""Tests for ResponsesApiRequestWrapper."""

import json
from typing import Any, List, cast

import pytest
from langchain_core.messages import AIMessage, AnyMessage

from languagemodelcommon.schema.openai.responses import ResponsesRequest
from languagemodelcommon.structures.openai.request.responses_api_request_wrapper import (
    ResponsesApiRequestWrapper,
)


def _make_wrapper(
    *,
    input_: str | list[dict[str, Any]] = "hello",
    model: str = "gpt-4",
    stream: bool = False,
    enable_debug_logging: bool = False,
    instructions: str | None = None,
    previous_response_id: str | None = None,
    store: bool | None = False,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    parallel_tool_calls: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> ResponsesApiRequestWrapper:
    request = ResponsesRequest(
        model=model,
        input=cast(Any, input_),
        stream=stream,
        instructions=instructions,
        previous_response_id=previous_response_id,
        store=store,
        temperature=temperature,
        top_p=None,
        max_output_tokens=max_output_tokens,
        tools=tools,
        parallel_tool_calls=parallel_tool_calls,
        metadata=metadata,
    )
    return ResponsesApiRequestWrapper(
        chat_request=request,
        enable_debug_logging=enable_debug_logging,
    )


class TestHardcodedProperties:
    """Tests for properties that return hardcoded values rather than delegating."""

    def test_response_format_always_json_object(self) -> None:
        wrapper = _make_wrapper()
        assert wrapper.response_format == "json_object"

    def test_response_json_schema_always_none(self) -> None:
        wrapper = _make_wrapper()
        assert wrapper.response_json_schema is None

    def test_max_tokens_always_none(self) -> None:
        wrapper = _make_wrapper()
        assert wrapper.max_tokens is None


class TestMessageConversion:
    """Tests for message conversion from string and list inputs."""

    def test_string_input_creates_single_user_message(self) -> None:
        wrapper = _make_wrapper(input_="What is AI?")
        assert len(wrapper.messages) == 1
        assert wrapper.messages[0].role == "user"
        assert wrapper.messages[0].content == "What is AI?"

    def test_list_input_creates_multiple_messages(self) -> None:
        wrapper = _make_wrapper(
            input_=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        )
        assert len(wrapper.messages) == 2
        assert wrapper.messages[0].role == "user"
        assert wrapper.messages[1].role == "assistant"

    def test_append_message_adds_system_message(self) -> None:
        wrapper = _make_wrapper(input_="hello")
        system_msg = wrapper.create_system_message(content="You are helpful")
        wrapper.append_message(message=system_msg)
        assert len(wrapper.messages) == 2
        assert wrapper.messages[1].system_message is True


class TestDebugPrefixToggle:
    """Tests for the DEBUG: prefix detection and stripping."""

    def test_debug_prefix_enables_logging_and_strips_content(self) -> None:
        wrapper = _make_wrapper(input_="DEBUG: What is AI?", enable_debug_logging=False)
        assert wrapper.enable_debug_logging is True
        assert wrapper.messages[0].content == "What is AI?"
        assert wrapper.request.input == "What is AI?"

    def test_no_debug_prefix_keeps_logging_disabled(self) -> None:
        wrapper = _make_wrapper(input_="What is AI?", enable_debug_logging=False)
        assert wrapper.enable_debug_logging is False

    def test_debug_prefix_with_list_input(self) -> None:
        wrapper = _make_wrapper(
            input_=[
                {
                    "role": "user",
                    "content": "DEBUG: Tell me something",
                    "type": "message",
                },
            ],
            enable_debug_logging=False,
        )
        assert wrapper.enable_debug_logging is True
        assert wrapper.messages[0].content == "Tell me something"


class TestUserInput:
    """Tests for user_input property extraction."""

    def test_user_input_from_string(self) -> None:
        wrapper = _make_wrapper(input_="hello world")
        assert wrapper.user_input == "hello world"

    def test_user_input_from_list_with_content(self) -> None:
        wrapper = _make_wrapper(input_=[{"role": "user", "content": "hello from list"}])
        assert wrapper.user_input == "hello from list"

    def test_user_input_from_list_with_multiple_parts(self) -> None:
        wrapper = _make_wrapper(
            input_=[
                {"role": "user", "content": "part one"},
                {"role": "user", "content": "part two"},
            ]
        )
        user_input = wrapper.user_input
        assert user_input is not None
        assert "part one" in user_input
        assert "part two" in user_input


class TestSSEMessages:
    """Tests for SSE message creation methods."""

    def test_create_first_sse_message_contains_response_created(self) -> None:
        wrapper = _make_wrapper()
        result = wrapper.create_first_sse_message(request_id="req-1", source="test")
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        payload = json.loads(result[len("data: ") :])
        assert payload["type"] == "response.created"
        assert payload["response"]["id"] == "req-1"
        assert payload["response"]["model"] == "gpt-4"
        assert payload["response"]["status"] == "in_progress"

    def test_create_sse_message_with_content(self) -> None:
        wrapper = _make_wrapper()
        result = wrapper.create_sse_message(
            request_id="req-1", content="Hello", usage_metadata=None, source="test"
        )
        assert result.startswith("data: ")
        payload = json.loads(result[len("data: ") :])
        assert payload["type"] == "response.output_text.delta"
        assert payload["delta"] == "Hello"
        assert payload["item_id"] == "req-1"

    def test_create_sse_message_with_none_content_returns_empty(self) -> None:
        wrapper = _make_wrapper()
        result = wrapper.create_sse_message(
            request_id="req-1", content=None, usage_metadata=None, source="test"
        )
        assert result == ""

    def test_create_final_sse_message_contains_done_event(self) -> None:
        wrapper = _make_wrapper()
        result = wrapper.create_final_sse_message(
            request_id="req-1", usage_metadata=None, source="test"
        )
        payload = json.loads(result[len("data: ") :])
        assert payload["type"] == "response.output_text.done"
        assert payload["item_id"] == "req-1"

    def test_create_debug_sse_message_returns_none_when_debug_disabled(self) -> None:
        wrapper = _make_wrapper(enable_debug_logging=False)
        result = wrapper.create_debug_sse_message(
            request_id="req-1", content="debug info", usage_metadata=None, source="test"
        )
        assert result is None

    def test_create_debug_sse_message_returns_content_when_debug_enabled(self) -> None:
        wrapper = _make_wrapper(enable_debug_logging=True)
        result = wrapper.create_debug_sse_message(
            request_id="req-1", content="debug info", usage_metadata=None, source="test"
        )
        assert result is not None
        assert "debug info" in result


class TestNonStreamingResponse:
    """Tests for non-streaming response creation."""

    def test_single_ai_message(self) -> None:
        wrapper = _make_wrapper()
        response = wrapper.create_non_streaming_response(
            request_id="req-1",
            json_output_requested=False,
            responses=[AIMessage(content="Hello world")],
        )
        assert response["id"] == "req-1"
        assert response["object"] == "response"
        assert response["model"] == "gpt-4"
        assert len(response["output"]) == 1
        output_msg = response["output"][0]
        assert output_msg["role"] == "assistant"
        assert output_msg["status"] == "completed"
        assert output_msg["content"][0]["text"] == "Hello world"

    def test_multiple_ai_messages(self) -> None:
        wrapper = _make_wrapper()
        response = wrapper.create_non_streaming_response(
            request_id="req-2",
            json_output_requested=False,
            responses=[
                AIMessage(content="First"),
                AIMessage(content="Second"),
            ],
        )
        assert len(response["output"]) == 2
        assert response["output"][0]["content"][0]["text"] == "First"
        assert response["output"][1]["content"][0]["text"] == "Second"

    def test_empty_responses(self) -> None:
        wrapper = _make_wrapper()
        response = wrapper.create_non_streaming_response(
            request_id="req-3",
            json_output_requested=False,
            responses=[],
        )
        assert response["output"] == []


class TestConvertMessageContent:
    """Tests for the static convert_message_content method."""

    def test_string_content(self) -> None:
        result = ResponsesApiRequestWrapper.convert_message_content(
            input_content="Hello"
        )
        assert len(result) == 1
        assert result[0].type == "output_text"
        assert result[0].text == "Hello"

    def test_list_of_strings(self) -> None:
        result = ResponsesApiRequestWrapper.convert_message_content(
            input_content=["Hello", "World"]
        )
        assert len(result) == 2
        assert hasattr(result[0], "text") and result[0].text == "Hello"
        assert hasattr(result[1], "text") and result[1].text == "World"

    def test_list_with_dicts(self) -> None:
        result = ResponsesApiRequestWrapper.convert_message_content(
            input_content=[
                {"text": "From dict", "type": "output_text", "annotations": []}
            ]
        )
        assert len(result) == 1
        assert hasattr(result[0], "text") and result[0].text == "From dict"

    def test_unsupported_type_returns_empty(self) -> None:
        result = ResponsesApiRequestWrapper.convert_message_content(
            input_content=123  # type: ignore[arg-type]
        )
        assert result == []


class TestToDict:
    """Tests for the to_dict serialization."""

    def test_to_dict_contains_model_and_input(self) -> None:
        wrapper = _make_wrapper(input_="hello", model="gpt-4o")
        d = wrapper.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["input"] == "hello"


class TestGetTools:
    """Tests for MCP tool extraction."""

    def test_no_tools_returns_empty(self) -> None:
        wrapper = _make_wrapper(tools=None)
        assert wrapper.get_tools() == []

    def test_mcp_tool_extracted(self) -> None:
        wrapper = _make_wrapper(
            tools=[
                {
                    "type": "mcp",
                    "server_url": "http://localhost:8080",
                    "server_label": "my-server",
                    "allowed_tools": [{"name": "tool_a"}, {"name": "tool_b"}],
                }
            ]
        )
        configs = wrapper.get_tools()
        assert len(configs) == 1
        assert configs[0].url == "http://localhost:8080"
        assert configs[0].name == "my-server"
        tools_str = configs[0].tools or ""
        assert "tool_a" in tools_str
        assert "tool_b" in tools_str

    def test_non_mcp_tool_ignored(self) -> None:
        wrapper = _make_wrapper(
            tools=[
                {
                    "type": "function",
                    "name": "some_function",
                }
            ]
        )
        assert wrapper.get_tools() == []

    def test_mcp_tool_missing_fields_ignored(self) -> None:
        wrapper = _make_wrapper(
            tools=[
                {
                    "type": "mcp",
                    "server_url": "http://localhost:8080",
                    # Missing server_label and allowed_tools
                }
            ]
        )
        assert wrapper.get_tools() == []


class TestStreamResponse:
    """Tests for the stream_response method."""

    @pytest.mark.asyncio
    async def test_stream_response_yields_created_then_deltas_then_done(self) -> None:
        wrapper = _make_wrapper(model="gpt-4o")
        messages: List[AnyMessage] = [
            AIMessage(content="Hello"),
            AIMessage(content="World"),
        ]
        stream = wrapper.stream_response(response_messages1=messages)
        chunks = [chunk async for chunk in stream]

        # created + 2 deltas + done = 4
        assert len(chunks) == 4

        created = json.loads(chunks[0][len("data: ") :])
        assert created["type"] == "response.created"
        assert created["response"]["model"] == "gpt-4o"
        assert created["response"]["status"] == "in_progress"
        assert created["sequence_number"] == 0

        delta1 = json.loads(chunks[1][len("data: ") :])
        assert delta1["type"] == "response.output_text.delta"
        assert delta1["delta"] == "Hello\n"
        assert delta1["sequence_number"] == 1

        delta2 = json.loads(chunks[2][len("data: ") :])
        assert delta2["type"] == "response.output_text.delta"
        assert delta2["delta"] == "World\n"
        assert delta2["sequence_number"] == 2

        done = json.loads(chunks[3][len("data: ") :])
        assert done["type"] == "response.output_text.done"
        assert done["text"] == ""
        assert done["sequence_number"] == 3

    @pytest.mark.asyncio
    async def test_stream_response_skips_empty_content(self) -> None:
        wrapper = _make_wrapper()
        messages: List[AnyMessage] = [
            AIMessage(content=""),
            AIMessage(content="Real content"),
        ]
        stream = wrapper.stream_response(response_messages1=messages)
        chunks = [chunk async for chunk in stream]

        # created + 1 delta (empty skipped) + done = 3
        assert len(chunks) == 3
        delta = json.loads(chunks[1][len("data: ") :])
        assert delta["delta"] == "Real content\n"

    @pytest.mark.asyncio
    async def test_stream_response_with_no_messages(self) -> None:
        wrapper = _make_wrapper()
        stream = wrapper.stream_response(response_messages1=[])
        chunks = [chunk async for chunk in stream]

        # created + done = 2
        assert len(chunks) == 2
        assert "response.created" in chunks[0]
        assert "response.output_text.done" in chunks[1]
