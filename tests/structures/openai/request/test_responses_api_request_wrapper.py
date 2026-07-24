"""Tests for ResponsesApiRequestWrapper."""

import json
from typing import Any, List, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from languagemodelcommon.schema.openai.responses import ResponsesRequest
from languagemodelcommon.structures.openai.request.responses_api_request_wrapper import (
    ResponsesApiRequestWrapper,
)


def _make_env() -> MagicMock:
    env = MagicMock()
    env.debug_prefixes = ("DEBUG:", "/debug ")
    return env


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
    tool_choice: str | dict[str, Any] | None = None,
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
        tool_choice=tool_choice,
        metadata=metadata,
    )
    return ResponsesApiRequestWrapper(
        chat_request=request,
        enable_debug_logging=enable_debug_logging,
        environment_variables=_make_env(),
    )


class TestHardcodedProperties:
    """Tests for properties that return hardcoded values rather than delegating."""

    def test_response_format_always_json_object(self) -> None:
        wrapper = _make_wrapper()
        assert wrapper.response_format == "json_object"


class TestToolChoice:
    """Tests for the tool_choice property — pass-through of the OpenAI parameter."""

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "none",
            "auto",
            "required",
            {"type": "function", "function": {"name": "lookup"}},
        ],
    )
    def test_tool_choice_passthrough(self, value: str | dict[str, Any] | None) -> None:
        wrapper = _make_wrapper(tool_choice=value)
        assert wrapper.tool_choice == value


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


class TestDebugPrefixToggle:
    """Tests for the DEBUG: prefix detection and stripping."""

    def test_debug_prefix_enables_logging_and_strips_content(self) -> None:
        wrapper = _make_wrapper(input_="DEBUG: What is AI?", enable_debug_logging=False)
        assert wrapper.enable_debug_logging is True
        assert wrapper.messages[0].content == "What is AI?"
        assert wrapper.request.input == "What is AI?"

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

    def test_slash_debug_prefix_enables_logging_and_strips_content(self) -> None:
        wrapper = _make_wrapper(input_="/debug What is AI?", enable_debug_logging=False)
        assert wrapper.enable_debug_logging is True
        assert wrapper.messages[0].content == "What is AI?"
        assert wrapper.request.input == "What is AI?"

    def test_slash_debug_prefix_with_list_input(self) -> None:
        wrapper = _make_wrapper(
            input_=[
                {
                    "role": "user",
                    "content": "/debug Tell me something",
                    "type": "message",
                },
            ],
            enable_debug_logging=False,
        )
        assert wrapper.enable_debug_logging is True
        assert wrapper.messages[0].content == "Tell me something"


class TestUserInput:
    """Tests for user_input property extraction."""

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
        # Final message contains multiple SSE events: text.done, response.completed, [DONE]
        events = [
            line[len("data: ") :]
            for line in result.strip().split("\n")
            if line.startswith("data: ")
        ]
        assert len(events) == 3
        text_done = json.loads(events[0])
        assert text_done["type"] == "response.output_text.done"
        assert text_done["item_id"] == "req-1"
        completed = json.loads(events[1])
        assert completed["type"] == "response.completed"
        assert events[2] == "[DONE]"

    def test_create_debug_sse_message_returns_none_when_debug_disabled(self) -> None:
        wrapper = _make_wrapper(enable_debug_logging=False)
        result = wrapper.create_debug_sse_message(
            request_id="req-1", content="debug info", usage_metadata=None, source="test"
        )
        assert result is None


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

    def test_filters_out_system_and_human_messages(self) -> None:
        wrapper = _make_wrapper()
        response = wrapper.create_non_streaming_response(
            request_id="req-2",
            json_output_requested=False,
            responses=[
                SystemMessage(content="Current date and time: Thursday, May 28, 2026"),
                HumanMessage(content="Write a story"),
                AIMessage(content="I can only help with healthcare topics."),
            ],
        )
        assert len(response["output"]) == 1
        assert (
            response["output"][0]["content"][0]["text"]
            == "I can only help with healthcare topics."
        )

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

    def test_list_of_dicts_with_text_type(self) -> None:
        result = ResponsesApiRequestWrapper.convert_message_content(
            input_content=[{"text": "Hello from LLM", "type": "text"}]
        )
        assert len(result) == 1
        assert result[0].type == "output_text"
        assert result[0].text == "Hello from LLM"
        assert result[0].annotations == []

    def test_list_of_dicts_with_output_text_type(self) -> None:
        result = ResponsesApiRequestWrapper.convert_message_content(
            input_content=[
                {"text": "Already correct", "type": "output_text", "annotations": []}
            ]
        )
        assert len(result) == 1
        assert result[0].type == "output_text"
        assert result[0].text == "Already correct"

    def test_unsupported_type_returns_empty(self) -> None:
        result = ResponsesApiRequestWrapper.convert_message_content(
            input_content=123  # type: ignore[arg-type]
        )
        assert result == []


class TestGetTools:
    """Tests for MCP tool extraction."""

    def test_mcp_tool_with_server_url(self) -> None:
        wrapper = _make_wrapper(
            tools=[
                {
                    "type": "mcp",
                    "server_url": "http://localhost:8080",
                    "server_label": "my-server",
                    "allowed_tools": [{"name": "tool_a"}, {"name": "tool_b"}],
                    "headers": {"X-Token": "secret"},
                }
            ]
        )
        configs = wrapper.get_tools()
        assert len(configs) == 1
        assert configs[0].url == "http://localhost:8080"
        assert configs[0].name == "my-server"
        assert configs[0].mcp_server is None
        assert configs[0].auth == "headers"
        assert configs[0].headers == {"X-Token": "secret"}
        assert configs[0].tools == "tool_a,tool_b"

    def test_mcp_tool_label_only_resolves_via_mcp_server_reference(self) -> None:
        """Without server_url, the config carries mcp_server set to server_label
        so the caller resolves the URL from .mcp.json at load time."""
        wrapper = _make_wrapper(
            tools=[
                {
                    "type": "mcp",
                    "server_label": "github",
                    "allowed_tools": ["search_repos", "get_issue"],
                }
            ]
        )
        configs = wrapper.get_tools()
        assert len(configs) == 1
        assert configs[0].url is None
        assert configs[0].name == "github"
        assert configs[0].mcp_server == "github"
        assert configs[0].tools == "search_repos,get_issue"

    def test_mcp_tool_without_server_label_is_skipped(self) -> None:
        wrapper = _make_wrapper(tools=[{"type": "mcp", "server_url": "http://x"}])
        assert wrapper.get_tools() == []

    def test_non_mcp_tool_ignored(self) -> None:
        wrapper = _make_wrapper(tools=[{"type": "function", "name": "some_function"}])
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
        stream = wrapper.stream_response(
            request_id="req-stream-1", response_messages1=messages
        )
        chunks = [chunk async for chunk in stream]

        # created + 2 deltas + final (text.done + completed + [DONE]) = 4
        assert len(chunks) == 4

        created = json.loads(chunks[0][len("data: ") :])
        assert created["type"] == "response.created"
        assert created["response"]["id"] == "req-stream-1"
        assert created["response"]["model"] == "gpt-4o"
        assert created["response"]["status"] == "in_progress"

        delta1 = json.loads(chunks[1][len("data: ") :])
        assert delta1["type"] == "response.output_text.delta"
        assert delta1["delta"] == "Hello\n"

        delta2 = json.loads(chunks[2][len("data: ") :])
        assert delta2["type"] == "response.output_text.delta"
        assert delta2["delta"] == "World\n"

        # Final chunk contains multiple SSE events
        final_events = [
            line[len("data: ") :]
            for line in chunks[3].strip().split("\n")
            if line.startswith("data: ")
        ]
        text_done = json.loads(final_events[0])
        assert text_done["type"] == "response.output_text.done"
        assert text_done["text"] == ""
        completed = json.loads(final_events[1])
        assert completed["type"] == "response.completed"
        assert final_events[2] == "[DONE]"

    @pytest.mark.asyncio
    async def test_stream_response_skips_empty_content(self) -> None:
        wrapper = _make_wrapper()
        messages: List[AnyMessage] = [
            AIMessage(content=""),
            AIMessage(content="Real content"),
        ]
        stream = wrapper.stream_response(
            request_id="req-stream-2", response_messages1=messages
        )
        chunks = [chunk async for chunk in stream]

        # created + 1 delta (empty skipped) + done = 3
        assert len(chunks) == 3
        delta = json.loads(chunks[1][len("data: ") :])
        assert delta["delta"] == "Real content\n"


class TestCreateToolEndSseEvent:
    """Tests for create_tool_end_sse_event's runtime/output/error reporting."""

    def test_completed_call_includes_runtime_and_output(self) -> None:
        wrapper = _make_wrapper()
        raw = wrapper.create_tool_end_sse_event(
            request_id="req-1",
            tool_name="load_skill",
            tool_input={"skill_name": "pss"},
            runtime_seconds=1.5,
            output="Loaded skill pss.",
            is_error=False,
        )
        assert raw is not None
        event = json.loads(raw[len("data: ") :])
        item = event["item"]
        assert item["status"] == "completed"
        assert item["runtime_seconds"] == 1.5
        assert item["output"] == "Loaded skill pss."
        assert item["is_error"] is False

    def test_failed_call_marks_status_failed(self) -> None:
        wrapper = _make_wrapper()
        raw = wrapper.create_tool_end_sse_event(
            request_id="req-1",
            tool_name="call_tool",
            tool_input={"name": "propose_skill", "arguments": {}},
            runtime_seconds=0.8,
            output="Tool call failed:\nSkill validation failed: missing description",
            is_error=True,
        )
        assert raw is not None
        event = json.loads(raw[len("data: ") :])
        item = event["item"]
        assert item["status"] == "failed"
        assert item["is_error"] is True
        assert "Skill validation failed" in item["output"]

    def test_defaults_to_no_output_and_not_error(self) -> None:
        wrapper = _make_wrapper()
        raw = wrapper.create_tool_end_sse_event(
            request_id="req-1",
            tool_name="load_skill",
            tool_input=None,
            runtime_seconds=None,
        )
        assert raw is not None
        event = json.loads(raw[len("data: ") :])
        item = event["item"]
        assert item["status"] == "completed"
        assert item["runtime_seconds"] is None
        assert item["output"] == ""
        assert item["is_error"] is False
