"""Tests for ResponsesApiRequestWrapper."""

import json
import logging
import re
from typing import Any, List, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from languagemodelcommon.schema.openai.responses import (
    MAX_CONVERSATION_ID_LENGTH,
    ResponsesRequest,
)
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
    conversation: str | dict[str, Any] | None = None,
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
        conversation=cast(Any, conversation),
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

    def test_messages_preserve_role_separation_for_multi_turn_input(self) -> None:
        """`user_input` joins every item's text regardless of role (see test above) --
        that's the known flattening behavior consumed by the old
        AssembleResponsesMessagesStep. `wrapper.messages` is the role-aware,
        per-item accessor that replaced it (mirroring
        ChatCompletionApiRequestWrapper.messages): each Responses API `input`
        item becomes its own LangChain message with role preserved, instead of
        being collapsed into one string.
        """
        wrapper = _make_wrapper(
            input_=[
                {"role": "user", "content": "first user turn"},
                {"role": "assistant", "content": "assistant reply"},
                {"role": "user", "content": "second user turn"},
            ]
        )

        converted: List[AnyMessage] = [
            m.to_langchain_message() for m in wrapper.messages
        ]

        assert len(converted) == 3
        assert isinstance(converted[0], HumanMessage)
        assert converted[0].content == "first user turn"
        assert isinstance(converted[1], AIMessage)
        assert converted[1].content == "assistant reply"
        assert isinstance(converted[2], HumanMessage)
        assert converted[2].content == "second user turn"

    def test_messages_preserve_system_role_for_multi_turn_input(self) -> None:
        wrapper = _make_wrapper(
            input_=[
                {"role": "system", "content": "system turn"},
                {"role": "user", "content": "user turn"},
            ]
        )

        converted: List[AnyMessage] = [
            m.to_langchain_message() for m in wrapper.messages
        ]

        assert len(converted) == 2
        assert isinstance(converted[0], SystemMessage)
        assert converted[0].content == "system turn"
        assert isinstance(converted[1], HumanMessage)
        assert converted[1].content == "user turn"


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

    @pytest.mark.parametrize(
        ("tool_extras", "expected_auth", "expected_headers"),
        [
            pytest.param({}, None, None, id="no-headers-key"),
            pytest.param({"headers": None}, None, None, id="headers-none"),
            pytest.param({"headers": {}}, None, None, id="headers-empty"),
            pytest.param(
                {"headers": {"X-Token": "secret"}},
                "headers",
                {"X-Token": "secret"},
                id="static-header",
            ),
            pytest.param(
                {"headers": {"Authorization": "Bearer abc"}},
                "headers",
                {"Authorization": "Bearer abc"},
                id="authorization-header",
            ),
        ],
    )
    def test_mcp_tool_with_server_url_sets_auth_headers_only_when_headers_supplied(
        self,
        tool_extras: dict[str, Any],
        expected_auth: str | None,
        expected_headers: dict[str, str] | None,
    ) -> None:
        wrapper = _make_wrapper(
            tools=[
                {
                    "type": "mcp",
                    "server_url": "http://localhost:8080",
                    "server_label": "my-server",
                    **tool_extras,
                }
            ]
        )

        configs = wrapper.get_tools()

        assert len(configs) == 1
        assert configs[0].auth == expected_auth
        assert configs[0].headers == expected_headers

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

    def test_includes_structured_output_when_present(self) -> None:
        """BAI-879/BAI-882: a tool's own MCP structuredContent rides alongside
        the text output on the same event."""
        wrapper = _make_wrapper()
        raw = wrapper.create_tool_end_sse_event(
            request_id="req-1",
            tool_name="search_connections",
            tool_input={"query": "labs"},
            runtime_seconds=0.5,
            output="Found 2 connections.",
            is_error=False,
            structured_output={"count": 2, "connections": ["a", "b"]},
        )
        assert raw is not None
        event = json.loads(raw[len("data: ") :])
        assert event["item"]["structured_output"] == {
            "count": 2,
            "connections": ["a", "b"],
        }

    def test_defaults_structured_output_to_none(self) -> None:
        wrapper = _make_wrapper()
        raw = wrapper.create_tool_end_sse_event(
            request_id="req-1",
            tool_name="load_skill",
            tool_input=None,
            runtime_seconds=None,
        )
        assert raw is not None
        event = json.loads(raw[len("data: ") :])
        assert event["item"]["structured_output"] is None


class TestCreateLlmCallSseEvents:
    """Tests for create_llm_call_start_sse_event/create_llm_call_end_sse_event
    (BAI-882): one pair per individual model invocation within a turn."""

    def test_start_event_carries_request_messages(self) -> None:
        wrapper = _make_wrapper()
        request_messages = [
            {"role": "system", "content": "You are Bailey."},
            {"role": "user", "content": "Hello"},
        ]
        raw = wrapper.create_llm_call_start_sse_event(
            request_id="req-1",
            request_messages=request_messages,
        )
        assert raw is not None
        event = json.loads(raw[len("data: ") :])
        assert event["type"] == "response.output_item.added"
        item = event["item"]
        assert item["type"] == "llm_call"
        assert item["status"] == "in_progress"
        assert item["request"] == request_messages

    def test_end_event_carries_response_text(self) -> None:
        wrapper = _make_wrapper()
        raw = wrapper.create_llm_call_end_sse_event(
            request_id="req-1",
            response_text="Hi there!",
        )
        assert raw is not None
        event = json.loads(raw[len("data: ") :])
        assert event["type"] == "response.output_item.done"
        item = event["item"]
        assert item["type"] == "llm_call"
        assert item["status"] == "completed"
        assert item["output"] == "Hi there!"

    def test_end_event_defaults_output_to_empty_string(self) -> None:
        wrapper = _make_wrapper()
        raw = wrapper.create_llm_call_end_sse_event(
            request_id="req-1",
            response_text=None,
        )
        assert raw is not None
        event = json.loads(raw[len("data: ") :])
        assert event["item"]["output"] == ""


class TestCreateImageOutputSseEvent:
    """Tests for create_image_output_sse_event (BAI-806)."""

    def test_emits_output_item_done_with_output_image_item(self) -> None:
        wrapper = _make_wrapper()
        raw = wrapper.create_image_output_sse_event(
            request_id="req-1",
            image_part={
                "type": "output_image",
                "image_url": "https://example.com/chart.png",
                "mime_type": "image/png",
            },
        )
        assert raw is not None
        event = json.loads(raw[len("data: ") :])
        assert event["type"] == "response.output_item.done"
        item = event["item"]
        assert item["type"] == "output_image"
        assert item["status"] == "completed"
        assert item["image_url"] == "https://example.com/chart.png"
        assert item["mime_type"] == "image/png"
        assert item["id"].startswith("img_req-1_")

    def test_multiple_images_in_same_turn_get_distinct_ids(self) -> None:
        """Regression for BAI-806 review finding: len(self._messages) alone
        is constant across calls within one turn, so without a per-image
        discriminator two images emitted back-to-back would collide on id."""
        wrapper = _make_wrapper()
        image_part = {
            "type": "output_image",
            "image_url": "https://example.com/chart.png",
            "mime_type": "image/png",
        }
        first_raw = wrapper.create_image_output_sse_event(
            request_id="req-1", image_part=image_part
        )
        second_raw = wrapper.create_image_output_sse_event(
            request_id="req-1", image_part=image_part
        )
        assert first_raw is not None
        assert second_raw is not None
        first = json.loads(first_raw[len("data: ") :])
        second = json.loads(second_raw[len("data: ") :])
        assert first["item"]["id"] != second["item"]["id"]


def _event_lines(*, chunk: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in chunk.split("\n"):
        if line.startswith("data: ") and line[6:] != "[DONE]":
            events.append(json.loads(line[6:]))
    return events


def _completed_response(*, wrapper: ResponsesApiRequestWrapper) -> dict[str, Any]:
    chunk = wrapper.create_final_sse_message(
        request_id="r", usage_metadata=None, source="t"
    )
    (completed,) = [
        e for e in _event_lines(chunk=chunk) if e["type"] == "response.completed"
    ]
    return cast(dict[str, Any], completed["response"])


def _created_response(*, wrapper: ResponsesApiRequestWrapper) -> dict[str, Any]:
    chunk = wrapper.create_first_sse_message(request_id="r", source="t")
    (created,) = _event_lines(chunk=chunk)
    assert created["type"] == "response.created"
    return cast(dict[str, Any], created["response"])


def _non_streaming(*, wrapper: ResponsesApiRequestWrapper) -> dict[str, Any]:
    return wrapper.create_non_streaming_response(
        request_id="r",
        json_output_requested=False,
        responses=[AIMessage(content="ok")],
    )


class TestConversationEcho:
    """The server-chosen conversation id is echoed on every Response object (BAI-1244)."""

    def test_created_event_carries_conversation_id(self) -> None:
        wrapper = _make_wrapper()
        wrapper.conversation_id = "u1:chat-ab12"

        assert _created_response(wrapper=wrapper)["conversation"] == {
            "id": "u1:chat-ab12"
        }

    def test_completed_event_carries_conversation_id(self) -> None:
        wrapper = _make_wrapper()
        wrapper.conversation_id = "u1:chat-ab12"

        assert _completed_response(wrapper=wrapper)["conversation"] == {
            "id": "u1:chat-ab12"
        }
        chunk = wrapper.create_final_sse_message(
            request_id="r", usage_metadata=None, source="t"
        )
        assert chunk.count("data: ") == 3
        assert chunk.endswith("data: [DONE]\n\n")

    def test_non_streaming_body_carries_conversation_id(self) -> None:
        wrapper = _make_wrapper()
        wrapper.conversation_id = "u1:chat-ab12"

        assert _non_streaming(wrapper=wrapper)["conversation"] == {"id": "u1:chat-ab12"}

    @pytest.mark.asyncio
    async def test_conversation_id_is_identical_across_all_three_payloads(
        self,
    ) -> None:
        wrapper = _make_wrapper(stream=True)
        wrapper.conversation_id = "u1:chat-ab12"
        expected = {"id": "u1:chat-ab12"}

        assert _created_response(wrapper=wrapper)["conversation"] == expected
        assert _completed_response(wrapper=wrapper)["conversation"] == expected
        assert _non_streaming(wrapper=wrapper)["conversation"] == expected

        streamed = [
            event
            async for chunk in wrapper.stream_response(
                request_id="r", response_messages1=[AIMessage(content="ok")]
            )
            for event in _event_lines(chunk=chunk)
        ]
        conversations = {
            e["type"]: e["response"]["conversation"]
            for e in streamed
            if e["type"] in ("response.created", "response.completed")
        }
        assert conversations == {
            "response.created": expected,
            "response.completed": expected,
        }

    def test_unset_conversation_id_leaves_payload_conversation_null(self) -> None:
        wrapper = _make_wrapper()

        assert _created_response(wrapper=wrapper)["conversation"] is None
        assert _completed_response(wrapper=wrapper)["conversation"] is None
        assert _non_streaming(wrapper=wrapper)["conversation"] is None
        assert wrapper.conversation_id is None

    @pytest.mark.parametrize("conversation", ["u1:chat-zz", {"id": "u1:chat-zz"}])
    def test_request_conversation_is_not_echoed_unless_the_caller_sets_conversation_id(
        self, conversation: str | dict[str, Any]
    ) -> None:
        wrapper = _make_wrapper(conversation=conversation)

        assert _created_response(wrapper=wrapper)["conversation"] is None
        assert _completed_response(wrapper=wrapper)["conversation"] is None
        assert _non_streaming(wrapper=wrapper)["conversation"] is None

    @pytest.mark.parametrize(
        "value", ["", "   ", "x" * (MAX_CONVERSATION_ID_LENGTH + 1)]
    )
    def test_setting_blank_conversation_id_raises(self, value: str) -> None:
        wrapper = _make_wrapper()

        with pytest.raises(ValueError):
            wrapper.conversation_id = value

    def test_setting_none_clears_conversation_id(self) -> None:
        wrapper = _make_wrapper()
        wrapper.conversation_id = "u1:chat-ab12"
        wrapper.conversation_id = None

        assert wrapper.conversation_id is None
        assert _completed_response(wrapper=wrapper)["conversation"] is None

    def test_completed_chunk_survives_a_top_level_key_rewrite(self) -> None:
        """Mirrors baileyai's attach_message_id_to_completed_event round trip."""
        wrapper = _make_wrapper()
        wrapper.conversation_id = "u1:chat-ab12"
        chunk = wrapper.create_final_sse_message(
            request_id="r", usage_metadata=None, source="t"
        )

        rewritten: list[str] = []
        for line in chunk.split("\n"):
            if line.startswith("data: ") and line[6:] != "[DONE]":
                event = json.loads(line[6:])
                if event["type"] == "response.completed":
                    event["message_id"] = "m1"
                    line = f"data: {json.dumps(event)}"
            rewritten.append(line)

        completed = [
            e
            for e in _event_lines(chunk="\n".join(rewritten))
            if e["type"] == "response.completed"
        ][0]
        assert completed["response"]["conversation"] == {"id": "u1:chat-ab12"}
        assert completed["message_id"] == "m1"
        assert "message_id" not in completed["response"]


class TestHistoryOutcomeEcho:
    """The history load outcome rides in `conversation` on every Response (BAI-1263)."""

    @pytest.mark.parametrize(
        ("status", "loaded"),
        [("loaded", 7), ("empty", 0), ("failed", 0)],
    )
    @pytest.mark.parametrize(
        "builder", [_created_response, _completed_response, _non_streaming]
    )
    def test_status_and_count_are_emitted_in_all_three_payloads(
        self, status: str, loaded: int, builder: Any
    ) -> None:
        wrapper = _make_wrapper()
        wrapper.conversation_id = "u1:chat-ab12"
        wrapper.history_status = status
        wrapper.history_loaded = loaded

        assert builder(wrapper=wrapper)["conversation"] == {
            "id": "u1:chat-ab12",
            "history_status": status,
            "history_loaded": loaded,
        }

    @pytest.mark.parametrize(
        "builder", [_created_response, _completed_response, _non_streaming]
    )
    def test_unset_status_leaves_conversation_identical_to_id_only(
        self, builder: Any
    ) -> None:
        wrapper = _make_wrapper()
        wrapper.conversation_id = "u1:chat-ab12"
        wrapper.history_loaded = 5  # a count without a status is never emitted

        assert builder(wrapper=wrapper)["conversation"] == {"id": "u1:chat-ab12"}

    @pytest.mark.parametrize(
        "builder", [_created_response, _completed_response, _non_streaming]
    )
    def test_status_without_conversation_id_keeps_conversation_null(
        self, builder: Any
    ) -> None:
        wrapper = _make_wrapper()
        wrapper.history_status = "failed"

        assert builder(wrapper=wrapper)["conversation"] is None

    def test_failed_status_with_zero_loaded_emits_the_zero(self) -> None:
        wrapper = _make_wrapper()
        wrapper.conversation_id = "u1:chat-ab12"
        wrapper.history_status = "failed"

        assert _completed_response(wrapper=wrapper)["conversation"] == {
            "id": "u1:chat-ab12",
            "history_status": "failed",
            "history_loaded": 0,
        }

    def test_key_order_and_raw_serialization_are_stable(self) -> None:
        wrapper = _make_wrapper()
        wrapper.conversation_id = "u1:chat-ab12"
        wrapper.history_status = "loaded"
        wrapper.history_loaded = 2

        chunk = wrapper.create_first_sse_message(request_id="r", source="t")

        assert (
            '"conversation":{"id":"u1:chat-ab12","history_status":"loaded",'
            '"history_loaded":2}'
        ) in chunk
        assert chunk.count("\n") == 2  # single-line data: event

    def test_payload_without_history_is_byte_identical_to_before(self) -> None:
        plain = _make_wrapper()
        plain.conversation_id = "u1:chat-ab12"
        touched = _make_wrapper()
        touched.conversation_id = "u1:chat-ab12"
        touched.history_status = "empty"
        touched.history_status = None  # cleared again

        def normalise(chunk: str) -> str:
            return re.sub(r'"created_at":[^,}]*', "", chunk)

        assert normalise(
            plain.create_final_sse_message(
                request_id="r", usage_metadata=None, source="t"
            )
        ) == normalise(
            touched.create_final_sse_message(
                request_id="r", usage_metadata=None, source="t"
            )
        )

    @pytest.mark.asyncio
    async def test_live_stream_start_and_completed_events_both_carry_outcome(
        self,
    ) -> None:
        wrapper = _make_wrapper(stream=True)
        wrapper.conversation_id = "u1:chat-ab12"
        wrapper.history_status = "failed"
        expected = {
            "id": "u1:chat-ab12",
            "history_status": "failed",
            "history_loaded": 0,
        }

        started = wrapper.create_stream_start_sse_event(request_id="r", source="t")
        assert started is not None
        assert _event_lines(chunk=started)[0]["response"]["conversation"] == expected
        assert _completed_response(wrapper=wrapper)["conversation"] == expected

    def test_completed_chunk_survives_a_top_level_key_rewrite(self) -> None:
        wrapper = _make_wrapper()
        wrapper.conversation_id = "u1:chat-ab12"
        wrapper.history_status = "loaded"
        wrapper.history_loaded = 3
        chunk = wrapper.create_final_sse_message(
            request_id="r", usage_metadata=None, source="t"
        )

        event = next(
            e for e in _event_lines(chunk=chunk) if e["type"] == "response.completed"
        )
        event["message_id"] = "m1"
        reparsed = json.loads(json.dumps(event))
        assert reparsed["response"]["conversation"]["history_loaded"] == 3

    @pytest.mark.parametrize("value", ["", "LOADED", "partial"])
    def test_unknown_status_is_ignored_not_raised(self, value: str) -> None:
        wrapper = _make_wrapper()
        wrapper.conversation_id = "u1:chat-ab12"

        wrapper.history_status = value

        assert wrapper.history_status is None
        assert _completed_response(wrapper=wrapper)["conversation"] == {
            "id": "u1:chat-ab12"
        }

    def test_created_event_only_carries_status_set_before_it_is_built(self) -> None:
        wrapper = _make_wrapper()
        wrapper.conversation_id = "u1:chat-ab12"
        before = _created_response(wrapper=wrapper)["conversation"]
        wrapper.history_status = "failed"

        assert before == {"id": "u1:chat-ab12"}
        assert (
            _completed_response(wrapper=wrapper)["conversation"]["history_status"]
            == "failed"
        )

    def test_upstream_conversation_model_still_allows_extra_keys(self) -> None:
        from openai.types.responses.response import Conversation

        assert Conversation.model_config.get("extra") == "allow", (
            "history fields rely on openai Conversation allowing extras"
        )

    @pytest.mark.parametrize("value", [-1, True, 1.5, "3", None])
    def test_invalid_count_is_dropped_and_logged_not_raised(
        self, value: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Same policy as an unknown status: an observability field must not fail the turn."""
        wrapper = _make_wrapper()
        wrapper.history_loaded = 4

        with caplog.at_level(logging.WARNING):
            wrapper.history_loaded = value

        assert wrapper.history_loaded == 0
        assert "Ignoring invalid history count" in caplog.text
        assert repr(value) not in caplog.text

    @pytest.mark.parametrize("value", [0, 1, 250])
    def test_valid_count_is_kept(self, value: int) -> None:
        wrapper = _make_wrapper()

        wrapper.history_loaded = value

        assert wrapper.history_loaded == value


class TestUsageConversion:
    """Usage metadata must serialize on every response path.

    Newer openai releases require `cache_write_tokens` on `InputTokensDetails`; building it
    from `cached_tokens` alone raised a ValidationError whenever a turn carried usage metadata.
    """

    @pytest.mark.parametrize(
        ("usage_metadata", "expected_input", "expected_output", "expected_total"),
        [
            ({"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}, 5, 7, 12),
            ({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, 0, 0, 0),
        ],
    )
    def test_streamed_completed_event_carries_usage(
        self,
        usage_metadata: dict[str, int],
        expected_input: int,
        expected_output: int,
        expected_total: int,
    ) -> None:
        wrapper = _make_wrapper()
        result = wrapper.create_final_sse_message(
            request_id="req-1", usage_metadata=cast(Any, usage_metadata), source="test"
        )
        completed = next(
            json.loads(line[len("data: ") :])
            for line in result.strip().split("\n")
            if line.startswith("data: ") and '"response.completed"' in line
        )
        usage = completed["response"]["usage"]
        assert usage["input_tokens"] == expected_input
        assert usage["output_tokens"] == expected_output
        assert usage["total_tokens"] == expected_total
        assert usage["input_tokens_details"]["cached_tokens"] == 0
        assert usage["input_tokens_details"]["cache_write_tokens"] == 0

    def test_non_streaming_response_sums_usage_across_messages(self) -> None:
        wrapper = _make_wrapper()
        first = AIMessage(content="a")
        first.usage_metadata = {
            "input_tokens": 3,
            "output_tokens": 4,
            "total_tokens": 7,
        }
        second = AIMessage(content="b")
        second.usage_metadata = {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
        }
        result = wrapper.create_non_streaming_response(
            request_id="req-1",
            json_output_requested=False,
            responses=cast(List[AnyMessage], [first, second]),
        )
        usage = result["usage"]
        assert usage["input_tokens"] == 4
        assert usage["output_tokens"] == 6
        assert usage["total_tokens"] == 10
        assert usage["input_tokens_details"]["cache_write_tokens"] == 0

    @pytest.mark.parametrize(
        ("input_token_details", "expected_cached", "expected_cache_write"),
        [
            ({"cache_read": 100, "cache_creation": 200}, 100, 200),
            ({"cache_read": 100}, 100, 0),
            ({"cache_creation": 200}, 0, 200),
            ({"audio": 10}, 0, 0),
            ({}, 0, 0),
            ({"cache_read": None, "cache_creation": None}, 0, 0),
            (
                {
                    "cache_read": 100,
                    "cache_creation": 0,
                    "ephemeral_5m_input_tokens": 120,
                    "ephemeral_1h_input_tokens": 80,
                },
                100,
                200,
            ),
        ],
    )
    def test_streamed_completed_event_reports_cache_tokens_from_input_token_details(
        self,
        input_token_details: dict[str, int | None],
        expected_cached: int,
        expected_cache_write: int,
    ) -> None:
        wrapper = _make_wrapper()
        usage_metadata = {
            "input_tokens": 300,
            "output_tokens": 7,
            "total_tokens": 307,
            "input_token_details": input_token_details,
        }
        result = wrapper.create_final_sse_message(
            request_id="req-1", usage_metadata=cast(Any, usage_metadata), source="test"
        )
        completed = next(
            json.loads(line[len("data: ") :])
            for line in result.strip().split("\n")
            if line.startswith("data: ") and '"response.completed"' in line
        )
        details = completed["response"]["usage"]["input_tokens_details"]
        assert details["cached_tokens"] == expected_cached
        assert details["cache_write_tokens"] == expected_cache_write

    def test_non_streaming_response_sums_cache_tokens_across_messages(self) -> None:
        wrapper = _make_wrapper()
        first = AIMessage(content="a")
        first.usage_metadata = {
            "input_tokens": 300,
            "output_tokens": 4,
            "total_tokens": 304,
            "input_token_details": {"cache_read": 100, "cache_creation": 50},
        }
        second = AIMessage(content="b")
        second.usage_metadata = {
            "input_tokens": 200,
            "output_tokens": 2,
            "total_tokens": 202,
            "input_token_details": {"cache_read": 25},
        }
        third = AIMessage(content="c")
        third.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 1,
            "total_tokens": 11,
        }
        result = wrapper.create_non_streaming_response(
            request_id="req-1",
            json_output_requested=False,
            responses=cast(List[AnyMessage], [first, second, third]),
        )
        details = result["usage"]["input_tokens_details"]
        assert details["cached_tokens"] == 125
        assert details["cache_write_tokens"] == 50

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("first_details", "second_details", "expected_cached", "expected_cache_write"),
        [
            ({"cache_read": 100, "cache_creation": 50}, None, 100, 50),
            ({"cache_read": 100, "cache_creation": 50}, {"cache_read": 25}, 125, 50),
            (
                {
                    "cache_read": 100,
                    "cache_creation": 0,
                    "ephemeral_5m_input_tokens": 30,
                },
                {"cache_creation": 20, "ephemeral_1h_input_tokens": 10},
                100,
                60,
            ),
            (None, None, 0, 0),
        ],
    )
    async def test_stream_response_completed_event_reports_cache_tokens(
        self,
        first_details: dict[str, int] | None,
        second_details: dict[str, int] | None,
        expected_cached: int,
        expected_cache_write: int,
    ) -> None:
        wrapper = _make_wrapper(stream=True)
        first = AIMessage(content="a")
        first.usage_metadata = cast(
            Any,
            {"input_tokens": 300, "output_tokens": 4, "total_tokens": 304}
            | ({"input_token_details": first_details} if first_details else {}),
        )
        second = AIMessage(content="b")
        second.usage_metadata = cast(
            Any,
            {"input_tokens": 200, "output_tokens": 2, "total_tokens": 202}
            | ({"input_token_details": second_details} if second_details else {}),
        )

        completed = [
            json.loads(line[len("data: ") :])
            async for chunk in wrapper.stream_response(
                request_id="r", response_messages1=[first, second]
            )
            for line in chunk.split("\n")
            if line.startswith("data: ") and '"response.completed"' in line
        ]

        (event,) = completed
        usage = event["response"]["usage"]
        assert usage["input_tokens"] == 500
        assert usage["input_tokens_details"]["cached_tokens"] == expected_cached
        assert (
            usage["input_tokens_details"]["cache_write_tokens"] == expected_cache_write
        )

    def test_non_streaming_response_without_usage_has_no_usage(self) -> None:
        wrapper = _make_wrapper()
        result = wrapper.create_non_streaming_response(
            request_id="req-1",
            json_output_requested=False,
            responses=cast(List[AnyMessage], [AIMessage(content="a")]),
        )
        assert result["usage"] is None
