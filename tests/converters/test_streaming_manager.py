from collections.abc import Callable
from typing import Any, Optional, cast
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.runnables.schema import CustomStreamEvent, StandardStreamEvent

from languagemodelcommon.converters.stream_buffer import StreamBufferManager
from languagemodelcommon.converters.stream_debug_output_manager import (
    StreamDebugOutputManager,
)
from languagemodelcommon.converters.streaming_manager import LangGraphStreamingManager
from languagemodelcommon.converters.tool_event_handlers import ToolEventHandler
from languagemodelcommon.file_managers.file_writer import FileWriter
from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)
from languagemodelcommon.utilities.token_reducer.token_reducer import TokenReducer
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.request_information import RequestInformation
from languagemodelcommon.utilities.tool_display_name_mapper import ToolDisplayNameMapper


class _FakeChatRequestWrapper:
    def __init__(self, *, enable_debug_logging: bool) -> None:
        self.enable_debug_logging = enable_debug_logging

    def create_sse_message(
        self,
        *,
        request_id: str,
        content: str | None,
        usage_metadata: Optional[dict[str, Any]],
        source: str,
    ) -> str:
        return content or ""

    def create_debug_sse_message(
        self,
        *,
        request_id: str,
        content: str | None,
        usage_metadata: Optional[dict[str, Any]],
        source: str,
    ) -> str | None:
        return content

    def create_final_sse_message(
        self,
        *,
        request_id: str,
        usage_metadata: Optional[dict[str, Any]],
        source: str,
    ) -> str:
        return "final"

    def create_tool_heartbeat_sse_event(
        self,
        *,
        request_id: str,
        tool_name: str,
        elapsed_seconds: float,
    ) -> str | None:
        return f"heartbeat:{tool_name}:{elapsed_seconds:.0f}"

    def create_image_output_sse_event(
        self,
        *,
        request_id: str,
        image_part: dict[str, Any],
    ) -> str | None:
        return f"image:{image_part.get('image_url')}"

    def create_llm_call_start_sse_event(
        self,
        *,
        request_id: str,
        request_messages: list[dict[str, Any]],
    ) -> str | None:
        return None

    def create_llm_call_end_sse_event(
        self,
        *,
        request_id: str,
        response_text: str | None,
    ) -> str | None:
        return None


@pytest.fixture()
def streaming_manager_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], LangGraphStreamingManager]:
    def _factory() -> LangGraphStreamingManager:
        monkeypatch.setenv("BUFFER_FLUSH_INTERVAL_SECONDS", "10.0")
        monkeypatch.setenv("WRITE_TOOL_OUTPUT_TO_FILE", "false")
        environment_variables = LanguageModelCommonEnvironmentVariables()
        mock_file_writer = AsyncMock(spec=FileWriter)
        mock_file_writer.write_to_file_async = AsyncMock(return_value=None)
        stream_buffer_manager = StreamBufferManager(
            flush_interval_seconds=10.0,
            enabled=True,
        )
        stream_debug_output_manager = StreamDebugOutputManager()
        tool_event_handler = ToolEventHandler(
            debug_file_writer=mock_file_writer,
            environment_variables=environment_variables,
            tool_display_name_mapper=ToolDisplayNameMapper(),
            stream_buffer_manager=stream_buffer_manager,
            stream_debug_output_manager=stream_debug_output_manager,
        )
        return LangGraphStreamingManager(
            token_reducer=TokenReducer(),
            environment_variables=environment_variables,
            debug_file_writer=mock_file_writer,
            tool_event_handler=tool_event_handler,
            stream_buffer_manager=stream_buffer_manager,
            stream_debug_output_manager=stream_debug_output_manager,
        )

    return _factory


@pytest.mark.asyncio
async def test_chat_model_end_includes_streamed_text_when_debug_enabled(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-1")
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=True),
    )

    stream_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content="Hello world")},
        },
    )
    streamed_chunks = [
        chunk
        async for chunk in manager._handle_on_chat_model_stream(
            event=stream_event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
        )
    ]
    assert streamed_chunks == []

    end_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_end",
            "data": {"input": {"messages": []}},
        },
    )
    debug_chunks = [
        chunk
        async for chunk in manager._handle_on_chat_model_end(
            event=end_event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
        )
        if chunk is not None
    ]

    assert len(debug_chunks) == 1
    assert "Streamed assistant output" in debug_chunks[0]
    assert "Hello world" in debug_chunks[0]


@pytest.mark.asyncio
async def test_chain_end_clears_streamed_text_when_chat_model_end_not_called(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-2")
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=False),
    )

    stream_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content="partial response")},
        },
    )
    _ = [
        chunk
        async for chunk in manager._handle_on_chat_model_stream(
            event=stream_event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
        )
    ]

    chain_end_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chain_end",
            "data": {},
        },
    )
    _ = [
        chunk
        async for chunk in manager._handle_on_chain_end(
            event=chain_end_event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
        )
    ]

    # Verify debug output was cleared
    assert manager._stream_debug_output_manager.pop_text() is None


@pytest.mark.asyncio
async def test_custom_event_mcp_tool_heartbeat_forwards_to_wrapper(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-1")
    chat_request_wrapper = _FakeChatRequestWrapper(enable_debug_logging=False)

    event = cast(
        CustomStreamEvent,
        {
            "event": "on_custom_event",
            "name": "mcp_tool_heartbeat",
            "data": {"tool_name": "propose_skill", "elapsed_seconds": 15.0},
        },
    )

    chunks = [
        chunk
        async for chunk in manager.handle_langchain_event(
            event=event,
            chat_request_wrapper=cast(ChatRequestWrapper, chat_request_wrapper),
            request_information=request_information,
            tool_start_times={},
        )
    ]
    assert chunks == ["heartbeat:propose_skill:15"]


@pytest.mark.asyncio
async def test_resuming_after_tool_call_inserts_missing_separator(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    """Regression test for BAI-726: LangGraph re-invokes the chat model after
    a tool call, and the resumed completion is not guaranteed to start with
    whitespace against the text already streamed -- without the on_chat_model_start
    boundary check, "have access to." + "Let me search" renders as
    "have access to.Let me search"."""
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-3")
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=False),
    )

    async def _drive(event: StandardStreamEvent | CustomStreamEvent) -> list[str]:
        return [
            chunk
            async for chunk in manager.handle_langchain_event(
                event=event,
                chat_request_wrapper=chat_request_wrapper,
                request_information=request_information,
                tool_start_times={},
            )
        ]

    # First chat-model invocation streams text, then the graph calls a tool.
    await _drive(cast(StandardStreamEvent, {"event": "on_chat_model_start"}))
    await _drive(
        cast(
            StandardStreamEvent,
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="have access to.")},
            },
        )
    )

    # LangGraph re-invokes the chat model after the tool call completes.
    await _drive(cast(StandardStreamEvent, {"event": "on_chat_model_start"}))
    resumed_chunks = await _drive(
        cast(
            StandardStreamEvent,
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="Let me search")},
            },
        )
    )
    resumed_chunks += await _drive(
        cast(StandardStreamEvent, {"event": "on_chain_end", "data": {}})
    )

    rendered = "".join(chunk for chunk in resumed_chunks if chunk)
    assert "to.Let" not in rendered
    assert "to. Let" in rendered


@pytest.mark.asyncio
async def test_chat_model_stream_emits_image_event_for_image_only_chunk(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    """BAI-806: an image content block must produce an atomic image SSE
    event instead of being silently dropped by the text-delta path."""
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-4")
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=False),
    )

    stream_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": AIMessageChunk(
                    content=[
                        {
                            "type": "image",
                            "url": "https://example.com/chart.png",
                            "mime_type": "image/png",
                        }
                    ]
                )
            },
        },
    )
    chunks = [
        chunk
        async for chunk in manager._handle_on_chat_model_stream(
            event=stream_event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
        )
    ]
    assert chunks == ["image:https://example.com/chart.png"]


@pytest.mark.asyncio
async def test_chat_model_stream_emits_text_and_image_for_mixed_chunk(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-5")
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=False),
    )

    stream_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": AIMessageChunk(
                    content=[
                        {"type": "text", "text": "Here is the chart:"},
                        {
                            "type": "image",
                            "url": "https://example.com/chart.png",
                        },
                    ]
                )
            },
        },
    )
    chunks = [
        chunk
        async for chunk in manager._handle_on_chat_model_stream(
            event=stream_event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
        )
    ]
    # The image event rides its own atomic path and is not subject to the
    # text buffer's flush timing (unlike the text chunk alongside it, which
    # may still be buffered -- see test_chat_model_end_includes_streamed_text_
    # when_debug_enabled's equivalent "streamed_chunks == []" assertion).
    assert "image:https://example.com/chart.png" in chunks


@pytest.mark.asyncio
async def test_chat_model_stream_text_only_emits_no_image_event(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-6")
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=False),
    )

    stream_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content="just text")},
        },
    )
    chunks = [
        chunk
        async for chunk in manager._handle_on_chat_model_stream(
            event=stream_event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
        )
    ]
    assert not any(c and c.startswith("image:") for c in chunks)


class _RecordingLlmCallChatRequestWrapper(_FakeChatRequestWrapper):
    """Records create_llm_call_start/end_sse_event calls and returns a
    distinguishable sentinel, so a test can assert the streaming manager's
    on_chat_model_start/end handlers actually invoke and yield them -- BAI-882
    review finding: the shared _FakeChatRequestWrapper returns None for both,
    so no existing test could tell whether this wiring worked at all."""

    def __init__(self, *, enable_debug_logging: bool) -> None:
        super().__init__(enable_debug_logging=enable_debug_logging)
        self.llm_call_start_calls: list[list[dict[str, Any]]] = []
        self.llm_call_end_calls: list[str | None] = []

    def create_llm_call_start_sse_event(
        self,
        *,
        request_id: str,
        request_messages: list[dict[str, Any]],
    ) -> str | None:
        self.llm_call_start_calls.append(request_messages)
        return f"llm_call_start:{len(self.llm_call_start_calls)}"

    def create_llm_call_end_sse_event(
        self,
        *,
        request_id: str,
        response_text: str | None,
    ) -> str | None:
        self.llm_call_end_calls.append(response_text)
        return f"llm_call_end:{len(self.llm_call_end_calls)}"


@pytest.mark.asyncio
async def test_chat_model_start_yields_llm_call_start_with_request_messages(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-1")
    chat_request_wrapper = _RecordingLlmCallChatRequestWrapper(
        enable_debug_logging=False
    )

    event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_start",
            "data": {
                "input": {
                    "messages": [
                        [
                            HumanMessage(content="Hello"),
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "search",
                                        "args": {"query": "labs"},
                                        "id": "tc1",
                                    }
                                ],
                            ),
                        ]
                    ]
                }
            },
        },
    )

    chunks = [
        chunk
        async for chunk in manager._handle_on_chat_model_start(
            event=event,
            chat_request_wrapper=cast(ChatRequestWrapper, chat_request_wrapper),
            request_information=request_information,
        )
    ]

    assert chunks == ["llm_call_start:1"]
    assert chat_request_wrapper.llm_call_start_calls == [
        [
            {"role": "human", "content": "Hello"},
            {
                "role": "ai",
                "content": "",
                "tool_calls": [{"name": "search", "args": {"query": "labs"}}],
            },
        ]
    ]


@pytest.mark.asyncio
async def test_chat_model_end_yields_llm_call_end_unconditionally(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    """The llm_call end event must fire even when debug logging is disabled --
    unlike the pre-existing debug-only messages-log chunk emitted later in the
    same handler."""
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-1")
    chat_request_wrapper = _RecordingLlmCallChatRequestWrapper(
        enable_debug_logging=False
    )

    start_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent, {"event": "on_chat_model_start", "data": {}}
    )
    async for _ in manager._handle_on_chat_model_start(
        event=start_event,
        chat_request_wrapper=cast(ChatRequestWrapper, chat_request_wrapper),
        request_information=request_information,
    ):
        pass

    stream_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content="Hi there!")},
        },
    )
    async for _ in manager._handle_on_chat_model_stream(
        event=stream_event,
        chat_request_wrapper=cast(ChatRequestWrapper, chat_request_wrapper),
        request_information=request_information,
    ):
        pass

    end_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {"event": "on_chat_model_end", "data": {"input": {"messages": []}}},
    )
    chunks = [
        chunk
        async for chunk in manager._handle_on_chat_model_end(
            event=end_event,
            chat_request_wrapper=cast(ChatRequestWrapper, chat_request_wrapper),
            request_information=request_information,
        )
    ]

    assert chunks == ["llm_call_end:1"]
    assert chat_request_wrapper.llm_call_end_calls == ["Hi there!"]
