from collections.abc import Callable
from typing import Any, Optional, cast
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessageChunk
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
