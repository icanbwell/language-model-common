from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.runnables.schema import StandardStreamEvent

from languagemodelcommon.converters.stream_buffer import StreamBufferManager
from languagemodelcommon.converters.stream_debug_output_manager import (
    StreamDebugOutputManager,
)
from languagemodelcommon.converters.tool_event_handlers import ToolEventHandler
from languagemodelcommon.file_managers.file_writer import FileWriter
from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.request_information import RequestInformation
from languagemodelcommon.utilities.tool_display_name_mapper import ToolDisplayNameMapper


class _FakeChatRequestWrapper:
    def __init__(self, *, enable_debug_logging: bool = False) -> None:
        self.enable_debug_logging = enable_debug_logging
        self.last_tool_end_output: str | None = None
        self.last_tool_end_is_error: bool = False

    def create_sse_message(
        self, *, request_id: str, content: str | None, usage_metadata: Any, source: str
    ) -> str:
        return content or ""

    def create_debug_sse_message(
        self, *, request_id: str, content: str | None, usage_metadata: Any, source: str
    ) -> str | None:
        return content

    def create_tool_start_sse_event(
        self, *, request_id: str, tool_name: str, tool_input: Any
    ) -> str | None:
        return None

    def create_tool_end_sse_event(
        self,
        *,
        request_id: str,
        tool_name: str,
        tool_input: Any,
        runtime_seconds: Any,
        output: str | None = None,
        is_error: bool = False,
    ) -> str | None:
        self.last_tool_end_output = output
        self.last_tool_end_is_error = is_error
        return None

    def create_mcp_app_sse_event(self, **kwargs: Any) -> str | None:
        return None


@pytest.fixture
def tool_event_handler(monkeypatch: pytest.MonkeyPatch) -> ToolEventHandler:
    monkeypatch.setenv("WRITE_TOOL_OUTPUT_TO_FILE", "false")
    environment_variables = LanguageModelCommonEnvironmentVariables()
    mock_file_writer = AsyncMock(spec=FileWriter)
    mock_file_writer.write_to_file_async = AsyncMock(return_value=None)
    stream_buffer_manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=False,
    )
    stream_debug_output_manager = StreamDebugOutputManager()
    return ToolEventHandler(
        debug_file_writer=mock_file_writer,
        environment_variables=environment_variables,
        tool_display_name_mapper=ToolDisplayNameMapper(),
        stream_buffer_manager=stream_buffer_manager,
        stream_debug_output_manager=stream_debug_output_manager,
    )


@pytest.mark.asyncio
async def test_tool_start_records_start_time(
    tool_event_handler: ToolEventHandler,
) -> None:
    event = cast(
        StandardStreamEvent,
        {
            "event": "on_tool_start",
            "name": "search_tool",
            "data": {"input": {"query": "test"}},
        },
    )
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=False),
    )
    request_information = RequestInformation(request_id="req-1")
    tool_start_times: dict[str, float] = {}

    # Consume the generator to trigger the side effect
    async for _ in tool_event_handler.handle_tool_start(
        event=event,
        chat_request_wrapper=chat_request_wrapper,
        request_information=request_information,
        tool_start_times=tool_start_times,
    ):
        pass

    assert len(tool_start_times) == 1


@pytest.mark.asyncio
async def test_tool_end_yields_content(
    tool_event_handler: ToolEventHandler,
) -> None:
    event = cast(
        StandardStreamEvent,
        {
            "event": "on_tool_end",
            "name": "search_tool",
            "data": {
                "input": {"query": "test"},
                "output": ToolMessage(
                    content="search results here",
                    tool_call_id="tc1",
                    name="search_tool",
                ),
            },
        },
    )
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=False),
    )
    request_information = RequestInformation(request_id="req-1")
    tool_start_times: dict[str, float] = {}

    chunks = [
        chunk
        async for chunk in tool_event_handler.handle_tool_end(
            event=event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
            tool_start_times=tool_start_times,
        )
        if chunk
    ]

    assert isinstance(chunks, list)
    fake_wrapper = cast(_FakeChatRequestWrapper, chat_request_wrapper)
    assert (fake_wrapper.last_tool_end_output or "").strip() == "search results here"
    assert fake_wrapper.last_tool_end_is_error is False


@pytest.mark.asyncio
async def test_tool_end_surfaces_call_tool_artifact_error(
    tool_event_handler: ToolEventHandler,
) -> None:
    """CallToolTool reports failures via artifact={'is_error': True} rather than
    raising, since it catches its own exceptions — handle_tool_end must still
    flag this as an error on the emitted tool_end SSE event."""
    event = cast(
        StandardStreamEvent,
        {
            "event": "on_tool_end",
            "name": "call_tool",
            "data": {
                "input": {"name": "propose_skill", "arguments": {}},
                "output": ToolMessage(
                    content="Tool call failed:\nSkill validation failed: missing description",
                    tool_call_id="tc2",
                    name="call_tool",
                    artifact={"is_error": True},
                ),
            },
        },
    )
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=False),
    )
    request_information = RequestInformation(request_id="req-1")
    tool_start_times: dict[str, float] = {}

    async for _ in tool_event_handler.handle_tool_end(
        event=event,
        chat_request_wrapper=chat_request_wrapper,
        request_information=request_information,
        tool_start_times=tool_start_times,
    ):
        pass

    fake_wrapper = cast(_FakeChatRequestWrapper, chat_request_wrapper)
    assert fake_wrapper.last_tool_end_is_error is True
    assert "Skill validation failed" in (fake_wrapper.last_tool_end_output or "")


@pytest.mark.asyncio
async def test_tool_error_yields_error_message(
    tool_event_handler: ToolEventHandler,
) -> None:
    event = cast(
        StandardStreamEvent,
        {
            "event": "on_tool_error",
            "name": "failing_tool",
            "data": {
                "input": {"param": "value"},
                "error": "Something went wrong",
            },
        },
    )
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=False),
    )
    request_information = RequestInformation(request_id="req-1")
    tool_start_times: dict[str, float] = {}

    chunks = [
        chunk
        async for chunk in tool_event_handler.handle_tool_error(
            event=event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
            tool_start_times=tool_start_times,
        )
        if chunk
    ]

    assert len(chunks) >= 1
    assert "Something went wrong" in chunks[0]
