from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.runnables.schema import StandardStreamEvent
from langchain_core.tools import ToolException

from languagemodelcommon.converters.stream_buffer import StreamBufferManager
from languagemodelcommon.converters.stream_debug_output_manager import (
    StreamDebugOutputManager,
)
from languagemodelcommon.converters.tool_event_handlers import ToolEventHandler
from languagemodelcommon.file_managers.file_writer import (
    DebugFileWriteResult,
    FileWriter,
)
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
        self.last_tool_end_structured_output: dict[str, Any] | None = None
        self.image_output_events: list[dict[str, Any]] = []

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
        structured_output: dict[str, Any] | None = None,
    ) -> str | None:
        self.last_tool_end_output = output
        self.last_tool_end_is_error = is_error
        self.last_tool_end_structured_output = structured_output
        return None

    def create_mcp_app_sse_event(self, **kwargs: Any) -> str | None:
        return None

    def create_image_output_sse_event(
        self, *, request_id: str, image_part: dict[str, Any]
    ) -> str | None:
        self.image_output_events.append(image_part)
        return f"data: {image_part}\n\n"


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
async def test_tool_end_emits_image_output_event_for_tool_returned_image(
    tool_event_handler: ToolEventHandler,
) -> None:
    """A tool's own returned image (e.g. create_health_link's QR code) lives on
    ToolMessage.content, not `artifact` -- handle_tool_end must extract it and emit
    an output_image event, since BAI-806's streaming fix only covers the model's
    own AIMessage chunks, never a ToolMessage."""
    event = cast(
        StandardStreamEvent,
        {
            "event": "on_tool_end",
            "name": "create_health_link",
            "data": {
                "input": {"label": "For my doctor"},
                "output": ToolMessage(
                    content=[
                        {"type": "text", "text": "Health link created."},
                        {
                            "type": "image",
                            "base64": "aGVsbG8=",
                            "mime_type": "image/png",
                        },
                    ],
                    tool_call_id="tc3",
                    name="create_health_link",
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
    assert len(fake_wrapper.image_output_events) == 1
    image_part = fake_wrapper.image_output_events[0]
    assert image_part["type"] == "output_image"
    assert image_part["image_url"] == "data:image/png;base64,aGVsbG8="
    assert image_part["mime_type"] == "image/png"
    # The redacted text placeholder, not the raw image, is what rides the
    # existing tool_end text event -- the image only reaches the client via
    # the new atomic output_image event.
    assert "aGVsbG8=" not in (fake_wrapper.last_tool_end_output or "")


@pytest.mark.asyncio
async def test_tool_end_yields_no_image_event_for_text_only_output(
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

    async for _ in tool_event_handler.handle_tool_end(
        event=event,
        chat_request_wrapper=chat_request_wrapper,
        request_information=request_information,
        tool_start_times=tool_start_times,
    ):
        pass

    fake_wrapper = cast(_FakeChatRequestWrapper, chat_request_wrapper)
    assert fake_wrapper.image_output_events == []


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


@pytest.mark.asyncio
async def test_tool_error_with_exception_hides_raw_message_by_default(
    tool_event_handler: ToolEventHandler,
) -> None:
    """BAI-708 regression: this handler fires unconditionally on every
    on_tool_error callback -- regardless of whether the tool's own
    handle_tool_error setting will later convert the exception into normal
    ToolMessage content or let it propagate. It must never leak the raw
    exception text (which can include internal detail like a raw MCP
    JSON-RPC error payload) to a user who hasn't opted into debug logging."""
    event = cast(
        StandardStreamEvent,
        {
            "event": "on_tool_error",
            "name": "call_tool",
            "data": {
                "input": {"name": "get_clinical_notes", "arguments": {}},
                "error": ToolException(
                    "Invalid arguments - resource: Input should be "
                    "'Procedure' or 'Condition'"
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
        async for chunk in tool_event_handler.handle_tool_error(
            event=event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
            tool_start_times=tool_start_times,
        )
        if chunk
    ]

    assert len(chunks) >= 1
    assert "Invalid arguments" not in chunks[0]
    assert "I ran into an issue processing your request" in chunks[0]


@pytest.mark.asyncio
async def test_tool_error_with_exception_shows_raw_message_when_debug_enabled(
    tool_event_handler: ToolEventHandler,
) -> None:
    """Debug-enabled requests should still see the real error for
    troubleshooting -- only the default (non-debug) path hides it."""
    event = cast(
        StandardStreamEvent,
        {
            "event": "on_tool_error",
            "name": "call_tool",
            "data": {
                "input": {"name": "get_clinical_notes", "arguments": {}},
                "error": ToolException(
                    "Invalid arguments - resource: Input should be "
                    "'Procedure' or 'Condition'"
                ),
            },
        },
    )
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=True),
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
    assert "Invalid arguments" in chunks[0]


@pytest.mark.asyncio
async def test_tool_error_write_to_file_hides_raw_message_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BAI-708 regression: the write-to-file path is a second, separate
    surface that streams a download link to the user -- it must go through
    the same sanitization as the inline SSE message, not the raw exception.
    Previously this path used the raw `error_message` directly, so a debug
    download link could still leak internal detail (e.g. a raw MCP JSON-RPC
    error payload) even though the inline chat message was already sanitized."""
    monkeypatch.setenv("WRITE_TOOL_OUTPUT_TO_FILE", "true")
    environment_variables = LanguageModelCommonEnvironmentVariables()
    mock_file_writer = AsyncMock(spec=FileWriter)
    mock_file_writer.write_to_file_async = AsyncMock(
        return_value=DebugFileWriteResult(
            file_path="/mock-debug-storage/tool-error.txt",
            file_url="https://example.com/tool-error.txt",
            url_error_message=None,
        )
    )
    stream_buffer_manager = StreamBufferManager(
        flush_interval_seconds=10.0, enabled=False
    )
    stream_debug_output_manager = StreamDebugOutputManager()
    tool_event_handler = ToolEventHandler(
        debug_file_writer=mock_file_writer,
        environment_variables=environment_variables,
        tool_display_name_mapper=ToolDisplayNameMapper(),
        stream_buffer_manager=stream_buffer_manager,
        stream_debug_output_manager=stream_debug_output_manager,
    )

    raw_message = (
        "Invalid arguments - resource: Input should be 'Procedure' or 'Condition'"
    )
    event = cast(
        StandardStreamEvent,
        {
            "event": "on_tool_error",
            "name": "call_tool",
            "data": {
                "input": {"name": "get_clinical_notes", "arguments": {}},
                "error": ToolException(raw_message),
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

    # The download-link SSE chunk itself never contains the raw message.
    assert not any(raw_message in chunk for chunk in chunks)
    assert any("Click to download" in chunk for chunk in chunks)

    # Neither does the content actually written to the debug file...
    written_content = mock_file_writer.write_to_file_async.call_args.kwargs["content"]
    assert raw_message not in written_content
    assert "I ran into an issue processing your request" in written_content

    # ...nor the debug-output fragment recorded alongside it.
    debug_text = stream_debug_output_manager.pop_text()
    assert debug_text is not None
    assert raw_message not in debug_text


@pytest.mark.asyncio
async def test_tool_start_prefers_request_scoped_display_name_mapper(
    tool_event_handler: ToolEventHandler,
) -> None:
    """A per-request mapper (e.g. built via ToolDisplayNameMapper.with_tools()
    from that request's live MCP tools) must take precedence over the
    singleton mapper the handler was constructed with, which only ever
    carries the static config.
    """
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
    request_scoped_mapper = ToolDisplayNameMapper.from_mapping(
        name_to_display_name={"search_tool": "🔍 Request-Scoped Search"}
    )
    request_information = RequestInformation(
        request_id="req-1", tool_display_name_mapper=request_scoped_mapper
    )
    tool_start_times: dict[str, float] = {}

    chunks = [
        chunk
        async for chunk in tool_event_handler.handle_tool_start(
            event=event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
            tool_start_times=tool_start_times,
        )
        if chunk
    ]

    assert any("🔍 Request-Scoped Search" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_tool_start_falls_back_to_singleton_mapper_when_request_has_none(
    tool_event_handler: ToolEventHandler,
) -> None:
    """When a request carries no per-request mapper, the handler must still
    work using the singleton it was constructed with (backwards compatible).
    """
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

    chunks = [
        chunk
        async for chunk in tool_event_handler.handle_tool_start(
            event=event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
            tool_start_times=tool_start_times,
        )
        if chunk
    ]

    assert any("Search Tool" in chunk for chunk in chunks)
