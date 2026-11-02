import json
import logging
import time
from collections.abc import Callable
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.runnables.schema import CustomStreamEvent, StandardStreamEvent

from languagemodelcommon.converters.stream_buffer import StreamBufferManager
from languagemodelcommon.converters.stream_debug_output_manager import (
    StreamDebugOutputManager,
)
from languagemodelcommon.converters.streaming_manager import (
    LLM_CALLS_ABANDONED_EVENT,
    LangGraphStreamingManager,
)
from languagemodelcommon.converters.tool_event_handlers import ToolEventHandler
from languagemodelcommon.file_managers.file_writer import FileWriter
from languagemodelcommon.schema.openai.responses import ResponsesRequest
from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)
from languagemodelcommon.structures.openai.request.responses_api_request_wrapper import (
    ResponsesApiRequestWrapper,
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
        request_messages: list[dict[str, Any]] | None,
        invocation_id: str | None = None,
    ) -> str | None:
        return None

    def create_llm_call_end_sse_event(
        self,
        *,
        request_id: str,
        response_text: str | None,
        runtime_seconds: float | None = None,
        invocation_id: str | None = None,
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
async def test_custom_event_tool_result_compacted_yields_inline_chat_text(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    """BAI-904: baileyai's OldToolResultTruncator dispatches this custom event
    when it truncates an old tool result before resending it to the model.
    There's no dedicated SSE event type for it -- it must render as an inline
    chat-text bubble via create_sse_message, the same path tool-run messages
    use, so it needs zero frontend changes.
    """
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-1")
    chat_request_wrapper = _FakeChatRequestWrapper(enable_debug_logging=False)

    event = cast(
        CustomStreamEvent,
        {
            "event": "on_custom_event",
            "name": "tool_result_compacted",
            "data": {"tool_name": "search_records", "estimated_tokens": 4096},
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

    assert len(chunks) == 1
    assert "Compacting conversation history" in chunks[0]


@pytest.mark.asyncio
async def test_custom_event_tool_result_compacted_notice_shown_once_per_request(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    """BAI-920: OldToolResultTruncator has no cross-call memory and re-dispatches
    tool_result_compacted on every subsequent model call for as long as a message
    stays outside its keep-recent window, so a single request can see this event
    multiple times. The UI notice must surface at most once per request -- later
    dispatches within the same request (same RequestInformation instance) should
    be swallowed.
    """
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-1")
    chat_request_wrapper = _FakeChatRequestWrapper(enable_debug_logging=False)

    def _make_event(tool_name: str) -> CustomStreamEvent:
        return cast(
            CustomStreamEvent,
            {
                "event": "on_custom_event",
                "name": "tool_result_compacted",
                "data": {"tool_name": tool_name, "estimated_tokens": 4096},
            },
        )

    first_chunks = [
        chunk
        async for chunk in manager.handle_langchain_event(
            event=_make_event("search_records"),
            chat_request_wrapper=cast(ChatRequestWrapper, chat_request_wrapper),
            request_information=request_information,
            tool_start_times={},
        )
    ]
    second_chunks = [
        chunk
        async for chunk in manager.handle_langchain_event(
            event=_make_event("list_skills"),
            chat_request_wrapper=cast(ChatRequestWrapper, chat_request_wrapper),
            request_information=request_information,
            tool_start_times={},
        )
    ]

    assert len(first_chunks) == 1
    assert "Compacting conversation history" in first_chunks[0]
    assert second_chunks == []
    assert request_information.compaction_notice_shown is True


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
        self.llm_call_start_calls: list[list[dict[str, Any]] | None] = []
        self.llm_call_end_calls: list[str | None] = []
        self.llm_call_end_runtimes: list[float | None] = []

    def create_llm_call_start_sse_event(
        self,
        *,
        request_id: str,
        request_messages: list[dict[str, Any]] | None,
        invocation_id: str | None = None,
    ) -> str | None:
        self.llm_call_start_calls.append(request_messages)
        return f"llm_call_start:{len(self.llm_call_start_calls)}"

    def create_llm_call_end_sse_event(
        self,
        *,
        request_id: str,
        response_text: str | None,
        runtime_seconds: float | None = None,
        invocation_id: str | None = None,
    ) -> str | None:
        self.llm_call_end_calls.append(response_text)
        self.llm_call_end_runtimes.append(runtime_seconds)
        return f"llm_call_end:{len(self.llm_call_end_calls)}"


@pytest.mark.asyncio
async def test_chat_model_start_yields_llm_call_start_with_request_messages(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-1")
    chat_request_wrapper = _RecordingLlmCallChatRequestWrapper(
        enable_debug_logging=True
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
async def test_chat_model_end_yields_llm_call_end_when_debug_logging_enabled(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    """The llm_call end event fires when debug logging is enabled, same as
    the pre-existing debug-only messages-log chunk emitted later in the same
    handler."""
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-1")
    chat_request_wrapper = _RecordingLlmCallChatRequestWrapper(
        enable_debug_logging=True
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

    assert chunks == [
        "llm_call_end:1",
        "\n\n<details>\n<summary>Messages log</summary>\n\n"
        "```\n--- Streamed assistant output ---\nHi there!\n\n```\n\n"
        "</details>\n\n",
    ]
    assert chat_request_wrapper.llm_call_end_calls == ["Hi there!"]


async def _run_llm_call(
    *,
    manager: LangGraphStreamingManager,
    chat_request_wrapper: _RecordingLlmCallChatRequestWrapper,
    request_information: RequestInformation,
    prompt_text: str,
    response_text: str,
) -> tuple[list[str | None], list[str | None]]:
    """Drives one model invocation (start, stream, end) and returns the chunks
    yielded by the start and end handlers."""
    wrapper = cast(ChatRequestWrapper, chat_request_wrapper)
    start_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_start",
            "run_id": "run-1",
            "data": {"input": {"messages": [[HumanMessage(content=prompt_text)]]}},
        },
    )
    start_chunks = [
        chunk
        async for chunk in manager._handle_on_chat_model_start(
            event=start_event,
            chat_request_wrapper=wrapper,
            request_information=request_information,
        )
    ]
    stream_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content=response_text)},
        },
    )
    async for _ in manager._handle_on_chat_model_stream(
        event=stream_event,
        chat_request_wrapper=wrapper,
        request_information=request_information,
    ):
        pass
    end_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_end",
            "run_id": "run-1",
            "data": {"input": {"messages": [[HumanMessage(content=prompt_text)]]}},
        },
    )
    end_chunks = [
        chunk
        async for chunk in manager._handle_on_chat_model_end(
            event=end_event,
            chat_request_wrapper=wrapper,
            request_information=request_information,
        )
    ]
    return start_chunks, end_chunks


@pytest.mark.asyncio
async def test_llm_call_items_emitted_without_content_when_debug_logging_disabled(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BAI-1440: the llm_call start/done items are always emitted so clients
    can show model-call timing, but prompt and response text (potential PHI)
    must not reach the wrapper when debug logging is off."""
    clock = iter([100.0, 102.5])
    monkeypatch.setattr(time, "perf_counter", lambda: next(clock))
    manager = streaming_manager_factory()
    chat_request_wrapper = _RecordingLlmCallChatRequestWrapper(
        enable_debug_logging=False
    )

    start_chunks, end_chunks = await _run_llm_call(
        manager=manager,
        chat_request_wrapper=chat_request_wrapper,
        request_information=RequestInformation(request_id="req-1"),
        prompt_text="synthetic prompt text",
        response_text="synthetic response text",
    )

    assert start_chunks == ["llm_call_start:1"]
    assert end_chunks == ["llm_call_end:1"]
    assert chat_request_wrapper.llm_call_start_calls == [None]
    assert chat_request_wrapper.llm_call_end_calls == [None]
    assert chat_request_wrapper.llm_call_end_runtimes == [2.5]


@pytest.mark.asyncio
async def test_llm_call_end_includes_runtime_when_debug_logging_enabled(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([100.0, 102.5])
    monkeypatch.setattr(time, "perf_counter", lambda: next(clock))
    manager = streaming_manager_factory()
    chat_request_wrapper = _RecordingLlmCallChatRequestWrapper(
        enable_debug_logging=True
    )

    await _run_llm_call(
        manager=manager,
        chat_request_wrapper=chat_request_wrapper,
        request_information=RequestInformation(request_id="req-1"),
        prompt_text="synthetic prompt text",
        response_text="synthetic response text",
    )

    assert chat_request_wrapper.llm_call_end_runtimes == [2.5]
    assert chat_request_wrapper.llm_call_end_calls == ["synthetic response text"]


class _PrepareRecordingChatRequestWrapper(_FakeChatRequestWrapper):
    """Records the order of create_prepare/llm_call_start events so tests can
    assert the prepare item is emitted once, ahead of the first llm_call."""

    def __init__(self, *, enable_debug_logging: bool = False) -> None:
        super().__init__(enable_debug_logging=enable_debug_logging)
        self.calls: list[str] = []
        self.prepare_runtimes: list[float] = []

    def create_prepare_sse_event(
        self,
        *,
        request_id: str,
        runtime_seconds: float,
    ) -> str | None:
        self.calls.append("prepare")
        self.prepare_runtimes.append(runtime_seconds)
        return "prepare"

    def create_llm_call_start_sse_event(
        self,
        *,
        request_id: str,
        request_messages: list[dict[str, Any]] | None,
        invocation_id: str | None = None,
    ) -> str | None:
        self.calls.append("llm_call_start")
        return "llm_call_start"


async def _model_start_chunks(
    *,
    manager: LangGraphStreamingManager,
    wrapper: _PrepareRecordingChatRequestWrapper,
    request_information: RequestInformation,
) -> list[str | None]:
    event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent, {"event": "on_chat_model_start", "data": {}}
    )
    return [
        chunk
        async for chunk in manager._handle_on_chat_model_start(
            event=event,
            chat_request_wrapper=cast(ChatRequestWrapper, wrapper),
            request_information=request_information,
        )
    ]


@pytest.mark.asyncio
async def test_prepare_emitted_once_before_first_llm_call_start(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory()
    wrapper = _PrepareRecordingChatRequestWrapper()
    request_information = RequestInformation(
        request_id="req-1", request_started_at=time.perf_counter() - 2.0
    )

    first = await _model_start_chunks(
        manager=manager, wrapper=wrapper, request_information=request_information
    )
    second = await _model_start_chunks(
        manager=manager, wrapper=wrapper, request_information=request_information
    )
    third = await _model_start_chunks(
        manager=manager, wrapper=wrapper, request_information=request_information
    )

    assert first == ["prepare", "llm_call_start"]
    assert second == ["llm_call_start"]
    assert third == ["llm_call_start"]
    assert wrapper.calls == [
        "prepare",
        "llm_call_start",
        "llm_call_start",
        "llm_call_start",
    ]
    (runtime_seconds,) = wrapper.prepare_runtimes
    assert runtime_seconds >= 2.0


@pytest.mark.asyncio
async def test_prepare_not_emitted_when_request_started_at_is_none(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory()
    wrapper = _PrepareRecordingChatRequestWrapper()

    chunks = await _model_start_chunks(
        manager=manager,
        wrapper=wrapper,
        request_information=RequestInformation(request_id="req-1"),
    )

    assert chunks == ["llm_call_start"]
    assert wrapper.prepare_runtimes == []


@pytest.mark.asyncio
async def test_prepare_runtime_clamped_to_zero_for_future_start(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory()
    wrapper = _PrepareRecordingChatRequestWrapper()

    await _model_start_chunks(
        manager=manager,
        wrapper=wrapper,
        request_information=RequestInformation(
            request_id="req-1", request_started_at=time.perf_counter() + 1000.0
        ),
    )

    assert wrapper.prepare_runtimes == [0.0]


@pytest.mark.asyncio
async def test_prepare_state_is_per_request(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    """The once-per-turn flag must not leak across requests sharing a manager."""
    manager = streaming_manager_factory()
    wrapper = _PrepareRecordingChatRequestWrapper()

    for request_id in ("req-1", "req-2"):
        await _model_start_chunks(
            manager=manager,
            wrapper=wrapper,
            request_information=RequestInformation(
                request_id=request_id, request_started_at=time.perf_counter()
            ),
        )

    assert wrapper.calls == ["prepare", "llm_call_start", "prepare", "llm_call_start"]


class _LlmCallIdRecordingChatRequestWrapper(_FakeChatRequestWrapper):
    def __init__(self) -> None:
        super().__init__(enable_debug_logging=False)
        self.start_ids: list[str | None] = []
        self.end_calls: list[tuple[str | None, float | None]] = []

    def create_llm_call_start_sse_event(
        self,
        *,
        request_id: str,
        request_messages: list[dict[str, Any]] | None,
        invocation_id: str | None = None,
    ) -> str | None:
        self.start_ids.append(invocation_id)
        return "start"

    def create_llm_call_end_sse_event(
        self,
        *,
        request_id: str,
        response_text: str | None,
        runtime_seconds: float | None = None,
        invocation_id: str | None = None,
    ) -> str | None:
        self.end_calls.append((invocation_id, runtime_seconds))
        return "end"


@pytest.mark.asyncio
async def test_overlapping_llm_calls_are_timed_and_paired_by_run_id(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parallel model invocations must not share one timing slot: each end
    event pairs with its own start via the run id and gets its own runtime."""
    clock = iter([0.0, 10.0, 15.0, 40.0])
    monkeypatch.setattr(time, "perf_counter", lambda: next(clock))
    manager = streaming_manager_factory()
    wrapper = _LlmCallIdRecordingChatRequestWrapper()
    request_information = RequestInformation(request_id="req-1")
    cast_wrapper = cast(ChatRequestWrapper, wrapper)

    async def _run(*, event_name: str, run_id: str) -> None:
        event: StandardStreamEvent | CustomStreamEvent = cast(
            StandardStreamEvent,
            {"event": event_name, "run_id": run_id, "data": {}},
        )
        handler = (
            manager._handle_on_chat_model_start
            if event_name == "on_chat_model_start"
            else manager._handle_on_chat_model_end
        )
        async for _ in handler(
            event=event,
            chat_request_wrapper=cast_wrapper,
            request_information=request_information,
        ):
            pass

    await _run(event_name="on_chat_model_start", run_id="run-a")  # t=0
    await _run(event_name="on_chat_model_start", run_id="run-b")  # t=10
    await _run(event_name="on_chat_model_end", run_id="run-b")  # t=15
    await _run(event_name="on_chat_model_end", run_id="run-a")  # t=40

    assert wrapper.start_ids == ["run-a", "run-b"]
    assert wrapper.end_calls == [("run-b", 5.0), ("run-a", 40.0)]


@pytest.mark.asyncio
async def test_llm_call_end_without_matching_start_has_no_runtime(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory()
    wrapper = _LlmCallIdRecordingChatRequestWrapper()
    event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {"event": "on_chat_model_end", "run_id": "never-started", "data": {}},
    )

    async for _ in manager._handle_on_chat_model_end(
        event=event,
        chat_request_wrapper=cast(ChatRequestWrapper, wrapper),
        request_information=RequestInformation(request_id="req-1"),
    ):
        pass

    assert wrapper.end_calls == [("never-started", None)]


@pytest.mark.asyncio
async def test_llm_call_without_run_id_skips_timing_and_warns(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = streaming_manager_factory()
    wrapper = _LlmCallIdRecordingChatRequestWrapper()
    cast_wrapper = cast(ChatRequestWrapper, wrapper)
    request_information = RequestInformation(request_id="req-1")
    start_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent, {"event": "on_chat_model_start", "data": {}}
    )
    end_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent, {"event": "on_chat_model_end", "data": {}}
    )

    with caplog.at_level(logging.WARNING):
        async for _ in manager._handle_on_chat_model_start(
            event=start_event,
            chat_request_wrapper=cast_wrapper,
            request_information=request_information,
        ):
            pass
    async for _ in manager._handle_on_chat_model_end(
        event=end_event,
        chat_request_wrapper=cast_wrapper,
        request_information=request_information,
    ):
        pass

    assert wrapper.start_ids == [None]
    assert wrapper.end_calls == [(None, None)]
    assert "no run_id" in caplog.text


@pytest.mark.asyncio
async def test_manager_with_real_responses_wrapper_pairs_items_and_times_them(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wires the streaming manager to the real ResponsesApiRequestWrapper and
    parses the SSE frames: prepare once ahead of the first llm_call, start and
    done items sharing an id, runtime on the done item, and no prompt or
    response text with debug logging off."""
    clock = iter([5.0, 7.0, 9.5])
    monkeypatch.setattr(time, "perf_counter", lambda: next(clock))
    manager = streaming_manager_factory()
    environment_variables = MagicMock()
    environment_variables.debug_prefixes = ("DEBUG:", "/debug ")
    wrapper = ResponsesApiRequestWrapper(
        chat_request=ResponsesRequest(model="gpt-4", input="hello", stream=True),
        enable_debug_logging=False,
        environment_variables=environment_variables,
    )
    request_information = RequestInformation(request_id="req-1", request_started_at=1.0)
    cast_wrapper = cast(ChatRequestWrapper, wrapper)

    frames: list[str | None] = []
    start_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_start",
            "run_id": "run-a",
            "data": {"input": {"messages": [[HumanMessage(content="secret prompt")]]}},
        },
    )
    async for chunk in manager._handle_on_chat_model_start(
        event=start_event,
        chat_request_wrapper=cast_wrapper,
        request_information=request_information,
    ):
        frames.append(chunk)
    stream_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content="secret response")},
        },
    )
    async for _ in manager._handle_on_chat_model_stream(
        event=stream_event,
        chat_request_wrapper=cast_wrapper,
        request_information=request_information,
    ):
        pass
    end_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {"event": "on_chat_model_end", "run_id": "run-a", "data": {}},
    )
    async for chunk in manager._handle_on_chat_model_end(
        event=end_event,
        chat_request_wrapper=cast_wrapper,
        request_information=request_information,
    ):
        frames.append(chunk)

    events = [json.loads(f[len("data: ") :]) for f in frames if f]
    prepare, llm_start, llm_done = (e["item"] for e in events)
    assert prepare == {
        "type": "prepare",
        "id": "prepare_req-1",
        "status": "completed",
        "runtime_seconds": 4.0,
    }
    assert llm_start["type"] == llm_done["type"] == "llm_call"
    assert llm_start["id"] == llm_done["id"]
    assert llm_done["runtime_seconds"] == 2.5
    assert "request" not in llm_start
    assert "output" not in llm_done
    assert "secret" not in "".join(f for f in frames if f)


class _StatusRecordingChatRequestWrapper(_FakeChatRequestWrapper):
    def __init__(self) -> None:
        super().__init__(enable_debug_logging=False)
        self.end_calls: list[tuple[str | None, float | None, str]] = []

    def create_llm_call_end_sse_event(
        self,
        *,
        request_id: str,
        response_text: str | None,
        runtime_seconds: float | None = None,
        invocation_id: str | None = None,
        status: str = "completed",
    ) -> str | None:
        self.end_calls.append((invocation_id, runtime_seconds, status))
        return f"end:{invocation_id}:{status}"


@pytest.mark.asyncio
async def test_abandoned_event_closes_only_open_llm_calls_as_failed(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed or retried attempt must not leave llm_call items in_progress:
    calls that already ended are untouched, open ones are closed as failed."""
    clock = iter([0.0, 1.0, 2.0, 12.0])
    monkeypatch.setattr(time, "perf_counter", lambda: next(clock))
    manager = streaming_manager_factory()
    wrapper = _StatusRecordingChatRequestWrapper()
    cast_wrapper = cast(ChatRequestWrapper, wrapper)
    request_information = RequestInformation(request_id="req-1")

    async def _send(*, event_name: str, run_id: str) -> list[str]:
        event: StandardStreamEvent | CustomStreamEvent = cast(
            StandardStreamEvent,
            {"event": event_name, "run_id": run_id, "data": {}},
        )
        return [
            chunk
            async for chunk in manager.handle_langchain_event(
                event=event,
                chat_request_wrapper=cast_wrapper,
                request_information=request_information,
                tool_start_times={},
                user_id=None,
            )
        ]

    await _send(event_name="on_chat_model_start", run_id="run-a")  # t=0
    await _send(event_name="on_chat_model_start", run_id="run-b")  # t=1
    ended = await _send(event_name="on_chat_model_end", run_id="run-a")  # t=2
    abandoned = await _send(event_name=LLM_CALLS_ABANDONED_EVENT, run_id="")  # t=12
    again = await _send(event_name=LLM_CALLS_ABANDONED_EVENT, run_id="")

    assert ended == ["end:run-a:completed"]
    assert abandoned == ["end:run-b:failed"]
    assert again == []
    assert wrapper.end_calls[-1] == ("run-b", 11.0, "failed")
