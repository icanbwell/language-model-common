from collections.abc import Callable
from typing import Generator, Any, Optional, cast

import boto3
import pytest
from boto3 import Session
from botocore.client import BaseClient
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables.schema import CustomStreamEvent, StandardStreamEvent
from moto import mock_aws
from types_boto3_s3.client import S3Client

from languagemodelcommon.aws.aws_client_factory import AwsClientFactory
from languagemodelcommon.file_managers.file_writer import FileWriter
from languagemodelcommon.converters.streaming_manager import LangGraphStreamingManager
from languagemodelcommon.file_managers.aws_s3_file_manager import AwsS3FileManager
from languagemodelcommon.file_managers.file_manager_factory import FileManagerFactory
from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)
from languagemodelcommon.utilities.token_reducer.token_reducer import TokenReducer
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.request_information import RequestInformation
from languagemodelcommon.mocks.mock_aws_client_factory import MockAwsClientFactory
from languagemodelcommon.utilities.tool_friendly_name_mapper import (
    ToolFriendlyNameMapper,
)


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


class _FakeClock:
    def __init__(self) -> None:
        self._current = 0.0

    def advance(self, delta: float) -> None:
        self._current += delta

    def monotonic(self) -> float:
        return self._current


@pytest.fixture
def mock_s3() -> Generator[S3Client, Any, None]:
    """Create a mock S3 client using moto."""
    with mock_aws():
        session: Session = boto3.Session()
        s3_client: S3Client = session.client(
            service_name="s3",
            region_name="us-east-1",
        )
        yield s3_client


@pytest.fixture
def aws_client_factory(mock_s3: BaseClient) -> AwsClientFactory:
    """Create a mock AWS client factory."""
    return MockAwsClientFactory(aws_client=mock_s3)


@pytest.fixture
def aws_s3_file_manager(aws_client_factory: AwsClientFactory) -> AwsS3FileManager:
    """Create an instance of AwsS3FileManager for testing."""
    return AwsS3FileManager(aws_client_factory=aws_client_factory)


@pytest.fixture()
def streaming_manager_factory(
    monkeypatch: pytest.MonkeyPatch,
    aws_client_factory: AwsClientFactory,
) -> Callable[[float], LangGraphStreamingManager]:
    def _factory(interval_seconds: float) -> LangGraphStreamingManager:
        monkeypatch.setenv("BUFFER_FLUSH_INTERVAL_SECONDS", str(interval_seconds))
        environment_variables = LanguageModelCommonEnvironmentVariables()
        file_manager_factory = FileManagerFactory(
            aws_client_factory=aws_client_factory,
        )
        return LangGraphStreamingManager(
            token_reducer=TokenReducer(),
            environment_variables=environment_variables,
            debug_file_writer=FileWriter(
                file_manager_factory=file_manager_factory,
            ),
            tool_friendly_name_mapper=ToolFriendlyNameMapper(),
        )

    return _factory


@pytest.mark.asyncio
async def test_buffer_flushes_on_newline(
    streaming_manager_factory: Callable[[float], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory(10.0)

    assert (
        await manager._buffer_stream_content(
            request_id="req",
            content_text="Hello",
        )
        is None
    )

    flushed = await manager._buffer_stream_content(
        request_id="req",
        content_text=" world\n",
    )

    assert flushed == "Hello world\n"


@pytest.mark.asyncio
async def test_buffer_flushes_after_interval(
    monkeypatch: pytest.MonkeyPatch,
    streaming_manager_factory: Callable[[float], LangGraphStreamingManager],
) -> None:
    fake_clock = _FakeClock()
    monkeypatch.setattr(
        "languagemodelcommon.converters.streaming_manager.time.monotonic",
        fake_clock.monotonic,
    )

    manager = streaming_manager_factory(0.05)

    assert (
        await manager._buffer_stream_content(
            request_id="req",
            content_text="a",
        )
        is None
    )

    fake_clock.advance(0.051)

    flushed = await manager._buffer_stream_content(
        request_id="req",
        content_text="b",
    )

    assert flushed == "ab"


@pytest.mark.asyncio
async def test_buffer_is_disabled_when_buffering_env_flag_is_false(
    monkeypatch: pytest.MonkeyPatch,
    streaming_manager_factory: Callable[[float], LangGraphStreamingManager],
) -> None:
    monkeypatch.setenv("ENABLE_STREAMING_BUFFERING", "false")
    manager = streaming_manager_factory(10.0)

    first_chunk = await manager._buffer_stream_content(
        request_id="req",
        content_text="Hello",
    )
    second_chunk = await manager._buffer_stream_content(
        request_id="req",
        content_text=" world",
    )

    assert first_chunk == "Hello"
    assert second_chunk == " world"
    assert "req" not in manager._stream_buffers


@pytest.mark.asyncio
async def test_chat_model_end_includes_streamed_text_when_debug_enabled(
    streaming_manager_factory: Callable[[float], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory(10.0)
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
    assert "req-1" not in manager._streamed_text_fragments


@pytest.mark.asyncio
async def test_chain_end_clears_streamed_text_when_chat_model_end_not_called(
    streaming_manager_factory: Callable[[float], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory(10.0)
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
    assert "req-2" in manager._streamed_text_fragments

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

    assert "req-2" not in manager._streamed_text_fragments
