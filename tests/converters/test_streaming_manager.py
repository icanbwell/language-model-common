from collections.abc import Callable
from typing import Generator, Any

import boto3
import pytest
from boto3 import Session
from botocore.client import BaseClient
from moto import mock_aws
from types_boto3_s3.client import S3Client

from languagemodelcommon.aws.aws_client_factory import AwsClientFactory
from languagemodelcommon.converters.streaming_manager import LangGraphStreamingManager
from languagemodelcommon.file_managers.aws_s3_file_manager import AwsS3FileManager
from languagemodelcommon.file_managers.file_manager_factory import FileManagerFactory
from languagemodelcommon.utilities.token_reducer.token_reducer import TokenReducer
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.mocks.mock_aws_client_factory import MockAwsClientFactory


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
        return LangGraphStreamingManager(
            token_reducer=TokenReducer(),
            environment_variables=environment_variables,
            file_manager_factory=FileManagerFactory(
                aws_client_factory=aws_client_factory,
            ),
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
