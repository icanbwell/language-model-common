from collections.abc import Callable

import pytest

from languagemodelcommon.converters.streaming_manager import LangGraphStreamingManager
from languagemodelcommon.utilities.token_reducer.token_reducer import TokenReducer
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)


class _FakeClock:
    def __init__(self) -> None:
        self._current = 0.0

    def advance(self, delta: float) -> None:
        self._current += delta

    def monotonic(self) -> float:
        return self._current


@pytest.fixture()
def streaming_manager_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[float], LangGraphStreamingManager]:
    def _factory(interval_seconds: float) -> LangGraphStreamingManager:
        monkeypatch.setenv("BUFFER_FLUSH_INTERVAL_SECONDS", str(interval_seconds))
        environment_variables = LanguageModelCommonEnvironmentVariables()
        return LangGraphStreamingManager(
            token_reducer=TokenReducer(),
            environment_variables=environment_variables,
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
