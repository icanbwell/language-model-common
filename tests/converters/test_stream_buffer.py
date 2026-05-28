import pytest

from languagemodelcommon.converters.stream_buffer import StreamBufferManager


class _FakeClock:
    def __init__(self) -> None:
        self._current = 0.0

    def advance(self, delta: float) -> None:
        self._current += delta

    def monotonic(self) -> float:
        return self._current


@pytest.mark.asyncio
async def test_buffer_flushes_on_newline() -> None:
    manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=True,
    )

    assert await manager.buffer_content(content_text="Hello") is None

    flushed = await manager.buffer_content(content_text=" world\n")

    assert flushed == "Hello world\n"


@pytest.mark.asyncio
async def test_buffer_flushes_after_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_clock = _FakeClock()
    monkeypatch.setattr(
        "languagemodelcommon.converters.stream_buffer.time.monotonic",
        fake_clock.monotonic,
    )

    manager = StreamBufferManager(
        flush_interval_seconds=0.05,
        enabled=True,
    )

    assert await manager.buffer_content(content_text="a") is None

    fake_clock.advance(0.051)

    flushed = await manager.buffer_content(content_text="b")

    assert flushed == "ab"


@pytest.mark.asyncio
async def test_buffer_disabled_returns_content_immediately() -> None:
    manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=False,
    )

    first = await manager.buffer_content(content_text="Hello")
    second = await manager.buffer_content(content_text=" world")

    assert first == "Hello"
    assert second == " world"


@pytest.mark.asyncio
async def test_force_flush_returns_buffered_content() -> None:
    manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=True,
    )

    await manager.buffer_content(content_text="buffered")
    flushed = await manager.buffer_content(content_text="", force_flush=True)

    assert flushed == "buffered"


@pytest.mark.asyncio
async def test_force_flush_empty_buffer_returns_none() -> None:
    manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=True,
    )

    result = await manager.buffer_content(content_text="", force_flush=True)

    assert result is None
