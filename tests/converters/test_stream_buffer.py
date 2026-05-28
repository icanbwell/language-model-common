from datetime import datetime, timezone

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

    assert await manager.buffer_content(request_id="req", content_text="Hello") is None

    flushed = await manager.buffer_content(request_id="req", content_text=" world\n")

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

    assert await manager.buffer_content(request_id="req", content_text="a") is None

    fake_clock.advance(0.051)

    flushed = await manager.buffer_content(request_id="req", content_text="b")

    assert flushed == "ab"


@pytest.mark.asyncio
async def test_buffer_disabled_returns_content_immediately() -> None:
    manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=False,
    )

    first = await manager.buffer_content(request_id="req", content_text="Hello")
    second = await manager.buffer_content(request_id="req", content_text=" world")

    assert first == "Hello"
    assert second == " world"


@pytest.mark.asyncio
async def test_force_flush_returns_buffered_content() -> None:
    manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=True,
    )

    await manager.buffer_content(request_id="req", content_text="buffered")
    flushed = await manager.buffer_content(
        request_id="req", content_text="", force_flush=True
    )

    assert flushed == "buffered"


@pytest.mark.asyncio
async def test_force_flush_empty_buffer_returns_none() -> None:
    manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=True,
    )

    result = await manager.buffer_content(
        request_id="req", content_text="", force_flush=True
    )

    assert result is None


class TestStreamedTextFragments:
    def test_append_and_pop(self) -> None:
        manager = StreamBufferManager(flush_interval_seconds=1.0, enabled=True)
        manager.append_streamed_text_fragment(request_id="req", text="hello ")
        manager.append_streamed_text_fragment(request_id="req", text="world")
        result = manager.pop_streamed_text("req")
        assert result == "hello world"

    def test_pop_empty_returns_none(self) -> None:
        manager = StreamBufferManager(flush_interval_seconds=1.0, enabled=True)
        assert manager.pop_streamed_text("req") is None

    def test_clear_removes_fragments(self) -> None:
        manager = StreamBufferManager(flush_interval_seconds=1.0, enabled=True)
        manager.append_streamed_text_fragment(request_id="req", text="text")
        manager.clear_request_streamed_text("req")
        assert manager.pop_streamed_text("req") is None

    def test_append_empty_string_is_noop(self) -> None:
        manager = StreamBufferManager(flush_interval_seconds=1.0, enabled=True)
        manager.append_streamed_text_fragment(request_id="req", text="")
        assert manager.pop_streamed_text("req") is None

    def test_start_streamed_output_sets_metadata(self) -> None:
        manager = StreamBufferManager(flush_interval_seconds=1.0, enabled=True)
        start = datetime(2026, 5, 28, 6, 30, 0, tzinfo=timezone.utc)
        manager.start_streamed_output(
            request_id="req",
            event_name="agent_node",
            start_time=start,
        )
        manager.append_streamed_text_fragment(request_id="req", text="response")
        output = manager.pop_streamed_output("req")
        assert output is not None
        assert output.event_name == "agent_node"
        assert output.start_time == start
        assert "".join(output.text_fragments) == "response"
