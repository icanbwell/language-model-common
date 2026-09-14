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


@pytest.mark.asyncio
async def test_invocation_boundary_noop_before_any_content_streamed() -> None:
    """The first chat-model invocation of a request has nothing to glue onto."""
    manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=True,
    )

    manager.mark_new_invocation_boundary()
    flushed = await manager.buffer_content(content_text="Hello", force_flush=True)

    assert flushed == "Hello"


@pytest.mark.parametrize(
    "prior_text,resumed_text,expected",
    [
        pytest.param(
            "have access to.",
            "Let me search",
            "have access to. Let me search",
            id="glued_sentence_gets_separator",
        ),
        pytest.param(
            "categories of tools:",
            "Based on",
            "categories of tools: Based on",
            id="glued_after_colon_gets_separator",
        ),
        pytest.param(
            "have access to.\n\n",
            "Let me search",
            "have access to.\n\nLet me search",
            id="trailing_whitespace_needs_no_separator",
        ),
        pytest.param(
            "have access to.",
            " Let me search",
            "have access to. Let me search",
            id="leading_whitespace_needs_no_separator",
        ),
    ],
)
@pytest.mark.asyncio
async def test_invocation_boundary_inserts_separator_only_when_missing(
    prior_text: str,
    resumed_text: str,
    expected: str,
) -> None:
    """Resuming after a tool call must not glue words together (BAI-726)."""
    manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=True,
    )

    # A newline inside prior_text triggers an early flush inside the first
    # call, so the two calls' return values -- not just the second -- must
    # be concatenated to see the full reconstructed text.
    first_flushed = await manager.buffer_content(content_text=prior_text)
    manager.mark_new_invocation_boundary()
    second_flushed = await manager.buffer_content(
        content_text=resumed_text, force_flush=True
    )

    assert (first_flushed or "") + (second_flushed or "") == expected


@pytest.mark.asyncio
async def test_invocation_boundary_ignored_when_disabled() -> None:
    """Boundary handling must also apply when buffering itself is disabled."""
    manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=False,
    )

    await manager.buffer_content(content_text="have access to.")
    manager.mark_new_invocation_boundary()
    resumed = await manager.buffer_content(content_text="Let me search")

    assert resumed == " Let me search"
