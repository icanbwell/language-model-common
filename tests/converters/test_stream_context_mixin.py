import pytest

from languagemodelcommon.converters.stream_buffer import StreamBufferManager
from languagemodelcommon.converters.stream_context_mixin import StreamContextMixin
from languagemodelcommon.converters.stream_debug_output_manager import (
    StreamDebugOutputManager,
)


class _ConcreteWithMixin(StreamContextMixin):
    def __init__(
        self,
        *,
        stream_buffer_manager: StreamBufferManager | None = None,
        stream_debug_output_manager: StreamDebugOutputManager | None = None,
    ) -> None:
        self._static_stream_buffer_manager = stream_buffer_manager
        self._static_stream_debug_output_manager = stream_debug_output_manager


class TestStreamContextMixin:
    def test_returns_static_buffer_manager_when_set(self) -> None:
        buffer = StreamBufferManager(flush_interval_seconds=1.0, enabled=True)
        obj = _ConcreteWithMixin(stream_buffer_manager=buffer)
        assert obj._stream_buffer_manager is buffer

    def test_returns_static_debug_output_manager_when_set(self) -> None:
        debug = StreamDebugOutputManager()
        obj = _ConcreteWithMixin(stream_debug_output_manager=debug)
        assert obj._stream_debug_output_manager is debug

    def test_raises_when_no_static_and_no_context(self) -> None:
        obj = _ConcreteWithMixin()
        with pytest.raises(RuntimeError, match="StreamBufferManager not set"):
            _ = obj._stream_buffer_manager

    def test_debug_raises_when_no_static_and_no_context(self) -> None:
        obj = _ConcreteWithMixin()
        with pytest.raises(RuntimeError, match="StreamDebugOutputManager not set"):
            _ = obj._stream_debug_output_manager
