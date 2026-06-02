from languagemodelcommon.converters.stream_buffer import StreamBufferManager
from languagemodelcommon.converters.stream_debug_output_manager import (
    StreamDebugOutputManager,
)


class StreamContextMixin:
    """Provides per-request stream components via contextvar with test-override support."""

    _static_stream_buffer_manager: StreamBufferManager | None
    _static_stream_debug_output_manager: StreamDebugOutputManager | None

    @property
    def _stream_buffer_manager(self) -> StreamBufferManager:
        if self._static_stream_buffer_manager is not None:
            return self._static_stream_buffer_manager
        from languagemodelcommon.context.request_context import (
            get_stream_buffer_manager,
        )

        return get_stream_buffer_manager()

    @property
    def _stream_debug_output_manager(self) -> StreamDebugOutputManager:
        if self._static_stream_debug_output_manager is not None:
            return self._static_stream_debug_output_manager
        from languagemodelcommon.context.request_context import (
            get_stream_debug_output_manager,
        )

        return get_stream_debug_output_manager()
