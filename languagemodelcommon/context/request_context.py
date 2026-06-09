from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from languagemodelcommon.converters.stream_buffer import StreamBufferManager
    from languagemodelcommon.converters.stream_debug_output_manager import (
        StreamDebugOutputManager,
    )


_stream_debug_output_var: ContextVar["StreamDebugOutputManager | None"] = ContextVar(
    "stream_debug_output", default=None
)
_stream_buffer_var: ContextVar["StreamBufferManager | None"] = ContextVar(
    "stream_buffer", default=None
)


def get_stream_debug_output_manager() -> "StreamDebugOutputManager":
    mgr = _stream_debug_output_var.get()
    if mgr is None:
        raise RuntimeError(
            "StreamDebugOutputManager not set — call init_request_context() first"
        )
    return mgr


def get_stream_buffer_manager() -> "StreamBufferManager":
    mgr = _stream_buffer_var.get()
    if mgr is None:
        raise RuntimeError(
            "StreamBufferManager not set — call init_request_context() first"
        )
    return mgr


def init_request_context(
    *,
    stream_debug_output_manager: "StreamDebugOutputManager",
    stream_buffer_manager: "StreamBufferManager",
) -> None:
    _stream_debug_output_var.set(stream_debug_output_manager)
    _stream_buffer_var.set(stream_buffer_manager)


def reset_request_context() -> None:
    _stream_debug_output_var.set(None)
    _stream_buffer_var.set(None)
