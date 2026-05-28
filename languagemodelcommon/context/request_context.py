from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

from oidcauthlib.auth.models.auth import AuthInformation

if TYPE_CHECKING:
    from languagemodelcommon.converters.stream_buffer import StreamBufferManager
    from languagemodelcommon.converters.stream_debug_output_manager import (
        StreamDebugOutputManager,
    )


@dataclass(frozen=True)
class McpRequestContext:
    headers: Dict[str, str]
    auth_information: AuthInformation
    user_id: str | None = None


_request_context_var: ContextVar["McpRequestContext | None"] = ContextVar(
    "mcp_request_context", default=None
)
_stream_debug_output_var: ContextVar["StreamDebugOutputManager | None"] = ContextVar(
    "stream_debug_output", default=None
)
_stream_buffer_var: ContextVar["StreamBufferManager | None"] = ContextVar(
    "stream_buffer", default=None
)


def get_request_context() -> "McpRequestContext":
    ctx = _request_context_var.get()
    if ctx is None:
        raise RuntimeError(
            "McpRequestContext not set — call init_request_context() first"
        )
    return ctx


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
    headers: Dict[str, str],
    auth_information: "AuthInformation",
    user_id: str | None = None,
    stream_debug_output_manager: "StreamDebugOutputManager",
    stream_buffer_manager: "StreamBufferManager",
) -> None:
    _request_context_var.set(
        McpRequestContext(
            headers=headers,
            auth_information=auth_information,
            user_id=user_id,
        )
    )
    _stream_debug_output_var.set(stream_debug_output_manager)
    _stream_buffer_var.set(stream_buffer_manager)


def reset_request_context() -> None:
    _request_context_var.set(None)
    _stream_debug_output_var.set(None)
    _stream_buffer_var.set(None)
