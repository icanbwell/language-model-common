"""Callback types for MCP client notifications.

Replaces the callback types previously imported from langchain-mcp-adapters.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from mcp.client.session import LoggingFnT as MCPLoggingFnT
from mcp.shared.session import ProgressFnT as MCPProgressFnT
from mcp.types import LoggingMessageNotificationParams


@dataclass
class CallbackContext:
    """Context passed to callback functions."""

    server_name: str | None = None
    tool_name: str | None = None


@runtime_checkable
class LoggingMessageCallback(Protocol):
    async def __call__(
        self,
        params: LoggingMessageNotificationParams,
        context: CallbackContext,
    ) -> None: ...


@runtime_checkable
class ProgressCallback(Protocol):
    async def __call__(
        self,
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ) -> None: ...


@dataclass
class _MCPCallbacks:
    """Callbacks in MCP SDK format. For internal use."""

    logging_callback: MCPLoggingFnT | None = None
    progress_callback: MCPProgressFnT | None = None


@dataclass
class Callbacks:
    """Callbacks for MCP client notifications."""

    on_logging_message: LoggingMessageCallback | None = None
    on_progress: ProgressCallback | None = None

    def to_mcp_format(self, *, context: CallbackContext) -> _MCPCallbacks:
        """Convert to MCP SDK callback format, injecting context."""
        mcp_logging_cb: MCPLoggingFnT | None = None
        mcp_progress_cb: MCPProgressFnT | None = None

        if (on_logging_message := self.on_logging_message) is not None:

            async def _logging_cb(
                params: LoggingMessageNotificationParams,
            ) -> None:
                await on_logging_message(params, context)

            mcp_logging_cb = _logging_cb

        if (on_progress := self.on_progress) is not None:

            async def _progress_cb(
                progress: float, total: float | None, message: str | None
            ) -> None:
                await on_progress(progress, total, message, context)

            mcp_progress_cb = _progress_cb

        return _MCPCallbacks(
            logging_callback=mcp_logging_cb,
            progress_callback=mcp_progress_cb,
        )
