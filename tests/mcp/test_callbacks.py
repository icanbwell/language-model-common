"""Tests for MCP callback types and conversion."""

import pytest
from mcp.types import LoggingMessageNotificationParams

from languagemodelcommon.mcp.callbacks import (
    Callbacks,
    CallbackContext,
)


class TestCallbackContext:
    def test_defaults(self) -> None:
        ctx = CallbackContext()
        assert ctx.server_name is None
        assert ctx.tool_name is None

    def test_with_values(self) -> None:
        ctx = CallbackContext(server_name="s1", tool_name="t1")
        assert ctx.server_name == "s1"
        assert ctx.tool_name == "t1"


class TestCallbacks:
    def test_to_mcp_format_no_callbacks(self) -> None:
        cb = Callbacks()
        mcp = cb.to_mcp_format(context=CallbackContext())
        assert mcp.logging_callback is None
        assert mcp.progress_callback is None

    @pytest.mark.asyncio
    async def test_logging_callback_conversion(self) -> None:
        received_contexts: list[CallbackContext] = []

        async def on_log(
            params: LoggingMessageNotificationParams,
            context: CallbackContext,
        ) -> None:
            received_contexts.append(context)

        cb = Callbacks(on_logging_message=on_log)
        ctx = CallbackContext(server_name="test_server")
        mcp = cb.to_mcp_format(context=ctx)
        assert mcp.logging_callback is not None

        await mcp.logging_callback(
            LoggingMessageNotificationParams(level="info", data="test")
        )
        assert len(received_contexts) == 1
        assert received_contexts[0].server_name == "test_server"

    @pytest.mark.asyncio
    async def test_progress_callback_conversion(self) -> None:
        received: list[tuple[float, float | None, str | None]] = []

        async def on_progress(
            progress: float,
            total: float | None,
            message: str | None,
            context: CallbackContext,
        ) -> None:
            received.append((progress, total, message))

        cb = Callbacks(on_progress=on_progress)
        mcp = cb.to_mcp_format(context=CallbackContext())
        assert mcp.progress_callback is not None

        await mcp.progress_callback(0.5, 1.0, "halfway")
        assert received == [(0.5, 1.0, "halfway")]
