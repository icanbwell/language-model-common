import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult, TextContent

from languagemodelcommon.mcp.mcp_client.tool_invocation import (
    _execute_tool_call_with_heartbeat,
)


@pytest.mark.asyncio
async def test_fast_call_emits_no_heartbeat() -> None:
    session = AsyncMock()
    fast_result = CallToolResult(content=[TextContent(type="text", text="ok")])
    session.call_tool = AsyncMock(return_value=fast_result)

    with patch(
        "languagemodelcommon.mcp.mcp_client.tool_invocation.adispatch_custom_event",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        result = await _execute_tool_call_with_heartbeat(
            session=session,
            name="propose_skill",
            arguments={},
            progress_callback=None,
            server_name="skills-publisher",
            heartbeat_interval_seconds=10.0,
        )

    assert result is fast_result
    mock_dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_slow_call_emits_periodic_heartbeats() -> None:
    session = AsyncMock()
    slow_result = CallToolResult(content=[TextContent(type="text", text="ok")])

    async def _slow_call_tool(
        name: str,
        arguments: dict[str, Any],
        progress_callback: Any = None,
    ) -> CallToolResult:
        await asyncio.sleep(0.25)
        return slow_result

    session.call_tool = _slow_call_tool

    with patch(
        "languagemodelcommon.mcp.mcp_client.tool_invocation.adispatch_custom_event",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        result = await _execute_tool_call_with_heartbeat(
            session=session,
            name="slow_tool",
            arguments={},
            progress_callback=None,
            server_name="slow-server",
            heartbeat_interval_seconds=0.1,
        )

    assert result is slow_result
    assert mock_dispatch.await_count >= 2
    first_call = mock_dispatch.call_args_list[0]
    assert first_call.args[0] == "mcp_tool_heartbeat"
    assert first_call.args[1]["tool_name"] == "slow_tool"
    assert first_call.args[1]["server_name"] == "slow-server"


@pytest.mark.asyncio
async def test_outer_cancellation_cancels_inner_call_task() -> None:
    session = AsyncMock()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _hanging_call_tool(
        name: str,
        arguments: dict[str, Any],
        progress_callback: Any = None,
    ) -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    session.call_tool = _hanging_call_tool

    task = asyncio.ensure_future(
        _execute_tool_call_with_heartbeat(
            session=session,
            name="hanging_tool",
            arguments={},
            progress_callback=None,
            server_name="hanging-server",
            heartbeat_interval_seconds=5.0,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(cancelled.wait(), timeout=1.0)


class _CleanupError(RuntimeError):
    """Distinct exception raised by a fake tool's cancellation cleanup path."""


@pytest.mark.asyncio
async def test_inner_cleanup_exception_after_cancellation_is_not_swallowed() -> None:
    """If the inner call_task raises a different exception while handling its
    own CancelledError (e.g. a failure during session teardown), that
    exception must propagate rather than being silently dropped when the
    outer coroutine is cancelled and cancels the inner task in turn.
    """
    session = AsyncMock()
    started = asyncio.Event()

    async def _call_tool_with_failing_cleanup(
        name: str,
        arguments: dict[str, Any],
        progress_callback: Any = None,
    ) -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise _CleanupError("cleanup failed") from None

    session.call_tool = _call_tool_with_failing_cleanup

    task = asyncio.ensure_future(
        _execute_tool_call_with_heartbeat(
            session=session,
            name="hanging_tool",
            arguments={},
            progress_callback=None,
            server_name="hanging-server",
            heartbeat_interval_seconds=5.0,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(_CleanupError, match="cleanup failed"):
        await task


@pytest.mark.asyncio
async def test_non_runtime_error_from_dispatch_still_cancels_inner_call_task() -> None:
    """A heartbeat-dispatch exception other than RuntimeError must still
    cancel and await the in-flight call_task rather than leaving it
    orphaned, mirroring the cleanup already done for CancelledError.
    """
    session = AsyncMock()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _hanging_call_tool(
        name: str,
        arguments: dict[str, Any],
        progress_callback: Any = None,
    ) -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    session.call_tool = _hanging_call_tool

    with patch(
        "languagemodelcommon.mcp.mcp_client.tool_invocation.adispatch_custom_event",
        new_callable=AsyncMock,
        side_effect=ValueError("unexpected dispatch failure"),
    ):
        with pytest.raises(ValueError, match="unexpected dispatch failure"):
            await _execute_tool_call_with_heartbeat(
                session=session,
                name="hanging_tool",
                arguments={},
                progress_callback=None,
                server_name="hanging-server",
                heartbeat_interval_seconds=0.05,
            )

    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
