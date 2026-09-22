"""Tests for BAI-889: retrying transient MCP session-establishment failures.

`open_initialized_mcp_session` wraps `create_mcp_session` + `session.initialize()`
with bounded, backed-off retry. Scope is strictly session establishment --
before any tool call reaches the server -- so retrying is safe even for
non-idempotent tools; a failure during `call_tool` itself must never be
retried by this function.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

from languagemodelcommon.mcp.mcp_client.session import (
    MCPConnectionConfig,
    open_initialized_mcp_session,
)


def _make_session() -> AsyncMock:
    session = AsyncMock()
    session.initialize = AsyncMock()
    return session


@asynccontextmanager
async def _session_cm(session: AsyncMock) -> AsyncIterator[AsyncMock]:
    yield session


class TestOpenInitializedMcpSession:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt_without_sleeping(self) -> None:
        session = _make_session()

        with (
            patch(
                "languagemodelcommon.mcp.mcp_client.session.create_mcp_session",
                side_effect=lambda *args, **kwargs: _session_cm(session),
            ),
            patch(
                "languagemodelcommon.mcp.mcp_client.session.asyncio.sleep"
            ) as mock_sleep,
        ):
            cm, returned_session = await open_initialized_mcp_session(
                MCPConnectionConfig(url="https://example.test/mcp")
            )
            await cm.__aexit__(None, None, None)

        assert returned_session is session
        session.initialize.assert_awaited_once()
        mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retries_transient_initialize_failure_then_succeeds(self) -> None:
        good_session = _make_session()
        attempts: list[AsyncMock] = []

        def _next_cm(*args: Any, **kwargs: Any) -> Any:
            if len(attempts) == 0:
                failing_session = AsyncMock()
                failing_session.initialize = AsyncMock(
                    side_effect=ConnectionError("connection refused")
                )
                attempts.append(failing_session)
                return _session_cm(failing_session)
            attempts.append(good_session)
            return _session_cm(good_session)

        with (
            patch(
                "languagemodelcommon.mcp.mcp_client.session.create_mcp_session",
                side_effect=_next_cm,
            ),
            patch(
                "languagemodelcommon.mcp.mcp_client.session.asyncio.sleep"
            ) as mock_sleep,
        ):
            cm, returned_session = await open_initialized_mcp_session(
                MCPConnectionConfig(url="https://example.test/mcp"),
                max_attempts=3,
            )
            await cm.__aexit__(None, None, None)

        assert returned_session is good_session
        assert len(attempts) == 2
        mock_sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_all_attempts(self) -> None:
        call_count = 0

        def _always_failing_cm(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1

            @asynccontextmanager
            async def _cm() -> AsyncIterator[AsyncMock]:
                raise ConnectionError(f"connection refused #{call_count}")
                yield  # pragma: no cover

            return _cm()

        with (
            patch(
                "languagemodelcommon.mcp.mcp_client.session.create_mcp_session",
                side_effect=_always_failing_cm,
            ),
            patch("languagemodelcommon.mcp.mcp_client.session.asyncio.sleep"),
        ):
            with pytest.raises(ConnectionError, match="connection refused #3"):
                await open_initialized_mcp_session(
                    MCPConnectionConfig(url="https://example.test/mcp"),
                    max_attempts=3,
                )

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_cancellation_during_initialize_closes_session_and_propagates(
        self,
    ) -> None:
        """A CancelledError during `initialize()` must still close the
        underlying session (matching the base branch's `async with`
        cleanup) and must propagate immediately, without retrying."""
        session = AsyncMock()
        session.initialize = AsyncMock(side_effect=asyncio.CancelledError())
        aexit_calls: list[Any] = []

        @asynccontextmanager
        async def _cm() -> AsyncIterator[AsyncMock]:
            try:
                yield session
            except BaseException as exc:
                aexit_calls.append(exc)
                raise

        call_count = 0

        def _next_cm(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return _cm()

        with (
            patch(
                "languagemodelcommon.mcp.mcp_client.session.create_mcp_session",
                side_effect=_next_cm,
            ),
            patch(
                "languagemodelcommon.mcp.mcp_client.session.asyncio.sleep"
            ) as mock_sleep,
        ):
            with pytest.raises(asyncio.CancelledError):
                await open_initialized_mcp_session(
                    MCPConnectionConfig(url="https://example.test/mcp"),
                    max_attempts=3,
                )

        assert call_count == 1
        assert len(aexit_calls) == 1
        assert isinstance(aexit_calls[0], asyncio.CancelledError)
        mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_does_not_retry_call_tool_failures(self) -> None:
        """Only session establishment retries. A failure raised by the
        caller from `call_tool` (after this function has already returned)
        must never cause `create_mcp_session` to be invoked again."""
        session = _make_session()
        session.call_tool = AsyncMock(side_effect=RuntimeError("tool blew up"))
        call_count = 0

        def _cm(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return _session_cm(session)

        with patch(
            "languagemodelcommon.mcp.mcp_client.session.create_mcp_session",
            side_effect=_cm,
        ):
            cm, returned_session = await open_initialized_mcp_session(
                MCPConnectionConfig(url="https://example.test/mcp")
            )
            try:
                with pytest.raises(RuntimeError, match="tool blew up"):
                    await returned_session.call_tool("some_tool", {})
            finally:
                await cm.__aexit__(None, None, None)

        assert call_count == 1
