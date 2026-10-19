"""create_mcp_session forwards an optional max_sse_event_size (PHR-3775)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from languagemodelcommon.mcp.mcp_client.session import (
    MCPConnectionConfig,
    create_mcp_session,
)

SESSION_MODULE = "languagemodelcommon.mcp.mcp_client.session"
SIXTEEN_MIB = 16 * 1024 * 1024


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra_config", "expected_kwargs"),
    [
        ({}, {}),
        ({"max_sse_event_size": SIXTEEN_MIB}, {"max_sse_event_size": SIXTEEN_MIB}),
        ({"max_sse_event_size": None}, {"max_sse_event_size": None}),
    ],
    ids=["absent-keeps-mcp-default", "int-forwarded", "none-forwarded-to-disable"],
)
async def test_create_mcp_session_forwards_max_sse_event_size_only_when_set(
    *,
    extra_config: dict[str, Any],
    expected_kwargs: dict[str, Any],
) -> None:
    # Arrange
    @asynccontextmanager
    async def _transport(*_: Any, **__: Any) -> AsyncIterator[tuple[Any, Any]]:
        yield MagicMock(), MagicMock()

    @asynccontextmanager
    async def _client_session(*_: Any, **__: Any) -> AsyncIterator[Any]:
        yield MagicMock()

    config: MCPConnectionConfig = {
        "url": "http://example.test/mcp",
        "transport": "streamable_http",
        **extra_config,  # type: ignore[typeddict-item]
    }

    # Act
    with (
        patch(f"{SESSION_MODULE}.streamable_http_client") as streamable,
        patch(f"{SESSION_MODULE}.ClientSession", new=_client_session),
    ):
        streamable.side_effect = _transport
        async with create_mcp_session(config):
            pass

    # Assert
    streamable.assert_called_once()
    assert streamable.call_args.args == ("http://example.test/mcp",)
    kwargs = dict(streamable.call_args.kwargs)
    kwargs.pop("http_client")
    assert kwargs == expected_kwargs
