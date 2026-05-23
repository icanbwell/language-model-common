"""Tests for MCP App proxy service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from languagemodelcommon.mcp.mcp_client.mcp_app_proxy import (
    McpProxyToolCallRequest,
    McpProxyResourceReadRequest,
    proxy_tool_call,
    proxy_resource_read,
)


@pytest.mark.asyncio
class TestProxyToolCall:
    @patch("languagemodelcommon.mcp.mcp_client.mcp_app_proxy.create_mcp_session")
    async def test_calls_tool_and_returns_result(
        self, mock_create_session: MagicMock
    ) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "content": [{"type": "text", "text": "hello"}],
            "isError": False,
        }

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_create_session.return_value = mock_ctx

        request = McpProxyToolCallRequest(
            tool_name="get_data",
            arguments={"id": "123"},
            server_url="http://mcp-server:8080/sse",
        )
        result = await proxy_tool_call(request)

        assert result == {
            "content": [{"type": "text", "text": "hello"}],
            "isError": False,
        }
        mock_session.call_tool.assert_called_once_with("get_data", {"id": "123"})


@pytest.mark.asyncio
class TestProxyResourceRead:
    @patch("languagemodelcommon.mcp.mcp_client.mcp_app_proxy.create_mcp_session")
    async def test_reads_resource_and_returns_result(
        self, mock_create_session: MagicMock
    ) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "contents": [{"text": "<html>Resource</html>", "uri": "ui://server/app"}]
        }

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.read_resource = AsyncMock(return_value=mock_result)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_create_session.return_value = mock_ctx

        request = McpProxyResourceReadRequest(
            uri="ui://server/app",
            server_url="http://mcp-server:8080/sse",
        )
        result = await proxy_resource_read(request)

        assert result == {
            "contents": [{"text": "<html>Resource</html>", "uri": "ui://server/app"}]
        }
        mock_session.read_resource.assert_called_once()
