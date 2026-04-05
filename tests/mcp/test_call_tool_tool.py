"""Tests for CallToolTool — meta-tool for calling specific MCP tools by name."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)

from languagemodelcommon.mcp.call_tool_tool import (
    CallToolTool,
    _call_tool_result_to_text,
)
from languagemodelcommon.mcp.tool_catalog import ToolCatalog
from languagemodelcommon.configs.schemas.config_schema import AgentConfig
from mcp.types import Tool as MCPTool


def _agent_config() -> AgentConfig:
    return AgentConfig(name="server1", url="https://example.com/mcp")


class TestCallToolResultToText:
    def test_text_content(self) -> None:
        result = CallToolResult(content=[TextContent(type="text", text="Hello world")])
        assert _call_tool_result_to_text(result) == "Hello world"

    def test_multiple_text_blocks(self) -> None:
        result = CallToolResult(
            content=[
                TextContent(type="text", text="Line 1"),
                TextContent(type="text", text="Line 2"),
            ]
        )
        assert _call_tool_result_to_text(result) == "Line 1\nLine 2"

    def test_image_content(self) -> None:
        result = CallToolResult(
            content=[ImageContent(type="image", data="base64", mimeType="image/png")]
        )
        assert _call_tool_result_to_text(result) == "[Image: image/png]"

    def test_embedded_text_resource(self) -> None:
        result = CallToolResult(
            content=[
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="file://test.txt",
                        text="resource text",
                    ),
                )
            ]
        )
        assert _call_tool_result_to_text(result) == "resource text"

    def test_error_result(self) -> None:
        result = CallToolResult(
            content=[TextContent(type="text", text="Something failed")],
            isError=True,
        )
        text = _call_tool_result_to_text(result)
        assert text.startswith("Tool call failed:")
        assert "Something failed" in text


class TestCallToolTool:
    @pytest.mark.asyncio
    async def test_tool_not_found(self) -> None:
        catalog = ToolCatalog()
        tool = CallToolTool(
            catalog=catalog,
            mcp_tool_provider=MagicMock(),
            auth_interceptor=MagicMock(),
        )
        result = await tool._arun(name="nonexistent", arguments={})
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_successful_call(self) -> None:
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="my_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="tool output")]
            )
        )

        tool = CallToolTool(
            catalog=catalog,
            mcp_tool_provider=mock_provider,
            auth_interceptor=MagicMock(),
        )
        result = await tool._arun(name="my_tool", arguments={"key": "value"})
        assert result == "tool output"
        mock_provider.execute_mcp_tool.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_call_failure_returns_error_string(self) -> None:
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="failing_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )

        tool = CallToolTool(
            catalog=catalog,
            mcp_tool_provider=mock_provider,
            auth_interceptor=MagicMock(),
        )
        result = await tool._arun(name="failing_tool")
        assert "RuntimeError" in result
        assert "connection lost" in result

    def test_sync_run_raises(self) -> None:
        catalog = ToolCatalog()
        tool = CallToolTool(
            catalog=catalog,
            mcp_tool_provider=MagicMock(),
            auth_interceptor=MagicMock(),
        )
        with pytest.raises(NotImplementedError):
            tool._run(name="test")

    @pytest.mark.asyncio
    async def test_default_arguments(self) -> None:
        """Arguments default to empty dict when None."""
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="my_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(content=[TextContent(type="text", text="ok")])
        )

        tool = CallToolTool(
            catalog=catalog,
            mcp_tool_provider=mock_provider,
            auth_interceptor=MagicMock(),
        )
        await tool._arun(name="my_tool", arguments=None)
        call_kwargs = mock_provider.execute_mcp_tool.call_args.kwargs
        assert call_kwargs["arguments"] == {}
