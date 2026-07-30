"""Tests for CallToolTool — meta-tool for calling specific MCP tools by name."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import ToolException
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl

from languagemodelcommon.tools.mcp.call_tool_tool import (
    CallToolInput,
    CallToolTool,
    _call_tool_result_to_text,
)
from languagemodelcommon.mcp.tool_catalog import ToolCatalog
from languagemodelcommon.configs.schemas.config_schema import AgentConfig
from mcp.types import Tool as MCPTool


def _agent_config() -> AgentConfig:
    return AgentConfig(name="server1", url="https://example.com/mcp")


def _make_call_tool_tool(
    catalog: ToolCatalog,
    mcp_tool_provider: Any = None,
    auth_interceptor: Any = None,
) -> CallToolTool:
    """Create a CallToolTool bypassing Pydantic validation for mock dependencies."""
    return CallToolTool.model_construct(
        name="call_tool",
        description="Call a specific tool by name with the given arguments.",
        args_schema=CallToolInput,
        response_format="content",
        catalog=catalog,
        mcp_tool_provider=mcp_tool_provider or MagicMock(),
        auth_interceptor=auth_interceptor or MagicMock(),
    )


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
                        uri=AnyUrl("file://test.txt"),
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
        tool = _make_call_tool_tool(catalog=catalog)
        with pytest.raises(ToolException, match="not found"):
            await tool._arun(name="nonexistent", arguments={})

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
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = _make_call_tool_tool(catalog=catalog, mcp_tool_provider=mock_provider)
        result = await tool._arun(name="my_tool", arguments={"key": "value"})
        text, artifact = result
        assert text == "tool output"
        assert artifact is None
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

        tool = _make_call_tool_tool(catalog=catalog, mcp_tool_provider=mock_provider)
        with pytest.raises(ToolException) as exc_info:
            await tool._arun(name="failing_tool")
        assert "RuntimeError" in str(exc_info.value)
        assert "connection lost" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_inner_tool_error_result_raises_tool_exception(self) -> None:
        """The wrapped MCP tool itself can return isError=True without raising —
        CallToolTool must turn that into a ToolException so the resulting
        ToolMessage carries status="error" instead of looking like a success."""
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="rejecting_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="validation failed")],
                isError=True,
            )
        )
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = _make_call_tool_tool(catalog=catalog, mcp_tool_provider=mock_provider)
        with pytest.raises(ToolException) as exc_info:
            await tool._arun(name="rejecting_tool", arguments={})
        assert str(exc_info.value).startswith("Tool call failed:")
        assert "validation failed" in str(exc_info.value)
        # fetch_mcp_app_embed is a no-op on the error path
        mock_provider.fetch_mcp_app_embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_inner_tool_error_result_sets_tool_message_status_error(
        self,
    ) -> None:
        """End-to-end: going through the real BaseTool.arun (not just _arun
        directly) must produce a ToolMessage with status="error", since that's
        the actual signal the calling model/harness reacts to."""
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="rejecting_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="validation failed")],
                isError=True,
            )
        )
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = CallToolTool.model_construct(
            name="call_tool",
            description="Call a specific tool by name with the given arguments.",
            args_schema=CallToolInput,
            response_format="content_and_artifact",
            handle_tool_error=True,
            catalog=catalog,
            mcp_tool_provider=mock_provider,
            auth_interceptor=MagicMock(),
        )
        message = await tool.arun(
            tool_input={"name": "rejecting_tool", "arguments": {}},
            tool_call_id="call-1",
        )
        assert message.status == "error"
        assert "validation failed" in message.content

    def test_sync_run_raises(self) -> None:
        catalog = ToolCatalog()
        tool = _make_call_tool_tool(catalog=catalog)
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
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = _make_call_tool_tool(catalog=catalog, mcp_tool_provider=mock_provider)
        await tool._arun(name="my_tool", arguments=None)
        call_kwargs = mock_provider.execute_mcp_tool.call_args.kwargs
        assert call_kwargs["arguments"] == {}
