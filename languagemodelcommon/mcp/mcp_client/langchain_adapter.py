"""LangChain BaseTool adapter for MCP tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from mcp.types import Tool as MCPTool

from languagemodelcommon.mcp.callbacks import Callbacks, CallbackContext, _MCPCallbacks
from languagemodelcommon.mcp.interceptors.types import (
    MCPToolCallRequest,
    ToolCallInterceptor,
)
from languagemodelcommon.mcp.mcp_client.content_conversion import (
    ToolMessageContentBlock,
    convert_call_tool_result,
)
from languagemodelcommon.mcp.mcp_client.session import MCPConnectionConfig
from languagemodelcommon.mcp.mcp_client.session_pool import McpSessionPool
from languagemodelcommon.mcp.mcp_client.tool_invocation import (
    _make_execute_tool,
    build_interceptor_chain,
)


def mcp_tool_to_langchain_tool(
    tool: MCPTool,
    *,
    connection: MCPConnectionConfig,
    callbacks: Callbacks | None = None,
    tool_interceptors: list[ToolCallInterceptor] | None = None,
    server_name: str | None = None,
    session_pool: McpSessionPool | None = None,
) -> BaseTool:
    """Convert an MCP Tool to a LangChain BaseTool.

    Creates a StructuredTool that establishes a new session per invocation
    and applies the interceptor chain.

    When ``session_pool`` is provided, sessions are reused across calls
    to the same MCP server URL within the pool's scope.
    """

    mcp_callbacks = (
        callbacks.to_mcp_format(
            context=CallbackContext(server_name=server_name, tool_name=tool.name)
        )
        if callbacks is not None
        else _MCPCallbacks()
    )
    execute_tool = _make_execute_tool(
        connection, mcp_callbacks, session_pool=session_pool
    )
    handler = build_interceptor_chain(execute_tool, tool_interceptors)

    async def call_tool(
        **arguments: dict[str, Any],
    ) -> tuple[list[ToolMessageContentBlock], None]:
        request = MCPToolCallRequest(
            name=tool.name,
            args=arguments,
            server_name=server_name or "unknown",
            headers=None,
        )
        call_tool_result = await handler(request)
        content = convert_call_tool_result(call_tool_result)
        return content, None

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.inputSchema,
        coroutine=call_tool,
        response_format="content_and_artifact",
    )
