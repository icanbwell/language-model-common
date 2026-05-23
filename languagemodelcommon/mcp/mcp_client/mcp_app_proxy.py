"""MCP App proxy service for tools/call and resources/read from iframes.

Provides the server-side proxy that MCP App iframes use to call tools
and read resources on the MCP server.  The injected bridge JavaScript
sends these requests via fetch() to the host, and this service handles
the actual MCP communication.
"""

import logging
from dataclasses import dataclass
from typing import Any

from mcp.types import CallToolResult
from pydantic import AnyUrl

from languagemodelcommon.mcp.mcp_client.session import (
    MCPConnectionConfig,
    create_mcp_session,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


@dataclass
class McpProxyToolCallRequest:
    """Request to call a tool via the MCP proxy."""

    tool_name: str
    arguments: dict[str, Any]
    server_url: str


@dataclass
class McpProxyResourceReadRequest:
    """Request to read a resource via the MCP proxy."""

    uri: str
    server_url: str


async def proxy_tool_call(request: McpProxyToolCallRequest) -> dict[str, Any]:
    """Execute a tool call on the MCP server and return the result.

    This is used by the MCP App proxy endpoint to forward tools/call
    requests from app iframes back to the originating MCP server.
    """
    config = MCPConnectionConfig(url=request.server_url)
    async with create_mcp_session(config) as session:
        await session.initialize()
        result: CallToolResult = await session.call_tool(
            request.tool_name, request.arguments
        )
        return result.model_dump(mode="json")


async def proxy_resource_read(request: McpProxyResourceReadRequest) -> dict[str, Any]:
    """Read a resource on the MCP server and return the result.

    This is used by the MCP App proxy endpoint to forward resources/read
    requests from app iframes back to the originating MCP server.
    """
    config = MCPConnectionConfig(url=request.server_url)
    async with create_mcp_session(config) as session:
        await session.initialize()
        result = await session.read_resource(AnyUrl(request.uri))
        return result.model_dump(mode="json")
