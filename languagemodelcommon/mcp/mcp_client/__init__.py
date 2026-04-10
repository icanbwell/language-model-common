"""Custom MCP client replacing langchain-mcp-adapters.

This package provides session management, tool listing, tool invocation,
and LangChain BaseTool conversion using the MCP SDK directly.

All public names are re-exported here for backward compatibility.
"""

from languagemodelcommon.mcp.mcp_client.content_conversion import (
    ToolMessageContentBlock,
    convert_call_tool_result,
    convert_mcp_content_to_lc_block,
)
from languagemodelcommon.mcp.mcp_client.langchain_adapter import (
    mcp_tool_to_langchain_tool,
)
from languagemodelcommon.mcp.mcp_client.session import (
    MCPConnectionConfig,
    McpHttpClientFactory,
    McpSessionError,
    create_mcp_session,
)
from languagemodelcommon.mcp.mcp_client.session_pool import McpSessionPool
from languagemodelcommon.mcp.mcp_client.tool_invocation import (
    build_interceptor_chain,
    call_mcp_tool_raw,
)
from languagemodelcommon.mcp.mcp_client.tool_list_cache import (
    ToolListCache,
    list_all_tools,
    list_all_tools_cached,
)

__all__ = [
    "MCPConnectionConfig",
    "McpHttpClientFactory",
    "McpSessionError",
    "McpSessionPool",
    "ToolListCache",
    "ToolMessageContentBlock",
    "build_interceptor_chain",
    "call_mcp_tool_raw",
    "convert_call_tool_result",
    "convert_mcp_content_to_lc_block",
    "create_mcp_session",
    "list_all_tools",
    "list_all_tools_cached",
    "mcp_tool_to_langchain_tool",
]
