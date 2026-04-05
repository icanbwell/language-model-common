"""Meta-discovery tool: call_tool.

Proxies tool invocations to the correct MCP server. The LLM calls this
after discovering tools via search_tools.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Type

from langchain_core.tools import BaseTool
from mcp.types import (
    CallToolResult,
    TextContent,
    ImageContent,
    EmbeddedResource,
    TextResourceContents,
)
from pydantic import BaseModel, Field

from languagemodelcommon.mcp.interceptors.auth import AuthMcpCallInterceptor
from languagemodelcommon.mcp.mcp_tool_provider import MCPToolProvider
from languagemodelcommon.mcp.tool_catalog import ToolCatalog
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


class CallToolInput(BaseModel):
    name: str = Field(
        ...,
        description="The exact name of the tool to call (from search_tools results).",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool as a JSON object.",
    )


def _call_tool_result_to_text(result: CallToolResult) -> str:
    """Convert a CallToolResult to a text representation for the LLM."""
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ImageContent):
            parts.append(f"[Image: {block.mimeType}]")
        elif isinstance(block, EmbeddedResource):
            if isinstance(block.resource, TextResourceContents):
                parts.append(block.resource.text)
            else:
                parts.append(f"[Resource: {block.resource.uri}]")
        else:
            parts.append(str(block))
    text = "\n".join(parts)

    if result.isError:
        return f"Tool call failed:\n{text}"
    return text


class CallToolTool(BaseTool):
    """Call a specific MCP tool by name with the given arguments."""

    name: str = "call_tool"
    description: str = (
        "Call a specific tool by name with the given arguments. "
        "Use search_tools first to find the tool name and its required parameters."
    )
    args_schema: Type[BaseModel] = CallToolInput
    response_format: Literal["content", "content_and_artifact"] = "content"

    catalog: ToolCatalog
    mcp_tool_provider: MCPToolProvider
    auth_interceptor: AuthMcpCallInterceptor

    class Config:
        arbitrary_types_allowed = True

    def _run(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        raise NotImplementedError("Use async version")

    async def _arun(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        if arguments is None:
            arguments = {}

        entry = self.catalog.get_tool(name)
        if entry is None:
            return f"Tool '{name}' not found. Use search_tools to find available tools."

        try:
            result: CallToolResult = await self.mcp_tool_provider.execute_mcp_tool(
                tool_name=name,
                arguments=arguments,
                agent_config=entry.agent_config,
                auth_interceptor=self.auth_interceptor,
            )
            return _call_tool_result_to_text(result)
        except Exception as e:
            logger.error(
                "call_tool failed for %s on %s: %s: %s",
                name,
                entry.server_name,
                type(e).__name__,
                e,
            )
            return f"Error calling tool '{name}': {type(e).__name__}: {e}"
