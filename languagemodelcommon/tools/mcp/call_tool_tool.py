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
from pydantic import BaseModel, ConfigDict, Field

from oidcauthlib.auth.exceptions.authorization_needed_exception import (
    AuthorizationNeededException,
)

from languagemodelcommon.mcp.interceptors.auth import AuthMcpCallInterceptor
from languagemodelcommon.mcp.mcp_client.session_pool import McpSessionPool
from languagemodelcommon.mcp.mcp_client.ui_resource import McpAppEmbed
from languagemodelcommon.mcp.mcp_tool_provider import MCPToolProvider
from languagemodelcommon.mcp.tool_catalog import ToolCatalog
from languagemodelcommon.utilities.logger.exception_logger import ExceptionLogger
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
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    catalog: ToolCatalog
    mcp_tool_provider: MCPToolProvider
    auth_interceptor: AuthMcpCallInterceptor
    session_pool: McpSessionPool | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        raise NotImplementedError("Use async version")

    async def _arun(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any] | None]:
        if arguments is None:
            arguments = {}

        entry = self.catalog.get_tool(name)
        if entry is None:
            return (
                f"Tool '{name}' not found. Use search_tools to find available tools.",
                None,
            )

        try:
            result: CallToolResult = await self.mcp_tool_provider.execute_mcp_tool(
                tool_name=name,
                arguments=arguments,
                agent_config=entry.agent_config,
                auth_interceptor=self.auth_interceptor,
                session_pool=self.session_pool,
            )
            text = _call_tool_result_to_text(result)

            # Best-effort: fetch MCP app UI resource if the tool declares one
            app_embed = await self.mcp_tool_provider.fetch_mcp_app_embed(
                tool=entry.tool,
                tool_name=name,
                tool_args=arguments,
                tool_result_text=text,
                agent_config=entry.agent_config,
                session_pool=self.session_pool,
            )

            artifact: dict[str, Any] | None = None
            if app_embed is not None:
                artifact = {"mcp_app_embed": app_embed}

            return text, artifact
        except AuthorizationNeededException:
            # Auth exceptions must propagate so the user sees login links
            raise
        except Exception as e:
            error_detail = ExceptionLogger.format_exception_message(e)
            logger.error(
                "call_tool failed for %s on %s: %s",
                name,
                entry.server_name,
                error_detail,
            )
            return f"Error calling tool '{name}': {error_detail}", None
