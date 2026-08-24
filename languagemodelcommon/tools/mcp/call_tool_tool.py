"""Meta-discovery tool: call_tool.

Proxies tool invocations to the correct MCP server. The LLM calls this
after discovering tools via search_tools.
"""

import logging
from typing import Any, Literal, Type

from langchain_core.tools import BaseTool, ToolException
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
from languagemodelcommon.mcp.mcp_tool_provider import MCPToolProvider
from languagemodelcommon.mcp.tool_catalog import ToolCatalog, ToolResolverProtocol
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
    # Required so a raised ToolException becomes a ToolMessage with status="error"
    # instead of re-raising (BaseTool's default). Without this, a failed MCP tool
    # call is indistinguishable from a successful one to the calling model.
    handle_tool_error: bool = True

    catalog: ToolCatalog
    mcp_tool_provider: MCPToolProvider
    auth_interceptor: AuthMcpCallInterceptor
    resolver: ToolResolverProtocol | None = None
    session_pool: McpSessionPool | None = None
    proxy_base_url: str | None = None
    session_token: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        raise NotImplementedError("Use async version")

    async def _arun(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any] | None]:
        if arguments is None:
            arguments = {}

        entry = self.catalog.get_tool(name)

        # The catalog is built fresh per request, so a server may still be
        # registered-but-unresolved here even though a prior search_tools
        # call (in an earlier turn, on a different catalog instance) already
        # surfaced this tool name to the model. Resolve on demand rather than
        # failing — mirrors the lazy resolution SearchToolsTool performs.
        if entry is None and self.resolver is not None:
            auth_exception: AuthorizationNeededException | None = None
            for server in self.catalog.get_unresolved_servers():
                try:
                    await self.catalog.resolve_server(
                        server_name=server.server_name, resolver=self.resolver
                    )
                except AuthorizationNeededException as e:
                    # An unrelated server needing auth must not block
                    # resolution of the rest -- the target tool may live on
                    # a different, no-auth server. Only surfaced below if
                    # the tool is still missing after every server has been
                    # attempted.
                    if auth_exception is None:
                        auth_exception = e
                except Exception as e:
                    logger.warning(
                        "Failed to resolve server %s while looking up tool '%s': %s",
                        server.server_name,
                        name,
                        ExceptionLogger.format_exception_message(e),
                    )
            entry = self.catalog.get_tool(name)
            if entry is None and auth_exception is not None:
                raise auth_exception

        if entry is None:
            raise ToolException(
                f"Tool '{name}' not found. Use search_tools to find available tools."
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

            # Surface the *inner* tool's own error status (result.isError), not just
            # transport-level failures — MCP servers report rejected/invalid calls
            # this way rather than raising, so this is the only place that signal
            # exists. Raising here (with handle_tool_error=True) is what makes the
            # resulting ToolMessage carry status="error" instead of looking like an
            # ordinary successful call to the model.
            if result.isError:
                raise ToolException(text)

            app_embed = await self.mcp_tool_provider.fetch_mcp_app_embed(
                tool=entry.tool,
                tool_name=name,
                tool_args=arguments,
                tool_result_text=text,
                agent_config=entry.agent_config,
                session_pool=self.session_pool,
                proxy_base_url=self.proxy_base_url,
                session_token=self.session_token,
            )

            artifact: dict[str, Any] | None = None
            if app_embed is not None:
                artifact = {"mcp_app_embed": app_embed}

            return text, artifact
        except AuthorizationNeededException:
            # Auth exceptions must propagate so the user sees login links
            raise
        except ToolException:
            raise
        except Exception as e:
            error_detail = ExceptionLogger.format_exception_message(e)
            logger.error(
                "call_tool failed for %s on %s: %s",
                name,
                entry.server_name,
                error_detail,
            )
            raise ToolException(f"Error calling tool '{name}': {error_detail}") from e
