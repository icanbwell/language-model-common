"""Interceptor types for MCP tool call lifecycle management.

Replaces the types previously imported from langchain-mcp-adapters.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol, Self, runtime_checkable

from mcp.types import CallToolResult
from typing_extensions import NotRequired, TypedDict, Unpack

# Result type — matches what interceptors and handlers return.
MCPToolCallResult = CallToolResult


class _MCPToolCallRequestOverrides(TypedDict, total=False):
    name: NotRequired[str]
    args: NotRequired[dict[str, Any]]
    headers: NotRequired[dict[str, Any] | None]


@dataclass
class MCPToolCallRequest:
    """Tool execution request passed to MCP tool call interceptors.

    Modifiable fields (override to change behavior):
        name: Tool name to invoke.
        args: Tool arguments as key-value pairs.
        headers: HTTP headers for applicable transports.

    Context fields (read-only, for routing/logging):
        server_name: Name of the MCP server handling the tool.
    """

    name: str
    args: dict[str, Any]
    server_name: str
    headers: dict[str, Any] | None = None

    def override(self, **overrides: Unpack[_MCPToolCallRequestOverrides]) -> Self:
        return replace(self, **overrides)


@runtime_checkable
class ToolCallInterceptor(Protocol):
    """Protocol for tool call interceptors using handler callback pattern.

    Interceptors wrap tool execution in an onion pattern.
    """

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    ) -> MCPToolCallResult: ...
