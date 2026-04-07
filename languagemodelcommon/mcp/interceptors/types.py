"""Interceptor types for MCP tool call and resource read lifecycle management.

Replaces the types previously imported from langchain-mcp-adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mcp.types import CallToolResult, ReadResourceResult
from typing_extensions import NotRequired, TypedDict, Unpack

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# ---------- Tool call types ----------

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

    def override(
        self, **overrides: Unpack[_MCPToolCallRequestOverrides]
    ) -> MCPToolCallRequest:
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


# ---------- Resource read types ----------

MCPResourceReadResult = ReadResourceResult


class _MCPResourceReadRequestOverrides(TypedDict, total=False):
    uri: NotRequired[str]
    headers: NotRequired[dict[str, Any] | None]


@dataclass
class MCPResourceReadRequest:
    """Resource read request passed to MCP resource read interceptors.

    Modifiable fields (override to change behavior):
        uri: The URI of the resource to read.
        headers: HTTP headers for applicable transports.

    Context fields (read-only, for routing/logging):
        server_name: Name of the MCP server hosting the resource.
    """

    uri: str
    server_name: str
    headers: dict[str, Any] | None = None

    def override(
        self, **overrides: Unpack[_MCPResourceReadRequestOverrides]
    ) -> MCPResourceReadRequest:
        return replace(self, **overrides)


@runtime_checkable
class ResourceReadInterceptor(Protocol):
    """Protocol for resource read interceptors using handler callback pattern.

    Interceptors wrap resource reads in an onion pattern, mirroring
    ToolCallInterceptor for the resource lifecycle.
    """

    async def __call__(
        self,
        request: MCPResourceReadRequest,
        handler: Callable[[MCPResourceReadRequest], Awaitable[MCPResourceReadResult]],
    ) -> MCPResourceReadResult: ...
