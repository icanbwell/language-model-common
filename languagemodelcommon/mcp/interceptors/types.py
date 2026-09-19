"""Interceptor types for MCP tool call lifecycle management.

Replaces the types previously imported from langchain-mcp-adapters.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol, Self, runtime_checkable

from mcp.types import CallToolResult, InputRequiredResult, InputResponses
from typing_extensions import NotRequired, TypedDict, Unpack

# Result type — matches what interceptors and handlers return. Widened for
# SEP-2322: a guard-tool-gated tool may return InputRequiredResult instead
# of a terminal CallToolResult.
MCPToolCallResult = CallToolResult | InputRequiredResult


class _MCPToolCallRequestOverrides(TypedDict, total=False):
    name: NotRequired[str]
    args: NotRequired[dict[str, Any]]
    headers: NotRequired[dict[str, Any] | None]
    input_responses: NotRequired["InputResponses | None"]
    request_state: NotRequired[str | None]
    allow_input_required: NotRequired[bool]


@dataclass
class MCPToolCallRequest:
    """Tool execution request passed to MCP tool call interceptors.

    Modifiable fields (override to change behavior):
        name: Tool name to invoke.
        args: Tool arguments as key-value pairs.
        headers: HTTP headers for applicable transports.
        input_responses: Answers to a prior call's InputRequiredResult
            (SEP-2322 guard-tool retry). None on a first-round call.
        request_state: Opaque state echoed from a prior InputRequiredResult.
            Must be passed through byte-exact; never inspected here.
        allow_input_required: Whether the caller is prepared to receive an
            InputRequiredResult instead of a terminal CallToolResult.
            Defaults to False so existing callers that don't handle
            InputRequiredResult keep getting the pre-SEP-2322 behavior
            (session.call_tool raises instead of returning one). Only
            callers that actually check isinstance(result,
            InputRequiredResult) should set this to True.

    Context fields (read-only, for routing/logging):
        server_name: Name of the MCP server handling the tool.
    """

    name: str
    args: dict[str, Any]
    server_name: str
    headers: dict[str, Any] | None = None
    input_responses: InputResponses | None = None
    request_state: str | None = None
    allow_input_required: bool = False

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
