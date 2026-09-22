"""LangChain BaseTool adapter for MCP tools."""

import re
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from mcp.types import InputRequest, InputRequiredResult
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
from languagemodelcommon.mcp.mcp_client.tool_list_cache import ToolListCache


_INVALID_TOOL_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


class MCPInputRequiredError(RuntimeError):
    """Raised when an MCP tool call returns InputRequiredResult (SEP-2322
    guard-tool ask) instead of a terminal CallToolResult.

    Carries everything a caller needs to resubmit the *exact same* call via
    ``call_mcp_tool_raw`` once it has collected an answer to
    ``input_requests`` -- this package has no opinion on how that answer is
    collected (a human-in-the-loop UI, a LangGraph ``interrupt()``, or
    anything else); it only guarantees the retry leg has what it needs.

    Deliberately a plain ``RuntimeError`` subclass, not ``BaseException``:
    unlike mcp-fhir-agent's server-side ``ElicitationRequired`` (which must
    survive broad `except Exception` handlers several call layers below the
    tool function), this exception is raised directly at the LangChain tool
    boundary with no intermediate layer *in this package* that could swallow
    it -- this package makes no guarantee beyond its own boundary.

    IMPORTANT -- this is the caller's responsibility, not something this
    package can enforce: if you execute this tool's coroutine through a
    framework that wraps tool execution in its own broad, default
    exception-to-error-message conversion, that framework's handling sits
    between this raise site and any "top-level catcher" you may have and
    WILL swallow this exception before your code ever sees it. The primary
    example is LangGraph's prebuilt ``ToolNode``, whose default
    ``handle_tool_errors=True`` wraps every tool call in `except Exception`
    and converts it to a generic error ``ToolMessage``, discarding
    ``input_requests``/``request_state`` in the process. If you wire this
    tool into a ``ToolNode``-style executor, you MUST catch
    ``MCPInputRequiredError`` yourself at or before the point where that
    framework invokes the tool -- do not rely on it propagating further up.
    baileyai's ``wrap_tool_for_guard_tool_bridge``
    (`baileyai/services/mcp/guard_tool_bridge.py`) is the correct pattern:
    it catches ``MCPInputRequiredError`` immediately around its own
    ``await tool.ainvoke(...)`` call, before ``ToolNode`` (or any other
    framework-level handler) ever gets a chance to intercept it -- see the
    companion baileyai ADR
    (`adrs/006-mcp-guard-tool-elicitation-support.md`) for how it's
    consumed.
    """

    def __init__(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        connection: "MCPConnectionConfig",
        server_name: str | None,
        input_requests: dict[str, InputRequest],
        request_state: str | None,
    ) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.connection = connection
        self.server_name = server_name
        self.input_requests = input_requests
        self.request_state = request_state
        super().__init__(
            f"MCP tool {tool_name!r} requires input before it can proceed "
            f"(fields: {sorted(input_requests)})"
        )


def _sanitize_tool_name(name: str) -> str:
    """Replace characters that violate LLM provider tool-name constraints.

    AWS Bedrock ConverseStream requires: [a-zA-Z0-9_-]+
    """
    return _INVALID_TOOL_NAME_CHARS.sub("_", name)


def _resolve_mcp_title(tool: MCPTool) -> str | None:
    """Return the best human-readable title from an MCP Tool, or None.

    Precedence (per MCP 2025-06-18 spec):
    1. ``tool.title``  — top-level, from BaseMetadata
    2. ``tool.annotations.title`` — hint from ToolAnnotations
    """
    if tool.title is not None:
        title = tool.title.strip()
        if title:
            return title
    if tool.annotations and tool.annotations.title is not None:
        title = tool.annotations.title.strip()
        if title:
            return title
    return None


def mcp_tool_to_langchain_tool(
    tool: MCPTool,
    *,
    connection: MCPConnectionConfig,
    callbacks: Callbacks | None = None,
    tool_interceptors: list[ToolCallInterceptor] | None = None,
    server_name: str | None = None,
    session_pool: McpSessionPool | None = None,
    tool_list_cache: ToolListCache | None = None,
    heartbeat_interval_seconds: float = 15.0,
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
        config=connection,
        mcp_callbacks=mcp_callbacks,
        session_pool=session_pool,
        tool_list_cache=tool_list_cache,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )
    handler = build_interceptor_chain(
        base_handler=execute_tool, tool_interceptors=tool_interceptors
    )

    async def call_tool(
        **arguments: dict[str, Any],
    ) -> tuple[list[ToolMessageContentBlock], dict[str, Any] | None]:
        request = MCPToolCallRequest(
            name=tool.name,
            args=arguments,
            server_name=server_name or "unknown",
            headers=None,
            allow_input_required=True,
        )
        call_tool_result = await handler(request)
        if isinstance(call_tool_result, InputRequiredResult):
            raise MCPInputRequiredError(
                tool_name=tool.name,
                arguments=arguments,
                connection=connection,
                server_name=server_name,
                input_requests=call_tool_result.input_requests or {},
                request_state=call_tool_result.request_state,
            )
        content = convert_call_tool_result(call_tool_result)
        # structured_content (MCP's structuredContent) never reaches the LLM --
        # LangChain only sends `content` back to the model, never `artifact` --
        # so this is purely for the UI's debugging details panel (BAI-882).
        # Previously discarded entirely: this call returned (content, None).
        return content, call_tool_result.structured_content

    metadata: dict[str, Any] = {}
    mcp_title = _resolve_mcp_title(tool)
    if mcp_title:
        metadata["mcp_title"] = mcp_title
    if tool.description:
        metadata["mcp_description"] = tool.description

    return StructuredTool(
        name=_sanitize_tool_name(tool.name),
        description=tool.description or "",
        args_schema=tool.input_schema,
        coroutine=call_tool,
        response_format="content_and_artifact",
        metadata=metadata or None,
    )
