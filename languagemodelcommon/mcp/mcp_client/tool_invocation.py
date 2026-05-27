"""Tool invocation — interceptor chain and raw MCP tool calls."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from mcp.types import CallToolResult

from languagemodelcommon.mcp.callbacks import Callbacks, CallbackContext, _MCPCallbacks
from languagemodelcommon.mcp.interceptors.types import (
    MCPToolCallRequest,
    MCPToolCallResult,
    ToolCallInterceptor,
)
from languagemodelcommon.mcp.mcp_client.session import (
    MCPConnectionConfig,
    create_mcp_session,
)
from languagemodelcommon.mcp.mcp_client.session_pool import McpSessionPool

logger = logging.getLogger(__name__)

_tool_task_support_cache: dict[str, dict[str, str | None]] = {}


def build_interceptor_chain(
    base_handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    tool_interceptors: list[ToolCallInterceptor] | None,
) -> Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]:
    """Build composed handler chain with interceptors in onion pattern."""
    handler = base_handler

    if tool_interceptors:
        for interceptor in reversed(tool_interceptors):
            current_handler = handler

            async def wrapped_handler(
                req: MCPToolCallRequest,
                _interceptor: ToolCallInterceptor = interceptor,
                _handler: Callable[
                    [MCPToolCallRequest], Awaitable[MCPToolCallResult]
                ] = current_handler,
            ) -> MCPToolCallResult:
                return await _interceptor(req, _handler)

            handler = wrapped_handler

    return handler


def _server_supports_tool_tasks(session: Any) -> bool:
    """Check whether the MCP server advertises task support for tool calls.

    Supports both spec versions:
    - Old (2025-11-25): capabilities.tasks.requests.tools.call
    - New (2026-07-28): capabilities.extensions["io.modelcontextprotocol/tasks"]
    """
    try:
        caps = session.get_server_capabilities()
        if not caps:
            return False

        # New spec (2026-07-28): extension-based capability
        extensions = getattr(caps, "extensions", None)
        if extensions and "io.modelcontextprotocol/tasks" in extensions:
            return True

        # Old spec (2025-11-25): nested capability hierarchy
        tasks_cap = getattr(caps, "tasks", None)
        if tasks_cap:
            requests = getattr(tasks_cap, "requests", None)
            if requests:
                tools = getattr(requests, "tools", None)
                if tools and getattr(tools, "call", None):
                    return True

        return False
    except Exception:
        return False


async def _tool_supports_tasks(session: Any, tool_name: str, server_url: str) -> bool:
    """Check whether a specific tool supports task-augmented execution.

    Server-level task capability is necessary but not sufficient. Each tool
    declares its own task support via execution.taskSupport in the tools/list
    response. Only tools with "optional" or "required" support tasks.

    Results are cached per server URL to avoid repeated list_tools calls.
    """
    if not _server_supports_tool_tasks(session):
        return False

    cache_key = server_url
    if cache_key not in _tool_task_support_cache:
        try:
            tools_result = await session.list_tools()
            tool_map: dict[str, str | None] = {}
            for tool in tools_result.tools:
                task_support: str | None = None
                if tool.execution is not None:
                    task_support = getattr(tool.execution, "taskSupport", None)
                tool_map[tool.name] = task_support
            _tool_task_support_cache[cache_key] = tool_map
        except Exception:
            logger.debug(
                "Failed to fetch tool list for task support check on %s",
                server_url,
            )
            return False

    tool_support = _tool_task_support_cache.get(cache_key, {}).get(tool_name)
    return tool_support in ("optional", "required")


def _extract_task_id_from_create_result(create_result: Any) -> str:
    """Extract taskId from CreateTaskResult, handling both spec versions.

    Old spec: result.task.taskId (nested under 'task' field)
    New spec: result.taskId (flat, Result & Task merged)
    """
    # New spec: taskId directly on result
    if hasattr(create_result, "taskId"):
        return str(create_result.taskId)

    # Old spec: nested under .task
    if hasattr(create_result, "task"):
        return str(create_result.task.taskId)

    raise ValueError(f"Cannot extract taskId from {type(create_result)}")


def _extract_result_from_task_status(task_status: Any) -> CallToolResult | None:
    """Extract inline result from a terminal task status (new spec).

    New spec: completed tasks carry result inline in tasks/get response.
    Old spec: requires separate tasks/result call — returns None.
    """
    if hasattr(task_status, "result") and task_status.result is not None:
        result = task_status.result
        if isinstance(result, CallToolResult):
            return result
        # New spec may return raw dict — wrap it
        if isinstance(result, dict):
            return CallToolResult(**result)
    return None


async def _execute_tool_as_task(
    session: Any,
    name: str,
    arguments: dict[str, Any],
    server_name: str,
) -> CallToolResult:
    """Execute a tool via the MCP task protocol, polling until completion.

    Supports both spec versions:
    - Old (experimental): call_tool_as_task → poll_task → get_task_result
    - New (extension): server returns resultType="task" → poll via
      tasks/get → result inline in completed status

    Emits ``mcp_task_progress`` custom events via LangChain's
    ``adispatch_custom_event`` so the streaming manager can forward
    progress updates to the client.
    """
    from langchain_core.callbacks.manager import adispatch_custom_event

    create_result = await session.experimental.call_tool_as_task(
        name, arguments, ttl=60000
    )
    task_id = _extract_task_id_from_create_result(create_result)

    inline_result: CallToolResult | None = None

    async for status in session.experimental.poll_task(task_id):
        try:
            await adispatch_custom_event(
                "mcp_task_progress",
                {
                    "task_id": task_id,
                    "status": status.status,
                    "message": getattr(status, "statusMessage", None),
                    "server_name": server_name,
                    "tool_name": name,
                },
            )
        except RuntimeError as e:
            logger.debug(
                "Skipping mcp_task_progress event dispatch: %s (task_id=%s, tool=%s)",
                e,
                task_id,
                name,
            )

        # New spec: result may be inline on terminal status
        inline_result = _extract_result_from_task_status(status)

    # If new spec provided result inline, use it
    if inline_result is not None:
        return inline_result

    # Old spec: fetch result via separate tasks/result call
    result: CallToolResult = await session.experimental.get_task_result(
        task_id, CallToolResult
    )
    return result


def _make_execute_tool(
    config: MCPConnectionConfig,
    mcp_callbacks: _MCPCallbacks,
    session_pool: McpSessionPool | None = None,
) -> Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]:
    """Create an execute_tool handler that opens a session and calls the tool.

    Shared by both ``call_mcp_tool_raw`` and ``mcp_tool_to_langchain_tool``
    to avoid duplicating session management logic.

    When a ``session_pool`` is provided, sessions are reused across calls
    to the same MCP server URL within the pool's scope.

    If the server and tool both advertise task support, the tool is
    executed via the MCP task protocol with polling and progress events.
    Per-tool ``execution.taskSupport`` is checked so that tools with
    ``"forbidden"`` (or no declaration) use normal call_tool.
    """

    async def execute_tool(request: MCPToolCallRequest) -> MCPToolCallResult:
        effective_config = config
        modified_headers = request.headers
        if modified_headers is not None:
            updated = dict(config)
            existing_headers = config.get("headers") or {}
            updated["headers"] = {**existing_headers, **modified_headers}
            effective_config = updated  # type: ignore[assignment]

        server_url = effective_config.get("url", "")

        if session_pool is not None:
            session = await session_pool.get_session(
                effective_config, mcp_callbacks=mcp_callbacks
            )
            try:
                if await _tool_supports_tasks(session, request.name, server_url):
                    return await _execute_tool_as_task(
                        session, request.name, request.args, request.server_name
                    )
                return await session.call_tool(
                    request.name,
                    request.args,
                    progress_callback=mcp_callbacks.progress_callback,
                )
            except Exception:
                await session_pool.evict(effective_config)
                raise

        # Fallback: create a one-shot session (original behavior)
        captured_exception = None
        async with create_mcp_session(
            effective_config, mcp_callbacks=mcp_callbacks
        ) as session:
            await session.initialize()
            try:
                if await _tool_supports_tasks(session, request.name, server_url):
                    result = await _execute_tool_as_task(
                        session, request.name, request.args, request.server_name
                    )
                else:
                    result = await session.call_tool(
                        request.name,
                        request.args,
                        progress_callback=mcp_callbacks.progress_callback,
                    )
            except Exception as e:
                captured_exception = e

        if captured_exception is not None:
            raise captured_exception
        return result

    return execute_tool


async def call_mcp_tool_raw(
    *,
    config: MCPConnectionConfig,
    tool_name: str,
    arguments: dict[str, Any],
    server_name: str,
    callbacks: Callbacks | None = None,
    tool_interceptors: list[ToolCallInterceptor] | None = None,
    session_pool: McpSessionPool | None = None,
) -> CallToolResult:
    """Call an MCP tool and return the raw CallToolResult.

    This is used by the call_tool meta-tool to proxy calls without
    converting to LangChain format.
    """
    mcp_callbacks = (
        callbacks.to_mcp_format(
            context=CallbackContext(server_name=server_name, tool_name=tool_name)
        )
        if callbacks is not None
        else _MCPCallbacks()
    )

    execute_tool = _make_execute_tool(config, mcp_callbacks, session_pool=session_pool)
    handler = build_interceptor_chain(execute_tool, tool_interceptors)
    request = MCPToolCallRequest(
        name=tool_name,
        args=arguments,
        server_name=server_name,
        headers=None,
    )
    return await handler(request)
