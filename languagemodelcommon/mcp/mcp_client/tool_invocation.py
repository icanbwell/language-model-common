"""Tool invocation — interceptor chain and raw MCP tool calls."""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.callbacks.manager import adispatch_custom_event
from mcp.types import CallToolResult, InputRequiredResult, InputResponses
from pydantic import ValidationError

from languagemodelcommon.mcp.callbacks import Callbacks, CallbackContext, _MCPCallbacks
from languagemodelcommon.mcp.interceptors.types import (
    MCPToolCallRequest,
    MCPToolCallResult,
    ToolCallInterceptor,
)
from languagemodelcommon.mcp.mcp_client.session import (
    MCPConnectionConfig,
    open_initialized_mcp_session,
)
from languagemodelcommon.mcp.mcp_client.session_pool import McpSessionPool
from languagemodelcommon.mcp.mcp_client.tool_list_cache import ToolListCache

logger = logging.getLogger(__name__)


def build_interceptor_chain(
    *,
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


async def _tool_supports_tasks(
    *,
    session: Any,
    tool_name: str,
    tool_list_cache: ToolListCache | None = None,
    cache_key: str | None = None,
) -> bool:
    """Check whether a specific tool supports task-augmented execution.

    Server-level task capability is necessary but not sufficient. Each tool
    declares its own task support via execution.task_support in the tools/list
    response. Only tools with "optional" or "required" support tasks.

    Reads from the existing ToolListCache (populated when tools are listed)
    rather than making a redundant list_tools call.
    """
    if not _server_supports_tool_tasks(session):
        return False

    if tool_list_cache is None or cache_key is None:
        return False

    tools = await tool_list_cache.get_async(key=cache_key)
    if tools is None:
        return False

    for tool in tools:
        if tool.name == tool_name:
            if tool.execution is not None:
                task_support = getattr(tool.execution, "task_support", None)
                return task_support in ("optional", "required")
            return False

    return False


def _extract_task_id_from_create_result(create_result: Any) -> str:
    """Extract task_id from CreateTaskResult, handling both spec versions.

    Old spec: result.task.task_id (nested under 'task' field)
    New spec: result.task_id (flat, Result & Task merged)
    """
    # New spec: task_id directly on result
    if hasattr(create_result, "task_id"):
        return str(create_result.task_id)

    # Old spec: nested under .task
    if hasattr(create_result, "task"):
        return str(create_result.task.task_id)

    raise ValueError(f"Cannot extract task_id from {type(create_result)}")


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


class TaskProtocolError(Exception):
    """Raised when the MCP task protocol fails and caller should fall back to call_tool."""


async def _execute_tool_as_task(
    *,
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

    Raises TaskProtocolError if the server does not respond with a valid
    CreateTaskResult (e.g., returns a plain CallToolResult instead).
    """
    try:
        create_result = await session.experimental.call_tool_as_task(
            name, arguments, ttl=60000
        )
    except (ValidationError, ValueError, TypeError) as e:
        raise TaskProtocolError(
            f"Server did not return a valid CreateTaskResult for '{name}': {e}"
        ) from e
    task_id = _extract_task_id_from_create_result(create_result)

    inline_result: CallToolResult | None = None

    async for status in session.experimental.poll_task(task_id):
        try:
            await adispatch_custom_event(
                "mcp_task_progress",
                {
                    "task_id": task_id,
                    "status": status.status,
                    "message": getattr(status, "status_message", None),
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


async def _execute_tool_call_with_heartbeat(
    *,
    session: Any,
    name: str,
    arguments: dict[str, Any],
    progress_callback: Any,
    server_name: str,
    heartbeat_interval_seconds: float,
    input_responses: InputResponses | None = None,
    request_state: str | None = None,
    allow_input_required: bool = False,
) -> CallToolResult | InputRequiredResult:
    """Call ``session.call_tool``, emitting a synthetic ``mcp_tool_heartbeat``
    custom event every ``heartbeat_interval_seconds`` while the call is in
    flight, regardless of whether the tool reports real progress.

    ``input_responses``/``request_state`` are SEP-2322 guard-tool retry
    fields, forwarded byte-exact to ``session.call_tool``. When
    ``allow_input_required`` is True, an ``InputRequiredResult`` is returned
    to the caller instead of raising -- see ``ClientSession.call_tool``'s own
    docstring for the raise-vs-return contract.

    Uses ``asyncio.shield`` so a heartbeat tick's wait_for timeout never
    cancels the underlying tool call -- only the local wait is abandoned and
    retried on the same call_task. If this coroutine itself is cancelled
    (e.g. the overall chat turn is aborted), the inner call_task is
    cancelled too rather than left running in the background.
    """
    call_task: "asyncio.Task[CallToolResult | InputRequiredResult]" = (
        asyncio.ensure_future(
            session.call_tool(
                name,
                arguments,
                progress_callback=progress_callback,
                input_responses=input_responses,
                request_state=request_state,
                allow_input_required=allow_input_required,
            )
        )
    )
    elapsed_seconds = 0.0
    try:
        while True:
            try:
                return await asyncio.wait_for(
                    asyncio.shield(call_task), timeout=heartbeat_interval_seconds
                )
            except asyncio.TimeoutError:
                elapsed_seconds += heartbeat_interval_seconds
                try:
                    await adispatch_custom_event(
                        "mcp_tool_heartbeat",
                        {
                            "server_name": server_name,
                            "tool_name": name,
                            "elapsed_seconds": elapsed_seconds,
                        },
                    )
                except RuntimeError as e:
                    logger.debug(
                        "Skipping mcp_tool_heartbeat event dispatch: %s (tool=%s)",
                        e,
                        name,
                    )
    finally:
        # Ensure call_task is never left orphaned, whether this coroutine
        # exits via CancelledError, a return, or any other exception raised
        # from the loop body (e.g. adispatch_custom_event).
        if not call_task.done():
            call_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await call_task


def _make_execute_tool(
    *,
    config: MCPConnectionConfig,
    mcp_callbacks: _MCPCallbacks,
    session_pool: McpSessionPool | None = None,
    tool_list_cache: ToolListCache | None = None,
    heartbeat_interval_seconds: float = 15.0,
) -> Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]:
    """Create an execute_tool handler that opens a session and calls the tool.

    Shared by both ``call_mcp_tool_raw`` and ``mcp_tool_to_langchain_tool``
    to avoid duplicating session management logic.

    When a ``session_pool`` is provided, sessions are reused across calls
    to the same MCP server URL within the pool's scope.

    If the server and tool both advertise task support, the tool is
    executed via the MCP task protocol with polling and progress events.
    Per-tool ``execution.task_support`` is checked from the already-cached
    tool list so that tools with ``"forbidden"`` (or no declaration) use
    normal call_tool.
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
        effective_headers = effective_config.get("headers") or {}
        cache_key = ToolListCache.make_key(
            server_url, auth_header=effective_headers.get("Authorization")
        )

        if session_pool is not None:
            session = await session_pool.get_session(
                effective_config, mcp_callbacks=mcp_callbacks
            )
            try:
                if await _tool_supports_tasks(
                    session=session,
                    tool_name=request.name,
                    tool_list_cache=tool_list_cache,
                    cache_key=cache_key,
                ):
                    try:
                        return await _execute_tool_as_task(
                            session=session,
                            name=request.name,
                            arguments=request.args,
                            server_name=request.server_name,
                        )
                    except TaskProtocolError:
                        logger.warning(
                            "Task protocol failed for %s on %s, falling back to call_tool",
                            request.name,
                            server_url,
                        )
                        if tool_list_cache is not None:
                            await tool_list_cache.invalidate_async(key=cache_key)
                return await _execute_tool_call_with_heartbeat(
                    session=session,
                    name=request.name,
                    arguments=request.args,
                    progress_callback=mcp_callbacks.progress_callback,
                    server_name=request.server_name,
                    heartbeat_interval_seconds=heartbeat_interval_seconds,
                    input_responses=request.input_responses,
                    request_state=request.request_state,
                    allow_input_required=request.allow_input_required,
                )
            except Exception:
                await session_pool.evict(effective_config)
                if tool_list_cache is not None:
                    await tool_list_cache.invalidate_async(key=cache_key)
                raise

        # Fallback: create a one-shot session (original behavior). Session
        # establishment (connect + initialize) retries transient failures
        # internally (BAI-889); the tool call itself below is never retried.
        captured_exception = None
        cm, session = await open_initialized_mcp_session(
            effective_config, mcp_callbacks=mcp_callbacks
        )
        try:
            result: CallToolResult | InputRequiredResult
            try:
                if await _tool_supports_tasks(
                    session=session,
                    tool_name=request.name,
                    tool_list_cache=tool_list_cache,
                    cache_key=cache_key,
                ):
                    try:
                        result = await _execute_tool_as_task(
                            session=session,
                            name=request.name,
                            arguments=request.args,
                            server_name=request.server_name,
                        )
                    except TaskProtocolError:
                        logger.warning(
                            "Task protocol failed for %s on %s, falling back to call_tool",
                            request.name,
                            server_url,
                        )
                        if tool_list_cache is not None:
                            await tool_list_cache.invalidate_async(key=cache_key)
                        result = await _execute_tool_call_with_heartbeat(
                            session=session,
                            name=request.name,
                            arguments=request.args,
                            progress_callback=mcp_callbacks.progress_callback,
                            server_name=request.server_name,
                            heartbeat_interval_seconds=heartbeat_interval_seconds,
                            input_responses=request.input_responses,
                            request_state=request.request_state,
                            allow_input_required=request.allow_input_required,
                        )
                else:
                    result = await _execute_tool_call_with_heartbeat(
                        session=session,
                        name=request.name,
                        arguments=request.args,
                        progress_callback=mcp_callbacks.progress_callback,
                        server_name=request.server_name,
                        heartbeat_interval_seconds=heartbeat_interval_seconds,
                        input_responses=request.input_responses,
                        request_state=request.request_state,
                        allow_input_required=request.allow_input_required,
                    )
            except Exception as e:
                captured_exception = e
        finally:
            # Always exit clean (None, None, None): captured_exception is
            # re-raised below, outside the session's lifecycle -- matching
            # the prior `async with` behavior -- so create_mcp_session's own
            # broad except clause never gets a chance to re-wrap a tool-call
            # error (e.g. a guard-tool InputRequired RuntimeError) as an
            # unrelated McpSessionError.
            await cm.__aexit__(None, None, None)

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
    tool_list_cache: ToolListCache | None = None,
    heartbeat_interval_seconds: float = 15.0,
    input_responses: InputResponses | None = None,
    request_state: str | None = None,
    allow_input_required: bool = False,
) -> MCPToolCallResult:
    """Call an MCP tool and return the raw CallToolResult (or, for a
    guard-tool-gated tool, an InputRequiredResult).

    This is used by the call_tool meta-tool to proxy calls without
    converting to LangChain format, and by callers resubmitting a SEP-2322
    guard-tool answer (``input_responses``/``request_state``) directly,
    bypassing the LangChain tool-call path entirely since the retry leg
    isn't driven by a fresh LLM tool call.

    ``allow_input_required`` defaults to False so existing callers that
    don't handle ``InputRequiredResult`` keep getting the pre-SEP-2322
    behavior: the underlying session raises rather than returning one.
    Pass True only if the caller actually checks
    ``isinstance(result, InputRequiredResult)``.
    """
    mcp_callbacks = (
        callbacks.to_mcp_format(
            context=CallbackContext(server_name=server_name, tool_name=tool_name)
        )
        if callbacks is not None
        else _MCPCallbacks()
    )

    execute_tool = _make_execute_tool(
        config=config,
        mcp_callbacks=mcp_callbacks,
        session_pool=session_pool,
        tool_list_cache=tool_list_cache,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )
    handler = build_interceptor_chain(
        base_handler=execute_tool, tool_interceptors=tool_interceptors
    )
    request = MCPToolCallRequest(
        name=tool_name,
        args=arguments,
        server_name=server_name,
        headers=None,
        input_responses=input_responses,
        request_state=request_state,
        allow_input_required=allow_input_required,
    )
    return await handler(request)
