"""Custom MCP client replacing langchain-mcp-adapters.

Provides session management, tool listing, tool invocation, and
LangChain BaseTool conversion using the MCP SDK directly.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import httpx
from langchain_core.messages.content import (
    FileContentBlock,
    ImageContentBlock,
    TextContentBlock,
    create_file_block,
    create_image_block,
    create_text_block,
)
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ReadResourceResult,
    Resource as MCPResource,
    ResourceLink,
    ResourceTemplate as MCPResourceTemplate,
    TextContent,
    TextResourceContents,
)
from mcp.types import Tool as MCPTool
from pydantic import AnyUrl
from typing_extensions import NotRequired, TypedDict

from languagemodelcommon.mcp.callbacks import Callbacks, CallbackContext, _MCPCallbacks
from languagemodelcommon.mcp.interceptors.types import (
    MCPResourceReadRequest,
    MCPResourceReadResult,
    MCPToolCallRequest,
    MCPToolCallResult,
    ResourceReadInterceptor,
    ToolCallInterceptor,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)

MAX_ITERATIONS = 1000

DEFAULT_TIMEOUT = timedelta(seconds=30)
DEFAULT_SSE_READ_TIMEOUT = timedelta(seconds=300)

ToolMessageContentBlock = TextContentBlock | ImageContentBlock | FileContentBlock


# ---------- Connection config ----------


class McpHttpClientFactory:
    """Protocol-compatible callable for creating httpx async clients."""

    def __call__(
        self,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            auth=auth,
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )


class MCPConnectionConfig(TypedDict, total=False):
    """Connection config for streamable HTTP MCP servers."""

    url: str
    transport: str
    headers: NotRequired[dict[str, Any] | None]
    timeout: NotRequired[timedelta]
    sse_read_timeout: NotRequired[timedelta]
    httpx_client_factory: NotRequired[Any]


# ---------- Session management ----------


@asynccontextmanager
async def create_mcp_session(
    config: MCPConnectionConfig,
    *,
    mcp_callbacks: _MCPCallbacks | None = None,
) -> AsyncIterator[ClientSession]:
    """Create an MCP client session using streamable HTTP transport."""
    url = config["url"]
    headers = config.get("headers")
    timeout = config.get("timeout", DEFAULT_TIMEOUT)
    sse_read_timeout = config.get("sse_read_timeout", DEFAULT_SSE_READ_TIMEOUT)
    httpx_client_factory = config.get("httpx_client_factory")

    kwargs: dict[str, Any] = {}
    if httpx_client_factory is not None:
        kwargs["httpx_client_factory"] = httpx_client_factory

    session_kwargs: dict[str, Any] = {}
    if mcp_callbacks is not None:
        if mcp_callbacks.logging_callback is not None:
            session_kwargs["logging_callback"] = mcp_callbacks.logging_callback

    async with (
        streamablehttp_client(
            url,
            headers,
            timeout,
            sse_read_timeout,
            **kwargs,
        ) as (read, write, _),
        ClientSession(read, write, **session_kwargs) as session,
    ):
        yield session


# ---------- Tool listing ----------


async def list_all_tools(session: ClientSession) -> list[MCPTool]:
    """List all tools from an MCP session with pagination."""
    cursor: str | None = None
    all_tools: list[MCPTool] = []
    iterations = 0

    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            raise RuntimeError("Exceeded max iterations while listing tools")

        result = await session.list_tools(cursor=cursor)
        if result.tools:
            all_tools.extend(result.tools)
        if not result.nextCursor:
            break
        cursor = result.nextCursor

    return all_tools


# ---------- Resource listing ----------


async def list_all_resources(session: ClientSession) -> list[MCPResource]:
    """List all resources from an MCP session with pagination."""
    cursor: str | None = None
    all_resources: list[MCPResource] = []
    iterations = 0

    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            raise RuntimeError("Exceeded max iterations while listing resources")

        result = await session.list_resources(cursor=cursor)
        if result.resources:
            all_resources.extend(result.resources)
        if not result.nextCursor:
            break
        cursor = result.nextCursor

    return all_resources


async def list_all_resource_templates(
    session: ClientSession,
) -> list[MCPResourceTemplate]:
    """List all resource templates from an MCP session with pagination."""
    cursor: str | None = None
    all_templates: list[MCPResourceTemplate] = []
    iterations = 0

    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            raise RuntimeError(
                "Exceeded max iterations while listing resource templates"
            )

        result = await session.list_resource_templates(cursor=cursor)
        if result.resourceTemplates:
            all_templates.extend(result.resourceTemplates)
        if not result.nextCursor:
            break
        cursor = result.nextCursor

    return all_templates


# ---------- Interceptor chain ----------


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


def build_resource_interceptor_chain(
    base_handler: Callable[[MCPResourceReadRequest], Awaitable[MCPResourceReadResult]],
    resource_interceptors: list[ResourceReadInterceptor] | None,
) -> Callable[[MCPResourceReadRequest], Awaitable[MCPResourceReadResult]]:
    """Build composed handler chain for resource reads in onion pattern."""
    handler = base_handler

    if resource_interceptors:
        for interceptor in reversed(resource_interceptors):
            current_handler = handler

            async def wrapped_handler(
                req: MCPResourceReadRequest,
                _interceptor: ResourceReadInterceptor = interceptor,
                _handler: Callable[
                    [MCPResourceReadRequest], Awaitable[MCPResourceReadResult]
                ] = current_handler,
            ) -> MCPResourceReadResult:
                return await _interceptor(req, _handler)

            handler = wrapped_handler

    return handler


# ---------- Raw tool invocation ----------


def _make_execute_tool(
    config: MCPConnectionConfig,
    mcp_callbacks: _MCPCallbacks,
) -> Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]:
    """Create an execute_tool handler that opens a session and calls the tool.

    Shared by both ``call_mcp_tool_raw`` and ``mcp_tool_to_langchain_tool``
    to avoid duplicating session management logic.
    """

    async def execute_tool(request: MCPToolCallRequest) -> MCPToolCallResult:
        effective_config = config
        modified_headers = request.headers
        if modified_headers is not None:
            updated = dict(config)
            existing_headers = config.get("headers") or {}
            updated["headers"] = {**existing_headers, **modified_headers}
            effective_config = updated  # type: ignore[assignment]

        captured_exception = None
        async with create_mcp_session(
            effective_config, mcp_callbacks=mcp_callbacks
        ) as session:
            await session.initialize()
            try:
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

    execute_tool = _make_execute_tool(config, mcp_callbacks)
    handler = build_interceptor_chain(execute_tool, tool_interceptors)
    request = MCPToolCallRequest(
        name=tool_name,
        args=arguments,
        server_name=server_name,
        headers=None,
    )
    return await handler(request)


# ---------- Raw resource read ----------


def _make_read_resource(
    config: MCPConnectionConfig,
    mcp_callbacks: _MCPCallbacks,
) -> Callable[[MCPResourceReadRequest], Awaitable[MCPResourceReadResult]]:
    """Create a read_resource handler that opens a session and reads the resource."""

    async def read_resource(
        request: MCPResourceReadRequest,
    ) -> MCPResourceReadResult:
        effective_config = config
        modified_headers = request.headers
        if modified_headers is not None:
            updated = dict(config)
            existing_headers = config.get("headers") or {}
            updated["headers"] = {**existing_headers, **modified_headers}
            effective_config = updated  # type: ignore[assignment]

        captured_exception = None
        async with create_mcp_session(
            effective_config, mcp_callbacks=mcp_callbacks
        ) as session:
            await session.initialize()
            try:
                result = await session.read_resource(AnyUrl(request.uri))
            except Exception as e:
                captured_exception = e

        if captured_exception is not None:
            raise captured_exception
        return result

    return read_resource


async def read_mcp_resource_raw(
    *,
    config: MCPConnectionConfig,
    uri: str,
    server_name: str,
    callbacks: Callbacks | None = None,
    resource_interceptors: list[ResourceReadInterceptor] | None = None,
) -> ReadResourceResult:
    """Read an MCP resource and return the raw ReadResourceResult.

    This is used by the read_resource meta-tool to proxy reads without
    converting to LangChain format.
    """
    mcp_callbacks = (
        callbacks.to_mcp_format(
            context=CallbackContext(server_name=server_name)
        )
        if callbacks is not None
        else _MCPCallbacks()
    )

    execute_read = _make_read_resource(config, mcp_callbacks)
    handler = build_resource_interceptor_chain(execute_read, resource_interceptors)
    request = MCPResourceReadRequest(
        uri=uri,
        server_name=server_name,
        headers=None,
    )
    return await handler(request)


# ---------- Resource content conversion ----------


def convert_resource_contents_to_text(result: ReadResourceResult) -> str:
    """Convert a ReadResourceResult to a text representation for the LLM."""
    parts: list[str] = []
    for content in result.contents:
        if isinstance(content, TextResourceContents):
            parts.append(content.text)
        elif isinstance(content, BlobResourceContents):
            mime_type = content.mimeType or "application/octet-stream"
            parts.append(f"[Binary content: {mime_type}, uri: {content.uri}]")
        else:
            parts.append(f"[Resource content: {content.uri}]")
    return "\n".join(parts)


# ---------- Content conversion ----------


def convert_mcp_content_to_lc_block(
    content: ContentBlock,
) -> ToolMessageContentBlock:
    """Convert an MCP content block to a LangChain content block."""
    if isinstance(content, TextContent):
        return create_text_block(text=content.text)

    if isinstance(content, ImageContent):
        return create_image_block(base64=content.data, mime_type=content.mimeType)

    if isinstance(content, AudioContent):
        raise NotImplementedError(
            f"AudioContent conversion not supported. Mime type: {content.mimeType}"
        )

    if isinstance(content, ResourceLink):
        mime_type = content.mimeType or None
        if mime_type and mime_type.startswith("image/"):
            return create_image_block(url=str(content.uri), mime_type=mime_type)
        return create_file_block(url=str(content.uri), mime_type=mime_type)

    if isinstance(content, EmbeddedResource):
        resource = content.resource
        if isinstance(resource, TextResourceContents):
            return create_text_block(text=resource.text)
        if isinstance(resource, BlobResourceContents):
            mime_type = resource.mimeType or None
            if mime_type and mime_type.startswith("image/"):
                return create_image_block(base64=resource.blob, mime_type=mime_type)
            return create_file_block(base64=resource.blob, mime_type=mime_type)
        raise ValueError(f"Unknown embedded resource type: {type(resource).__name__}")

    raise ValueError(f"Unknown MCP content type: {type(content).__name__}")


def convert_call_tool_result(
    result: CallToolResult,
) -> list[ToolMessageContentBlock]:
    """Convert a CallToolResult to LangChain content blocks.

    Raises ToolException if the result indicates an error.
    """
    tool_content: list[ToolMessageContentBlock] = [
        convert_mcp_content_to_lc_block(c) for c in result.content
    ]

    if result.isError:
        error_parts = [
            block.text for block in result.content if isinstance(block, TextContent)
        ]
        raise ToolException(
            "\n".join(error_parts) if error_parts else str(tool_content)
        )

    return tool_content


# ---------- MCP Tool -> LangChain Tool ----------


def mcp_tool_to_langchain_tool(
    tool: MCPTool,
    *,
    connection: MCPConnectionConfig,
    callbacks: Callbacks | None = None,
    tool_interceptors: list[ToolCallInterceptor] | None = None,
    server_name: str | None = None,
) -> BaseTool:
    """Convert an MCP Tool to a LangChain BaseTool.

    Creates a StructuredTool that establishes a new session per invocation
    and applies the interceptor chain.
    """

    mcp_callbacks = (
        callbacks.to_mcp_format(
            context=CallbackContext(server_name=server_name, tool_name=tool.name)
        )
        if callbacks is not None
        else _MCPCallbacks()
    )
    execute_tool = _make_execute_tool(connection, mcp_callbacks)
    handler = build_interceptor_chain(execute_tool, tool_interceptors)

    async def call_tool(
        **arguments: dict[str, Any],
    ) -> tuple[list[ToolMessageContentBlock], None]:
        request = MCPToolCallRequest(
            name=tool.name,
            args=arguments,
            server_name=server_name or "unknown",
            headers=None,
        )
        call_tool_result = await handler(request)
        content = convert_call_tool_result(call_tool_result)
        return content, None

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.inputSchema,
        coroutine=call_tool,
        response_format="content_and_artifact",
    )
