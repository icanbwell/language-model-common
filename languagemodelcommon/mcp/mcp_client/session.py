"""MCP session management — creating and connecting to MCP servers."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from typing_extensions import NotRequired, TypedDict

from languagemodelcommon.mcp.callbacks import _MCPCallbacks
from languagemodelcommon.utilities.logger.exception_logger import ExceptionLogger
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)

DEFAULT_TIMEOUT = timedelta(seconds=30)
DEFAULT_SSE_READ_TIMEOUT = timedelta(seconds=300)


class McpSessionError(Exception):
    """Raised when an MCP session cannot be established or fails unexpectedly.

    Wraps lower-level exceptions with actionable context (URL, timeout values,
    HTTP status) so operators can diagnose connectivity problems in production.
    """

    def __init__(self, message: str, *, url: str | None = None) -> None:
        super().__init__(message)
        self.url = url


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

    try:
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
    except httpx.ConnectError as e:
        raise McpSessionError(
            f"Connection refused — is the MCP server running at {url}? "
            f"({type(e).__name__}: {e})",
            url=url,
        ) from e
    except httpx.ConnectTimeout as e:
        raise McpSessionError(
            f"Connection timed out reaching MCP server at {url} "
            f"(timeout={timeout}). ({type(e).__name__}: {e})",
            url=url,
        ) from e
    except httpx.ReadTimeout as e:
        raise McpSessionError(
            f"Read timed out waiting for MCP server at {url} "
            f"(sse_read_timeout={sse_read_timeout}). "
            f"({type(e).__name__}: {e})",
            url=url,
        ) from e
    except httpx.HTTPStatusError as e:
        raise McpSessionError(
            f"MCP server at {url} returned HTTP {e.response.status_code}. "
            f"({type(e).__name__}: {e})",
            url=url,
        ) from e
    except Exception as e:
        # Unwrap ExceptionGroups so the error message surfaces the real
        # cause (e.g. an HTTP 401) instead of the opaque "unhandled
        # errors in a TaskGroup (1 sub-exception)" wrapper.
        first = ExceptionLogger.get_first_exception(e)
        msg = ExceptionLogger.format_exception_message(e)
        if url not in msg:
            raise McpSessionError(
                f"MCP session failed for {url}: {msg}",
                url=url,
            ) from first
        raise
