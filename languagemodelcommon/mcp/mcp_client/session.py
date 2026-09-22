"""MCP session management — creating and connecting to MCP servers."""

import asyncio
import contextlib
import logging
import random
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import timedelta
from typing import Any

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from typing_extensions import NotRequired, TypedDict

from languagemodelcommon.mcp.callbacks import _MCPCallbacks
from languagemodelcommon.utilities.logger.exception_logger import ExceptionLogger
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)

DEFAULT_TIMEOUT = timedelta(seconds=30)
DEFAULT_SSE_READ_TIMEOUT = timedelta(seconds=300)

# BAI-889: a backend MCP server scaling (HPA) or redeploying can reset an
# in-flight connect/initialize handshake before any tool call is sent.
# These defaults bound the added latency to a couple of seconds worst case
# while giving a fresh request a chance to land on a healthy pod.
DEFAULT_SESSION_RETRY_MAX_ATTEMPTS = 3
DEFAULT_SESSION_RETRY_BASE_DELAY_SECONDS = 0.5


class McpSessionError(Exception):
    """Raised when an MCP session cannot be established or fails unexpectedly.

    Wraps lower-level exceptions with actionable context (URL, timeout values,
    HTTP status) so operators can diagnose connectivity problems in production.
    """

    def __init__(self, message: str, *, url: str | None = None) -> None:
        super().__init__(message)
        self.url = url


class McpHttpClientFactory:
    """Protocol-compatible callable for creating httpx2 async clients."""

    def __call__(
        self,
        headers: dict[str, str] | None = None,
        timeout: httpx2.Timeout | None = None,
        auth: httpx2.Auth | None = None,
    ) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
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
    httpx_client_factory = config.get("httpx_client_factory") or McpHttpClientFactory()

    session_kwargs: dict[str, Any] = {}
    if mcp_callbacks is not None:
        if mcp_callbacks.logging_callback is not None:
            session_kwargs["logging_callback"] = mcp_callbacks.logging_callback

    try:
        # mcp>=2.0's streamable_http_client takes a ready-made
        # httpx2.AsyncClient (http_client=) instead of individual
        # headers/timeout/sse_read_timeout kwargs. connect/write/pool keep
        # `timeout`; `read` gets the longer `sse_read_timeout`, since the
        # streamable HTTP connection is a long-lived read that would
        # otherwise be cut short by `timeout`. Built inside this try block
        # (not before it) so a construction-time failure -- a malformed
        # proxy env var, invalid headers, or a caller-supplied factory that
        # raises on its own -- is still wrapped in McpSessionError below,
        # same as a connection-time failure.
        http_client = httpx_client_factory(
            headers=headers,
            timeout=httpx2.Timeout(
                timeout.total_seconds(),
                read=sse_read_timeout.total_seconds(),
            ),
        )
        # streamable_http_client only manages the http_client's lifecycle
        # when it creates one itself (http_client=None) -- since we always
        # pass a pre-built client, we own closing it.
        async with (
            http_client,
            streamable_http_client(
                url,
                http_client=http_client,
            ) as (read, write),
            ClientSession(read, write, **session_kwargs) as session,
        ):
            yield session
    except httpx2.ConnectError as e:
        raise McpSessionError(
            f"Connection refused — is the MCP server running at {url}? "
            f"({type(e).__name__}: {e})",
            url=url,
        ) from e
    except httpx2.ConnectTimeout as e:
        raise McpSessionError(
            f"Connection timed out reaching MCP server at {url} "
            f"(timeout={timeout}). ({type(e).__name__}: {e})",
            url=url,
        ) from e
    except httpx2.ReadTimeout as e:
        raise McpSessionError(
            f"Read timed out waiting for MCP server at {url} "
            f"(sse_read_timeout={sse_read_timeout}). "
            f"({type(e).__name__}: {e})",
            url=url,
        ) from e
    except httpx2.HTTPStatusError as e:
        raise McpSessionError(
            f"MCP server at {url} returned HTTP {e.response.status_code}. "
            f"({type(e).__name__}: {e})",
            url=url,
        ) from e
    except (Exception, BaseExceptionGroup) as e:
        # Unwrap ExceptionGroups so the error message surfaces the real
        # cause (e.g. an HTTP 401) instead of the opaque "unhandled
        # errors in a TaskGroup (1 sub-exception)" wrapper.
        # BaseExceptionGroup must be listed explicitly: anyio's TaskGroup
        # always raises BaseExceptionGroup (not ExceptionGroup), even when
        # all inner exceptions are plain Exceptions, so `except Exception`
        # alone would miss it.
        msg = ExceptionLogger.format_exception_message(e)
        # Use str(e) for the URL guard — not the unwrapped msg.
        # format_exception_message recursively extracts leaf messages
        # which often contain the URL, causing the guard to pass
        # incorrectly and re-raising the raw ExceptionGroup.
        # Chain from e (not first) to preserve the full exception tree
        # so _contains_http_auth_error can traverse all children.
        if url not in str(e):
            raise McpSessionError(
                f"MCP session failed for {url}: {msg}",
                url=url,
            ) from e
        raise


def _compute_session_retry_delay_seconds(
    attempt: int, *, base_delay_seconds: float
) -> float:
    """Exponential backoff with full jitter, keyed off a 0-indexed attempt
    number (the delay before retrying after attempt's failure)."""
    return float(base_delay_seconds * (2**attempt) * random.random())


async def open_initialized_mcp_session(
    config: MCPConnectionConfig,
    *,
    mcp_callbacks: _MCPCallbacks | None = None,
    max_attempts: int = DEFAULT_SESSION_RETRY_MAX_ATTEMPTS,
    base_delay_seconds: float = DEFAULT_SESSION_RETRY_BASE_DELAY_SECONDS,
) -> tuple[AbstractAsyncContextManager[ClientSession], ClientSession]:
    """Open an MCP session and complete its ``initialize()`` handshake,
    retrying transient failures with exponential backoff and full jitter.

    Retries are scoped strictly to session *establishment* — connect and
    ``initialize()`` — which happens before any tool call reaches the
    server, so retrying here is safe even for non-idempotent tools. A
    failure during ``call_tool`` itself must never be retried by this
    function or its callers (BAI-889).

    Returns the entered context manager and the initialized session. The
    caller owns the context manager's lifecycle from here — it must call
    ``await cm.__aexit__(...)`` itself (e.g. in a ``finally`` block, or via
    `McpSessionPool`'s own close/evict path) since this function cannot
    express "yield across retries" as a plain ``@asynccontextmanager``.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        cm = create_mcp_session(config, mcp_callbacks=mcp_callbacks)
        try:
            session = await cm.__aenter__()
            await session.initialize()
        except BaseException as exc:
            with contextlib.suppress(Exception):
                await cm.__aexit__(type(exc), exc, exc.__traceback__)
            # Cancellation (and other non-Exception BaseExceptions) must
            # propagate immediately — retrying here would swallow a
            # shutdown/timeout signal the caller is relying on.
            if not isinstance(exc, Exception):
                raise
            last_exc = exc
            if attempt == max_attempts - 1:
                raise
            delay = _compute_session_retry_delay_seconds(
                attempt, base_delay_seconds=base_delay_seconds
            )
            logger.warning(
                "MCP session establishment attempt %d/%d failed for %s "
                "(retrying in %.2fs): %s",
                attempt + 1,
                max_attempts,
                config.get("url"),
                delay,
                exc,
            )
            await asyncio.sleep(delay)
            continue
        return cm, session

    # Unreachable: the loop above always either returns or raises on its
    # final attempt. Satisfies mypy's control-flow analysis.
    assert last_exc is not None
    raise last_exc
