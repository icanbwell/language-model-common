"""MCP session pooling — reuse sessions per server URL within a request scope."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession

from languagemodelcommon.mcp.callbacks import _MCPCallbacks
from languagemodelcommon.mcp.mcp_client.session import (
    MCPConnectionConfig,
    create_mcp_session,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


@dataclass
class _PooledSession:
    """A live MCP session whose CM lifecycle runs in a dedicated task.

    ``streamablehttp_client`` uses anyio task groups whose cancel scopes
    enforce that ``__aexit__`` is called in the same asyncio task that
    called ``__aenter__``.  To satisfy this, a background task runs the
    CM and keeps it alive until ``close_event`` is set.  The session
    reference is published back to the caller via ``ready_event``.
    """

    url: str
    session: ClientSession = field(init=False)
    _task: asyncio.Task[None] = field(init=False)
    _close_event: asyncio.Event = field(default_factory=asyncio.Event)
    _ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    _error: BaseException | None = field(default=None, init=False)

    async def start(
        self,
        config: MCPConnectionConfig,
        *,
        mcp_callbacks: _MCPCallbacks | None = None,
    ) -> None:
        """Launch the background task and wait until the session is ready."""
        self._task = asyncio.create_task(self._run(config, mcp_callbacks=mcp_callbacks))
        await self._ready_event.wait()
        if self._error is not None:
            raise self._error

    async def _run(
        self,
        config: MCPConnectionConfig,
        *,
        mcp_callbacks: _MCPCallbacks | None = None,
    ) -> None:
        """Enter the session CM, signal readiness, then wait for close."""
        cm: AbstractAsyncContextManager[ClientSession] = create_mcp_session(
            config, mcp_callbacks=mcp_callbacks
        )
        try:
            session = await cm.__aenter__()
            try:
                await session.initialize()
            except BaseException:
                await cm.__aexit__(None, None, None)
                raise
            self.session = session
            self._ready_event.set()
            await self._close_event.wait()
        except BaseException as exc:
            self._error = exc
            self._ready_event.set()
            return
        finally:
            if hasattr(self, "session"):
                try:
                    await cm.__aexit__(None, None, None)
                except Exception as e:
                    logger.warning("Error closing MCP session for %s: %s", self.url, e)

    async def close(self) -> None:
        """Signal the background task to exit and wait for it."""
        self._close_event.set()
        try:
            await self._task
        except Exception as e:
            logger.warning("Error in session task for %s: %s", self.url, e)


class McpSessionPool:
    """Pools MCP sessions per server URL and headers within a request scope.

    Usage::

        async with McpSessionPool() as pool:
            session = await pool.get_session(config, mcp_callbacks)
            result = await session.call_tool(...)
            # session is reused for subsequent calls with the same URL + headers

    The pool keeps sessions open until ``__aexit__``, which closes them
    all.  This avoids the TCP + TLS + ``initialize()`` cost on every
    tool call when the agent invokes multiple tools from the same server.

    Sessions are keyed by ``(url, headers)`` because the underlying
    ``httpx.AsyncClient`` is created once at session-open time with
    fixed default headers.  Different auth tokens to the same URL
    therefore require separate sessions.

    Each session's context manager lifecycle runs in a dedicated asyncio
    task so that ``__aenter__`` and ``__aexit__`` execute in the same
    task — required by anyio cancel scopes used inside
    ``streamablehttp_client``.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _PooledSession] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _cache_key(config: MCPConnectionConfig) -> str:
        """Derive a pool key from the config's URL and headers."""
        url = config["url"]
        headers = config.get("headers")
        if not headers:
            return url
        # Sort for deterministic key regardless of dict insertion order
        sorted_items = sorted(headers.items())
        return f"{url}|{sorted_items}"

    async def __aenter__(self) -> McpSessionPool:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        errors: list[str] = []
        for pooled in reversed(list(self._sessions.values())):
            try:
                await pooled.close()
            except BaseException as e:
                errors.append(
                    f"MCP session failed for {pooled.url}: {type(e).__name__}: {e}"
                )
        self._sessions.clear()
        if errors:
            logger.warning(
                "Errors closing %d pooled MCP sessions: %s",
                len(errors),
                "; ".join(errors),
            )

    async def get_session(
        self,
        config: MCPConnectionConfig,
        *,
        mcp_callbacks: _MCPCallbacks | None = None,
    ) -> ClientSession:
        """Get or create a pooled session for the given config.

        Sessions are keyed by ``(url, headers)`` because the underlying
        ``httpx.AsyncClient`` is created once with fixed default headers.
        Different auth tokens to the same URL get separate sessions.
        """
        key = self._cache_key(config)
        url = config["url"]

        pooled = self._sessions.get(key)
        if pooled is not None:
            return pooled.session

        async with self._lock:
            pooled = self._sessions.get(key)
            if pooled is not None:
                return pooled.session

            pooled = _PooledSession(url=url)
            await pooled.start(config, mcp_callbacks=mcp_callbacks)
            self._sessions[key] = pooled
            logger.info("Pooled new MCP session for %s", url)
            return pooled.session

    async def evict(self, config: MCPConnectionConfig) -> None:
        """Remove and close the pooled session for *config*, if any.

        Call this when a session is known to be broken (e.g. after a
        ``call_tool`` failure) so the next ``get_session`` creates a
        fresh connection instead of reusing the broken one.
        """
        key = self._cache_key(config)
        async with self._lock:
            pooled = self._sessions.pop(key, None)
            if pooled is None:
                return
            try:
                await pooled.close()
            except Exception as e:
                logger.warning(
                    "Error closing evicted session for %s: %s", pooled.url, e
                )
        logger.info("Evicted pooled MCP session for %s", pooled.url)
