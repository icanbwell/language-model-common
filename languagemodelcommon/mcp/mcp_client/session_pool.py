"""MCP session pooling — reuse sessions per server URL within a request scope."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
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
    """A live MCP session held open for reuse."""

    session: ClientSession
    url: str
    cache_key: str


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
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _PooledSession] = {}
        self._exit_stack: list[Any] = []
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
        # Close all pooled sessions in reverse order
        errors: list[BaseException] = []
        for cm in reversed(self._exit_stack):
            try:
                await cm.__aexit__(None, None, None)
            except BaseException as e:
                errors.append(e)
        self._sessions.clear()
        self._exit_stack.clear()
        if errors:
            logger.warning(
                "Errors closing %d pooled MCP sessions: %s",
                len(errors),
                "; ".join(str(e) for e in errors),
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

        # Fast path: check without lock for already-pooled sessions
        pooled = self._sessions.get(key)
        if pooled is not None:
            return pooled.session

        async with self._lock:
            # Re-check after acquiring the lock — another coroutine may
            # have created the session while we were waiting.
            pooled = self._sessions.get(key)
            if pooled is not None:
                return pooled.session

            # Create a new session and keep its context manager alive.
            # Enter the CM first, then initialize.  If initialize fails,
            # clean up the CM so we don't leak a transport connection.
            cm = create_mcp_session(config, mcp_callbacks=mcp_callbacks)
            session = await cm.__aenter__()
            try:
                await session.initialize()
            except BaseException:
                await cm.__aexit__(None, None, None)
                raise
            self._exit_stack.append(cm)
            self._sessions[key] = _PooledSession(
                session=session, url=url, cache_key=key
            )
            logger.info("Pooled new MCP session for %s", url)
            return session

    async def evict(self, config: MCPConnectionConfig) -> None:
        """Remove and close the pooled session for *config*, if any.

        Call this when a session is known to be broken (e.g. after a
        ``call_tool`` failure) so the next ``get_session`` creates a
        fresh connection instead of reusing the broken one.
        """
        key = self._cache_key(config)
        url = config["url"]
        async with self._lock:
            pooled = self._sessions.pop(key, None)
            if pooled is None:
                return
            # Find and remove the matching CM from the exit stack
            for i, cm in enumerate(self._exit_stack):
                try:
                    await cm.__aexit__(None, None, None)
                except Exception as e:
                    logger.warning("Error closing evicted session for %s: %s", url, e)
                del self._exit_stack[i]
                break
        logger.info("Evicted pooled MCP session for %s", url)
