"""MCP session pooling — reuse sessions per server URL within a request scope."""

from __future__ import annotations

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


class McpSessionPool:
    """Pools MCP sessions per server URL within a request scope.

    Usage::

        async with McpSessionPool() as pool:
            session = await pool.get_session(config, mcp_callbacks)
            result = await session.call_tool(...)
            # session is reused for subsequent calls to the same URL

    The pool keeps sessions open until ``__aexit__``, which closes them
    all.  This avoids the TCP + TLS + ``initialize()`` cost on every
    tool call when the agent invokes multiple tools from the same server.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _PooledSession] = {}
        self._exit_stack: list[Any] = []

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
            except Exception as e:
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
        """Get or create a pooled session for the given config's URL.

        Sessions are keyed by URL.  Auth headers are part of the config
        and applied at the HTTP transport level, so sessions with
        different auth to the same URL are safe (the auth header is
        sent per-request by httpx, not baked into the session).
        """
        url = config["url"]
        pooled = self._sessions.get(url)
        if pooled is not None:
            return pooled.session

        # Create a new session and keep its context manager alive
        cm = create_mcp_session(config, mcp_callbacks=mcp_callbacks)
        session = await cm.__aenter__()
        self._exit_stack.append(cm)
        await session.initialize()
        self._sessions[url] = _PooledSession(session=session, url=url)
        logger.info("Pooled new MCP session for %s", url)
        return session
