"""Tool list caching and listing — avoids redundant MCP list_tools round-trips."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from mcp import ClientSession
from mcp.types import Tool as MCPTool

from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)

MAX_ITERATIONS = 1000


@dataclass
class _CachedToolList:
    """A cached list_tools result with expiry."""

    tools: list[MCPTool]
    expires_at: float


class ToolListCache:
    """TTL cache for MCP ``list_tools`` results, keyed by cache key.

    Tool schemas rarely change during a user session. Caching avoids
    redundant ``list_tools`` round-trips when the same server is queried
    multiple times (e.g. lazy discovery retry after token refresh).

    Keys should include both the server URL and any auth context
    (e.g. Authorization header) so that users with different
    permissions do not share cached tool lists.
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, _CachedToolList] = {}

    @staticmethod
    def make_key(url: str, *, auth_header: str | None = None) -> str:
        """Build a cache key from the URL and optional auth header."""
        if not auth_header:
            return url
        return f"{url}|{auth_header}"

    def get(self, key: str) -> list[MCPTool] | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._cache[key]
            return None
        return list(entry.tools)

    def put(self, key: str, tools: list[MCPTool]) -> None:
        self._cache[key] = _CachedToolList(
            tools=list(tools),
            expires_at=time.monotonic() + self._ttl,
        )

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()


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


async def list_all_tools_cached(
    session: ClientSession,
    *,
    url: str,
    cache: ToolListCache | None,
    cache_key: str | None = None,
) -> list[MCPTool]:
    """List tools with optional TTL caching.

    If ``cache`` is provided and contains a fresh entry for the key,
    returns the cached result without contacting the MCP server.

    ``cache_key`` overrides the default key (``url``) to allow
    auth-aware caching.
    """
    key = cache_key or url
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            logger.info("Tool list cache hit for %s (%d tools)", url, len(cached))
            return cached

    tools = await list_all_tools(session)

    if cache is not None:
        cache.put(key, tools)

    return tools
