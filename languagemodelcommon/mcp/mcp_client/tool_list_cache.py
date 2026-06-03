"""Tool list caching and listing — avoids redundant MCP list_tools round-trips."""

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from mcp import ClientSession
from mcp.types import Tool as MCPTool

from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)

MAX_ITERATIONS = 1000


class ToolListStoreProtocol(Protocol):
    """Async persistent backend for tool list caching.

    Implementations store serialized MCP tool lists keyed by server URL.
    The store does not expire entries — callers must explicitly clear.
    """

    async def get_tools(self, *, key: str) -> list[MCPTool] | None: ...

    async def put_tools(self, *, key: str, tools: list[MCPTool]) -> None: ...

    async def invalidate(self, *, key: str) -> None: ...

    async def clear(self) -> None: ...


@dataclass
class _CachedToolList:
    """A cached list_tools result with optional expiry."""

    tools: list[MCPTool]
    expires_at: float | None


class ToolListCache:
    """TTL cache for MCP ``list_tools`` results, keyed by server URL.

    Tool schemas rarely change during runtime. Caching avoids
    redundant ``list_tools`` round-trips when the same server is queried
    multiple times.

    When a persistent ``store`` is provided, the cache writes through to the
    store on put and falls back to the store on in-memory miss. The store is
    never expired — entries persist until explicitly cleared via ``clear_async``.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        store: ToolListStoreProtocol | None = None,
    ) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, _CachedToolList] = {}
        self._store = store

    @staticmethod
    def make_key(url: str, *, auth_header: str | None = None) -> str:
        """Build a cache key from the URL only.

        The auth_header parameter is accepted for backwards compatibility
        but is no longer included in the key. Tool lists are identical
        regardless of which user fetches them.
        """
        return url

    def get(self, key: str) -> list[MCPTool] | None:
        """Synchronous in-memory cache lookup."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.expires_at is not None and time.monotonic() > entry.expires_at:
            del self._cache[key]
            return None
        return list(entry.tools)

    async def get_async(self, *, key: str) -> list[MCPTool] | None:
        """Async cache lookup: in-memory first, then persistent store."""
        result = self.get(key)
        if result is not None:
            return result

        if self._store is not None:
            tools = await self._store.get_tools(key=key)
            if tools is not None:
                self._cache[key] = _CachedToolList(tools=list(tools), expires_at=None)
                logger.info(
                    "Tool list cache hit from persistent store for %s (%d tools)",
                    key,
                    len(tools),
                )
                return list(tools)

        return None

    def put(self, key: str, tools: list[MCPTool]) -> None:
        """Write to in-memory cache (synchronous)."""
        expires_at = time.monotonic() + self._ttl if self._store is None else None
        self._cache[key] = _CachedToolList(
            tools=list(tools),
            expires_at=expires_at,
        )

    async def put_async(self, *, key: str, tools: list[MCPTool]) -> None:
        """Write to both in-memory cache and persistent store."""
        self._cache[key] = _CachedToolList(tools=list(tools), expires_at=None)

        if self._store is not None:
            await self._store.put_tools(key=key, tools=tools)

    def invalidate(self, key: str) -> None:
        """Remove from in-memory cache (synchronous)."""
        self._cache.pop(key, None)

    async def invalidate_async(self, *, key: str) -> None:
        """Remove from both in-memory cache and persistent store."""
        self._cache.pop(key, None)
        if self._store is not None:
            await self._store.invalidate(key=key)

    def get_all_tool_names(self) -> set[str]:
        """Return a set of all tool names currently cached in-memory."""
        names: set[str] = set()
        for entry in self._cache.values():
            if entry.expires_at is not None and time.monotonic() > entry.expires_at:
                continue
            for tool in entry.tools:
                names.add(tool.name)
        return names

    def clear(self) -> None:
        """Clear in-memory cache only (synchronous)."""
        self._cache.clear()

    async def clear_async(self) -> None:
        """Clear both in-memory cache and persistent store."""
        self._cache.clear()
        if self._store is not None:
            await self._store.clear()


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
    """List tools with optional caching.

    If ``cache`` is provided and contains a fresh entry for the key,
    returns the cached result without contacting the MCP server.

    ``cache_key`` overrides the default key (``url``) to allow
    custom cache key strategies.
    """
    key = cache_key or url
    if cache is not None:
        cached = await cache.get_async(key=key)
        if cached is not None:
            logger.info("Tool list cache hit for %s (%d tools)", url, len(cached))
            return cached

    tools = await list_all_tools(session)

    if cache is not None:
        await cache.put_async(key=key, tools=tools)

    return tools
