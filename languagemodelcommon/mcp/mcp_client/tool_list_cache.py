"""Tool list caching and listing — avoids redundant MCP list_tools round-trips."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
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
    Entries may carry a TTL (implementation-defined) and can also be
    cleared explicitly on demand.
    """

    async def get_tools(self, *, key: str) -> list[MCPTool] | None: ...

    async def get_all_tools(self) -> list[MCPTool]: ...

    async def put_tools(
        self, *, key: str, tools: list[MCPTool], fetched_at: float | None = None
    ) -> None: ...

    async def invalidate(self, *, key: str) -> None: ...

    async def clear(self) -> None: ...


@dataclass
class _CachedToolList:
    """A cached list_tools result with optional expiry."""

    tools: list[MCPTool]
    expires_at: float | None
    fetched_at: float | None = None


class ToolListCache:
    """Cache for MCP ``list_tools`` results, keyed by server URL.

    Tool schemas rarely change during runtime. Caching avoids
    redundant ``list_tools`` round-trips when the same server is queried
    multiple times.

    When a persistent ``store`` is provided, all reads and writes go through
    the store exclusively. No in-memory layer is used because production
    runs multiple workers — in-process caches diverge across instances.

    When no store is provided (e.g., tests or single-process dev), falls back
    to an in-memory TTL cache.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        store: ToolListStoreProtocol | None = None,
    ) -> None:
        self._ttl = ttl_seconds
        self._fallback_cache: dict[str, _CachedToolList] = {}
        self._fallback_cleared_at: float = 0.0
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
        """Synchronous cache lookup (fallback only, no-op when store is configured)."""
        if self._store is not None:
            return None
        entry = self._fallback_cache.get(key)
        if entry is None:
            return None
        if entry.expires_at is not None and time.monotonic() > entry.expires_at:
            del self._fallback_cache[key]
            return None
        if (
            entry.fetched_at is not None
            and entry.fetched_at < self._fallback_cleared_at
        ):
            del self._fallback_cache[key]
            return None
        return list(entry.tools)

    async def get_async(self, *, key: str) -> list[MCPTool] | None:
        """Async cache lookup via persistent store (or fallback)."""
        if self._store is not None:
            tools = await self._store.get_tools(key=key)
            if tools is not None:
                logger.info(
                    "Tool list cache hit from persistent store for %s (%d tools)",
                    key,
                    len(tools),
                )
                return list(tools)
            return None

        return self.get(key)

    def put(
        self, key: str, tools: list[MCPTool], *, fetched_at: float | None = None
    ) -> None:
        """Write to fallback cache (no-op when store is configured)."""
        if self._store is not None:
            return
        expires_at = time.monotonic() + self._ttl
        self._fallback_cache[key] = _CachedToolList(
            tools=list(tools),
            expires_at=expires_at,
            fetched_at=fetched_at,
        )

    async def put_async(
        self, *, key: str, tools: list[MCPTool], fetched_at: float | None = None
    ) -> None:
        """Write to persistent store (or fallback).

        ``fetched_at`` should be the time the underlying ``list_tools`` call
        *started* (not when this write happens) — see ``list_all_tools_cached``.
        A slow fetch that started before a concurrent ``clear_async()`` and
        only finishes afterward must not resurrect stale data; both the
        persistent store and the in-memory fallback use ``fetched_at`` to
        detect and reject that case on read.
        """
        resolved_fetched_at = fetched_at if fetched_at is not None else time.time()
        if self._store is not None:
            await self._store.put_tools(
                key=key,
                tools=tools,
                fetched_at=resolved_fetched_at,
            )
        else:
            self.put(key, tools, fetched_at=resolved_fetched_at)

    def invalidate(self, key: str) -> None:
        """Remove from fallback cache (no-op when store is configured)."""
        if self._store is not None:
            return
        self._fallback_cache.pop(key, None)

    async def invalidate_async(self, *, key: str) -> None:
        """Remove from persistent store (or fallback)."""
        if self._store is not None:
            await self._store.invalidate(key=key)
        else:
            self._fallback_cache.pop(key, None)

    def get_all_tool_names(self) -> set[str]:
        """Return tool names from fallback cache (no-op when store is configured).

        When a persistent store is used, callers should use
        ``get_all_tool_names_async`` instead.
        """
        if self._store is not None:
            return set()
        names: set[str] = set()
        for entry in self._fallback_cache.values():
            if entry.expires_at is not None and time.monotonic() > entry.expires_at:
                continue
            if (
                entry.fetched_at is not None
                and entry.fetched_at < self._fallback_cleared_at
            ):
                continue
            for tool in entry.tools:
                names.add(tool.name)
        return names

    async def get_all_tool_names_async(self) -> set[str]:
        """Return all tool names from the persistent store."""
        if self._store is None:
            return self.get_all_tool_names()
        names: set[str] = set()
        tools = await self._store.get_all_tools()
        for tool in tools:
            names.add(tool.name)
        return names

    def clear(self) -> None:
        """Clear fallback cache (no-op when store is configured)."""
        self._fallback_cache.clear()
        self._fallback_cleared_at = time.time()

    async def clear_async(self) -> None:
        """Clear persistent store (or fallback)."""
        if self._store is not None:
            await self._store.clear()
        else:
            self.clear()

    async def reload_async(
        self, *, fetchers: dict[str, Callable[[], Awaitable[list[MCPTool]]]]
    ) -> dict[str, BaseException | None]:
        """Clear the cache, then eagerly repopulate every given key.

        Avoids the cold-cache miss callers would otherwise see on the first
        request after a manual reload (e.g. the ``/reload`` command) by
        refetching immediately rather than waiting for lazy repopulation.
        Preserves the same clear-vs-concurrent-write ordering as
        ``clear_async()``/``put_async()``: each fetcher's start time is
        captured before it runs, so a fetch here that's somehow still
        in flight when a later, unrelated ``clear_async()`` lands is still
        correctly rejected on read.

        Keys are refetched concurrently (matching the ``asyncio.gather``
        pattern ``MCPToolProvider`` already uses for multi-server fetches),
        so total latency is bounded by the slowest server, not the sum of
        all of them. A failure fetching *or writing* one key does not abort
        the others -- failures are reported in the returned mapping
        (key -> exception, or None on success) rather than raised, so one
        unreachable server can't block reload of the rest.
        """
        await self.clear_async()

        async def _refresh_one(
            key: str, fetch: Callable[[], Awaitable[list[MCPTool]]]
        ) -> BaseException | None:
            started_at = time.time()
            try:
                tools = await fetch()
                await self.put_async(key=key, tools=tools, fetched_at=started_at)
            except Exception as exc:
                logger.warning(
                    "Failed to eagerly refetch tool list for %s during reload",
                    key,
                    exc_info=True,
                )
                return exc
            return None

        keys = list(fetchers.keys())
        outcomes = await asyncio.gather(
            *(_refresh_one(key, fetchers[key]) for key in keys)
        )
        return dict(zip(keys, outcomes))


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

    # Captured before the live round trip, not after: a fetch that started
    # before a concurrent clear_async() must be recognizable as stale even if
    # it doesn't finish (and write) until after the clear completes.
    fetched_at = time.time()
    tools = await list_all_tools(session)

    if cache is not None:
        await cache.put_async(key=key, tools=tools, fetched_at=fetched_at)

    return tools
