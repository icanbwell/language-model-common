from __future__ import annotations

import time

from mcp.types import Tool as MCPTool

from languagemodelcommon.mcp.mcp_client.tool_list_cache import ToolListCache


def _make_mcp_tool(*, name: str) -> MCPTool:
    return MCPTool(name=name, inputSchema={})


class TestGetAllToolNames:
    def test_returns_empty_when_cache_is_empty(self) -> None:
        cache = ToolListCache(ttl_seconds=300)
        assert cache.get_all_tool_names() == set()

    def test_returns_names_from_single_server(self) -> None:
        cache = ToolListCache(ttl_seconds=300)
        tools = [_make_mcp_tool(name="tool-a"), _make_mcp_tool(name="tool-b")]
        cache.put("http://server1/", tools)

        assert cache.get_all_tool_names() == {"tool-a", "tool-b"}

    def test_returns_names_from_multiple_servers(self) -> None:
        cache = ToolListCache(ttl_seconds=300)
        cache.put("http://server1/", [_make_mcp_tool(name="tool-a")])
        cache.put(
            "http://server2/",
            [_make_mcp_tool(name="tool-b"), _make_mcp_tool(name="tool-c")],
        )

        assert cache.get_all_tool_names() == {"tool-a", "tool-b", "tool-c"}

    def test_deduplicates_tool_names_across_servers(self) -> None:
        cache = ToolListCache(ttl_seconds=300)
        cache.put("http://server1/", [_make_mcp_tool(name="shared-tool")])
        cache.put("http://server2/", [_make_mcp_tool(name="shared-tool")])

        assert cache.get_all_tool_names() == {"shared-tool"}

    def test_excludes_expired_entries(self) -> None:
        cache = ToolListCache(ttl_seconds=0.001)
        cache.put("http://server1/", [_make_mcp_tool(name="expired-tool")])
        time.sleep(0.01)

        assert cache.get_all_tool_names() == set()

    def test_returns_empty_after_clear(self) -> None:
        cache = ToolListCache(ttl_seconds=300)
        cache.put("http://server1/", [_make_mcp_tool(name="tool-a")])
        cache.clear()

        assert cache.get_all_tool_names() == set()


class TestClearVsConcurrentWriteRaceFallback:
    """The in-memory fallback must reject the same race window that
    McpToolListStore rejects: a fetch that started before clear() but only
    writes (via put_async) afterward must not resurrect stale data."""

    def test_get_rejects_entry_fetched_before_last_clear(self) -> None:
        cache = ToolListCache(ttl_seconds=300)
        fetch_started_at = time.time()
        time.sleep(0.01)
        cache.clear()  # _fallback_cleared_at is now > fetch_started_at

        # Simulates a slow fetch that began before clear() but only writes
        # (and lands) afterward.
        cache.put(
            "http://server1/",
            [_make_mcp_tool(name="stale-tool")],
            fetched_at=fetch_started_at,
        )

        assert cache.get("http://server1/") is None
        assert cache.get_all_tool_names() == set()

    def test_get_accepts_entry_fetched_after_last_clear(self) -> None:
        cache = ToolListCache(ttl_seconds=300)
        cache.clear()
        fetch_started_at = time.time()

        cache.put(
            "http://server1/",
            [_make_mcp_tool(name="fresh-tool")],
            fetched_at=fetch_started_at,
        )

        assert cache.get("http://server1/") is not None
        assert cache.get_all_tool_names() == {"fresh-tool"}

    async def test_put_async_fallback_rejects_stale_fetch_after_clear(self) -> None:
        cache = ToolListCache(ttl_seconds=300)
        fetch_started_at = time.time()
        time.sleep(0.01)
        await cache.clear_async()

        await cache.put_async(
            key="http://server1/",
            tools=[_make_mcp_tool(name="stale-tool")],
            fetched_at=fetch_started_at,
        )

        assert await cache.get_async(key="http://server1/") is None


class TestReloadAsync:
    """reload_async() clears the cache and eagerly repopulates it, so the
    next request after a manual reload doesn't see a cold-cache miss."""

    async def test_repopulates_every_key_from_its_fetcher(self) -> None:
        cache = ToolListCache(ttl_seconds=300)
        cache.put("http://stale-server/", [_make_mcp_tool(name="stale-tool")])

        async def fetch_a() -> list[MCPTool]:
            return [_make_mcp_tool(name="tool-a")]

        async def fetch_b() -> list[MCPTool]:
            return [_make_mcp_tool(name="tool-b")]

        results = await cache.reload_async(
            fetchers={
                "http://server-a/": fetch_a,
                "http://server-b/": fetch_b,
            }
        )

        assert results == {"http://server-a/": None, "http://server-b/": None}
        assert await cache.get_async(key="http://server-a/") == [
            _make_mcp_tool(name="tool-a")
        ]
        assert await cache.get_async(key="http://server-b/") == [
            _make_mcp_tool(name="tool-b")
        ]
        # The pre-existing entry was wiped by the clear, not merely
        # overwritten -- it wasn't among the fetchers, so it must be gone.
        assert await cache.get_async(key="http://stale-server/") is None

    async def test_one_fetcher_failure_does_not_block_the_others(self) -> None:
        cache = ToolListCache(ttl_seconds=300)

        async def fetch_ok() -> list[MCPTool]:
            return [_make_mcp_tool(name="tool-ok")]

        async def fetch_broken() -> list[MCPTool]:
            raise ConnectionError("server unreachable")

        results = await cache.reload_async(
            fetchers={
                "http://ok-server/": fetch_ok,
                "http://broken-server/": fetch_broken,
            }
        )

        assert results["http://ok-server/"] is None
        assert isinstance(results["http://broken-server/"], ConnectionError)
        assert await cache.get_async(key="http://ok-server/") == [
            _make_mcp_tool(name="tool-ok")
        ]
        assert await cache.get_async(key="http://broken-server/") is None

    async def test_empty_fetchers_just_clears(self) -> None:
        cache = ToolListCache(ttl_seconds=300)
        cache.put("http://server1/", [_make_mcp_tool(name="tool-a")])

        results = await cache.reload_async(fetchers={})

        assert results == {}
        assert cache.get_all_tool_names() == set()
