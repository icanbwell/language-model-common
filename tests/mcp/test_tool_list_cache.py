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
