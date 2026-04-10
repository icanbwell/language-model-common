"""Tests for MCP client performance features: ToolListCache and McpSessionPool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import Tool as MCPTool

from languagemodelcommon.mcp.mcp_client.session_pool import McpSessionPool
from languagemodelcommon.mcp.mcp_client.tool_list_cache import (
    ToolListCache,
    list_all_tools_cached,
)


# ---------- ToolListCache ----------


class TestToolListCache:
    def _make_tool(self, name: str = "test_tool") -> MCPTool:
        return MCPTool(
            name=name,
            description="A test tool",
            inputSchema={"type": "object", "properties": {}},
        )

    def test_cache_miss_returns_none(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        assert cache.get("https://example.com") is None

    def test_cache_hit_returns_tools(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        tools = [self._make_tool()]
        cache.put("https://example.com", tools)
        result = cache.get("https://example.com")
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "test_tool"

    def test_cache_returns_copy(self) -> None:
        """Cached result should be independent of the original list."""
        cache = ToolListCache(ttl_seconds=60)
        tools = [self._make_tool()]
        cache.put("https://example.com", tools)
        tools.append(self._make_tool("extra"))
        result = cache.get("https://example.com")
        assert result is not None
        assert len(result) == 1

    def test_cache_expiry(self) -> None:
        cache = ToolListCache(ttl_seconds=0.0)
        cache.put("https://example.com", [self._make_tool()])
        # TTL of 0 means already expired
        assert cache.get("https://example.com") is None

    def test_cache_invalidate(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        cache.put("https://example.com", [self._make_tool()])
        cache.invalidate("https://example.com")
        assert cache.get("https://example.com") is None

    def test_cache_invalidate_nonexistent(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        cache.invalidate("https://nonexistent.com")  # Should not raise

    def test_cache_clear(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        cache.put("https://a.com", [self._make_tool("a")])
        cache.put("https://b.com", [self._make_tool("b")])
        cache.clear()
        assert cache.get("https://a.com") is None
        assert cache.get("https://b.com") is None

    def test_different_urls_are_independent(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        cache.put("https://a.com", [self._make_tool("a")])
        cache.put("https://b.com", [self._make_tool("b")])
        result_a = cache.get("https://a.com")
        result_b = cache.get("https://b.com")
        assert result_a is not None and result_a[0].name == "a"
        assert result_b is not None and result_b[0].name == "b"


# ---------- list_all_tools_cached ----------


class TestListAllToolsCached:
    def _make_tool(self, name: str = "test_tool") -> MCPTool:
        return MCPTool(
            name=name,
            description="A test tool",
            inputSchema={"type": "object", "properties": {}},
        )

    @pytest.mark.asyncio
    async def test_cache_miss_calls_session(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        tools = [self._make_tool()]
        session = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=MagicMock(tools=tools, nextCursor=None)
        )

        result = await list_all_tools_cached(
            session, url="https://example.com", cache=cache
        )
        assert len(result) == 1
        session.list_tools.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cache_hit_skips_session(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        tools = [self._make_tool()]
        cache.put("https://example.com", tools)

        session = AsyncMock()
        result = await list_all_tools_cached(
            session, url="https://example.com", cache=cache
        )
        assert len(result) == 1
        session.list_tools.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_cache_always_calls_session(self) -> None:
        tools = [self._make_tool()]
        session = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=MagicMock(tools=tools, nextCursor=None)
        )

        result = await list_all_tools_cached(
            session, url="https://example.com", cache=None
        )
        assert len(result) == 1
        session.list_tools.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_populates_cache_after_fetch(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        tools = [self._make_tool()]
        session = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=MagicMock(tools=tools, nextCursor=None)
        )

        await list_all_tools_cached(session, url="https://example.com", cache=cache)

        # Second call should hit cache
        result = await list_all_tools_cached(
            session, url="https://example.com", cache=cache
        )
        assert len(result) == 1
        # list_tools should only have been called once (first call)
        assert session.list_tools.await_count == 1


# ---------- McpSessionPool ----------


class TestMcpSessionPool:
    @pytest.mark.asyncio
    async def test_pool_context_manager(self) -> None:
        async with McpSessionPool() as pool:
            assert pool is not None

    @pytest.mark.asyncio
    async def test_pool_reuses_session_for_same_url(self) -> None:
        """Two get_session calls for the same URL should return the same session."""
        config = {"url": "https://example.com"}
        mock_session = AsyncMock()

        with patch(
            "languagemodelcommon.mcp.mcp_client.session_pool.create_mcp_session"
        ) as mock_create:
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_create.return_value = mock_cm

            async with McpSessionPool() as pool:
                s1 = await pool.get_session(config)
                s2 = await pool.get_session(config)
                assert s1 is s2
                # create_mcp_session should only be called once
                assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_pool_creates_separate_sessions_for_different_urls(self) -> None:
        config_a = {"url": "https://a.com"}
        config_b = {"url": "https://b.com"}

        sessions = {}
        call_count = 0

        with patch(
            "languagemodelcommon.mcp.mcp_client.session_pool.create_mcp_session"
        ) as mock_create:

            def make_cm(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                mock_session = AsyncMock()
                mock_session.name = f"session_{call_count}"
                mock_cm = AsyncMock()
                mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
                mock_cm.__aexit__ = AsyncMock(return_value=None)
                return mock_cm

            mock_create.side_effect = make_cm

            async with McpSessionPool() as pool:
                s1 = await pool.get_session(config_a)
                s2 = await pool.get_session(config_b)
                assert s1 is not s2
                assert call_count == 2
