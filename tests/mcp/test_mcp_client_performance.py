"""Tests for MCP client performance features: ToolListCache and McpSessionPool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import Tool as MCPTool

from languagemodelcommon.mcp.mcp_client.session import MCPConnectionConfig
from languagemodelcommon.mcp.mcp_client.session_pool import McpSessionPool
from languagemodelcommon.mcp.mcp_client.tool_list_cache import (
    ToolListCache,
    list_all_tools_cached,
)


def _make_tool(name: str = "test_tool") -> MCPTool:
    return MCPTool(
        name=name,
        description="A test tool",
        inputSchema={"type": "object", "properties": {}},
    )


# ---------- ToolListCache ----------


class TestToolListCache:
    def test_cache_miss_returns_none(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        assert cache.get("https://example.com") is None

    def test_cache_hit_returns_tools(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        cache.put("https://example.com", [_make_tool()])
        result = cache.get("https://example.com")
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "test_tool"

    def test_put_copies_input_list(self) -> None:
        """Mutating the original list after put should not affect the cache."""
        cache = ToolListCache(ttl_seconds=60)
        tools = [_make_tool()]
        cache.put("https://example.com", tools)
        tools.append(_make_tool("extra"))
        result = cache.get("https://example.com")
        assert result is not None
        assert len(result) == 1

    def test_get_returns_defensive_copy(self) -> None:
        """Mutating the returned list should not affect the cache."""
        cache = ToolListCache(ttl_seconds=60)
        cache.put("https://example.com", [_make_tool()])
        first = cache.get("https://example.com")
        assert first is not None
        first.append(_make_tool("extra"))
        second = cache.get("https://example.com")
        assert second is not None
        assert len(second) == 1

    def test_cache_expiry(self) -> None:
        cache = ToolListCache(ttl_seconds=0.0)
        cache.put("https://example.com", [_make_tool()])
        # TTL of 0 means already expired
        assert cache.get("https://example.com") is None

    def test_cache_invalidate(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        cache.put("https://example.com", [_make_tool()])
        cache.invalidate("https://example.com")
        assert cache.get("https://example.com") is None

    def test_cache_invalidate_nonexistent(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        cache.invalidate("https://nonexistent.com")  # Should not raise

    def test_cache_clear(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        cache.put("https://a.com", [_make_tool("a")])
        cache.put("https://b.com", [_make_tool("b")])
        cache.clear()
        assert cache.get("https://a.com") is None
        assert cache.get("https://b.com") is None

    def test_different_keys_are_independent(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        cache.put("https://a.com", [_make_tool("a")])
        cache.put("https://b.com", [_make_tool("b")])
        result_a = cache.get("https://a.com")
        result_b = cache.get("https://b.com")
        assert result_a is not None and result_a[0].name == "a"
        assert result_b is not None and result_b[0].name == "b"


class TestToolListCacheMakeKey:
    def test_url_only(self) -> None:
        assert ToolListCache.make_key("https://example.com") == "https://example.com"

    def test_url_with_auth_header(self) -> None:
        key = ToolListCache.make_key(
            "https://example.com", auth_header="Bearer token123"
        )
        assert key == "https://example.com|Bearer token123"

    def test_none_auth_header_returns_url(self) -> None:
        key = ToolListCache.make_key("https://example.com", auth_header=None)
        assert key == "https://example.com"

    def test_empty_auth_header_returns_url(self) -> None:
        key = ToolListCache.make_key("https://example.com", auth_header="")
        assert key == "https://example.com"

    def test_different_auth_headers_produce_different_keys(self) -> None:
        key_a = ToolListCache.make_key(
            "https://example.com", auth_header="Bearer user_a"
        )
        key_b = ToolListCache.make_key(
            "https://example.com", auth_header="Bearer user_b"
        )
        assert key_a != key_b


# ---------- list_all_tools_cached ----------


class TestListAllToolsCached:
    @pytest.mark.asyncio
    async def test_cache_miss_calls_session(self) -> None:
        cache = ToolListCache(ttl_seconds=60)
        tools = [_make_tool()]
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
        cache.put("https://example.com", [_make_tool()])

        session = AsyncMock()
        result = await list_all_tools_cached(
            session, url="https://example.com", cache=cache
        )
        assert len(result) == 1
        session.list_tools.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_cache_always_calls_session(self) -> None:
        tools = [_make_tool()]
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
        tools = [_make_tool()]
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

    @pytest.mark.asyncio
    async def test_cache_key_override(self) -> None:
        """When cache_key is provided, it is used instead of url."""
        cache = ToolListCache(ttl_seconds=60)
        tools = [_make_tool()]
        session = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=MagicMock(tools=tools, nextCursor=None)
        )

        await list_all_tools_cached(
            session,
            url="https://example.com",
            cache=cache,
            cache_key="https://example.com|Bearer token",
        )

        # Lookup with url alone should miss (different key)
        assert cache.get("https://example.com") is None
        # Lookup with the cache_key should hit
        assert cache.get("https://example.com|Bearer token") is not None


# ---------- McpSessionPool ----------


def _mock_create_mcp_session() -> tuple[AsyncMock, AsyncMock]:
    """Return (mock_create, mock_session) for patching create_mcp_session."""
    mock_session = AsyncMock()
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    mock_create = MagicMock(return_value=mock_cm)
    return mock_create, mock_session


class TestMcpSessionPool:
    @pytest.mark.asyncio
    async def test_pool_reuses_session_for_same_config(self) -> None:
        """Two get_session calls for the same config return the same session."""
        config: MCPConnectionConfig = {"url": "https://example.com"}
        mock_create, mock_session = _mock_create_mcp_session()

        with patch(
            "languagemodelcommon.mcp.mcp_client.session_pool.create_mcp_session",
            mock_create,
        ):
            async with McpSessionPool() as pool:
                s1 = await pool.get_session(config)
                s2 = await pool.get_session(config)
                assert s1 is s2
                assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_pool_creates_separate_sessions_for_different_urls(self) -> None:
        config_a: MCPConnectionConfig = {"url": "https://a.com"}
        config_b: MCPConnectionConfig = {"url": "https://b.com"}

        call_count = 0

        with patch(
            "languagemodelcommon.mcp.mcp_client.session_pool.create_mcp_session"
        ) as mock_create:

            def make_cm(*args: object, **kwargs: object) -> AsyncMock:
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

    @pytest.mark.asyncio
    async def test_pool_creates_separate_sessions_for_different_headers(self) -> None:
        """Same URL but different auth headers should yield different sessions."""
        config_a: MCPConnectionConfig = {
            "url": "https://example.com",
            "headers": {"Authorization": "Bearer user_a"},
        }
        config_b: MCPConnectionConfig = {
            "url": "https://example.com",
            "headers": {"Authorization": "Bearer user_b"},
        }

        call_count = 0

        with patch(
            "languagemodelcommon.mcp.mcp_client.session_pool.create_mcp_session"
        ) as mock_create:

            def make_cm(*args: object, **kwargs: object) -> AsyncMock:
                nonlocal call_count
                call_count += 1
                mock_session = AsyncMock()
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

    @pytest.mark.asyncio
    async def test_evict_removes_session_and_closes_transport(self) -> None:
        """After evict, the next get_session creates a fresh session."""
        config: MCPConnectionConfig = {"url": "https://example.com"}
        call_count = 0

        with patch(
            "languagemodelcommon.mcp.mcp_client.session_pool.create_mcp_session"
        ) as mock_create:

            def make_cm(*args: object, **kwargs: object) -> AsyncMock:
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
                s1 = await pool.get_session(config)
                assert call_count == 1

                await pool.evict(config)

                s2 = await pool.get_session(config)
                assert call_count == 2
                assert s1 is not s2

    @pytest.mark.asyncio
    async def test_evict_nonexistent_is_noop(self) -> None:
        config: MCPConnectionConfig = {"url": "https://example.com"}
        async with McpSessionPool() as pool:
            await pool.evict(config)  # Should not raise

    @pytest.mark.asyncio
    async def test_initialize_failure_does_not_leak_transport(self) -> None:
        """If session.initialize() fails, the CM should still be cleaned up."""
        config: MCPConnectionConfig = {"url": "https://example.com"}
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(side_effect=RuntimeError("init failed"))

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "languagemodelcommon.mcp.mcp_client.session_pool.create_mcp_session",
            return_value=mock_cm,
        ):
            async with McpSessionPool() as pool:
                with pytest.raises(RuntimeError, match="init failed"):
                    await pool.get_session(config)

                # CM should have been cleaned up
                mock_cm.__aexit__.assert_awaited_once()
