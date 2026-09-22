"""Tests for McpSessionPool's per-key locking and session reuse/eviction."""

import asyncio
from contextlib import AbstractAsyncContextManager
from typing import Any
from unittest.mock import AsyncMock

import pytest
from mcp import ClientSession

from languagemodelcommon.mcp.mcp_client.session import MCPConnectionConfig
from languagemodelcommon.mcp.mcp_client.session_pool import McpSessionPool


class _FakeSessionCm(AbstractAsyncContextManager[ClientSession]):
    """Minimal async CM standing in for the real session's CM."""

    async def __aenter__(self) -> ClientSession:
        return AsyncMock(spec=ClientSession)

    async def __aexit__(self, *args: Any) -> None:
        return None


def _config(*, url: str) -> MCPConnectionConfig:
    return {"url": url, "transport": "streamable_http"}


class TestGetSessionReuse:
    @pytest.mark.asyncio
    async def test_returns_same_session_for_same_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        open_calls = 0

        async def fake_open(
            config: MCPConnectionConfig, *, mcp_callbacks: Any = None
        ) -> tuple[AbstractAsyncContextManager[ClientSession], ClientSession]:
            nonlocal open_calls
            open_calls += 1
            cm = _FakeSessionCm()
            session = await cm.__aenter__()
            return cm, session

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.session_pool.open_initialized_mcp_session",
            fake_open,
        )

        async with McpSessionPool() as pool:
            config = _config(url="https://mcp.example.com")
            first = await pool.get_session(config)
            second = await pool.get_session(config)

        assert first is second
        assert open_calls == 1


class TestEvict:
    @pytest.mark.asyncio
    async def test_evict_closes_session_and_allows_fresh_connect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        open_calls = 0

        async def fake_open(
            config: MCPConnectionConfig, *, mcp_callbacks: Any = None
        ) -> tuple[AbstractAsyncContextManager[ClientSession], ClientSession]:
            nonlocal open_calls
            open_calls += 1
            cm = _FakeSessionCm()
            session = await cm.__aenter__()
            return cm, session

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.session_pool.open_initialized_mcp_session",
            fake_open,
        )

        async with McpSessionPool() as pool:
            config = _config(url="https://mcp.example.com")
            first = await pool.get_session(config)
            await pool.evict(config)
            second = await pool.get_session(config)

        assert first is not second
        assert open_calls == 2


class TestPerKeyLocking:
    @pytest.mark.asyncio
    async def test_slow_connect_for_one_server_does_not_block_another(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BAI-889 review finding: a single pool-wide lock held for the
        entire retrying connect meant a slow/unhealthy server blocked every
        other server's get_session() in the same pool. Locking must be
        scoped per cache key so unrelated, healthy servers aren't held up."""
        slow_started = asyncio.Event()
        release_slow = asyncio.Event()

        async def fake_open(
            config: MCPConnectionConfig, *, mcp_callbacks: Any = None
        ) -> tuple[AbstractAsyncContextManager[ClientSession], ClientSession]:
            if config["url"] == "https://slow.example.com":
                slow_started.set()
                await release_slow.wait()
            cm = _FakeSessionCm()
            session = await cm.__aenter__()
            return cm, session

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.session_pool.open_initialized_mcp_session",
            fake_open,
        )

        async with McpSessionPool() as pool:
            slow_task = asyncio.create_task(
                pool.get_session(_config(url="https://slow.example.com"))
            )
            await slow_started.wait()

            fast_result = await asyncio.wait_for(
                pool.get_session(_config(url="https://fast.example.com")),
                timeout=1.0,
            )
            assert fast_result is not None

            release_slow.set()
            await slow_task
