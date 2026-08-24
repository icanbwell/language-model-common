from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import AsyncMock

import pytest
from mcp.types import Tool as MCPTool

from key_value.aio.stores.base import BaseDestroyCollectionStore, BaseStore
from key_value.aio.stores.memory import MemoryStore

from languagemodelcommon.mcp.mcp_client.mcp_tool_list_store import McpToolListStore


class TestClearResilience:
    """McpToolListStore.clear() must tolerate an uninitialized collection.

    py-key-value-aio's MongoDBStore lazy-registers collections in
    `_collections_by_name` on first read/write. Calling destroy_collection
    before any read/write raises KeyError. /reload must not surface that
    as `An error occurred processing your request. (Code: 102)`.
    """

    @pytest.mark.asyncio
    async def test_clear_swallows_keyerror_when_collection_never_initialized(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backing_store = AsyncMock(spec=BaseDestroyCollectionStore)
        backing_store._setup_collection_complete = {}
        backing_store.destroy_collection.side_effect = KeyError("mcp-tool-cache")

        store = McpToolListStore(store=backing_store, collection="mcp-tool-cache")

        with caplog.at_level(logging.DEBUG, logger="languagemodelcommon"):
            await store.clear()

        backing_store.destroy_collection.assert_awaited_once_with(
            collection="mcp-tool-cache"
        )
        assert backing_store._setup_collection_complete == {}

    @pytest.mark.asyncio
    async def test_clear_destroys_collection_when_initialized(self) -> None:
        backing_store = AsyncMock(spec=BaseDestroyCollectionStore)
        backing_store._setup_collection_complete = {"mcp-tool-cache": True}

        store = McpToolListStore(store=backing_store, collection="mcp-tool-cache")

        await store.clear()

        backing_store.destroy_collection.assert_awaited_once_with(
            collection="mcp-tool-cache"
        )
        assert backing_store._setup_collection_complete["mcp-tool-cache"] is False


class TestPutToolsTtl:
    """put_tools must forward the configured TTL so entries expire —
    otherwise a server's tool list can never be rediscovered without an
    explicit /reload, even after the server's actual tools change.
    """

    @pytest.mark.asyncio
    async def test_put_tools_forwards_configured_ttl(self) -> None:
        backing_store = AsyncMock(spec=BaseStore)
        store = McpToolListStore(
            store=backing_store, collection="mcp-tool-cache", ttl_seconds=3600.0
        )

        await store.put_tools(key="https://mcp.example.com", tools=[])

        backing_store.put.assert_awaited_once()
        _, kwargs = backing_store.put.call_args
        assert kwargs["ttl"] == 3600.0
        assert kwargs["collection"] == "mcp-tool-cache"

    @pytest.mark.asyncio
    async def test_put_tools_defaults_to_no_ttl(self) -> None:
        """Backward compatible: omitting ttl_seconds means no expiry."""
        backing_store = AsyncMock(spec=BaseStore)
        store = McpToolListStore(store=backing_store, collection="mcp-tool-cache")

        await store.put_tools(
            key="https://mcp.example.com",
            tools=[MCPTool(name="search", inputSchema={"type": "object"})],
        )

        backing_store.put.assert_awaited_once()
        _, kwargs = backing_store.put.call_args
        assert kwargs["ttl"] is None

    @pytest.mark.parametrize("non_positive_ttl", [0.0, -1.0])
    @pytest.mark.asyncio
    async def test_non_positive_ttl_is_treated_as_no_expiry(
        self, non_positive_ttl: float
    ) -> None:
        """A non-positive TTL (e.g. an operator setting
        MCP_TOOLS_METADATA_CACHE_TTL_SECONDS=0 to try to disable caching)
        must not be forwarded as-is: py-key-value-aio's underlying store
        raises InvalidTTLError for ttl <= 0, which would otherwise crash
        every put_tools call and break MCP tool discovery entirely."""
        backing_store = AsyncMock(spec=BaseStore)
        store = McpToolListStore(
            store=backing_store,
            collection="mcp-tool-cache",
            ttl_seconds=non_positive_ttl,
        )

        await store.put_tools(key="https://mcp.example.com", tools=[])

        backing_store.put.assert_awaited_once()
        _, kwargs = backing_store.put.call_args
        assert kwargs["ttl"] is None


class TestClearVsConcurrentWriteRace:
    """A fetch that started before clear() but writes after it must not
    resurrect stale data.

    This is the exact bug seen in production: /reload cleared the cache,
    but a resolution already in flight (started before the clear) wrote its
    stale result back afterward, so the next reader still saw the old tool
    list. put_tools() now records when the *fetch* started (fetched_at);
    clear() stamps a cleared_at marker in a separate collection (so it
    survives destroy_collection); get_tools() rejects any entry whose
    fetched_at predates the most recent cleared_at.
    """

    @pytest.mark.asyncio
    async def test_get_tools_rejects_entry_fetched_before_last_clear(self) -> None:
        backing_store = MemoryStore(default_collection="mcp-tool-cache")
        store = McpToolListStore(store=backing_store, collection="mcp-tool-cache")
        key = "https://mcp.example.com"

        fetch_started_at = time.time()
        await asyncio.sleep(0.01)  # simulate the live list_tools() round trip
        await store.clear()  # cleared_at is now > fetch_started_at

        # Simulates a slow fetch that began before clear() but only writes
        # (and lands) afterward.
        await store.put_tools(
            key=key,
            tools=[MCPTool(name="search", inputSchema={"type": "object"})],
            fetched_at=fetch_started_at,
        )

        assert await store.get_tools(key=key) is None

    @pytest.mark.asyncio
    async def test_get_tools_accepts_entry_fetched_after_last_clear(self) -> None:
        backing_store = MemoryStore(default_collection="mcp-tool-cache")
        store = McpToolListStore(store=backing_store, collection="mcp-tool-cache")
        key = "https://mcp.example.com"

        await store.clear()
        fetch_started_at = time.time()

        await store.put_tools(
            key=key,
            tools=[MCPTool(name="search", inputSchema={"type": "object"})],
            fetched_at=fetch_started_at,
        )

        tools = await store.get_tools(key=key)
        assert tools is not None
        assert [t.name for t in tools] == ["search"]

    @pytest.mark.asyncio
    async def test_clear_stamps_epoch_marker_even_without_destroy_collection_support(
        self,
    ) -> None:
        """Backends without destroy_collection previously made clear() a
        total no-op. The epoch marker now makes clear() effective even
        there, since staleness is rejected on read rather than relying on
        physical deletion.
        """
        backing_store = AsyncMock(spec=BaseStore)
        store = McpToolListStore(store=backing_store, collection="mcp-tool-cache")

        await store.clear()

        backing_store.put.assert_awaited_once()
        args, kwargs = backing_store.put.call_args
        assert args[0] == "cleared_at"
        assert kwargs["collection"] == "mcp-tool-cache__epoch"
        assert isinstance(args[1]["cleared_at"], float)

    @pytest.mark.asyncio
    async def test_second_clear_invalidates_entries_written_after_first_clear(
        self,
    ) -> None:
        """The marker lives in a separate collection from the cached tool
        lists (so destroy_collection(self._collection) can't wipe it) and
        each clear() advances it — a second clear must invalidate entries
        that looked fresh relative to the first.
        """
        backing_store = MemoryStore(default_collection="mcp-tool-cache")
        store = McpToolListStore(store=backing_store, collection="mcp-tool-cache")
        key = "https://mcp.example.com"

        await store.clear()
        await store.put_tools(
            key=key,
            tools=[MCPTool(name="search", inputSchema={"type": "object"})],
            fetched_at=time.time(),
        )
        assert await store.get_tools(key=key) is not None

        await asyncio.sleep(0.01)
        await store.clear()

        assert await store.get_tools(key=key) is None
