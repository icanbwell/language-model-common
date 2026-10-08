from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from unittest.mock import AsyncMock, patch

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
            tools=[MCPTool(name="search", input_schema={"type": "object"})],
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
            tools=[MCPTool(name="search", input_schema={"type": "object"})],
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
            tools=[MCPTool(name="search", input_schema={"type": "object"})],
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
            tools=[MCPTool(name="search", input_schema={"type": "object"})],
            fetched_at=time.time(),
        )
        assert await store.get_tools(key=key) is not None

        await asyncio.sleep(0.01)
        await store.clear()

        assert await store.get_tools(key=key) is None


async def _seeded_store(
    *, keys: list[str]
) -> tuple[MemoryStore, McpToolListStore, list[str]]:
    """A store with one single-tool entry per key (tool names tool-0, tool-1, ...),
    with ``_get_all_keys`` left to the caller to patch (MemoryStore can't list keys)."""
    backing_store = MemoryStore(default_collection="mcp-tool-cache")
    store = McpToolListStore(store=backing_store, collection="mcp-tool-cache")
    for i, key in enumerate(keys):
        await store.put_tools(
            key=key,
            tools=[MCPTool(name=f"tool-{i}", input_schema={"type": "object"})],
            fetched_at=time.time(),
        )
    return backing_store, store, keys


_KEYS = ["https://a.example.com", "https://b.example.com", "https://c.example.com"]


class TestGetAllToolsRoundTrips:
    """get_all_tools() batches every entry read into one get_many and reads
    the epoch marker once, instead of one get (and epoch read) per key."""

    @pytest.mark.asyncio
    async def test_get_all_tools_fetches_cleared_at_once_for_multiple_keys(
        self,
    ) -> None:
        _, store, keys = await _seeded_store(keys=_KEYS)

        with (
            patch.object(store, "_get_all_keys", AsyncMock(return_value=keys)),
            patch.object(
                store, "_get_cleared_at", AsyncMock(wraps=store._get_cleared_at)
            ) as cleared_at_spy,
        ):
            tools = await store.get_all_tools()

        assert len(tools) == 3
        cleared_at_spy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_all_tools_uses_one_get_many_not_a_get_per_key(self) -> None:
        backing_store, store, keys = await _seeded_store(keys=_KEYS[:2])

        with (
            patch.object(store, "_get_all_keys", AsyncMock(return_value=keys)),
            patch.object(
                backing_store, "get_many", AsyncMock(wraps=backing_store.get_many)
            ) as get_many_spy,
            patch.object(
                backing_store, "get", AsyncMock(wraps=backing_store.get)
            ) as get_spy,
        ):
            tools = await store.get_all_tools()

        assert sorted(t.name for t in tools) == ["tool-0", "tool-1"]
        get_many_spy.assert_awaited_once()
        # Only the epoch marker is read with get(); no per-key get().
        assert get_spy.await_count == 1

    @pytest.mark.asyncio
    async def test_get_all_tools_returns_empty_without_reading_when_no_keys(
        self,
    ) -> None:
        backing_store, store, _ = await _seeded_store(keys=[])

        with (
            patch.object(store, "_get_all_keys", AsyncMock(return_value=[])),
            patch.object(backing_store, "get_many", AsyncMock()) as get_many_spy,
        ):
            tools = await store.get_all_tools()

        assert tools == []
        get_many_spy.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_all_tools_skips_key_missing_from_the_batch(self) -> None:
        _, store, keys = await _seeded_store(keys=_KEYS[:2])

        with patch.object(
            store,
            "_get_all_keys",
            AsyncMock(return_value=[*keys, "https://gone.example.com"]),
        ):
            tools = await store.get_all_tools()

        assert sorted(t.name for t in tools) == ["tool-0", "tool-1"]

    @pytest.mark.asyncio
    async def test_get_all_tools_returns_empty_and_warns_when_get_many_raises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backing_store, store, keys = await _seeded_store(keys=_KEYS[:2])

        with (
            patch.object(store, "_get_all_keys", AsyncMock(return_value=keys)),
            patch.object(
                backing_store,
                "get_many",
                AsyncMock(side_effect=RuntimeError("mongo down")),
            ),
            caplog.at_level(logging.WARNING, logger="languagemodelcommon"),
        ):
            tools = await store.get_all_tools()

        assert tools == []
        assert "Failed to retrieve all tools from persistent store" in caplog.text

    @pytest.mark.asyncio
    async def test_get_all_tools_skips_entries_fetched_before_last_clear(
        self,
    ) -> None:
        backing_store = MemoryStore(default_collection="mcp-tool-cache")
        store = McpToolListStore(store=backing_store, collection="mcp-tool-cache")
        keys = _KEYS[:2]
        stale_fetched_at = time.time()
        await asyncio.sleep(0.01)
        await store.clear()
        await store.put_tools(
            key=keys[0],
            tools=[MCPTool(name="stale", input_schema={"type": "object"})],
            fetched_at=stale_fetched_at,
        )
        await store.put_tools(
            key=keys[1],
            tools=[MCPTool(name="fresh", input_schema={"type": "object"})],
            fetched_at=time.time(),
        )

        with patch.object(store, "_get_all_keys", AsyncMock(return_value=keys)):
            tools = await store.get_all_tools()

        assert [t.name for t in tools] == ["fresh"]

    @pytest.mark.asyncio
    async def test_get_all_tools_reads_entries_before_epoch_marker(self) -> None:
        """Same entry-before-marker ordering as get_tools: the marker read
        must start only after the batched entry read has finished."""
        backing_store, store, keys = await _seeded_store(keys=_KEYS[:2])
        events: list[str] = []
        real_get_many = backing_store.get_many
        real_get = backing_store.get

        async def recording_get_many(*args: Any, **kwargs: Any) -> Any:
            events.append("entries:start")
            result = await real_get_many(*args, **kwargs)
            events.append("entries:end")
            return result

        async def recording_get(*args: Any, **kwargs: Any) -> Any:
            events.append("epoch:start")
            return await real_get(*args, **kwargs)

        with (
            patch.object(store, "_get_all_keys", AsyncMock(return_value=keys)),
            patch.object(backing_store, "get_many", recording_get_many),
            patch.object(backing_store, "get", recording_get),
        ):
            await store.get_all_tools()

        assert events == ["entries:start", "entries:end", "epoch:start"]


class TestGetToolsReadOrdering:
    """get_tools() reads the epoch marker only after the entry, and only for
    a hit: reading them concurrently could pair a marker read from before a
    clear() with an entry written after it, serving a pre-clear entry."""

    @pytest.mark.asyncio
    async def test_get_tools_reads_epoch_marker_after_entry(self) -> None:
        backing_store, store, _ = await _seeded_store(keys=[_KEYS[0]])
        events: list[str] = []
        real_get = backing_store.get

        async def recording_get(*args: Any, **kwargs: Any) -> Any:
            events.append(f"start:{kwargs['collection']}")
            result = await real_get(*args, **kwargs)
            events.append(f"end:{kwargs['collection']}")
            return result

        with patch.object(backing_store, "get", recording_get):
            tools = await store.get_tools(key=_KEYS[0])

        assert tools is not None
        assert events == [
            "start:mcp-tool-cache",
            "end:mcp-tool-cache",
            "start:mcp-tool-cache__epoch",
            "end:mcp-tool-cache__epoch",
        ]

    @pytest.mark.asyncio
    async def test_get_tools_miss_does_not_read_epoch_marker(self) -> None:
        backing_store, store, _ = await _seeded_store(keys=[])

        with patch.object(
            store, "_get_cleared_at", AsyncMock(wraps=store._get_cleared_at)
        ) as cleared_at_spy:
            tools = await store.get_tools(key="https://absent.example.com")

        assert tools is None
        cleared_at_spy.assert_not_awaited()
