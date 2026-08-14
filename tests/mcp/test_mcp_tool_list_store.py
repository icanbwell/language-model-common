from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest
from mcp.types import Tool as MCPTool

from key_value.aio.stores.base import BaseDestroyCollectionStore, BaseStore

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
