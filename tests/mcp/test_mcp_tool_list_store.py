from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from key_value.aio.stores.base import BaseDestroyCollectionStore

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
