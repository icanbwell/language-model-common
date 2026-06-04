"""Persistent store for MCP tool list caching.

Uses py-key-value-aio's store abstraction to persist tool schemas across
process restarts. Entries never expire — they persist until explicitly
cleared (e.g. via /reload).
"""

import logging
from typing import Any

from mcp.types import Tool as MCPTool

from key_value.aio.stores.base import BaseDestroyCollectionStore, BaseStore


logger = logging.getLogger(__name__)


class McpToolListStore:
    """Persistent MCP tool list store backed by py-key-value-aio.

    Implements ToolListStoreProtocol using any py-key-value-aio BaseStore
    (MongoDBStore, MemoryStore, etc.). Serializes MCPTool objects to dicts
    for storage and deserializes on retrieval.
    """

    def __init__(self, *, store: BaseStore, collection: str) -> None:
        self._store = store
        self._collection = collection

    async def get_tools(self, *, key: str) -> list[MCPTool] | None:
        result: dict[str, Any] | None = await self._store.get(
            key, collection=self._collection
        )
        if result is None:
            return None

        tools_data = result.get("tools")
        if not isinstance(tools_data, list):
            return None

        try:
            return [MCPTool.model_validate(t) for t in tools_data]
        except Exception:
            logger.warning(
                "Failed to deserialize cached tools for key %s, treating as miss",
                key,
            )
            return None

    async def get_all_tools(self) -> list[MCPTool]:
        all_tools: list[MCPTool] = []
        try:
            keys = await self._get_all_keys()
            for key in keys:
                tools = await self.get_tools(key=key)
                if tools:
                    all_tools.extend(tools)
        except Exception:
            logger.warning(
                "Failed to retrieve all tools from persistent store",
                exc_info=True,
            )
        return all_tools

    async def _get_all_keys(self) -> list[str]:
        from key_value.aio.stores.mongodb import MongoDBStore

        if isinstance(self._store, MongoDBStore):
            mongo_collection = self._store._collections_by_name.get(self._collection)
            if mongo_collection is None:
                return []
            cursor = mongo_collection.find({}, {"key": 1})
            return [doc["key"] async for doc in cursor if "key" in doc]
        return []

    async def put_tools(self, *, key: str, tools: list[MCPTool]) -> None:
        value: dict[str, Any] = {
            "tools": [t.model_dump(mode="python") for t in tools],
        }
        await self._store.put(key, value, collection=self._collection)

    async def invalidate(self, *, key: str) -> None:
        await self._store.delete(key, collection=self._collection)

    async def clear(self) -> None:
        if isinstance(self._store, BaseDestroyCollectionStore):
            await self._store.destroy_collection(collection=self._collection)
        else:
            logger.warning(
                "Tool list cache store does not support destroy_collection; "
                "cache clear is a no-op for this backend"
            )
