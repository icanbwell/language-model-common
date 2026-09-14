"""Persistent store for MCP tool list caching.

Uses py-key-value-aio's store abstraction to persist tool schemas across
process restarts. Entries expire after ``ttl_seconds`` (so a server's tool
list is periodically rediscovered) and can also be cleared early on demand
(e.g. via /reload).
"""

import logging
import time
from typing import Any

from mcp.types import Tool as MCPTool

from key_value.aio.stores.base import BaseDestroyCollectionStore, BaseStore

logger = logging.getLogger(__name__)


class McpToolListStore:
    """Persistent MCP tool list store backed by py-key-value-aio.

    Implements ToolListStoreProtocol using any py-key-value-aio BaseStore
    (MongoDBStore, MemoryStore, etc.). Serializes MCPTool objects to dicts
    for storage and deserializes on retrieval.

    Guards against a clear()-vs-concurrent-write race: this store is shared
    across every pod and every request (it's keyed by server URL, not by
    caller), so a resolution that started fetching before a clear() call but
    only finishes (and writes) afterward would otherwise silently resurrect
    stale data right after an operator explicitly cleared it. Every write
    records the time its *fetch* started (``fetched_at``, supplied by the
    caller — see ``put_tools``); clear() stamps a ``cleared_at`` marker that
    survives the collection wipe, and reads reject any entry whose
    ``fetched_at`` predates the most recent ``cleared_at``.
    """

    SCHEMA_VERSION = 1
    _EPOCH_KEY = "cleared_at"

    def __init__(
        self,
        *,
        store: BaseStore,
        collection: str,
        ttl_seconds: float | None = None,
    ) -> None:
        self._store = store
        self._collection = collection
        # Separate collection so the epoch marker survives clear()'s
        # destroy_collection(self._collection) call.
        self._epoch_collection = f"{collection}__epoch"
        # A non-positive value (e.g. an operator setting 0, a natural way to
        # try to disable caching) would otherwise crash every put_tools call:
        # py-key-value-aio's BaseStore.put raises InvalidTTLError for
        # ttl <= 0. Treat it as "no expiry" instead of propagating a config
        # value that breaks tool discovery for every server.
        self._ttl_seconds = ttl_seconds if ttl_seconds and ttl_seconds > 0 else None

    async def _get_cleared_at(self) -> float:
        result: dict[str, Any] | None = await self._store.get(
            self._EPOCH_KEY, collection=self._epoch_collection
        )
        if result is None:
            return 0.0
        cleared_at = result.get("cleared_at")
        return float(cleared_at) if isinstance(cleared_at, (int, float)) else 0.0

    async def get_tools(
        self, *, key: str, cleared_at: float | None = None
    ) -> list[MCPTool] | None:
        """Look up a cached tool list.

        ``cleared_at`` lets a bulk caller (``get_all_tools``) supply an
        already-fetched epoch marker instead of every call re-querying the
        epoch collection.
        """
        result: dict[str, Any] | None = await self._store.get(
            key, collection=self._collection
        )
        if result is None:
            return None

        if result.get("schema_version") != self.SCHEMA_VERSION:
            return None

        tools_data = result.get("tools")
        if not isinstance(tools_data, list):
            return None

        fetched_at = result.get("fetched_at")
        if isinstance(fetched_at, (int, float)):
            effective_cleared_at = (
                cleared_at if cleared_at is not None else await self._get_cleared_at()
            )
            if fetched_at < effective_cleared_at:
                logger.info(
                    "Cached tools for key %s were fetched before the last clear(); "
                    "treating as miss",
                    key,
                )
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
            cleared_at = await self._get_cleared_at()
            for key in keys:
                tools = await self.get_tools(key=key, cleared_at=cleared_at)
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

    async def put_tools(
        self, *, key: str, tools: list[MCPTool], fetched_at: float | None = None
    ) -> None:
        value: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "tools": [t.model_dump(mode="python") for t in tools],
            "fetched_at": fetched_at if fetched_at is not None else time.time(),
        }
        await self._store.put(
            key, value, collection=self._collection, ttl=self._ttl_seconds
        )

    async def invalidate(self, *, key: str) -> None:
        await self._store.delete(key, collection=self._collection)

    async def clear(self) -> None:
        # Stamp the epoch marker first and unconditionally: this is what
        # makes clear() effective even for backends that don't support
        # destroy_collection (previously a silent no-op there), and it's
        # what rejects a write from a fetch that started before this clear
        # but lands after it. See get_tools()/_get_cleared_at().
        await self._store.put(
            self._EPOCH_KEY,
            {"cleared_at": time.time()},
            collection=self._epoch_collection,
            ttl=None,
        )

        if isinstance(self._store, BaseDestroyCollectionStore):
            try:
                await self._store.destroy_collection(collection=self._collection)
            except KeyError:
                # py-key-value-aio's MongoDBStore lazy-registers collections in
                # _collections_by_name on first read/write and raises KeyError
                # from destroy_collection when the collection was never touched.
                # Treat as a no-op: nothing was stored, nothing to clear.
                logger.debug(
                    "Collection %s not initialized; clear is a no-op",
                    self._collection,
                )
                return
            self._store._setup_collection_complete[self._collection] = False
        else:
            logger.warning(
                "Tool list cache store does not support destroy_collection; "
                "entries will be rejected by the cleared_at epoch check on "
                "read instead of being physically removed"
            )
