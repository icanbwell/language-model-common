"""Tests for snapshot cache store factory and lifecycle."""

from __future__ import annotations

import pytest

from key_value.aio.stores.mongodb import MongoDBStore

from languagemodelcommon.utilities.cache.snapshot_cache_store import (
    MemoryStoreWithContextManager,
    create_cache_store,
)


class TestCreateCacheStoreDisabled:
    """When disabled or no mongo_url, returns MemoryStoreWithContextManager."""

    def test_disabled_returns_memory_store(self) -> None:
        store = create_cache_store(enabled=False)
        assert isinstance(store, MemoryStoreWithContextManager)

    def test_no_url_returns_memory_store(self) -> None:
        store = create_cache_store(enabled=True, mongo_url=None)
        assert isinstance(store, MemoryStoreWithContextManager)

    def test_empty_url_returns_memory_store(self) -> None:
        store = create_cache_store(enabled=True, mongo_url="")
        assert isinstance(store, MemoryStoreWithContextManager)


class TestCreateCacheStoreEnabled:
    """When enabled with mongo_url, returns MongoDBStore."""

    def test_returns_mongodb_store(self) -> None:
        store = create_cache_store(
            enabled=True,
            mongo_url="mongodb://localhost:27017",
            mongo_db_name="test_db",
            collection="test_cache",
        )
        assert isinstance(store, MongoDBStore)

    def test_custom_collection(self) -> None:
        store = create_cache_store(
            enabled=True,
            mongo_url="mongodb://localhost:27017",
            collection="custom_collection",
        )
        assert isinstance(store, MongoDBStore)
        assert store.default_collection == "custom_collection"


class TestMemoryStoreContextManager:
    """MemoryStoreWithContextManager supports async with."""

    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self) -> None:
        store = MemoryStoreWithContextManager(default_collection="test")
        async with store as s:
            assert s is store

    @pytest.mark.asyncio
    async def test_put_and_get(self) -> None:
        store = MemoryStoreWithContextManager(default_collection="test")
        async with store:
            await store.put("key1", {"data": "hello"})
            result = await store.get("key1")
            assert result is not None
            assert result["data"] == "hello"

    @pytest.mark.asyncio
    async def test_get_missing_key(self) -> None:
        store = MemoryStoreWithContextManager(default_collection="test")
        async with store:
            result = await store.get("nonexistent")
            assert result is None

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        store = MemoryStoreWithContextManager(default_collection="test")
        async with store:
            await store.put("key1", {"data": "hello"})
            await store.delete("key1")
            result = await store.get("key1")
            assert result is None
