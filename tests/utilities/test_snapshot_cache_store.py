"""Tests for snapshot cache store factory and store types."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from key_value.aio.stores.mongodb import MongoDBStore

from languagemodelcommon.utilities.cache.snapshot_cache_store import (
    FileStore,
    MemoryStoreWithContextManager,
    create_cache_store,
)


class TestCreateCacheStoreMemory:
    """cache_type='memory' returns MemoryStoreWithContextManager."""

    def test_default_returns_memory(self) -> None:
        store = create_cache_store()
        assert isinstance(store, MemoryStoreWithContextManager)

    def test_explicit_memory(self) -> None:
        store = create_cache_store(cache_type="memory")
        assert isinstance(store, MemoryStoreWithContextManager)

    def test_unknown_type_defaults_to_memory(self) -> None:
        store = create_cache_store(cache_type="redis")
        assert isinstance(store, MemoryStoreWithContextManager)


class TestCreateCacheStoreMongo:
    """cache_type='mongo' returns MongoDBStore."""

    def test_returns_mongodb_store(self) -> None:
        store = create_cache_store(
            cache_type="mongo",
            mongo_url="mongodb://localhost:27017",
            mongo_db_name="test_db",
            collection="test_cache",
        )
        assert isinstance(store, MongoDBStore)

    def test_custom_collection(self) -> None:
        store = create_cache_store(
            cache_type="mongo",
            mongo_url="mongodb://localhost:27017",
            collection="custom_collection",
        )
        assert isinstance(store, MongoDBStore)
        assert store.default_collection == "custom_collection"

    def test_raises_without_url(self) -> None:
        with pytest.raises(ValueError, match="no MongoDB URL"):
            create_cache_store(cache_type="mongo", mongo_url=None)


class TestCreateCacheStoreFile:
    """cache_type='file' returns FileStore."""

    def test_returns_file_store(self, tmp_path: Path) -> None:
        store = create_cache_store(
            cache_type="file",
            file_path=str(tmp_path / "cache.json"),
        )
        assert isinstance(store, FileStore)


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


class TestFileStore:
    """FileStore persists data to a JSON file."""

    @pytest.mark.asyncio
    async def test_put_and_get(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        async with FileStore(file_path=path) as store:
            await store.put("key1", {"data": "hello"})
            result = await store.get("key1")
            assert result is not None
            assert result["data"] == "hello"

    @pytest.mark.asyncio
    async def test_persists_to_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        async with FileStore(file_path=path) as store:
            await store.put("key1", {"value": 42})

        assert path.is_file()
        data = json.loads(path.read_text())
        assert "key1" in data.get("snapshots", {})

    @pytest.mark.asyncio
    async def test_restores_from_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"

        # Write data in first session
        async with FileStore(file_path=path) as store:
            await store.put("key1", {"persisted": True})

        # Read data in new session
        async with FileStore(file_path=path) as store:
            result = await store.get("key1")
            assert result is not None
            assert result["persisted"] is True

    @pytest.mark.asyncio
    async def test_handles_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent" / "cache.json"
        async with FileStore(file_path=path) as store:
            result = await store.get("key1")
            assert result is None

    @pytest.mark.asyncio
    async def test_handles_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text("not valid json{{{", encoding="utf-8")
        async with FileStore(file_path=path) as store:
            result = await store.get("key1")
            assert result is None
