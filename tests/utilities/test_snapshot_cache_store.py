"""Tests for snapshot cache store factory and resilience."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from key_value.aio.stores.base import BaseContextManagerStore

from languagemodelcommon.utilities.cache.snapshot_cache_store import (
    ResilientCacheStore,
    create_cache_store,
)


def _mock_primary(*, connect_error: Exception | None = None) -> MagicMock:
    """Create a mock that passes isinstance(x, BaseContextManagerStore)."""
    primary = MagicMock(spec=BaseContextManagerStore)
    if connect_error:
        primary.__aenter__ = AsyncMock(side_effect=connect_error)
    else:
        primary.__aenter__ = AsyncMock(return_value=primary)
    primary.__aexit__ = AsyncMock()
    return primary


class TestCreateCacheStore:
    """Factory returns ResilientCacheStore with appropriate primary backend."""

    def test_disabled_returns_no_primary(self) -> None:
        store = create_cache_store(enabled=False)
        assert isinstance(store, ResilientCacheStore)
        assert store._primary is None

    def test_no_url_returns_no_primary(self) -> None:
        store = create_cache_store(enabled=True, mongo_url=None)
        assert isinstance(store, ResilientCacheStore)
        assert store._primary is None

    def test_empty_url_returns_no_primary(self) -> None:
        store = create_cache_store(enabled=True, mongo_url="")
        assert isinstance(store, ResilientCacheStore)
        assert store._primary is None

    def test_enabled_with_url_has_primary(self) -> None:
        store = create_cache_store(
            enabled=True,
            mongo_url="mongodb://localhost:27017",
            mongo_db_name="test_db",
            collection="test_cache",
        )
        assert isinstance(store, ResilientCacheStore)
        assert store._primary is not None

    def test_custom_collection(self) -> None:
        store = create_cache_store(
            enabled=True,
            mongo_url="mongodb://localhost:27017",
            collection="custom_collection",
        )
        assert store.default_collection == "custom_collection"


class TestResilientCacheStoreLifecycle:
    """Context manager handles connection failures gracefully."""

    @pytest.mark.asyncio
    async def test_no_primary_enters_cleanly(self) -> None:
        store = ResilientCacheStore(primary=None)
        async with store as s:
            assert s is store

    @pytest.mark.asyncio
    async def test_primary_connect_failure_falls_back(self) -> None:
        primary = _mock_primary(connect_error=ConnectionError("unreachable"))

        store = ResilientCacheStore(primary=primary)
        async with store as s:
            assert s is store
            assert store._primary_available is False

    @pytest.mark.asyncio
    async def test_primary_connect_success(self) -> None:
        primary = _mock_primary()

        store = ResilientCacheStore(primary=primary)
        async with store as s:
            assert s is store
            assert store._primary_available is True


class TestResilientCacheStoreOperations:
    """get/put/delete delegate to primary with fallback."""

    @pytest.mark.asyncio
    async def test_put_and_get_without_primary(self) -> None:
        store = ResilientCacheStore(primary=None)
        async with store:
            await store.put("key1", {"data": "hello"})
            result = await store.get("key1")
            assert result is not None
            assert result["data"] == "hello"

    @pytest.mark.asyncio
    async def test_get_missing_key(self) -> None:
        store = ResilientCacheStore(primary=None)
        async with store:
            result = await store.get("nonexistent")
            assert result is None

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        store = ResilientCacheStore(primary=None)
        async with store:
            await store.put("key1", {"data": "hello"})
            await store.delete("key1")
            result = await store.get("key1")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_delegates_to_primary(self) -> None:
        primary = _mock_primary()
        primary.get = AsyncMock(return_value={"from": "primary"})

        store = ResilientCacheStore(primary=primary)
        async with store:
            result = await store.get("key1")
            assert result == {"from": "primary"}
            primary.get.assert_called_once_with("key1", collection=None)

    @pytest.mark.asyncio
    async def test_get_falls_back_on_primary_error(self) -> None:
        primary = _mock_primary()
        primary.get = AsyncMock(side_effect=RuntimeError("connection lost"))

        store = ResilientCacheStore(primary=primary)
        async with store:
            # Put into memory layer directly
            await super(ResilientCacheStore, store).put("key1", {"from": "memory"})
            # Primary fails, falls back to memory
            result = await store.get("key1")
            assert result == {"from": "memory"}

    @pytest.mark.asyncio
    async def test_put_writes_to_both(self) -> None:
        primary = _mock_primary()
        primary.put = AsyncMock()

        store = ResilientCacheStore(primary=primary)
        async with store:
            await store.put("key1", {"data": "value"}, ttl=60)
            primary.put.assert_called_once_with(
                "key1", {"data": "value"}, collection=None, ttl=60
            )

    @pytest.mark.asyncio
    async def test_put_survives_primary_failure(self) -> None:
        primary = _mock_primary()
        primary.put = AsyncMock(side_effect=RuntimeError("write failed"))
        primary.get = AsyncMock(side_effect=RuntimeError("read failed"))

        store = ResilientCacheStore(primary=primary)
        async with store:
            # Should not raise even though primary.put fails
            await store.put("key1", {"data": "value"})
            # Primary.get also fails, but memory fallback works
            result = await store.get("key1")
            assert result == {"data": "value"}
