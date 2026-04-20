"""Tests for SnapshotCacheStore with MemoryStore fallback (no MongoDB needed)."""

from __future__ import annotations

import pytest

from languagemodelcommon.utilities.cache.snapshot_cache_store import SnapshotCacheStore


class TestSnapshotCacheStoreDisabled:
    """When disabled, all operations are no-ops."""

    @pytest.fixture
    def store(self) -> SnapshotCacheStore:
        return SnapshotCacheStore(enabled=False)

    @pytest.mark.asyncio
    async def test_get_returns_none(self, store: SnapshotCacheStore) -> None:
        async with store:
            result = await store.get("any_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_put_does_not_raise(self, store: SnapshotCacheStore) -> None:
        async with store:
            await store.put("key", {"data": "value"})

    @pytest.mark.asyncio
    async def test_delete_does_not_raise(self, store: SnapshotCacheStore) -> None:
        async with store:
            await store.delete("key")

    @pytest.mark.asyncio
    async def test_is_enabled_false(self, store: SnapshotCacheStore) -> None:
        assert store.is_enabled is False


class TestSnapshotCacheStoreEnabledWithoutUrl:
    """When enabled=True but no mongo_url, effectively disabled."""

    @pytest.mark.asyncio
    async def test_no_url_means_disabled(self) -> None:
        store = SnapshotCacheStore(enabled=True, mongo_url=None)
        assert store.is_enabled is False
        async with store:
            result = await store.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_url_means_disabled(self) -> None:
        store = SnapshotCacheStore(enabled=True, mongo_url="")
        assert store.is_enabled is False
