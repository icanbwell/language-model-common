"""Tests for model config cache store factory."""

import pytest

from key_value.aio.errors.store import StoreSetupError
from key_value.aio.stores.mongodb import MongoDBStore

from languagemodelcommon.utilities.cache.model_config_cache_store import (
    ValidatingMongoDBStore,
    create_cache_store,
)


class TestCreateCacheStore:
    """create_cache_store returns ValidatingMongoDBStore."""

    def test_raises_without_url(self) -> None:
        with pytest.raises(ValueError, match="No MongoDB URL"):
            create_cache_store()

    def test_raises_with_none_url(self) -> None:
        with pytest.raises(ValueError, match="No MongoDB URL"):
            create_cache_store(mongo_url=None)

    def test_returns_validating_mongodb_store(self) -> None:
        store = create_cache_store(
            mongo_url="mongodb://localhost:27017",
            mongo_db_name="test_db",
            collection="test_cache",
        )
        assert isinstance(store, ValidatingMongoDBStore)
        assert isinstance(store, MongoDBStore)

    def test_custom_collection(self) -> None:
        store = create_cache_store(
            mongo_url="mongodb://localhost:27017",
            collection="custom_collection",
        )
        assert isinstance(store, ValidatingMongoDBStore)
        assert store.default_collection == "custom_collection"

    @pytest.mark.asyncio
    async def test_aenter_raises_on_unreachable_mongo(self) -> None:
        store = create_cache_store(
            mongo_url="mongodb://unreachable-host:27017",
            mongo_db_name="test_db",
            collection="test_cache",
        )
        with pytest.raises(
            StoreSetupError, match="Model config cache failed to connect"
        ):
            await store.__aenter__()
