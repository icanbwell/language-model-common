"""Factory for creating async key-value cache stores for parsed snapshots.

Provides a ``ResilientCacheStore`` that wraps a primary store (MongoDB) with
automatic fallback to in-memory storage.  If the primary fails to connect or
any operation throws, the store silently degrades — consumers never see errors
from the cache layer.

All returned stores support ``async with store:`` / ``await store.get(...)``
/ ``await store.put(...)`` uniformly regardless of the underlying backend.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, SupportsFloat

from key_value.aio.stores.base import BaseStore
from key_value.aio.stores.base import BaseContextManagerStore
from key_value.aio.stores.memory import MemoryStore
from key_value.aio.stores.mongodb import MongoDBStore

from languagemodelcommon.utilities.mongo_url_utils import MongoUrlHelpers

logger = logging.getLogger(__name__)


class ResilientCacheStore(MemoryStore):
    """A cache store that delegates to a primary backend with graceful fallback.

    On ``__aenter__``, attempts to connect the primary store (e.g. MongoDB).
    If it fails, logs a warning and operates as an in-memory store.
    On ``get``/``put``, delegates to the primary; on error, falls back to
    the in-memory layer.  Consumers never see exceptions from this store.
    """

    def __init__(
        self,
        *,
        primary: BaseStore | None = None,
        default_collection: str = "snapshots",
    ) -> None:
        super().__init__(default_collection=default_collection)
        self._primary = primary
        self._primary_available = False

    async def __aenter__(self) -> "ResilientCacheStore":
        if self._primary is not None and isinstance(self._primary, BaseContextManagerStore):
            try:
                await self._primary.__aenter__()
                self._primary_available = True
                logger.info("Snapshot cache: primary store connected")
            except Exception:
                logger.warning(
                    "Snapshot cache: primary store failed to connect; "
                    "falling back to in-memory",
                    exc_info=True,
                )
                self._primary_available = False
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._primary_available and isinstance(self._primary, BaseContextManagerStore):
            try:
                await self._primary.__aexit__(None, None, None)
            except Exception:
                logger.debug(
                    "Snapshot cache: error closing primary store", exc_info=True
                )
        self._primary_available = False

    async def get(
        self,
        key: str,
        *,
        collection: str | None = None,
    ) -> dict[str, Any] | None:
        if self._primary_available and self._primary is not None:
            try:
                result: dict[str, Any] | None = await self._primary.get(key, collection=collection)
                return result
            except Exception:
                logger.debug(
                    "Snapshot cache: primary get failed for key=%s; falling back",
                    key,
                    exc_info=True,
                )
        result = await super().get(key, collection=collection)
        return result

    async def put(
        self,
        key: str,
        value: Mapping[str, Any],
        *,
        collection: str | None = None,
        ttl: SupportsFloat | None = None,
    ) -> None:
        # Always store in memory as local fallback
        await super().put(key, value, collection=collection, ttl=ttl)
        # Also store in primary if available
        if self._primary_available and self._primary is not None:
            try:
                await self._primary.put(key, value, collection=collection, ttl=ttl)
            except Exception:
                logger.debug(
                    "Snapshot cache: primary put failed for key=%s",
                    key,
                    exc_info=True,
                )

    async def delete(
        self,
        key: str,
        *,
        collection: str | None = None,
    ) -> bool:
        deleted: bool = await super().delete(key, collection=collection)
        if self._primary_available and self._primary is not None:
            try:
                await self._primary.delete(key, collection=collection)
            except Exception:
                logger.debug(
                    "Snapshot cache: primary delete failed for key=%s",
                    key,
                    exc_info=True,
                )
        return deleted


def create_cache_store(
    *,
    enabled: bool = False,
    mongo_url: str | None = None,
    mongo_db_name: str = "language_model_gateway",
    mongo_username: str | None = None,
    mongo_password: str | None = None,
    collection: str = "snapshots",
) -> ResilientCacheStore:
    """Create a resilient cache store based on configuration.

    When ``enabled`` is True and a ``mongo_url`` is provided, the store
    delegates to MongoDB as the primary backend.  If MongoDB is unreachable
    at startup or during any operation, the store silently falls back to
    in-memory storage.

    Returns a ``ResilientCacheStore`` that always supports ``async with``
    and never raises on cache operations.
    """
    primary: BaseStore | None = None

    if enabled and mongo_url:
        connection_url = MongoUrlHelpers.add_credentials_to_mongo_url(
            mongo_url=mongo_url,
            username=mongo_username,
            password=mongo_password,
        )
        logger.info(
            "Snapshot cache configured with MongoDB primary: db=%s, collection=%s",
            mongo_db_name,
            collection,
        )
        primary = MongoDBStore(
            url=connection_url,
            db_name=mongo_db_name,
            default_collection=collection,
        )
    else:
        logger.info("Snapshot cache: no primary configured; using in-memory only")

    return ResilientCacheStore(primary=primary, default_collection=collection)
