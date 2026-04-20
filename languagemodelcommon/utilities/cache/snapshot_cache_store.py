"""Shared cache store backed by py-key-value-aio.

Provides a MongoDB-backed (or in-memory fallback) cache for parsed
snapshots.  The backend is selected via an ``enabled`` flag; when
enabled, uses the standard ``MONGO_URL`` / ``MONGO_DB_NAME`` /
``MONGO_DB_USERNAME`` / ``MONGO_DB_PASSWORD`` environment variables
for the connection, with a configurable collection name.

Usage::

    store = SnapshotCacheStore(
        enabled=True,
        mongo_url="mongodb://mongo:27017",
        mongo_db_name="language_model_gateway",
        mongo_username="root",
        mongo_password="secret",
        collection="config_snapshots",
        ttl_seconds=3600,
    )
    async with store:
        await store.put("configs", {"models": [...]})
        cached = await store.get("configs")
"""

from __future__ import annotations

import logging
from typing import Any

from key_value.aio.stores.memory import MemoryStore
from key_value.aio.stores.mongodb import MongoDBStore

from languagemodelcommon.utilities.mongo_url_utils import MongoUrlHelpers

logger = logging.getLogger(__name__)


class SnapshotCacheStore:
    """Async key-value cache for parsed snapshots.

    Wraps a ``py-key-value-aio`` store with a simple get/put interface
    that includes TTL support.  When ``enabled=False``, all operations
    are no-ops (returns None on get, ignores put/delete).
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        mongo_url: str | None = None,
        mongo_db_name: str = "language_model_gateway",
        mongo_username: str | None = None,
        mongo_password: str | None = None,
        collection: str = "snapshots",
        ttl_seconds: float = 3600,
    ) -> None:
        self._enabled = enabled and bool(mongo_url)
        self._mongo_url = mongo_url
        self._mongo_db_name = mongo_db_name
        self._mongo_username = mongo_username
        self._mongo_password = mongo_password
        self._collection = collection
        self._ttl_seconds = ttl_seconds
        self._store: MongoDBStore | MemoryStore | None = None

    @property
    def is_enabled(self) -> bool:
        """Return True if this store uses a real persistent backend (MongoDB)."""
        return self._enabled

    async def __aenter__(self) -> SnapshotCacheStore:
        if self._enabled:
            store = self._create_store()
            if isinstance(store, MongoDBStore):
                await store.__aenter__()
            self._store = store
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._store is not None:
            if isinstance(self._store, MongoDBStore):
                await self._store.__aexit__(None, None, None)
            self._store = None

    async def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve a cached snapshot by key.  Returns None on miss or error."""
        if not self._enabled or self._store is None:
            return None
        try:
            result: dict[str, Any] | None = await self._store.get(key, collection=self._collection)
            return result
        except Exception:
            logger.debug("Snapshot cache get failed for key '%s'", key, exc_info=True)
            return None

    async def put(self, key: str, value: dict[str, Any]) -> None:
        """Store a snapshot with TTL.  Fails silently on error."""
        if not self._enabled or self._store is None:
            return
        try:
            await self._store.put(
                key,
                value,
                collection=self._collection,
                ttl=self._ttl_seconds,
            )
        except Exception:
            logger.warning(
                "Snapshot cache put failed for key '%s'", key, exc_info=True
            )

    async def delete(self, key: str) -> None:
        """Remove a cached snapshot.  Fails silently on error."""
        if not self._enabled or self._store is None:
            return
        try:
            await self._store.delete(key, collection=self._collection)
        except Exception:
            logger.debug(
                "Snapshot cache delete failed for key '%s'", key, exc_info=True
            )

    def _create_store(self) -> MongoDBStore | MemoryStore:
        """Create the MongoDB store with credentials injected."""
        if not self._mongo_url:
            return MemoryStore()

        connection_url = MongoUrlHelpers.add_credentials_to_mongo_url(
            mongo_url=self._mongo_url,
            username=self._mongo_username,
            password=self._mongo_password,
        )
        logger.info(
            "Snapshot cache using MongoDB: db=%s, collection=%s, ttl=%ds",
            self._mongo_db_name,
            self._collection,
            int(self._ttl_seconds),
        )
        return MongoDBStore(
            url=connection_url,
            db_name=self._mongo_db_name,
            default_collection=self._collection,
        )
