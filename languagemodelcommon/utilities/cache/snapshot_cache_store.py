"""Factory for creating async key-value cache stores for parsed snapshots.

Provides a consistent interface (``BaseContextManagerStore``) regardless of
backend.  When MongoDB is configured, returns a ``MongoDBStore``; otherwise
returns a no-op ``MemoryStore`` with context manager support.

All returned stores support ``async with store:`` / ``await store.get(...)``
/ ``await store.put(...)`` uniformly.
"""

from __future__ import annotations

import logging
from typing import Any

from key_value.aio.stores.base import BaseContextManagerStore
from key_value.aio.stores.memory import MemoryStore
from key_value.aio.stores.mongodb import MongoDBStore

from languagemodelcommon.utilities.mongo_url_utils import MongoUrlHelpers

logger = logging.getLogger(__name__)


class MemoryStoreWithContextManager(MemoryStore):
    """MemoryStore with async context manager support for consistent interface."""

    async def __aenter__(self) -> "MemoryStoreWithContextManager":
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass


def create_cache_store(
    *,
    enabled: bool = False,
    mongo_url: str | None = None,
    mongo_db_name: str = "language_model_gateway",
    mongo_username: str | None = None,
    mongo_password: str | None = None,
    collection: str = "snapshots",
) -> MongoDBStore | MemoryStoreWithContextManager:
    """Create a cache store based on configuration.

    When ``enabled`` is True and a ``mongo_url`` is provided, returns a
    ``MongoDBStore`` backed by MongoDB.  Otherwise returns a lightweight
    in-memory store (useful for single-pod deployments or when caching is
    disabled).

    The returned store always supports ``async with store:`` for consistent
    lifecycle management.
    """
    if not enabled or not mongo_url:
        logger.info("Snapshot cache disabled; using in-memory store")
        return MemoryStoreWithContextManager(default_collection=collection)

    connection_url = MongoUrlHelpers.add_credentials_to_mongo_url(
        mongo_url=mongo_url,
        username=mongo_username,
        password=mongo_password,
    )
    logger.info(
        "Snapshot cache using MongoDB: db=%s, collection=%s",
        mongo_db_name,
        collection,
    )
    return MongoDBStore(
        url=connection_url,
        db_name=mongo_db_name,
        default_collection=collection,
    )
