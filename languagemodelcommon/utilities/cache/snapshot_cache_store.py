"""Factory for creating async key-value cache stores for parsed snapshots.

The ``SNAPSHOT_CACHE_TYPE`` env var selects the backend:

- ``mongo``  — ``MongoDBStore``, fails if MongoDB is unreachable
- ``memory`` — in-process ``MemoryStore``, no persistence across restarts
- ``file``   — JSON file-backed store, persists locally without MongoDB

Both ``MongoDBStore`` and ``MemoryStore`` are from ``py-key-value-aio``.
All returned stores support ``async with`` / ``get`` / ``put`` uniformly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, SupportsFloat

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


class FileStore(MemoryStoreWithContextManager):
    """A file-backed key-value store that loads/saves JSON on open/close.

    On ``__aenter__``, reads existing data from the file into memory.
    On ``put``/``delete``, persists the change to disk immediately.
    Between open and close, all reads run against the in-memory store.
    """

    def __init__(
        self,
        *,
        file_path: str | Path,
        default_collection: str = "snapshots",
    ) -> None:
        super().__init__(default_collection=default_collection)
        self._file_path = Path(file_path)
        # Shadow index: tracks {collection: {key: value}} for serialization.
        # MemoryStore internals are opaque; this is the source of truth for disk.
        self._shadow: dict[str, dict[str, dict[str, Any]]] = {}

    async def __aenter__(self) -> "FileStore":
        if self._file_path.is_file():
            try:
                data = json.loads(self._file_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for collection, entries in data.items():
                        if isinstance(entries, dict):
                            for key, value in entries.items():
                                await self.put(key, value, collection=collection)
                logger.info("FileStore loaded from %s", self._file_path)
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "FileStore: failed to read %s; starting empty",
                    self._file_path,
                    exc_info=True,
                )
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._flush_to_disk()

    async def put(
        self,
        key: str,
        value: Mapping[str, Any],
        *,
        collection: str | None = None,
        ttl: SupportsFloat | None = None,
    ) -> None:
        resolved_collection = collection or self.default_collection
        await super().put(key, value, collection=collection, ttl=ttl)
        self._shadow.setdefault(resolved_collection, {})[key] = dict(value)
        self._flush_to_disk()

    async def delete(
        self,
        key: str,
        *,
        collection: str | None = None,
    ) -> bool:
        resolved_collection = collection or self.default_collection
        result: bool = await super().delete(key, collection=collection)
        coll_data = self._shadow.get(resolved_collection)
        if coll_data:
            coll_data.pop(key, None)
        self._flush_to_disk()
        return result

    def _flush_to_disk(self) -> None:
        """Write shadow index to the JSON file."""
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_path.write_text(
                json.dumps(self._shadow, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            logger.debug(
                "FileStore: failed to flush to %s", self._file_path, exc_info=True
            )


def create_cache_store(
    *,
    cache_type: str = "memory",
    mongo_url: str | None = None,
    mongo_db_name: str = "language_model_gateway",
    mongo_username: str | None = None,
    mongo_password: str | None = None,
    collection: str = "snapshots",
    file_path: str | None = None,
) -> MongoDBStore | FileStore | MemoryStoreWithContextManager:
    """Create a cache store based on the specified type.

    Args:
        cache_type: Backend type — ``'mongo'``, ``'file'``, or ``'memory'``.
        mongo_url: MongoDB connection URL (required when cache_type='mongo').
        mongo_db_name: MongoDB database name.
        mongo_username: MongoDB username.
        mongo_password: MongoDB password.
        collection: Collection/namespace for cache entries.
        file_path: Path for JSON file store (required when cache_type='file').
    """
    cache_type = cache_type.strip().lower()

    if cache_type == "mongo":
        if not mongo_url:
            raise ValueError(
                "SNAPSHOT_CACHE_TYPE is 'mongo' but no MongoDB URL is configured"
            )
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

    if cache_type == "file":
        resolved_path = file_path or f"/tmp/snapshot_cache/{collection}.json"  # noqa: S108  # nosec B108
        logger.info("Snapshot cache using file: %s", resolved_path)
        return FileStore(file_path=resolved_path, default_collection=collection)

    # Default: in-memory (no persistence)
    if cache_type != "memory":
        logger.warning(
            "Unknown SNAPSHOT_CACHE_TYPE '%s'; defaulting to memory",
            cache_type,
        )
    logger.info("Snapshot cache using in-memory store")
    return MemoryStoreWithContextManager(default_collection=collection)
