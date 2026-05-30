"""Factory for creating async key-value cache stores for model config persistence.

The cache backend is always MongoDB via ``ValidatingMongoDBStore``
(wraps ``MongoDBStore`` from ``py-key-value-aio``). Pings MongoDB on
open and raises ``ConnectionError`` immediately if unreachable.
"""

import logging

from key_value.aio.stores.mongodb import MongoDBStore

from languagemodelcommon.utilities.mongo_url_utils import MongoUrlHelpers

logger = logging.getLogger(__name__)


class ValidatingMongoDBStore(MongoDBStore):
    """MongoDBStore that validates connectivity on open.

    Pings MongoDB during ``__aenter__`` so that a misconfigured
    model config cache fails fast at startup rather than silently
    swallowing errors at runtime.
    """

    async def _setup(self) -> None:
        await super()._setup()
        try:
            await self._db.command("ping")
        except Exception as e:
            raise ConnectionError(
                f"Model config cache failed to connect to MongoDB "
                f"(database: {self._db.name}). "
                f"Verify MONGO_LLM_STORAGE_URI / MONGO_URL, credentials, "
                f"and network access. Error: {e}"
            ) from e
        logger.info(
            "Model config cache MongoDB connection validated (db=%s)",
            self._db.name,
        )


def create_cache_store(
    *,
    mongo_url: str | None = None,
    mongo_db_name: str = "language_model_gateway",
    mongo_username: str | None = None,
    mongo_password: str | None = None,
    collection: str = "snapshots",
) -> ValidatingMongoDBStore:
    """Create a MongoDB cache store.

    Args:
        mongo_url: MongoDB connection URL (required).
        mongo_db_name: MongoDB database name.
        mongo_username: MongoDB username.
        mongo_password: MongoDB password.
        collection: Collection/namespace for cache entries.
    """
    if not mongo_url:
        raise ValueError("No MongoDB URL is configured for model config cache")
    connection_url = MongoUrlHelpers.add_credentials_to_mongo_url(
        mongo_url=mongo_url,
        username=mongo_username,
        password=mongo_password,
    )
    logger.info(
        "Model config cache using MongoDB: db=%s, collection=%s",
        mongo_db_name,
        collection,
    )
    return ValidatingMongoDBStore(
        url=connection_url,
        db_name=mongo_db_name,
        default_collection=collection,
    )
