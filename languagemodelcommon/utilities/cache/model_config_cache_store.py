"""Factory for creating async key-value cache stores for model config persistence.

The cache backend is always MongoDB via ``ValidatingMongoDBStore``
(wraps ``MongoDBStore`` from ``py-key-value-aio``). Pings MongoDB on
open, retrying transient connection failures a bounded number of times,
and raises ``ConnectionError`` if it is still unreachable.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from key_value.aio.stores.mongodb import MongoDBStore
from pymongo.errors import ConnectionFailure

from languagemodelcommon.utilities.mongo_url_utils import MongoUrlHelpers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PingRetryPolicy:
    """Bounds for retrying the connectivity ping on open.

    Worst-case time before giving up is roughly
    ``max_attempts * ping_timeout_seconds`` plus the backoff delays.
    """

    max_attempts: int = 4
    ping_timeout_seconds: float = 3.0
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 4.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def backoff_delay(self, *, attempt: int) -> float:
        """Exponential backoff with full jitter for the 1-based ``attempt``."""
        ceiling = min(
            self.max_delay_seconds, self.base_delay_seconds * 2 ** (attempt - 1)
        )
        return random.uniform(0.0, ceiling)


class ValidatingMongoDBStore(MongoDBStore):
    """MongoDBStore that validates connectivity on open.

    Pings MongoDB during ``__aenter__`` so that a misconfigured
    model config cache fails fast at startup rather than silently
    swallowing errors at runtime.

    A transient connection failure (for example ``AutoReconnect`` while a
    pod is starting) is retried here, inside ``_setup``. ``py-key-value-aio``
    closes the store, and the Mongo client it owns, whenever ``_setup``
    raises, and ``setup`` is then re-run on that closed client, so every
    later call fails with ``Cannot use AsyncMongoClient after close``. The
    retry must therefore happen before ``_setup`` raises.
    """

    def __init__(
        self,
        *,
        ping_retry_policy: PingRetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._ping_retry_policy = ping_retry_policy or PingRetryPolicy()
        self._sleep = sleep

    async def _ping_with_retry(self) -> None:
        policy = self._ping_retry_policy
        for attempt in range(1, policy.max_attempts + 1):
            try:
                async with asyncio.timeout(policy.ping_timeout_seconds):
                    await self._db.command("ping")
                return
            except (ConnectionFailure, TimeoutError) as e:
                if attempt == policy.max_attempts:
                    raise
                delay = policy.backoff_delay(attempt=attempt)
                logger.warning(
                    "Model config cache MongoDB ping failed (%s), "
                    "retrying in %.2fs (attempt %d/%d)",
                    type(e).__name__,
                    delay,
                    attempt,
                    policy.max_attempts,
                )
                await self._sleep(delay)

    async def _setup(self) -> None:
        await super()._setup()
        try:
            await self._ping_with_retry()
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
    ping_retry_policy: PingRetryPolicy | None = None,
) -> ValidatingMongoDBStore:
    """Create a MongoDB cache store.

    Args:
        mongo_url: MongoDB connection URL (required).
        mongo_db_name: MongoDB database name.
        mongo_username: MongoDB username.
        mongo_password: MongoDB password.
        collection: Collection/namespace for cache entries.
        ping_retry_policy: Bounds for retrying the open-time ping (defaults
            to ``PingRetryPolicy()``).
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
        ping_retry_policy=ping_retry_policy,
    )
