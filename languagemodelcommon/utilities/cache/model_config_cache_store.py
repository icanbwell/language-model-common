"""Factory for creating async key-value cache stores for model config persistence.

The cache backend is always MongoDB via ``ValidatingMongoDBStore``
(wraps ``MongoDBStore`` from ``py-key-value-aio``). Pings MongoDB on
open, retrying transient connection failures a bounded number of times,
and raises ``ConnectionError`` if it is still unreachable.
"""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from key_value.aio.stores.mongodb import MongoDBStore
from pymongo import AsyncMongoClient
from pymongo.errors import ConnectionFailure

from languagemodelcommon.utilities.mongo_url_utils import MongoUrlHelpers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PingRetryPolicy:
    """Bounds for retrying the connectivity ping on open, and for recovering
    after the retries are exhausted.

    Worst-case time before giving up is roughly
    ``max_attempts * ping_timeout_seconds`` plus the backoff delays. The
    defaults keep that at or above pymongo's 30 s default server-selection
    window, so a MongoDB that becomes reachable late is still waited out as
    it was before retries existed.

    After the retries are exhausted the store builds a fresh client for its
    next setup, but not sooner than ``recovery_cooldown_seconds`` after the
    failure, so a MongoDB that is still down does not make every caller wait
    out the full retry budget.
    """

    max_attempts: int = 4
    ping_timeout_seconds: float = 10.0
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 4.0
    recovery_cooldown_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.ping_timeout_seconds <= 0:
            raise ValueError("ping_timeout_seconds must be greater than 0")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must not be negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be at least base_delay_seconds")
        if self.recovery_cooldown_seconds < 0:
            raise ValueError("recovery_cooldown_seconds must not be negative")

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

    If the retries are exhausted anyway, the closed client would keep the
    store unusable for the life of the process. So the next setup, once the
    policy's cooldown has passed, replaces the closed client with a new one.
    A store given its own ``client`` is not rebuilt, since its lifecycle
    belongs to the caller.

    Args (beyond those of ``MongoDBStore``):
        ping_retry_policy: Bounds for retrying the open-time ping and for
            recovery after it fails. Defaults to ``PingRetryPolicy()``.
        sleep: Awaitable used to wait between retries. Injectable so tests
            do not wait in real time.
        client_factory: Builds the replacement client from the store's URL.
            Injectable for tests.
        monotonic: Clock used for the recovery cooldown. Injectable for tests.
    """

    def __init__(
        self,
        *,
        ping_retry_policy: PingRetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        client_factory: Callable[[str], AsyncMongoClient[dict[str, Any]]] = (
            AsyncMongoClient
        ),
        monotonic: Callable[[], float] = time.monotonic,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._ping_retry_policy = ping_retry_policy or PingRetryPolicy()
        self._sleep = sleep
        self._client_factory = client_factory
        self._monotonic = monotonic
        self._mongo_url: str | None = kwargs.get("url")
        self._last_setup_failure_at: float | None = None

    def _recover_client_after_failed_setup(self) -> None:
        """Replace the client that a failed setup left closed.

        Called at the start of ``_setup``. Does nothing unless an earlier setup
        failed. Within the cooldown it raises straight away instead of
        pinging, so callers fail fast while MongoDB is still down.
        """
        if self._last_setup_failure_at is None:
            return
        cooldown = self._ping_retry_policy.recovery_cooldown_seconds
        remaining = cooldown - (self._monotonic() - self._last_setup_failure_at)
        if remaining > 0:
            raise ConnectionError(
                f"Model config cache could not connect to MongoDB "
                f"(database: {self._db.name}) and will try again in "
                f"{remaining:.0f}s."
            )
        if self._client_provided_by_user or not self._mongo_url:
            return
        db_name = self._db.name
        self._client = self._client_factory(self._mongo_url)
        self._db = self._client[db_name]
        # Handles from the failed attempt belong to the closed client.
        self._collections_by_name.clear()
        logger.warning(
            "Model config cache rebuilt its MongoDB client after a failed "
            "setup (db=%s)",
            db_name,
        )

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
        self._recover_client_after_failed_setup()
        await super()._setup()
        try:
            await self._ping_with_retry()
        except Exception as e:
            self._last_setup_failure_at = self._monotonic()
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
        self._last_setup_failure_at = None


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
