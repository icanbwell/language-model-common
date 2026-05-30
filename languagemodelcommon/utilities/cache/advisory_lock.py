from __future__ import annotations

import logging
import os
import uuid
from types import TracebackType

from key_value.aio.protocols.key_value import AsyncKeyValueProtocol

logger = logging.getLogger(__name__)

LOCK_COLLECTION = "advisory_locks"


class AdvisoryLock:
    """Distributed advisory lock backed by any AsyncKeyValueProtocol store.

    Uses put-with-TTL so that stale locks auto-expire if the holder crashes.
    Acquire semantics rely on get-then-put — not atomic compare-and-swap —
    so there is a small race window. For leader-election among cooperating
    workers at startup this is acceptable; do not use for high-contention
    critical sections.

    Usage:
        async with AdvisoryLock(store, "skill_sync", ttl_seconds=300) as acquired:
            if acquired:
                await do_sync()
    """

    def __init__(
        self,
        store: AsyncKeyValueProtocol,
        lock_name: str,
        *,
        ttl_seconds: int = 300,
        collection: str = LOCK_COLLECTION,
    ) -> None:
        self._store = store
        self._lock_name = lock_name
        self._ttl_seconds = ttl_seconds
        self._collection = collection
        self._holder_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._acquired = False

    @property
    def acquired(self) -> bool:
        return self._acquired

    async def __aenter__(self) -> bool:
        existing = await self._store.get(self._lock_name, collection=self._collection)

        if existing is not None:
            logger.info(
                "AdvisoryLock: lock '%s' held by '%s' — skipping.",
                self._lock_name,
                existing.get("holder", "unknown"),
            )
            self._acquired = False
            return False

        await self._store.put(
            self._lock_name,
            {"holder": self._holder_id},
            collection=self._collection,
            ttl=self._ttl_seconds,
        )

        # Verify we won the race (read-after-write check)
        written = await self._store.get(self._lock_name, collection=self._collection)
        if written is not None and written.get("holder") == self._holder_id:
            self._acquired = True
            logger.info(
                "AdvisoryLock: acquired lock '%s' (holder=%s, ttl=%ds)",
                self._lock_name,
                self._holder_id,
                self._ttl_seconds,
            )
        else:
            self._acquired = False
            logger.info(
                "AdvisoryLock: lost race for lock '%s' — another holder won.",
                self._lock_name,
            )

        return self._acquired

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._acquired:
            await self._store.delete(self._lock_name, collection=self._collection)
            logger.info(
                "AdvisoryLock: released lock '%s' (holder=%s)",
                self._lock_name,
                self._holder_id,
            )
