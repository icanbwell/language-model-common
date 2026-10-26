"""Persistent store for MCP tool titles learned at runtime (BAI-1306).

A tool reached through the ``call_tool`` discovery meta-tool is never bound as
a LangChain tool, so its title is only learned from a ``search_tools`` or
``call_tool`` result. ``LearnedTitleCache`` shares those titles across requests
on one pod; this store shares them across pods and restarts so a cold pod can
render the first ``call_tool`` start line with the real title.

Only ``tool name -> title template`` from ``search_tools`` listings is stored
(catalog metadata, no PHI and no tenant or request data). Every operation is
bounded by a timeout and never raises: the display name is cosmetic and must
not break a stream. After a failure the store is skipped for a short cool-down
so a degraded Mongo costs one timeout rather than one per event.
"""

import asyncio
import logging
import time
from typing import Mapping, Protocol

from key_value.aio.stores.base import BaseStore

from languagemodelcommon.utilities.tool_display_name_mapper import LearnedTitleCache

logger = logging.getLogger(__name__)

_NEGATIVE_CACHE_SECONDS = 60.0
_NEGATIVE_CACHE_MAX_ENTRIES = 1024
_FAILURE_COOLDOWN_SECONDS = 30.0
_RECENT_WRITES_MAX_ENTRIES = 1024


class LearnedTitleStoreProtocol(Protocol):
    """Cross-pod store of learned MCP tool titles."""

    async def put_titles(self, *, titles: Mapping[str, str]) -> None: ...

    async def get_title(self, *, tool_name: str) -> str | None: ...


class LearnedTitleStore:
    """``LearnedTitleStoreProtocol`` backed by a py-key-value-aio store."""

    def __init__(
        self,
        *,
        store: BaseStore,
        collection: str,
        ttl_seconds: float,
        timeout_seconds: float,
    ) -> None:
        self._store = store
        self._collection = collection
        # A non-positive TTL would raise InvalidTTLError on every put; treat it
        # as "no expiry" like McpToolListStore does.
        self._ttl_seconds = ttl_seconds if ttl_seconds > 0 else None
        self._timeout_seconds = timeout_seconds
        # Names with no stored title are remembered briefly so a tool that has
        # no title does not cost a store read on every call_tool start event.
        self._misses: dict[str, float] = {}
        # name -> (title, written_at): a title persisted recently is not
        # rewritten on every search_tools result.
        self._recent_writes: dict[str, tuple[str, float]] = {}
        # After a store failure or timeout the store is skipped until this time.
        self._unavailable_until = 0.0

    async def put_titles(self, *, titles: Mapping[str, str]) -> None:
        """Persist titles concurrently; one timeout bounds the whole batch."""
        pending: dict[str, str] = {}
        for name, title in titles.items():
            stripped = title.strip()
            if (
                stripped
                and LearnedTitleCache.is_storable(tool_name=name, title=stripped)
                and not self._was_written_recently(tool_name=name, title=stripped)
            ):
                pending[name] = stripped
        if not pending or self._is_unavailable():
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *(
                        self._store.put(
                            name,
                            {"title": title},
                            collection=self._collection,
                            ttl=self._ttl_seconds,
                        )
                        for name, title in pending.items()
                    )
                ),
                timeout=self._timeout_seconds,
            )
        except Exception:
            self._mark_unavailable()
            logger.warning(
                "Failed to persist %d learned tool titles", len(pending), exc_info=True
            )
            return
        now = time.monotonic()
        if len(self._recent_writes) + len(pending) > _RECENT_WRITES_MAX_ENTRIES:
            self._recent_writes.clear()
        for name, title in pending.items():
            self._recent_writes[name] = (title, now)
            self._misses.pop(name, None)

    async def get_title(self, *, tool_name: str) -> str | None:
        # The name is model output on the call_tool path: bound it before it
        # reaches the store or the miss cache.
        if not LearnedTitleCache.is_storable(tool_name=tool_name, title=""):
            return None
        if self._is_recent_miss(tool_name=tool_name) or self._is_unavailable():
            return None
        try:
            entry = await asyncio.wait_for(
                self._store.get(tool_name, collection=self._collection),
                timeout=self._timeout_seconds,
            )
        except Exception:
            # Not cached as a miss (the entry may exist); the cool-down stops
            # every start event from paying the timeout again.
            self._mark_unavailable()
            logger.warning(
                "Failed to read learned title for tool %s", tool_name, exc_info=True
            )
            return None
        title = entry.get("title") if entry else None
        if not isinstance(title, str) or not title.strip():
            self._remember_miss(tool_name=tool_name)
            return None
        return title.strip()

    def _was_written_recently(self, *, tool_name: str, title: str) -> bool:
        recent = self._recent_writes.get(tool_name)
        if recent is None or recent[0] != title:
            return False
        if self._ttl_seconds is None:
            return True
        # Rewrite well before the store TTL so a stable title does not expire.
        return time.monotonic() - recent[1] < self._ttl_seconds / 2

    def _is_unavailable(self) -> bool:
        return time.monotonic() < self._unavailable_until

    def _mark_unavailable(self) -> None:
        self._unavailable_until = time.monotonic() + _FAILURE_COOLDOWN_SECONDS

    def _is_recent_miss(self, *, tool_name: str) -> bool:
        expires_at = self._misses.get(tool_name)
        if expires_at is None:
            return False
        if time.monotonic() >= expires_at:
            del self._misses[tool_name]
            return False
        return True

    def _remember_miss(self, *, tool_name: str) -> None:
        if len(self._misses) >= _NEGATIVE_CACHE_MAX_ENTRIES:
            self._misses.clear()
        self._misses[tool_name] = time.monotonic() + _NEGATIVE_CACHE_SECONDS
