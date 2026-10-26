"""Persistent store for MCP tool titles learned at runtime (BAI-1306).

A tool reached through the ``call_tool`` discovery meta-tool is never bound as
a LangChain tool, so its title is only learned from a ``search_tools`` or
``call_tool`` result. ``LearnedTitleCache`` shares those titles across requests
on one pod; this store shares them across pods and restarts so a cold pod can
render the first ``call_tool`` start line with the real title.

Only ``tool name -> title template`` is stored (catalog metadata, no PHI and
no tenant or request data). Every operation is bounded by a timeout and never
raises: the display name is cosmetic and must not break a stream.
"""

import asyncio
import logging
import time
from typing import Protocol

from key_value.aio.stores.base import BaseStore

from languagemodelcommon.utilities.tool_display_name_mapper import LearnedTitleCache

logger = logging.getLogger(__name__)

_NEGATIVE_CACHE_SECONDS = 60.0
_NEGATIVE_CACHE_MAX_ENTRIES = 1024


class LearnedTitleStoreProtocol(Protocol):
    """Cross-pod store of learned MCP tool titles."""

    async def put_title(self, *, tool_name: str, title: str) -> None: ...

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

    async def put_title(self, *, tool_name: str, title: str) -> None:
        stripped_title = title.strip()
        if not stripped_title or not LearnedTitleCache.is_storable(
            tool_name=tool_name, title=stripped_title
        ):
            return
        try:
            await asyncio.wait_for(
                self._store.put(
                    tool_name,
                    {"title": stripped_title},
                    collection=self._collection,
                    ttl=self._ttl_seconds,
                ),
                timeout=self._timeout_seconds,
            )
        except Exception:
            logger.warning(
                "Failed to persist learned title for tool %s", tool_name, exc_info=True
            )
            return
        self._misses.pop(tool_name, None)

    async def get_title(self, *, tool_name: str) -> str | None:
        if self._is_recent_miss(tool_name=tool_name):
            return None
        try:
            entry = await asyncio.wait_for(
                self._store.get(tool_name, collection=self._collection),
                timeout=self._timeout_seconds,
            )
        except Exception:
            # Store errors are not cached as misses: the store may recover.
            logger.warning(
                "Failed to read learned title for tool %s", tool_name, exc_info=True
            )
            return None
        title = entry.get("title") if entry else None
        if not isinstance(title, str) or not title.strip():
            self._remember_miss(tool_name=tool_name)
            return None
        return title.strip()

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
