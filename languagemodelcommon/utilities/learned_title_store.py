"""Persistent store for MCP tool titles (BAI-1306, BAI-1732).

A tool reached through the ``call_tool`` discovery meta-tool is never bound as
a LangChain tool, so its title can only come from the tool catalog. This store
shares those titles across pods and restarts, in MongoDB only: there is no
in-process cache of titles (ADR 0011).

Two writers feed it:

* ``put_titles``: titles learned from ``search_tools`` listings, written with
  the configured TTL.
* ``replace_titles``: the full catalog set pre-warmed from the catalog's
  ``list_tool_titles`` tool, written without a TTL and tracked in a manifest
  entry so a later reload can delete the ones that disappeared.

Only ``tool name -> title template`` is stored (catalog metadata, no PHI and no
tenant or request data). Every operation is bounded by a timeout and never
raises: the display name is cosmetic and must not break a stream. After a
failure the store is skipped for a short cool-down. That cool-down is a circuit
breaker (one state flag), not a cache of data.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Mapping, Protocol

from key_value.aio.stores.base import BaseStore

from languagemodelcommon.utilities.tool_display_name_mapper import is_storable_title

logger = logging.getLogger(__name__)

_FAILURE_COOLDOWN_SECONDS = 30.0
_DEFAULT_BULK_TIMEOUT_SECONDS = 10.0
# Key of the entry listing the names written by the last warm. Not a tool name:
# the model-supplied name check in ``get_title`` never reads it.
_MANIFEST_KEY = "__warmed_titles_manifest__"
_MANIFEST_FIELD = "names"
# Marks an entry written by a warm (no TTL); learned writes never replace it.
_WARMED_FIELD = "warmed"


@dataclass(frozen=True)
class TitleReplaceResult:
    """Outcome of ``replace_titles`` / ``clear``; never raised, always returned."""

    ok: bool
    written: int = 0
    deleted: int = 0


class LearnedTitleStoreProtocol(Protocol):
    """Cross-pod store of learned MCP tool titles (read and learn path)."""

    async def put_titles(self, *, titles: Mapping[str, str]) -> None: ...

    async def get_title(self, *, tool_name: str) -> str | None: ...


class LearnedTitleBulkStoreProtocol(Protocol):
    """Bulk maintenance of the stored titles (reload path)."""

    async def replace_titles(
        self, *, titles: Mapping[str, str], prune: bool = True
    ) -> TitleReplaceResult: ...

    async def clear(self) -> TitleReplaceResult: ...


class LearnedTitleStore:
    """``LearnedTitleStoreProtocol`` and ``LearnedTitleBulkStoreProtocol`` backed
    by a py-key-value-aio store."""

    def __init__(
        self,
        *,
        store: BaseStore,
        collection: str,
        ttl_seconds: float,
        timeout_seconds: float,
        bulk_timeout_seconds: float = _DEFAULT_BULK_TIMEOUT_SECONDS,
    ) -> None:
        self._store = store
        self._collection = collection
        # A non-positive TTL would raise InvalidTTLError on every put; treat it
        # as "no expiry" like McpToolListStore does.
        self._ttl_seconds = ttl_seconds if ttl_seconds > 0 else None
        self._timeout_seconds = timeout_seconds
        self._bulk_timeout_seconds = bulk_timeout_seconds
        # Circuit breaker for the read/learn path: after a store failure or
        # timeout the store is skipped until this time. State only, no data.
        self._unavailable_until = 0.0

    async def put_titles(self, *, titles: Mapping[str, str]) -> None:
        """Persist learned titles concurrently; one timeout bounds the batch.

        Names that already hold a warmed entry are skipped: the warmed entry has
        no TTL and a learned write would replace it with an expiring one. The
        check reads only the entries being written (never the manifest).
        """
        pending: dict[str, str] = {}
        for name, title in titles.items():
            stripped = title.strip()
            if (
                stripped
                and name != _MANIFEST_KEY
                and is_storable_title(tool_name=name, title=stripped)
            ):
                pending[name] = stripped
        if not pending or self._is_unavailable():
            return
        try:
            await asyncio.wait_for(
                self._put_learned(pending=pending), timeout=self._timeout_seconds
            )
        except Exception:
            self._mark_unavailable()
            logger.warning(
                "Failed to persist %d learned tool titles", len(pending), exc_info=True
            )

    async def _put_learned(self, *, pending: dict[str, str]) -> None:
        names = list(pending)
        existing = await self._store.get_many(names, collection=self._collection)
        warmed = {
            name
            for name, entry in zip(names, existing, strict=True)
            if entry and entry.get(_WARMED_FIELD) is True
        }
        await asyncio.gather(
            *(
                self._store.put(
                    name,
                    {"title": title},
                    collection=self._collection,
                    ttl=self._ttl_seconds,
                )
                for name, title in pending.items()
                if name not in warmed
            )
        )

    async def get_title(self, *, tool_name: str) -> str | None:
        # The name is model output on the call_tool path: bound it before it
        # reaches the store.
        if tool_name == _MANIFEST_KEY or not is_storable_title(
            tool_name=tool_name, title=""
        ):
            return None
        if self._is_unavailable():
            return None
        try:
            entry = await asyncio.wait_for(
                self._store.get(tool_name, collection=self._collection),
                timeout=self._timeout_seconds,
            )
        except Exception:
            self._mark_unavailable()
            logger.warning(
                "Failed to read learned title for tool %s", tool_name, exc_info=True
            )
            return None
        title = entry.get("title") if entry else None
        if not isinstance(title, str) or not title.strip():
            return None
        return title.strip()

    async def replace_titles(
        self, *, titles: Mapping[str, str], prune: bool = True
    ) -> TitleReplaceResult:
        """Write ``titles`` (no TTL), then delete warmed entries not in the set.

        Write first, delete after, so a concurrent reader never sees an empty
        window. ``prune=False`` (a partial catalog response) only upserts and
        keeps every previously warmed name. Only entries written by a previous
        warm are deleted; titles learned from ``search_tools`` carry a TTL and
        age out on their own. Never raises.
        """
        valid: dict[str, str] = {}
        for name, title in titles.items():
            stripped = title.strip() if isinstance(title, str) else ""
            if (
                stripped
                and name != _MANIFEST_KEY
                and is_storable_title(tool_name=name, title=stripped)
            ):
                valid[name] = stripped
        try:
            return await asyncio.wait_for(
                self._replace(valid=valid, prune=prune),
                timeout=self._bulk_timeout_seconds,
            )
        except Exception:
            logger.warning(
                "Failed to replace %d stored tool titles", len(valid), exc_info=True
            )
            return TitleReplaceResult(ok=False)

    async def _replace(
        self, *, valid: dict[str, str], prune: bool
    ) -> TitleReplaceResult:
        previous = await self._read_manifest()
        if valid:
            await self._store.put_many(
                list(valid),
                [{"title": title, _WARMED_FIELD: True} for title in valid.values()],
                collection=self._collection,
                ttl=None,
            )
        kept = set(valid) if prune else set(valid) | previous
        await self._store.put(
            _MANIFEST_KEY,
            {_MANIFEST_FIELD: sorted(kept)},
            collection=self._collection,
            ttl=None,
        )
        stale = sorted(previous - kept)
        deleted = (
            await self._store.delete_many(stale, collection=self._collection)
            if stale
            else 0
        )
        return TitleReplaceResult(ok=True, written=len(valid), deleted=deleted)

    async def clear(self) -> TitleReplaceResult:
        """Delete every warmed entry and the manifest. Never raises.

        Learned (TTL) entries are not enumerable through the store interface
        and age out on their own.
        """
        try:
            return await asyncio.wait_for(
                self._clear(), timeout=self._bulk_timeout_seconds
            )
        except Exception:
            logger.warning("Failed to clear stored tool titles", exc_info=True)
            return TitleReplaceResult(ok=False)

    async def _clear(self) -> TitleReplaceResult:
        previous = await self._read_manifest()
        deleted = (
            await self._store.delete_many(sorted(previous), collection=self._collection)
            if previous
            else 0
        )
        await self._store.delete(_MANIFEST_KEY, collection=self._collection)
        return TitleReplaceResult(ok=True, deleted=deleted)

    async def _read_manifest(self) -> set[str]:
        entry = await self._store.get(_MANIFEST_KEY, collection=self._collection)
        names = entry.get(_MANIFEST_FIELD) if entry else None
        if not isinstance(names, list):
            return set()
        return {name for name in names if isinstance(name, str)}

    def _is_unavailable(self) -> bool:
        return time.monotonic() < self._unavailable_until

    def _mark_unavailable(self) -> None:
        self._unavailable_until = time.monotonic() + _FAILURE_COOLDOWN_SECONDS
