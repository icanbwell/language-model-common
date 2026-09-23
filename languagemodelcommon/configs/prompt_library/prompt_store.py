import hashlib
import logging
from typing import Any

from key_value.aio.stores.base import BaseDestroyCollectionStore, BaseStore


logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 3600


class PromptStore:
    """Persistent prompt store backed by py-key-value-aio.

    Stores prompt content keyed by name in a configurable collection.
    Used as the primary lookup for prompt resolution, eliminating
    filesystem race conditions during multi-worker startup.

    Entries carry a TTL (default 3600s) and, when *source_ref* is given,
    a `v{SCHEMA_VERSION}:{ref_hash}:` key prefix -- the same scheme
    `languagemodelcommon.ConfigReader` uses for model configs (BAI-720).
    Before this, entries were cached under the bare prompt name forever
    (no TTL at all): a new prompt version pushed upstream was never
    picked up without a manual `clear()` (BAI-933). *source_ref* is
    optional and defaults to no prefix, preserving the exact bare-name
    key shape for callers that haven't been updated to pass it.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        store: BaseStore,
        collection: str = "prompts",
        source_ref: str | None = None,
        ttl_seconds: int | None = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._store = store
        self._collection = collection
        self._ref_hash = self._compute_ref_hash(source_ref) if source_ref else None
        # A non-positive value (e.g. an operator setting PROMPT_STORE_TTL_SECONDS=0,
        # a natural way to try to disable caching) would otherwise crash every
        # put_prompt call: py-key-value-aio's BaseStore.put raises InvalidTTLError
        # for ttl <= 0. Treat it as "no expiry" instead -- same guard the sibling
        # McpToolListStore already applies to the identical put(..., ttl=...) call
        # shape.
        self._ttl_seconds = ttl_seconds if ttl_seconds and ttl_seconds > 0 else None

    @staticmethod
    def _compute_ref_hash(source_ref: str) -> str:
        return hashlib.sha256(source_ref.encode("utf-8")).hexdigest()[:12]

    def _key(self, name: str) -> str:
        if self._ref_hash is None:
            return name
        return f"v{self.SCHEMA_VERSION}:{self._ref_hash}:{name}"

    async def get_prompt(self, *, name: str) -> str | None:
        result: dict[str, Any] | None = await self._store.get(
            self._key(name), collection=self._collection
        )
        if result is None:
            return None

        if result.get("schema_version") != self.SCHEMA_VERSION:
            return None

        content = result.get("content")
        if not isinstance(content, str):
            return None
        return content

    async def put_prompt(self, *, name: str, content: str) -> None:
        value: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "content": content,
        }
        await self._store.put(
            self._key(name),
            value,
            collection=self._collection,
            ttl=self._ttl_seconds,
        )

    async def delete_prompt(self, *, name: str) -> None:
        await self._store.delete(self._key(name), collection=self._collection)

    async def clear(self) -> None:
        if isinstance(self._store, BaseDestroyCollectionStore):
            await self._store.destroy_collection(collection=self._collection)
            self._store._setup_collection_complete[self._collection] = False
        else:
            logger.warning(
                "Prompt store does not support destroy_collection; "
                "clear is a no-op for this backend"
            )
