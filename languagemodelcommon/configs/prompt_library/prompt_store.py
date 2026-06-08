import logging
from typing import Any

from key_value.aio.stores.base import BaseDestroyCollectionStore, BaseStore


logger = logging.getLogger(__name__)


class PromptStore:
    """Persistent prompt store backed by py-key-value-aio.

    Stores prompt content keyed by name in a configurable collection.
    Used as the primary lookup for prompt resolution, eliminating
    filesystem race conditions during multi-worker startup.
    """

    SCHEMA_VERSION = 1

    def __init__(self, *, store: BaseStore, collection: str = "prompts") -> None:
        self._store = store
        self._collection = collection

    async def get_prompt(self, *, name: str) -> str | None:
        result: dict[str, Any] | None = await self._store.get(
            name, collection=self._collection
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
        await self._store.put(name, value, collection=self._collection)

    async def delete_prompt(self, *, name: str) -> None:
        await self._store.delete(name, collection=self._collection)

    async def clear(self) -> None:
        if isinstance(self._store, BaseDestroyCollectionStore):
            await self._store.destroy_collection(collection=self._collection)
            self._store._setup_collection_complete[self._collection] = False
        else:
            logger.warning(
                "Prompt store does not support destroy_collection; "
                "clear is a no-op for this backend"
            )
