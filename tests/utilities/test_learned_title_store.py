import asyncio
from typing import Any

import pytest
from key_value.aio.stores.memory import MemoryStore

from languagemodelcommon.utilities.learned_title_store import LearnedTitleStore

_COLLECTION = "learned_tool_titles"


def _make_store(
    *, store: Any | None = None, ttl_seconds: float = 3600, timeout_seconds: float = 1
) -> LearnedTitleStore:
    return LearnedTitleStore(
        store=store if store is not None else MemoryStore(),
        collection=_COLLECTION,
        ttl_seconds=ttl_seconds,
        timeout_seconds=timeout_seconds,
    )


class _FailingStore:
    async def put(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("store down")

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("store down")


class _SlowStore:
    async def put(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(5)

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(5)


class _CountingStore(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.get_calls = 0

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        self.get_calls += 1
        return await super().get(*args, **kwargs)


class TestLearnedTitleStore:
    @pytest.mark.asyncio
    async def test_round_trip_is_visible_to_a_second_store_instance(self) -> None:
        backing = MemoryStore()
        await _make_store(store=backing).put_title(
            tool_name="get_vitals", title="  🩺 Vitals: {vitals|all}  "
        )

        title = await _make_store(store=backing).get_title(tool_name="get_vitals")

        assert title == "🩺 Vitals: {vitals|all}"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_none(self) -> None:
        assert await _make_store().get_title(tool_name="nope") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "title"),
        [
            ("get_vitals", ""),
            ("get_vitals", "   "),
            ("get_vitals", "x" * 257),
            ("n" * 129, "🩺 Vitals"),
        ],
    )
    async def test_unstorable_titles_are_not_persisted(
        self, tool_name: str, title: str
    ) -> None:
        store = _make_store()

        await store.put_title(tool_name=tool_name, title=title)

        assert await store.get_title(tool_name=tool_name) is None

    @pytest.mark.asyncio
    async def test_expired_entry_is_a_miss(self) -> None:
        store = _make_store(ttl_seconds=0.05)
        await store.put_title(tool_name="get_vitals", title="🩺 Vitals")

        await asyncio.sleep(0.1)

        assert await store.get_title(tool_name="get_vitals") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backing", [_FailingStore(), _SlowStore()])
    async def test_store_failure_or_timeout_never_raises(self, backing: Any) -> None:
        store = _make_store(store=backing, timeout_seconds=0.05)

        await store.put_title(tool_name="get_vitals", title="🩺 Vitals")

        assert await store.get_title(tool_name="get_vitals") is None

    @pytest.mark.asyncio
    async def test_recent_miss_does_not_read_the_store_again(self) -> None:
        backing = _CountingStore()
        store = _make_store(store=backing)

        await store.get_title(tool_name="untitled_tool")
        await store.get_title(tool_name="untitled_tool")

        assert backing.get_calls == 1

    @pytest.mark.asyncio
    async def test_put_clears_a_remembered_miss(self) -> None:
        store = _make_store()
        await store.get_title(tool_name="get_vitals")

        await store.put_title(tool_name="get_vitals", title="🩺 Vitals")

        assert await store.get_title(tool_name="get_vitals") == "🩺 Vitals"
