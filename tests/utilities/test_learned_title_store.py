import asyncio
from typing import Any

import pytest
from key_value.aio.stores.memory import MemoryStore

from languagemodelcommon.utilities.learned_title_store import LearnedTitleStore

_COLLECTION = "learned_tool_titles"


def _make_store(
    *,
    store: Any | None = None,
    ttl_seconds: float = 3600,
    timeout_seconds: float = 1,
    bulk_timeout_seconds: float = 5,
) -> LearnedTitleStore:
    return LearnedTitleStore(
        store=store if store is not None else MemoryStore(),
        collection=_COLLECTION,
        ttl_seconds=ttl_seconds,
        timeout_seconds=timeout_seconds,
        bulk_timeout_seconds=bulk_timeout_seconds,
    )


class _FailingStore:
    async def put(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("store down")

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("store down")

    async def get_many(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("store down")

    async def put_many(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("store down")

    async def delete_many(self, *args: Any, **kwargs: Any) -> int:
        raise RuntimeError("store down")

    async def delete(self, *args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("store down")


class _SlowStore:
    async def put(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(5)

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(5)

    async def get_many(self, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(5)

    async def put_many(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(5)

    async def delete_many(self, *args: Any, **kwargs: Any) -> int:
        await asyncio.sleep(5)
        return 0

    async def delete(self, *args: Any, **kwargs: Any) -> bool:
        await asyncio.sleep(5)
        return False


class _CountingStore(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.get_calls = 0
        self.put_calls = 0

    async def put(self, *args: Any, **kwargs: Any) -> Any:
        self.put_calls += 1
        return await super().put(*args, **kwargs)

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        self.get_calls += 1
        return await super().get(*args, **kwargs)


class TestLearnedTitleStore:
    @pytest.mark.asyncio
    async def test_round_trip_is_visible_to_a_second_store_instance(self) -> None:
        backing = MemoryStore()
        await _make_store(store=backing).put_titles(
            titles={"get_vitals": "  🩺 Vitals: {vitals|all}  "}
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

        await store.put_titles(titles={tool_name: title})

        assert await store.get_title(tool_name=tool_name) is None

    @pytest.mark.asyncio
    async def test_expired_entry_is_a_miss(self) -> None:
        store = _make_store(ttl_seconds=0.05)
        await store.put_titles(titles={"get_vitals": "🩺 Vitals"})

        await asyncio.sleep(0.1)

        assert await store.get_title(tool_name="get_vitals") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backing", [_FailingStore(), _SlowStore()])
    async def test_store_failure_or_timeout_never_raises(self, backing: Any) -> None:
        store = _make_store(store=backing, timeout_seconds=0.05)

        await store.put_titles(titles={"get_vitals": "🩺 Vitals"})

        assert await store.get_title(tool_name="get_vitals") is None

    @pytest.mark.asyncio
    async def test_no_in_process_title_state(self) -> None:
        """Every read goes to the backing store: a second instance sees a write
        immediately and an untitled name is read again (no miss cache)."""
        backing = _CountingStore()
        store = _make_store(store=backing)

        await store.get_title(tool_name="get_vitals")
        await _make_store(store=backing).put_titles(titles={"get_vitals": "🩺 Vitals"})

        assert await store.get_title(tool_name="get_vitals") == "🩺 Vitals"
        assert not hasattr(store, "_misses")
        assert not hasattr(store, "_recent_writes")

    @pytest.mark.asyncio
    async def test_batch_is_written_in_one_call(self) -> None:
        store = _make_store()

        await store.put_titles(titles={"a_tool": "🅰️ A", "b_tool": "🅱️ B"})

        assert await store.get_title(tool_name="a_tool") == "🅰️ A"
        assert await store.get_title(tool_name="b_tool") == "🅱️ B"

    @pytest.mark.asyncio
    async def test_oversized_name_never_reaches_the_store(self) -> None:
        backing = _CountingStore()
        store = _make_store(store=backing)

        assert await store.get_title(tool_name="n" * 129) is None
        assert backing.get_calls == 0

    @pytest.mark.asyncio
    async def test_failure_starts_a_cooldown_that_skips_the_store(self) -> None:
        backing = _CountingStore()
        store = _make_store(store=backing, timeout_seconds=0.05)
        store._store = _SlowStore()  # type: ignore[assignment]
        await store.get_title(tool_name="get_vitals")
        store._store = backing

        assert await store.get_title(tool_name="get_vitals") is None
        await store.put_titles(titles={"get_vitals": "🩺 Vitals"})

        assert backing.get_calls == 0
        assert backing.put_calls == 0


class TestReplaceTitles:
    @pytest.mark.asyncio
    async def test_writes_the_new_set_and_deletes_names_no_longer_present(self) -> None:
        store = _make_store()
        await store.replace_titles(titles={"a_tool": "🅰️ A", "b_tool": "🅱️ B"})

        result = await store.replace_titles(titles={"b_tool": "🅱️ B2", "c_tool": "C"})

        assert (result.ok, result.written, result.deleted) == (True, 2, 1)
        assert await store.get_title(tool_name="a_tool") is None
        assert await store.get_title(tool_name="b_tool") == "🅱️ B2"
        assert await store.get_title(tool_name="c_tool") == "C"

    @pytest.mark.asyncio
    async def test_old_titles_stay_readable_until_after_the_new_set_is_written(
        self,
    ) -> None:
        seen: list[str | None] = []

        class _ObservingStore(MemoryStore):
            async def delete_many(self, *args: Any, **kwargs: Any) -> int:
                seen.append(
                    (await self.get("a_tool", collection=_COLLECTION) or {}).get(
                        "title"
                    )
                )
                seen.append(
                    (await self.get("c_tool", collection=_COLLECTION) or {}).get(
                        "title"
                    )
                )
                deleted: int = await super().delete_many(*args, **kwargs)
                return deleted

        store = _make_store(store=_ObservingStore())
        await store.replace_titles(titles={"a_tool": "A"})

        await store.replace_titles(titles={"c_tool": "C"})

        assert seen == ["A", "C"]

    @pytest.mark.asyncio
    async def test_prune_false_keeps_previously_warmed_names(self) -> None:
        store = _make_store()
        await store.replace_titles(titles={"a_tool": "A"})

        result = await store.replace_titles(titles={"b_tool": "B"}, prune=False)
        await store.replace_titles(titles={"b_tool": "B"}, prune=True)

        assert result.deleted == 0
        assert await store.get_title(tool_name="a_tool") is None

    @pytest.mark.asyncio
    async def test_prune_false_does_not_delete(self) -> None:
        store = _make_store()
        await store.replace_titles(titles={"a_tool": "A"})

        await store.replace_titles(titles={"b_tool": "B"}, prune=False)

        assert await store.get_title(tool_name="a_tool") == "A"
        assert await store.get_title(tool_name="b_tool") == "B"

    @pytest.mark.asyncio
    async def test_warmed_titles_have_no_ttl_and_learned_titles_keep_theirs(
        self,
    ) -> None:
        backing = MemoryStore()
        store = _make_store(store=backing, ttl_seconds=3600)

        await store.replace_titles(titles={"warmed_tool": "W"})
        await store.put_titles(titles={"learned_tool": "L"})

        _, warmed_ttl = await backing.ttl("warmed_tool", collection=_COLLECTION)
        _, learned_ttl = await backing.ttl("learned_tool", collection=_COLLECTION)
        assert warmed_ttl is None
        assert learned_ttl is not None

    @pytest.mark.asyncio
    async def test_learned_write_does_not_replace_a_warmed_entry_with_an_expiring_one(
        self,
    ) -> None:
        backing = MemoryStore()
        store = _make_store(store=backing, ttl_seconds=3600)
        await store.replace_titles(titles={"warmed_tool": "W"})

        await store.put_titles(titles={"warmed_tool": "W-from-search"})

        _, ttl = await backing.ttl("warmed_tool", collection=_COLLECTION)
        assert ttl is None
        assert await store.get_title(tool_name="warmed_tool") == "W"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("titles", "expected_written"),
        [
            ({"ok_tool": "T", "blank": "  ", "long": "x" * 257}, 1),
            ({"n" * 129: "T"}, 0),
            ({}, 0),
        ],
    )
    async def test_unstorable_entries_are_dropped(
        self, titles: dict[str, str], expected_written: int
    ) -> None:
        result = await _make_store().replace_titles(titles=titles)

        assert (result.ok, result.written) == (True, expected_written)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backing", [_FailingStore(), _SlowStore()])
    async def test_store_failure_or_timeout_returns_not_ok_and_never_raises(
        self, backing: Any
    ) -> None:
        store = _make_store(store=backing, bulk_timeout_seconds=0.05)

        assert (await store.replace_titles(titles={"a_tool": "A"})).ok is False
        assert (await store.clear()).ok is False


class TestClear:
    @pytest.mark.asyncio
    async def test_removes_warmed_titles(self) -> None:
        store = _make_store()
        await store.replace_titles(titles={"a_tool": "A", "b_tool": "B"})

        result = await store.clear()

        assert (result.ok, result.deleted) == (True, 2)
        assert await store.get_title(tool_name="a_tool") is None

    @pytest.mark.asyncio
    async def test_on_an_empty_store_is_ok(self) -> None:
        result = await _make_store().clear()

        assert (result.ok, result.deleted) == (True, 0)

    @pytest.mark.asyncio
    async def test_manifest_key_is_not_readable_as_a_title(self) -> None:
        store = _make_store()
        await store.replace_titles(titles={"a_tool": "A"})

        assert await store.get_title(tool_name="__warmed_titles_manifest__") is None


class TestLearnedWriteDoesNotReadManifest:
    @pytest.mark.asyncio
    async def test_put_titles_never_reads_the_manifest_key(self) -> None:
        reads: list[str] = []

        class _Spy(MemoryStore):
            async def get(self, key: str, *args: Any, **kwargs: Any) -> Any:
                reads.append(key)
                return await super().get(key, *args, **kwargs)

        store = _make_store(store=_Spy())
        await store.replace_titles(titles={"warmed_tool": "W"})
        reads.clear()

        await store.put_titles(titles={"a_tool": "A", "warmed_tool": "X"})

        assert "__warmed_titles_manifest__" not in reads
