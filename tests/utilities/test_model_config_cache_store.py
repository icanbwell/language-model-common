"""Tests for model config cache store factory."""

import asyncio
from typing import Any

import pytest
from key_value.aio.errors.store import StoreSetupError
from key_value.aio.stores.mongodb import MongoDBStore
from pymongo.errors import AutoReconnect, OperationFailure

from languagemodelcommon.utilities.cache.model_config_cache_store import (
    PingRetryPolicy,
    ValidatingMongoDBStore,
    create_cache_store,
)


class TestCreateCacheStore:
    """create_cache_store returns ValidatingMongoDBStore."""

    def test_raises_without_url(self) -> None:
        with pytest.raises(ValueError, match="No MongoDB URL"):
            create_cache_store()

    def test_raises_with_none_url(self) -> None:
        with pytest.raises(ValueError, match="No MongoDB URL"):
            create_cache_store(mongo_url=None)

    def test_returns_validating_mongodb_store(self) -> None:
        store = create_cache_store(
            mongo_url="mongodb://localhost:27017",
            mongo_db_name="test_db",
            collection="test_cache",
        )
        assert isinstance(store, ValidatingMongoDBStore)
        assert isinstance(store, MongoDBStore)

    def test_custom_collection(self) -> None:
        store = create_cache_store(
            mongo_url="mongodb://localhost:27017",
            collection="custom_collection",
        )
        assert isinstance(store, ValidatingMongoDBStore)
        assert store.default_collection == "custom_collection"

    @pytest.mark.asyncio
    async def test_aenter_raises_on_unreachable_mongo(self) -> None:
        store = create_cache_store(
            mongo_url="mongodb://unreachable-host:27017",
            mongo_db_name="test_db",
            collection="test_cache",
            ping_retry_policy=PingRetryPolicy(max_attempts=1, ping_timeout_seconds=1.0),
        )
        with pytest.raises(
            StoreSetupError, match="Model config cache failed to connect"
        ):
            await store.__aenter__()


class _FakeDb:
    """Stands in for the Mongo database; fails per a scripted sequence."""

    name = "test_db"

    def __init__(self, *, outcomes: list[BaseException | None]) -> None:
        self._outcomes = list(outcomes)
        self.ping_calls = 0

    async def command(self, name: str) -> dict[str, Any]:
        self.ping_calls += 1
        outcome = self._outcomes.pop(0) if self._outcomes else None
        if isinstance(outcome, BaseException):
            raise outcome
        return {"ok": 1}


def _make_store(
    *,
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[BaseException | None],
    policy: PingRetryPolicy,
) -> tuple[ValidatingMongoDBStore, _FakeDb, list[float]]:
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    store = ValidatingMongoDBStore(
        url="mongodb://localhost:27017",
        db_name="test_db",
        default_collection="test_cache",
        ping_retry_policy=policy,
        sleep=record_sleep,
    )
    fake_db = _FakeDb(outcomes=outcomes)
    monkeypatch.setattr(store, "_db", fake_db)
    return store, fake_db, sleeps


class TestPingRetry:
    """A transient ping failure must be retried before setup raises, because the
    base store closes its Mongo client when setup raises."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("transient_failures", [0, 1, 3])
    async def test_setup_succeeds_after_transient_failures(
        self, monkeypatch: pytest.MonkeyPatch, transient_failures: int
    ) -> None:
        store, fake_db, sleeps = _make_store(
            monkeypatch=monkeypatch,
            outcomes=[AutoReconnect("blip")] * transient_failures,
            policy=PingRetryPolicy(max_attempts=4),
        )

        await store.setup()

        assert fake_db.ping_calls == transient_failures + 1
        assert len(sleeps) == transient_failures
        assert store._exit_stack_entered is True
        await store.close()

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, fake_db, sleeps = _make_store(
            monkeypatch=monkeypatch,
            outcomes=[AutoReconnect("down")] * 10,
            policy=PingRetryPolicy(max_attempts=3),
        )

        with pytest.raises(
            StoreSetupError, match="Model config cache failed to connect"
        ):
            await store.setup()

        assert fake_db.ping_calls == 3
        assert len(sleeps) == 2

    @pytest.mark.asyncio
    async def test_non_transient_error_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, fake_db, sleeps = _make_store(
            monkeypatch=monkeypatch,
            outcomes=[OperationFailure("auth failed")],
            policy=PingRetryPolicy(max_attempts=4),
        )

        with pytest.raises(
            StoreSetupError, match="Model config cache failed to connect"
        ):
            await store.setup()

        assert fake_db.ping_calls == 1
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_hung_ping_times_out_and_is_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, fake_db, sleeps = _make_store(
            monkeypatch=monkeypatch,
            outcomes=[None],
            policy=PingRetryPolicy(max_attempts=2, ping_timeout_seconds=0.05),
        )
        calls = 0

        async def hang_once_then_answer(name: str) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if calls == 1:
                await asyncio.sleep(5)
            return {"ok": 1}

        monkeypatch.setattr(fake_db, "command", hang_once_then_answer)

        await store.setup()

        assert calls == 2
        assert len(sleeps) == 1
        await store.close()


class TestPingRetryPolicy:
    @pytest.mark.parametrize(
        ("attempt", "ceiling"),
        [(1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0), (5, 4.0), (9, 4.0)],
    )
    def test_backoff_delay_is_jittered_within_exponential_ceiling(
        self, attempt: int, ceiling: float
    ) -> None:
        policy = PingRetryPolicy(base_delay_seconds=0.5, max_delay_seconds=4.0)

        delays = [policy.backoff_delay(attempt=attempt) for _ in range(200)]

        assert all(0.0 <= delay <= ceiling for delay in delays)
        assert len(set(delays)) > 1

    def test_default_budget_covers_pymongo_server_selection_window(self) -> None:
        pymongo_default_server_selection_seconds = 30.0
        policy = PingRetryPolicy()

        total_ping_budget = policy.max_attempts * policy.ping_timeout_seconds

        assert total_ping_budget >= pymongo_default_server_selection_seconds

    def test_rejects_zero_attempts(self) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            PingRetryPolicy(max_attempts=0)
