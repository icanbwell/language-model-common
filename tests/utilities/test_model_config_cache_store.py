"""Tests for model config cache store factory."""

import asyncio
from typing import Any, Self, cast

import pytest
from key_value.aio.errors.store import StoreSetupError
from key_value.aio.stores.mongodb import MongoDBStore
from pymongo import AsyncMongoClient
from pymongo.errors import AutoReconnect, InvalidOperation, OperationFailure

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

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_attempts": 0}, "max_attempts"),
            ({"ping_timeout_seconds": 0.0}, "ping_timeout_seconds"),
            ({"ping_timeout_seconds": -1.0}, "ping_timeout_seconds"),
            ({"base_delay_seconds": -0.1}, "base_delay_seconds"),
            (
                {"base_delay_seconds": 2.0, "max_delay_seconds": 1.0},
                "max_delay_seconds",
            ),
        ],
    )
    def test_rejects_invalid_bounds(
        self, kwargs: dict[str, float], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            PingRetryPolicy(**kwargs)  # type: ignore[arg-type]


class TestCreateCacheStoreRetryPolicy:
    @pytest.mark.asyncio
    async def test_factory_passes_the_policy_to_the_store(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Three transient failures: the default policy (4 attempts) would
        # recover, a 2-attempt policy handed to the factory must give up.
        store = create_cache_store(
            mongo_url="mongodb://localhost:27017",
            ping_retry_policy=PingRetryPolicy(
                max_attempts=2, base_delay_seconds=0.0, max_delay_seconds=0.0
            ),
        )
        fake_db = _FakeDb(outcomes=[AutoReconnect("down")] * 3)
        monkeypatch.setattr(store, "_db", fake_db)

        with pytest.raises(
            StoreSetupError, match="Model config cache failed to connect"
        ):
            await store.setup()

        assert fake_db.ping_calls == 2


class _FakeClient:
    """Mimics AsyncMongoClient: unusable once closed, like the real driver."""

    def __init__(self, *, ping_failures: int = 0, fail_on_enter: bool = False) -> None:
        self.closed = False
        self._ping_failures = ping_failures
        self._fail_on_enter = fail_on_enter
        self.ping_calls = 0

    async def __aenter__(self) -> Self:
        if self._fail_on_enter:
            raise AutoReconnect("could not connect")
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.closed = True

    def __getitem__(self, db_name: str) -> "_FakeClientDb":
        return _FakeClientDb(client=self, name=db_name)


class _FakeClientDb:
    def __init__(self, *, client: _FakeClient, name: str) -> None:
        self._client = client
        self.name = name

    async def command(self, name: str) -> dict[str, Any]:
        if self._client.closed:
            raise InvalidOperation("Cannot use AsyncMongoClient after close")
        self._client.ping_calls += 1
        if self._client.ping_calls <= self._client._ping_failures:
            raise AutoReconnect("down")
        return {"ok": 1}


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestClientRecovery:
    """After the retry budget is exhausted the base store leaves its client
    closed; the next setup, past the cooldown, must build a fresh one."""

    @staticmethod
    def _make_store(
        *,
        first_client: _FakeClient,
        replacements: list[_FakeClient],
        clock: _Clock,
        cooldown: float = 30.0,
        use_own_client: bool = False,
        max_attempts: int = 1,
    ) -> tuple[ValidatingMongoDBStore, list[str]]:
        factory_urls: list[str] = []

        def client_factory(url: str) -> AsyncMongoClient[dict[str, Any]]:
            factory_urls.append(url)
            return cast(AsyncMongoClient[dict[str, Any]], replacements.pop(0))

        async def no_sleep(delay: float) -> None:
            return None

        policy = PingRetryPolicy(
            max_attempts=max_attempts, recovery_cooldown_seconds=cooldown
        )
        if use_own_client:
            store = ValidatingMongoDBStore(
                client=AsyncMongoClient("mongodb://localhost:27017"),
                db_name="test_db",
                ping_retry_policy=policy,
                sleep=no_sleep,
                client_factory=client_factory,
                monotonic=clock,
            )
        else:
            store = ValidatingMongoDBStore(
                url="mongodb://localhost:27017",
                db_name="test_db",
                ping_retry_policy=policy,
                sleep=no_sleep,
                client_factory=client_factory,
                monotonic=clock,
            )
        store._client = cast(AsyncMongoClient[dict[str, Any]], first_client)
        store._db = cast(Any, first_client["test_db"])
        return store, factory_urls

    @pytest.mark.asyncio
    async def test_builds_a_fresh_client_after_cooldown(self) -> None:
        clock = _Clock()
        first, second = _FakeClient(ping_failures=99), _FakeClient()
        store, factory_urls = self._make_store(
            first_client=first, replacements=[second], clock=clock
        )

        with pytest.raises(StoreSetupError):
            await store.setup()
        clock.now += 31.0
        await store.setup()

        assert first.closed is True
        assert second.closed is False
        assert factory_urls == ["mongodb://localhost:27017"]
        await store.close()

    @pytest.mark.asyncio
    async def test_fails_fast_without_pinging_during_cooldown(self) -> None:
        clock = _Clock()
        first, second = _FakeClient(ping_failures=99), _FakeClient()
        store, factory_urls = self._make_store(
            first_client=first, replacements=[second], clock=clock
        )

        with pytest.raises(StoreSetupError):
            await store.setup()
        clock.now += 5.0
        with pytest.raises(StoreSetupError, match="try again in"):
            await store.setup()

        assert factory_urls == []
        assert first.ping_calls == 1
        assert second.ping_calls == 0

    @pytest.mark.asyncio
    async def test_keeps_waiting_when_the_new_client_also_fails(self) -> None:
        clock = _Clock()
        first = _FakeClient(ping_failures=99)
        second, third = _FakeClient(ping_failures=99), _FakeClient()
        store, factory_urls = self._make_store(
            first_client=first, replacements=[second, third], clock=clock
        )

        with pytest.raises(StoreSetupError):
            await store.setup()
        clock.now += 31.0
        with pytest.raises(StoreSetupError):
            await store.setup()
        clock.now += 5.0
        with pytest.raises(StoreSetupError, match="try again in"):
            await store.setup()
        clock.now += 31.0
        await store.setup()

        assert len(factory_urls) == 2
        assert third.closed is False
        await store.close()

    @pytest.mark.asyncio
    async def test_never_rebuilds_a_healthy_store(self) -> None:
        clock = _Clock()
        first = _FakeClient()
        store, factory_urls = self._make_store(
            first_client=first, replacements=[], clock=clock
        )

        await store.setup()
        clock.now += 1000.0
        await store.setup()

        assert factory_urls == []
        assert first.closed is False
        await store.close()

    @pytest.mark.asyncio
    async def test_does_not_rebuild_a_client_it_does_not_own(self) -> None:
        clock = _Clock()
        first = _FakeClient(ping_failures=1)
        store, factory_urls = self._make_store(
            first_client=first,
            replacements=[_FakeClient()],
            clock=clock,
            use_own_client=True,
        )

        with pytest.raises(StoreSetupError):
            await store.setup()
        clock.now += 31.0
        await store.setup()

        assert factory_urls == []
        assert first.ping_calls == 2

    @pytest.mark.parametrize("cooldown", [-1.0, 0.0, float("nan")])
    def test_rejects_non_positive_cooldown(self, cooldown: float) -> None:
        with pytest.raises(ValueError, match="recovery_cooldown_seconds"):
            PingRetryPolicy(recovery_cooldown_seconds=cooldown)

    @pytest.mark.parametrize("timeout", [-1.0, 0.0, float("nan")])
    def test_rejects_non_positive_recovery_ping_timeout(self, timeout: float) -> None:
        with pytest.raises(ValueError, match="recovery_ping_timeout_seconds"):
            PingRetryPolicy(recovery_ping_timeout_seconds=timeout)

    @pytest.mark.asyncio
    async def test_recovery_is_a_single_probe_not_the_full_retry_budget(self) -> None:
        clock = _Clock()
        first = _FakeClient(ping_failures=99)
        second, third = _FakeClient(ping_failures=99), _FakeClient()
        store, _ = self._make_store(
            first_client=first,
            replacements=[second, third],
            clock=clock,
            max_attempts=3,
        )

        with pytest.raises(StoreSetupError):
            await store.setup()
        assert first.ping_calls == 3
        clock.now += 31.0
        with pytest.raises(StoreSetupError):
            await store.setup()

        assert second.ping_calls == 1
        clock.now += 31.0
        await store.setup()
        assert third.ping_calls == 1
        await store.close()

    @pytest.mark.asyncio
    async def test_failure_before_the_ping_still_leads_to_a_rebuild(self) -> None:
        clock = _Clock()
        first = _FakeClient(fail_on_enter=True)
        second = _FakeClient()
        store, factory_urls = self._make_store(
            first_client=first, replacements=[second], clock=clock
        )

        with pytest.raises(StoreSetupError):
            await store.setup()
        clock.now += 31.0
        await store.setup()

        assert len(factory_urls) == 1
        assert second.ping_calls == 1
        await store.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("elapsed", "still_cooling_down"),
        [(29.9, True), (30.0, False), (30.1, False)],
    )
    async def test_cooldown_boundary(
        self, elapsed: float, still_cooling_down: bool
    ) -> None:
        clock = _Clock()
        first, second = _FakeClient(ping_failures=99), _FakeClient()
        store, factory_urls = self._make_store(
            first_client=first, replacements=[second], clock=clock
        )

        with pytest.raises(StoreSetupError):
            await store.setup()
        clock.now += elapsed
        if still_cooling_down:
            with pytest.raises(StoreSetupError, match="try again in"):
                await store.setup()
            assert factory_urls == []
        else:
            await store.setup()
            assert len(factory_urls) == 1
            await store.close()

    @pytest.mark.asyncio
    async def test_concurrent_callers_share_one_attempt(self) -> None:
        clock = _Clock()
        first, second = _FakeClient(ping_failures=99), _FakeClient()
        store, factory_urls = self._make_store(
            first_client=first, replacements=[second], clock=clock
        )

        results = await asyncio.gather(
            store.setup(), store.setup(), return_exceptions=True
        )

        assert all(isinstance(result, StoreSetupError) for result in results)
        assert first.ping_calls == 1
        assert factory_urls == []
