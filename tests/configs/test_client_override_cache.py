"""Tests for the per-client override config cache (BAI-1380, ADR 0009).

The GitHub helper is the external boundary and is replaced by ``FakeGitHub``.
Everything else (ConfigReader, the cache, the file reader, the key-value store
and the advisory lock) is real.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from key_value.aio.stores.memory import MemoryStore
from oidcauthlib.cache_gate.advisory_lock import LOCK_COLLECTION

from languagemodelcommon.configs.config_reader import (
    config_reader as config_reader_module,
)
from languagemodelcommon.configs.config_reader.config_reader import ConfigReader
from languagemodelcommon.configs.config_reader.override_config_cache import (
    ClientOverrideConfigCache,
)
from languagemodelcommon.configs.config_reader.override_read_result import (
    DownloadLockHeldError,
    OverrideOutcome,
    OverrideReadResult,
)
from languagemodelcommon.configs.prompt_library.prompt_library_environment_variables import (
    PromptLibraryEnvironmentVariables,
)
from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
    PromptLibraryManager,
)
from languagemodelcommon.configs.schemas.config_schema import ChatModelConfig
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.ref_hash import compute_short_ref_hash

CONFIG_PATH = "github://org/repo/configs?ref=main"
COLLECTION = "models"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeGitHub:
    """Stands in for GitHubDirectoryHelper (the GitHub boundary)."""

    def __init__(self) -> None:
        self.dirs: dict[str, Path] = {}
        self.effects: dict[str, Any] = {}
        self.calls = 0
        self.absent_confirmed = True
        self.delay = 0.0
        self.lock_strictness: list[bool] = []

    @staticmethod
    def _client_of(path: str) -> str:
        return path.split("?")[0].rsplit("/", 1)[-1]

    async def resolve_github_path(
        self,
        path: str,
        *,
        log_diagnostics_on_not_found: bool = True,
        raise_on_lock_held: bool = False,
    ) -> Path | None:
        self.calls += 1
        self.lock_strictness.append(raise_on_lock_held)
        if self.delay:
            await asyncio.sleep(self.delay)
        client = self._client_of(path)
        effect = self.effects.get(client)
        if effect == "lock_held":
            if raise_on_lock_held:
                raise DownloadLockHeldError(path)
            return None
        if isinstance(effect, Exception):
            raise effect
        if client not in self.dirs:
            raise FileNotFoundError(path)
        return self.dirs[client]

    async def confirm_directory_absent(self, *, github_uri: str) -> bool:
        return self.absent_confirmed


class _StubPromptLibraryEnv(PromptLibraryEnvironmentVariables):
    @property
    def prompt_library_path(self) -> str | None:
        return None


def _write_override(
    *, root: Path, client_id: str, description: str, with_type: bool = False
) -> Path:
    directory = root / client_id
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "id": "model-a",
        "name": "Model A",
        "description": description,
    }
    if with_type:
        payload["type"] = "custom"
    (directory / "model.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


class Harness:
    def __init__(self, *, tmp_path: Path, enabled: bool) -> None:
        self.github = FakeGitHub()
        self.store = MemoryStore()
        self.clock = FakeClock()
        self.root = tmp_path
        self.env = LanguageModelCommonEnvironmentVariables()
        self.cache = ClientOverrideConfigCache(
            store=self.store,
            collection=COLLECTION,
            schema_version=ConfigReader.SCHEMA_VERSION,
            environment_variables=self.env,
            clock=self.clock,
        )
        self.reader = ConfigReader(
            prompt_library_manager=PromptLibraryManager(
                environment_variables=_StubPromptLibraryEnv()
            ),
            environment_variables=self.env,
            github_directory_helper=self.github,  # type: ignore[arg-type]
            override_config_cache=self.cache if enabled else None,
        )

    def add_client(self, client_id: str, description: str) -> None:
        self.github.dirs[client_id] = _write_override(
            root=self.root, client_id=client_id, description=description
        )

    async def read(self, client_id: str) -> list[ChatModelConfig]:
        return await self.reader._read_override_models_async(
            config_path=CONFIG_PATH, client_id=client_id
        )

    async def settle(self) -> None:
        pending = list(self.reader._inflight_override_refreshes.values())
        if pending:
            await asyncio.gather(*pending)

    def key(self, client_id: str) -> str:
        return self.cache.build_key(config_path=CONFIG_PATH, client_id=client_id)

    async def row(self, client_id: str) -> dict[str, Any] | None:
        row: dict[str, Any] | None = await self.store.get(
            self.key(client_id), collection=COLLECTION
        )
        return row


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    monkeypatch.setenv("MODEL_CONFIG_OVERRIDE_CACHE_ENABLED", "true")
    monkeypatch.setenv("MODELS_OFFICIAL_PATH", CONFIG_PATH)
    return Harness(tmp_path=tmp_path, enabled=True)


# --- flag off / no cache: today's behaviour ----------------------------------


@pytest.mark.parametrize(
    "flag, with_cache",
    [("false", True), ("true", False)],
    ids=["flag-off", "no-cache-store"],
)
async def test_disabled_reads_live_every_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, with_cache: bool
) -> None:
    monkeypatch.setenv("MODEL_CONFIG_OVERRIDE_CACHE_ENABLED", flag)
    harness = Harness(tmp_path=tmp_path, enabled=with_cache)
    harness.add_client("client-a", "override-a")

    first = await harness.read("client-a")
    second = await harness.read("client-a")

    assert [m.description for m in first] == ["override-a"]
    assert [m.description for m in second] == ["override-a"]
    assert harness.github.calls == 2
    assert await harness.row("client-a") is None


async def test_flag_off_keeps_pre_cache_lock_held_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_CONFIG_OVERRIDE_CACHE_ENABLED", "false")
    harness = Harness(tmp_path=tmp_path, enabled=True)
    harness.add_client("client-a", "override-a")

    await harness.read("client-a")

    # Flag off must not ask the downloader to raise on a held lock, so the
    # downloader's pod-local fallback still applies as before the cache.
    assert harness.github.lock_strictness == [False]


async def test_cold_miss_times_out_and_serves_base_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_CONFIG_OVERRIDE_CACHE_ENABLED", "true")
    monkeypatch.setattr(
        "languagemodelcommon.configs.config_reader.config_reader."
        "_COLD_MISS_TIMEOUT_SECONDS",
        0.05,
    )
    harness = Harness(tmp_path=tmp_path, enabled=True)
    harness.add_client("client-a", "override-a")
    harness.github.delay = 5.0

    assert await harness.read("client-a") == []

    for task in list(harness.reader._inflight_override_refreshes.values()):
        task.cancel()


async def test_flag_off_does_not_confirm_absence_or_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_CONFIG_OVERRIDE_CACHE_ENABLED", "false")
    harness = Harness(tmp_path=tmp_path, enabled=True)

    assert await harness.read("client-missing") == []
    assert await harness.read("client-missing") == []
    assert harness.github.calls == 2
    assert await harness.row("client-missing") is None


# --- read-through, negative cache --------------------------------------------


async def test_miss_then_hit_makes_one_download(harness: Harness) -> None:
    harness.add_client("client-a", "override-a")

    first = await harness.read("client-a")
    second = await harness.read("client-a")

    assert [m.description for m in first] == ["override-a"]
    assert [m.description for m in second] == ["override-a"]
    assert harness.github.calls == 1
    row = await harness.row("client-a")
    assert row is not None and row["status"] == "found"


async def test_confirmed_absent_is_negative_cached(harness: Harness) -> None:
    assert await harness.read("client-none") == []
    assert await harness.read("client-none") == []

    assert harness.github.calls == 1
    row = await harness.row("client-none")
    assert row is not None and row["status"] == "absent"


async def test_negative_row_expires_after_negative_ttl(harness: Harness) -> None:
    assert await harness.read("client-new") == []
    harness.add_client("client-new", "now-exists")
    harness.clock.advance(
        harness.env.model_config_override_cache_negative_ttl_seconds + 1
    )

    # Stale negative row is served once while a refresh runs, then the
    # override shows up.
    await harness.read("client-new")
    await harness.settle()
    result = await harness.read("client-new")

    assert [m.description for m in result] == ["now-exists"]


# --- hazard 1: unconfirmed FileNotFoundError must not become ABSENT ----------


async def test_unconfirmed_not_found_is_not_cached(harness: Harness) -> None:
    harness.github.absent_confirmed = False

    assert await harness.read("client-x") == []
    assert await harness.read("client-x") == []

    assert harness.github.calls == 2
    assert await harness.row("client-x") is None


async def test_unconfirmed_not_found_does_not_overwrite_found_row(
    harness: Harness,
) -> None:
    harness.add_client("client-a", "override-a")
    await harness.read("client-a")
    # GitHub starts answering with a bare FileNotFoundError (e.g. bad token).
    del harness.github.dirs["client-a"]
    harness.github.absent_confirmed = False
    harness.clock.advance(harness.env.model_config_override_cache_soft_ttl_seconds + 1)

    stale = await harness.read("client-a")
    await harness.settle()

    assert [m.description for m in stale] == ["override-a"]
    row = await harness.row("client-a")
    assert row is not None and row["status"] == "found"
    assert [m.description for m in await harness.read("client-a")] == ["override-a"]


async def test_confirmed_removal_replaces_found_row(harness: Harness) -> None:
    harness.add_client("client-a", "override-a")
    await harness.read("client-a")
    del harness.github.dirs["client-a"]
    harness.clock.advance(harness.env.model_config_override_cache_soft_ttl_seconds + 1)

    await harness.read("client-a")
    await harness.settle()

    row = await harness.row("client-a")
    assert row is not None and row["status"] == "absent"


# --- hazard 3: lock-held and failures are never cached as "no override" ------


@pytest.mark.parametrize(
    "effect",
    ["lock_held", ValueError("download failed"), RuntimeError("boom")],
    ids=["lock-held", "download-failed", "unexpected-error"],
)
async def test_non_definitive_outcomes_are_never_cached(
    harness: Harness, effect: Any
) -> None:
    harness.github.effects["client-a"] = effect

    assert await harness.read("client-a") == []
    assert await harness.read("client-a") == []

    assert harness.github.calls == 2
    assert await harness.row("client-a") is None


@pytest.mark.parametrize(
    "outcome",
    [OverrideOutcome.LOCK_HELD, OverrideOutcome.FAILED],
)
async def test_cache_rejects_non_definitive_results(
    harness: Harness, outcome: OverrideOutcome
) -> None:
    with pytest.raises(ValueError):
        await harness.cache.put(
            config_path=CONFIG_PATH,
            client_id="client-a",
            result=OverrideReadResult(outcome=outcome),
        )


# --- stale-while-revalidate, serve stale on failure --------------------------


async def test_stale_row_served_immediately_and_refreshed(harness: Harness) -> None:
    harness.add_client("client-a", "v1")
    await harness.read("client-a")
    harness.add_client("client-a", "v2")
    harness.clock.advance(harness.env.model_config_override_cache_soft_ttl_seconds + 1)

    stale = await harness.read("client-a")
    assert [m.description for m in stale] == ["v1"]
    await harness.settle()
    fresh = await harness.read("client-a")

    assert [m.description for m in fresh] == ["v2"]
    assert harness.github.calls == 2


async def test_failed_refresh_keeps_serving_stale_and_backs_off(
    harness: Harness,
) -> None:
    harness.add_client("client-a", "v1")
    await harness.read("client-a")
    harness.github.effects["client-a"] = ValueError("github down")
    harness.clock.advance(harness.env.model_config_override_cache_soft_ttl_seconds + 1)

    stale = await harness.read("client-a")
    await harness.settle()
    calls_after_failure = harness.github.calls
    again = await harness.read("client-a")
    await harness.settle()

    assert [m.description for m in stale] == ["v1"]
    assert [m.description for m in again] == ["v1"]
    # Back-off: the immediate next read does not retry GitHub.
    assert harness.github.calls == calls_after_failure
    row = await harness.row("client-a")
    assert row is not None and row["status"] == "found"


async def test_deferred_refresh_cannot_extend_past_hard_expiry(
    harness: Harness,
) -> None:
    harness.add_client("client-a", "v1")
    await harness.read("client-a")
    cached = await harness.cache.get(config_path=CONFIG_PATH, client_id="client-a")
    assert cached is not None

    await harness.cache.defer_refresh(
        config_path=CONFIG_PATH,
        client_id="client-a",
        cached=cached,
        retry_after_seconds=15,
    )
    deferred = await harness.cache.get(config_path=CONFIG_PATH, client_id="client-a")
    assert deferred is not None
    assert deferred.hard_expires_at == cached.hard_expires_at

    harness.clock.advance(harness.env.model_config_override_cache_hard_ttl_seconds + 1)
    await harness.store.delete(harness.key("client-a"), collection=COLLECTION)
    await harness.cache.defer_refresh(
        config_path=CONFIG_PATH,
        client_id="client-a",
        cached=cached,
        retry_after_seconds=15,
    )

    assert await harness.row("client-a") is None


# --- hazard 4: contended miss -------------------------------------------------


async def _hold_refresh_lock(harness: Harness, client_id: str) -> None:
    from hashlib import sha256

    lock_name = (
        f"override_refresh:{sha256(harness.key(client_id).encode()).hexdigest()[:16]}"
    )
    await harness.store.put(
        lock_name, {"holder": "other-worker"}, collection=LOCK_COLLECTION, ttl=60
    )


async def test_contended_miss_returns_lock_held_without_fake_fetch(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        config_reader_module, "_OVERRIDE_PEER_POLL_TIMEOUT_SECONDS", 0.1
    )
    harness.add_client("client-a", "override-a")
    await _hold_refresh_lock(harness, "client-a")

    assert await harness.read("client-a") == []

    assert harness.github.calls == 0
    assert await harness.row("client-a") is None


async def test_contended_miss_uses_peer_row_when_it_appears(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        config_reader_module, "_OVERRIDE_PEER_POLL_TIMEOUT_SECONDS", 2.0
    )
    harness.add_client("client-a", "override-a")
    await _hold_refresh_lock(harness, "client-a")
    peer_models = (
        ChatModelConfig.model_validate(
            {"id": "model-a", "name": "Model A", "description": "from-peer"}
        ),
    )

    async def peer_writes_row() -> None:
        await asyncio.sleep(0.1)
        await harness.cache.put(
            config_path=CONFIG_PATH,
            client_id="client-a",
            result=OverrideReadResult(
                outcome=OverrideOutcome.FOUND, models=peer_models
            ),
        )

    writer = asyncio.create_task(peer_writes_row())
    result = await harness.read("client-a")
    await writer

    assert [m.description for m in result] == ["from-peer"]
    assert harness.github.calls == 0


async def test_stale_row_served_when_refresh_lock_held(harness: Harness) -> None:
    harness.add_client("client-a", "v1")
    await harness.read("client-a")
    calls = harness.github.calls
    harness.clock.advance(harness.env.model_config_override_cache_soft_ttl_seconds + 1)
    await _hold_refresh_lock(harness, "client-a")

    stale = await harness.read("client-a")
    await harness.settle()

    assert [m.description for m in stale] == ["v1"]
    assert harness.github.calls == calls


# --- stampede ------------------------------------------------------------------


async def test_concurrent_misses_share_one_download(harness: Harness) -> None:
    harness.add_client("client-a", "override-a")
    harness.github.delay = 0.05

    results = await asyncio.gather(*[harness.read("client-a") for _ in range(8)])

    assert all([m.description for m in r] == ["override-a"] for r in results)
    assert harness.github.calls == 1


# --- tenant isolation -----------------------------------------------------------


async def test_clients_never_see_each_others_overrides(harness: Harness) -> None:
    harness.add_client("client-a", "override-a")
    harness.add_client("client-b", "override-b")

    a_first = await harness.read("client-a")
    b_first = await harness.read("client-b")
    none_first = await harness.read("client-c")
    a_second = await harness.read("client-a")
    b_second = await harness.read("client-b")
    none_second = await harness.read("client-c")

    assert [m.description for m in a_first + a_second] == ["override-a"] * 2
    assert [m.description for m in b_first + b_second] == ["override-b"] * 2
    assert none_first == none_second == []
    assert harness.key("client-a") != harness.key("client-b")


async def test_row_stored_under_another_clients_key_is_ignored(
    harness: Harness,
) -> None:
    harness.add_client("client-a", "override-a")
    harness.add_client("client-b", "override-b")
    await harness.read("client-a")
    row_a = await harness.row("client-a")
    assert row_a is not None
    # Corrupt: client-a's row ends up under client-b's key.
    await harness.store.put(harness.key("client-b"), row_a, collection=COLLECTION)

    result = await harness.read("client-b")

    assert [m.description for m in result] == ["override-b"]


async def test_different_config_refs_do_not_share_rows(harness: Harness) -> None:
    other = "github://org/repo/configs?ref=other"
    assert harness.cache.build_key(
        config_path=CONFIG_PATH, client_id="client-a"
    ) != harness.cache.build_key(config_path=other, client_id="client-a")


# --- hazard 2 (key namespace): override rows never collide with base scan ------


@pytest.mark.parametrize(
    "client_id",
    ["client-a", "abcdef012345", "ovr", "x"],
)
def test_override_key_never_matches_base_prefix_scan(
    harness: Harness, client_id: str
) -> None:
    ref_hash = compute_short_ref_hash(CONFIG_PATH)
    base_prefix = f"v{ConfigReader.SCHEMA_VERSION}:{ref_hash}:"
    key = harness.key(client_id)

    assert not key.startswith(base_prefix)
    # ...but clear_cache's prefix still removes it.
    assert key.startswith(f"v{ConfigReader.SCHEMA_VERSION}:")


async def test_clear_cache_deletes_override_rows(harness: Harness) -> None:
    class _Cursor:
        def __init__(self, keys: list[str]) -> None:
            self._it = iter(keys)

        def __aiter__(self) -> "_Cursor":
            return self

        async def __anext__(self) -> dict[str, str]:
            try:
                return {"key": next(self._it)}
            except StopIteration:
                raise StopAsyncIteration

    class _Collection:
        def __init__(self, keys: list[str]) -> None:
            self._keys = keys

        def find(self, *args: Any, **kwargs: Any) -> _Cursor:
            prefix = kwargs.get("filter", args[0])["key"]["$regex"].lstrip("^")
            return _Cursor([k for k in self._keys if k.startswith(prefix)])

    override_key = harness.key("client-a")

    class _Store:
        _collections_by_name = {COLLECTION: _Collection([override_key, "other:key"])}
        deleted: list[str] = []

        async def delete_many(self, keys: list[str], *, collection: str) -> int:
            self.deleted = list(keys)
            return len(keys)

    store = _Store()
    reader = ConfigReader(
        prompt_library_manager=PromptLibraryManager(
            environment_variables=_StubPromptLibraryEnv()
        ),
        environment_variables=harness.env,
        model_config_cache_store=store,  # type: ignore[arg-type]
    )

    await reader.clear_cache()

    assert store.deleted == [override_key]


async def test_clear_client_override_forces_refetch(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.add_client("client-a", "v1")
    await harness.read("client-a")
    harness.add_client("client-a", "v2")

    await harness.reader.clear_client_override(client_id="client-a")
    result = await harness.read("client-a")

    assert [m.description for m in result] == ["v2"]


# --- hazard (merge fidelity): exclude_unset ------------------------------------


async def test_cached_override_merges_like_fresh_override(
    tmp_path: Path, harness: Harness
) -> None:
    harness.github.dirs["client-a"] = _write_override(
        root=tmp_path, client_id="client-a", description="override-a"
    )
    base = [
        ChatModelConfig.model_validate(
            {
                "id": "model-a",
                "name": "Model A",
                "description": "base",
                "type": "custom",
            }
        )
    ]

    fresh = await harness.read("client-a")
    cached = await harness.read("client-a")
    merged_fresh = ConfigReader._merge_model_configs(
        base_models=base, override_models=fresh
    )
    merged_cached = ConfigReader._merge_model_configs(
        base_models=base, override_models=cached
    )

    assert harness.github.calls == 1
    assert merged_cached[0].model_dump() == merged_fresh[0].model_dump()
    assert merged_cached[0].description == "override-a"
    assert merged_cached[0].type == "custom"


# --- untrusted client ids --------------------------------------------------------


@pytest.mark.parametrize(
    "client_id",
    ["../x", "a/b", "", "a b", "x" * 65 + "/y"],
)
async def test_invalid_client_ids_never_reach_cache_or_github(
    harness: Harness, client_id: str
) -> None:
    assert await harness.read(client_id) == []

    assert harness.github.calls == 0


async def test_over_long_id_is_read_live_and_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_id = "c" * 65
    results = {}
    for flag in ("true", "false"):
        monkeypatch.setenv("MODEL_CONFIG_OVERRIDE_CACHE_ENABLED", flag)
        harness = Harness(tmp_path=tmp_path / flag, enabled=True)
        harness.add_client(long_id, "override-long")
        first = await harness.read(long_id)
        await harness.read(long_id)
        results[flag] = [m.description for m in first]
        assert harness.github.calls == 2
        assert await harness.row(long_id) is None

    # Same override with the flag on or off: the length cap is cache-only.
    assert results == {"true": ["override-long"], "false": ["override-long"]}


# --- store errors degrade to a live read -------------------------------------------


@pytest.mark.parametrize("failing_method", ["get", "put"])
async def test_store_errors_fall_back_to_live_read(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, failing_method: str
) -> None:
    harness.add_client("client-a", "override-a")
    real = getattr(harness.store, failing_method)

    async def boom(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("collection") == LOCK_COLLECTION:
            return await real(*args, **kwargs)
        raise ConnectionError("mongo unavailable")

    monkeypatch.setattr(harness.store, failing_method, boom)

    result = await harness.read("client-a")

    assert [m.description for m in result] == ["override-a"]


# --- unresolved MCP refs use a short soft TTL ---------------------------------------


@pytest.mark.parametrize(
    "unresolved, expected_soft_ttl",
    [(True, 60.0), (False, 900.0)],
)
async def test_unresolved_mcp_override_gets_short_soft_ttl(
    harness: Harness, unresolved: bool, expected_soft_ttl: float
) -> None:
    await harness.cache.put(
        config_path=CONFIG_PATH,
        client_id="client-a",
        result=OverrideReadResult(
            outcome=OverrideOutcome.FOUND, has_unresolved_mcp=unresolved
        ),
    )

    row = await harness.row("client-a")

    assert row is not None
    assert row["soft_ttl_seconds"] == expected_soft_ttl
