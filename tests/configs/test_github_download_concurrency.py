"""Tests for BAI-1380 step 1: the GitHub download no longer blocks the event
loop, retries back off with ``asyncio.sleep``, concurrent same-target downloads
are safe, and lock-held is reported explicitly.

GitHub (fsspec / HTTP) is the external boundary and is faked.
"""

import asyncio
import time
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from key_value.aio.stores.memory import MemoryStore
from oidcauthlib.cache_gate.advisory_lock import LOCK_COLLECTION

from languagemodelcommon.configs.config_reader.github_directory_downloader import (
    GithubDirectoryDownloader,
)
from languagemodelcommon.configs.config_reader.github_directory_helper import (
    GitHubDirectoryHelper,
)
from languagemodelcommon.configs.config_reader.override_read_result import (
    DownloadLockHeldError,
)
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)

HELPER_ASYNC_CLIENT = "languagemodelcommon.configs.config_reader.github_directory_helper.httpx.AsyncClient"
SOURCE_URI = "github://org/repo/configs/clients/client-a?ref=main"
LOCK_KEY = (
    "github_download:"
    + sha256(b"org/repo:main:configs/clients/client-a").hexdigest()[:16]
)


# --- event loop stays responsive ------------------------------------------------


async def test_blocking_fetch_does_not_stall_the_event_loop(tmp_path: Path) -> None:
    downloader = GithubDirectoryDownloader()

    def slow_fetch(*, target_dir: Path, **_: Any) -> None:
        time.sleep(0.4)
        target_dir.mkdir(parents=True, exist_ok=True)

    max_gap = 0.0

    async def ticker() -> None:
        nonlocal max_gap
        last = time.monotonic()
        for _ in range(30):
            await asyncio.sleep(0.01)
            now = time.monotonic()
            max_gap = max(max_gap, now - last)
            last = now

    with patch.object(downloader, "_fetch_to_directory", side_effect=slow_fetch):
        await asyncio.gather(
            downloader.download(
                source_uri=SOURCE_URI, github_token=None, cache_path=tmp_path
            ),
            ticker(),
        )

    assert max_gap < 0.2


# --- retry path uses asyncio.sleep -----------------------------------------------


@pytest.mark.parametrize(
    "side_effects, expected_fetches, expected_sleeps, raises",
    [
        ([None], 1, [], None),
        ([RuntimeError("x"), None], 2, [2.0], None),
        ([RuntimeError("x"), RuntimeError("y"), None], 3, [2.0, 4.0], None),
        ([RuntimeError("x")] * 3, 3, [2.0, 4.0], ValueError),
        ([FileNotFoundError("missing")], 1, [], FileNotFoundError),
    ],
    ids=["ok", "retry-once", "retry-twice", "exhausted", "not-found-not-retried"],
)
async def test_retry_backoff_uses_async_sleep(
    tmp_path: Path,
    side_effects: list[Any],
    expected_fetches: int,
    expected_sleeps: list[float],
    raises: type[Exception] | None,
) -> None:
    downloader = GithubDirectoryDownloader()
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    with (
        patch.object(
            downloader, "_fetch_to_directory", side_effect=side_effects
        ) as fetch,
        patch(
            "languagemodelcommon.configs.config_reader.github_directory_downloader.asyncio.sleep",
            side_effect=fake_sleep,
        ),
    ):
        coro = downloader.download(
            source_uri=SOURCE_URI, github_token=None, cache_path=tmp_path
        )
        if raises is None:
            await coro
        else:
            with pytest.raises(raises):
                await coro

    assert fetch.call_count == expected_fetches
    assert sleeps == expected_sleeps


# --- concurrent same-target downloads --------------------------------------------


class _FakeFilesystem:
    def get(self, source_path: str, dest: str, recursive: bool = False) -> None:
        time.sleep(0.05)
        content = Path(dest, source_path.rsplit("/", 1)[-1])
        content.mkdir(parents=True, exist_ok=True)
        (content / "model.json").write_text("{}", encoding="utf-8")


async def test_concurrent_downloads_of_one_target_do_not_corrupt_it(
    tmp_path: Path,
) -> None:
    downloader = GithubDirectoryDownloader()

    with patch(
        "languagemodelcommon.configs.config_reader.github_directory_downloader.fsspec.filesystem",
        return_value=_FakeFilesystem(),
    ):
        results = await asyncio.gather(
            *[
                downloader.download(
                    source_uri=SOURCE_URI, github_token=None, cache_path=tmp_path
                )
                for _ in range(6)
            ]
        )

    # Final state is what matters: a caller may briefly see the directory
    # mid-swap (pre-existing, also across workers), but once all downloads
    # settle the target must be complete and not corrupted by a nested
    # staging directory.
    assert all(r is not None for r in results)
    cache_name = (
        "org-repo-" + sha256(b"org/repo:main:configs/clients/client-a").hexdigest()[:12]
    )
    assert (tmp_path / cache_name / "client-a" / "model.json").exists()
    leftovers = [
        p.name for p in tmp_path.iterdir() if ".staging." in p.name or ".old." in p.name
    ]
    assert leftovers == []
    nested = [p for p in (tmp_path / cache_name).rglob("*") if ".staging." in p.name]
    assert nested == []


# --- hazard: lock-held is reported even when an on-disk copy exists -------------


async def test_lock_held_raises_when_asked_even_with_disk_copy(
    tmp_path: Path,
) -> None:
    store = MemoryStore()
    await store.put(LOCK_KEY, {"holder": "other"}, collection=LOCK_COLLECTION, ttl=60)
    downloader = GithubDirectoryDownloader()
    cache_name = (
        "org-repo-" + sha256(b"org/repo:main:configs/clients/client-a").hexdigest()[:12]
    )
    stale_dir = tmp_path / cache_name / "client-a"
    stale_dir.mkdir(parents=True)

    with pytest.raises(DownloadLockHeldError):
        await downloader.download(
            source_uri=SOURCE_URI,
            github_token=None,
            cache_path=tmp_path,
            store=store,
            raise_on_lock_held=True,
        )


async def test_lock_held_default_still_returns_disk_copy(tmp_path: Path) -> None:
    store = MemoryStore()
    await store.put(LOCK_KEY, {"holder": "other"}, collection=LOCK_COLLECTION, ttl=60)
    cache_name = (
        "org-repo-" + sha256(b"org/repo:main:configs/clients/client-a").hexdigest()[:12]
    )
    stale_dir = tmp_path / cache_name / "client-a"
    stale_dir.mkdir(parents=True)

    result = await GithubDirectoryDownloader().download(
        source_uri=SOURCE_URI, github_token=None, cache_path=tmp_path, store=store
    )

    assert result == stale_dir.resolve()


# --- helper: same-process calls share one download ------------------------------


async def test_helper_collapses_concurrent_downloads_of_one_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_CONFIG_CACHE_DIR", str(tmp_path))
    helper = GitHubDirectoryHelper(
        environment_variables=LanguageModelCommonEnvironmentVariables()
    )

    async def slow_download(**_: Any) -> Path:
        await asyncio.sleep(0.05)
        return tmp_path

    download = AsyncMock(side_effect=slow_download)

    with patch.object(GithubDirectoryDownloader, "download", download):
        results = await asyncio.gather(
            *[helper.download_github_directory(SOURCE_URI) for _ in range(6)]
        )

    assert download.await_count == 1
    assert results == [tmp_path] * 6
    # Entry removed once done: a later call downloads again.
    with patch.object(GithubDirectoryDownloader, "download", download):
        await helper.download_github_directory(SOURCE_URI)
    assert download.await_count == 2


# --- helper: confirm_directory_absent ---------------------------------------------


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (200, [{"name": "client-b"}], True),
        (200, [], True),
        (200, [{"name": "client-a"}], False),
        (404, {"message": "Not Found"}, False),
        (403, {"message": "rate limited"}, False),
        (401, {"message": "Bad credentials"}, False),
        (200, [{"name": f"c{i}"} for i in range(1000)], False),
        (200, {"message": "not a listing"}, False),
    ],
    ids=[
        "absent",
        "empty-parent",
        "present",
        "parent-404",
        "rate-limit",
        "bad-token",
        "truncated-listing",
        "unexpected-body",
    ],
)
async def test_confirm_directory_absent_requires_a_positive_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    body: Any,
    expected: bool,
) -> None:
    monkeypatch.setenv("GITHUB_CONFIG_CACHE_DIR", str(tmp_path))
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/repos/org/repo/contents/configs/clients")
        return httpx.Response(status, json=body)

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    helper = GitHubDirectoryHelper(
        environment_variables=LanguageModelCommonEnvironmentVariables()
    )
    with patch(HELPER_ASYNC_CLIENT, factory):
        result = await helper.confirm_directory_absent(github_uri=SOURCE_URI)

    assert result is expected


async def test_confirm_directory_absent_false_on_transport_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_CONFIG_CACHE_DIR", str(tmp_path))
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timeout")

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    helper = GitHubDirectoryHelper(
        environment_variables=LanguageModelCommonEnvironmentVariables()
    )
    with patch(HELPER_ASYNC_CLIENT, factory):
        assert await helper.confirm_directory_absent(github_uri=SOURCE_URI) is False
