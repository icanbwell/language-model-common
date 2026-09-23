import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch, MagicMock

from languagemodelcommon.configs.config_reader.github_directory_downloader import (
    GithubDirectoryDownloader,
)

import pytest

from languagemodelcommon.configs.config_reader.config_reader import ConfigReader
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.configs.schemas.mcp_json_schema import (
    McpJsonConfig,
    McpServerEntry,
)
from languagemodelcommon.configs.config_reader.github_directory_helper import (
    GitHubDirectoryHelper,
)
from languagemodelcommon.configs.prompt_library.prompt_library_environment_variables import (
    PromptLibraryEnvironmentVariables,
)
from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
    PromptLibraryManager,
)


class _StubPromptLibraryEnv(PromptLibraryEnvironmentVariables):
    def __init__(self, prompt_library_path: str) -> None:
        self._prompt_library_path = prompt_library_path

    @property
    def prompt_library_path(self) -> str | None:
        return self._prompt_library_path


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_prompt_library_manager(tmp_path: Path) -> PromptLibraryManager:
    return PromptLibraryManager(
        environment_variables=_StubPromptLibraryEnv(str(tmp_path))
    )


# --- github_url_to_uri tests ---


@pytest.mark.parametrize(
    "url, expected",
    [
        (
            "https://github.com/owner/repo/tree/main/configs/chat",
            "github://owner/repo/configs/chat?ref=main",
        ),
        (
            "https://github.com/icanbwell/language-model-gateway-configuration/tree/main/configs/chat_completions/official",
            "github://icanbwell/language-model-gateway-configuration/configs/chat_completions/official?ref=main",
        ),
        (
            "https://github.com/owner/repo/tree/develop/path",
            "github://owner/repo/path?ref=develop",
        ),
        (
            "https://github.com/owner/repo/tree/main",
            "github://owner/repo?ref=main",
        ),
        (
            "https://github.com/owner/repo/tree/feature%2Ffoo/configs",
            "github://owner/repo/configs?ref=feature/foo",
        ),
    ],
)
def test_github_url_to_uri(url: str, expected: str) -> None:
    assert GitHubDirectoryHelper.github_url_to_uri(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://notgithub.com/owner/repo/tree/main/configs",
        "https://github.com/owner/repo/branch/main/configs",
        "https://github.com/owner/repo/tree",
    ],
)
def test_github_url_to_uri_invalid(url: str) -> None:
    with pytest.raises(ValueError):
        GitHubDirectoryHelper.github_url_to_uri(url)


# --- is_github_path tests ---


@pytest.mark.parametrize(
    "path, expected",
    [
        ("github://org/repo/configs?ref=main", True),
        ("https://github.com/owner/repo/tree/main/path", True),
        ("/local/path/to/configs", False),
        ("s3://bucket/path", False),
        # api.github.com URLs are not convertible tree URLs
        ("https://api.github.com/repos/owner/repo/zipball/main", False),
        # GitHub URLs without /tree/ segment are not convertible
        ("https://github.com/owner/repo", False),
    ],
)
def test_is_github_path(path: str, expected: bool) -> None:
    assert GitHubDirectoryHelper.is_github_path(path) == expected


# --- resolve_github_path tests ---


@pytest.mark.asyncio
async def test_resolve_github_path_local(tmp_path: Path) -> None:
    helper = GitHubDirectoryHelper()
    result = await helper.resolve_github_path(str(tmp_path))
    assert result == tmp_path


@pytest.mark.asyncio
async def test_resolve_github_path_github_uri(tmp_path: Path) -> None:
    helper = GitHubDirectoryHelper()
    with patch.object(
        helper,
        "download_github_directory",
        new_callable=AsyncMock,
        return_value=tmp_path,
    ) as mock_download:
        result = await helper.resolve_github_path("github://org/repo/configs?ref=main")

    assert result == tmp_path
    mock_download.assert_called_once_with("github://org/repo/configs?ref=main")


@pytest.mark.asyncio
async def test_resolve_github_path_https_url(tmp_path: Path) -> None:
    helper = GitHubDirectoryHelper()
    with patch.object(
        helper,
        "download_github_directory",
        new_callable=AsyncMock,
        return_value=tmp_path,
    ) as mock_download:
        result = await helper.resolve_github_path(
            "https://github.com/owner/repo/tree/main/configs"
        )

    assert result == tmp_path
    mock_download.assert_called_once_with("github://owner/repo/configs?ref=main")


# --- ConfigReader integration tests ---


@pytest.mark.asyncio
async def test_read_models_from_github_uri(tmp_path: Path, monkeypatch: Any) -> None:
    """ConfigReader uses GithubDirectoryDownloader for github:// URIs."""
    local_dir = tmp_path / "downloaded"
    local_dir.mkdir()
    _write_json(
        local_dir / "model.json",
        {"id": "m1", "name": "Model One"},
    )

    monkeypatch.setenv("MODELS_OFFICIAL_PATH", "github://org/repo/configs?ref=main")
    monkeypatch.delenv("MODELS_ZIP_PATH", raising=False)

    mock_helper = MagicMock(spec=GitHubDirectoryHelper)
    mock_helper.resolve_github_path.return_value = local_dir

    prompt_mgr = _make_prompt_library_manager(tmp_path)
    reader = ConfigReader(
        prompt_library_manager=prompt_mgr,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        github_directory_helper=mock_helper,
    )

    models = await reader.read_models_from_path_async(
        config_path="github://org/repo/configs?ref=main"
    )

    assert len(models) == 1
    assert models[0].name == "Model One"
    mock_helper.resolve_github_path.assert_called_once_with(
        "github://org/repo/configs?ref=main"
    )


@pytest.mark.asyncio
async def test_read_models_from_https_github_url(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """ConfigReader converts https://github.com/ URLs and downloads via fsspec."""
    local_dir = tmp_path / "downloaded"
    local_dir.mkdir()
    _write_json(
        local_dir / "model.json",
        {"id": "m1", "name": "Model One"},
    )

    mock_helper = MagicMock(spec=GitHubDirectoryHelper)
    mock_helper.resolve_github_path.return_value = local_dir

    prompt_mgr = _make_prompt_library_manager(tmp_path)
    reader = ConfigReader(
        prompt_library_manager=prompt_mgr,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        github_directory_helper=mock_helper,
    )

    models = await reader.read_models_from_path_async(
        config_path="https://github.com/owner/repo/tree/main/configs"
    )

    assert len(models) == 1
    assert models[0].name == "Model One"
    mock_helper.resolve_github_path.assert_called_once_with(
        "https://github.com/owner/repo/tree/main/configs"
    )


@pytest.mark.asyncio
async def test_github_uri_resolves_mcp_via_fetcher(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """github:// download path resolves mcp_server refs via McpJsonFetcher."""
    local_dir = tmp_path / "downloaded"
    local_dir.mkdir()
    _write_json(
        local_dir / "model.json",
        {
            "id": "m1",
            "name": "Model One",
            "plugins": ["all-employees"],
            "tools": [{"name": "drive", "mcp_server": "google-drive"}],
        },
    )

    mock_helper = MagicMock(spec=GitHubDirectoryHelper)
    mock_helper.resolve_github_path.return_value = local_dir

    mock_fetcher = AsyncMock()
    mock_fetcher._url = "http://localhost:5000/plugin-marketplace/"
    mock_fetcher.fetch_plugins_async.return_value = (
        {
            "all-employees": McpJsonConfig(
                mcpServers={
                    "google-drive": McpServerEntry(
                        url="https://mcp.example.com/drive/"
                    ),
                }
            ),
        },
        [],
    )

    prompt_mgr = _make_prompt_library_manager(tmp_path)
    reader = ConfigReader(
        prompt_library_manager=prompt_mgr,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        github_directory_helper=mock_helper,
        mcp_json_fetcher=mock_fetcher,
    )

    models = await reader.read_models_from_path_async(
        config_path="github://org/repo/configs?ref=main"
    )

    assert len(models) == 1
    assert models[0].tools is not None
    assert models[0].tools[0].url == "https://mcp.example.com/drive/"
    mock_fetcher.fetch_plugins_async.assert_awaited_once_with(["all-employees"])


@pytest.mark.asyncio
async def test_read_model_configs_async_with_github_uri(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """End-to-end: MODELS_OFFICIAL_PATH=github:// goes through fsspec downloader."""
    local_dir = tmp_path / "downloaded"
    local_dir.mkdir()
    _write_json(
        local_dir / "model.json",
        {"id": "m1", "name": "Model One"},
    )

    monkeypatch.setenv("MODELS_OFFICIAL_PATH", "github://org/repo/configs?ref=main")
    monkeypatch.delenv("MODELS_ZIP_PATH", raising=False)

    mock_helper = MagicMock(spec=GitHubDirectoryHelper)
    mock_helper.resolve_github_path.return_value = local_dir

    prompt_mgr = _make_prompt_library_manager(tmp_path)
    reader = ConfigReader(
        prompt_library_manager=prompt_mgr,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        github_directory_helper=mock_helper,
    )

    models = await reader.read_model_configs_async()

    assert len(models) == 1
    assert models[0].name == "Model One"


def test_override_config_path_with_github_uri() -> None:
    result = ConfigReader._resolve_override_config_path(
        config_path="github://org/repo/configs?ref=main",
        client_id="client-123",
    )
    assert result == "github://org/repo/configs/clients/client-123?ref=main"


def test_override_config_path_with_https_github_url() -> None:
    result = ConfigReader._resolve_override_config_path(
        config_path="https://github.com/owner/repo/tree/main/configs",
        client_id="client-123",
    )
    assert result == "github://owner/repo/configs/clients/client-123?ref=main"


def test_join_path_preserves_github_query_params() -> None:
    result = GitHubDirectoryHelper.join_github_uri_path(
        base_uri="github://org/repo/configs?ref=main", suffix="clients/client-123"
    )
    assert result == "github://org/repo/configs/clients/client-123?ref=main"


def test_join_path_works_without_query_params() -> None:
    result = GitHubDirectoryHelper.join_github_uri_path(
        base_uri="github://org/repo/configs", suffix="clients/client-123"
    )
    assert result == "github://org/repo/configs/clients/client-123"


# --- _resolve_content_dir tests ---


def test_resolve_content_dir_descends_into_last_component(tmp_path: Path) -> None:
    """When fsspec creates a subdirectory matching the last path component, return it."""
    target_dir = tmp_path / "cache-abc123"
    target_dir.mkdir()
    content_subdir = target_dir / "prompts"
    content_subdir.mkdir()
    (content_subdir / "system_prompt.txt").write_text("hello")

    result = GithubDirectoryDownloader._resolve_content_dir(
        target_dir=target_dir, source_path="bailey/prompts"
    )

    assert result == content_subdir.resolve()


def test_resolve_content_dir_falls_back_when_no_subdir(tmp_path: Path) -> None:
    """When the expected subdirectory doesn't exist, return target_dir itself."""
    target_dir = tmp_path / "cache-abc123"
    target_dir.mkdir()
    (target_dir / "file.txt").write_text("content")

    result = GithubDirectoryDownloader._resolve_content_dir(
        target_dir=target_dir, source_path="bailey/prompts"
    )

    assert result == target_dir.resolve()


def test_resolve_content_dir_with_empty_source_path(tmp_path: Path) -> None:
    """When source_path is empty (root fetch), return target_dir."""
    target_dir = tmp_path / "cache-abc123"
    target_dir.mkdir()

    result = GithubDirectoryDownloader._resolve_content_dir(
        target_dir=target_dir, source_path=""
    )

    assert result == target_dir.resolve()


def test_resolve_content_dir_with_single_segment_path(tmp_path: Path) -> None:
    """When source_path is a single segment, descend into that directory."""
    target_dir = tmp_path / "cache-abc123"
    target_dir.mkdir()
    content_subdir = target_dir / "configs"
    content_subdir.mkdir()
    (content_subdir / "model.json").write_text("{}")

    result = GithubDirectoryDownloader._resolve_content_dir(
        target_dir=target_dir, source_path="configs"
    )

    assert result == content_subdir.resolve()


# --- End-to-end: PromptLibraryManager with GitHub-like nested directories ---


@pytest.mark.asyncio
async def test_prompt_library_resolves_github_nested_directory(
    tmp_path: Path,
) -> None:
    """PromptLibraryManager finds prompts when GitHub download nests them."""
    # Simulate what GithubDirectoryDownloader produces: target_dir/prompts/files
    cache_dir = tmp_path / "cache-abc123"
    cache_dir.mkdir()
    prompts_subdir = cache_dir / "prompts"
    prompts_subdir.mkdir()
    (prompts_subdir / "bailey_system_prompt.txt").write_text("You are Bailey.")
    (prompts_subdir / "skills.md").write_text("# Skills")

    # The helper should return the prompts_subdir (with the fix)
    mock_helper = MagicMock(spec=GitHubDirectoryHelper)
    mock_helper.resolve_github_path.return_value = prompts_subdir

    mgr = PromptLibraryManager(
        environment_variables=_StubPromptLibraryEnv(
            "github://org/repo/bailey/prompts?ref=2.0.3"
        ),
        github_directory_helper=mock_helper,
    )

    content = await mgr.get_prompt_async("bailey_system_prompt")
    assert content == "You are Bailey."

    content_md = await mgr.get_prompt_async("skills")
    assert content_md == "# Skills"


@pytest.mark.asyncio
async def test_download_returns_content_subdir(tmp_path: Path) -> None:
    """download() returns the nested content directory, not the cache root."""
    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    downloader = GithubDirectoryDownloader()

    def fake_fetch(
        *, git_location: Any, source_path: str, github_token: Any, target_dir: Path
    ) -> None:
        subdir = target_dir / Path(source_path).name
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / "system_prompt.txt").write_text("hello")

    with patch.object(downloader, "_fetch_to_directory", side_effect=fake_fetch):
        result = await downloader.download(
            source_uri="github://org/repo/bailey/prompts?ref=main",
            github_token="fake-token",
            cache_path=cache_path,
        )

    assert result is not None
    assert result.name == "prompts"
    assert (result / "system_prompt.txt").read_text() == "hello"


@pytest.mark.asyncio
async def test_download_cached_returns_content_subdir(tmp_path: Path) -> None:
    """download() always re-downloads (no local freshness check)."""
    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    downloader = GithubDirectoryDownloader()

    def fake_fetch(
        *, git_location: Any, source_path: str, github_token: Any, target_dir: Path
    ) -> None:
        subdir = target_dir / Path(source_path).name
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / "my_prompt.txt").write_text("fresh content")

    with patch.object(downloader, "_fetch_to_directory", side_effect=fake_fetch):
        result = await downloader.download(
            source_uri="github://org/repo/bailey/prompts?ref=main",
            github_token="fake-token",
            cache_path=cache_path,
        )

    assert result is not None
    assert result.name == "prompts"
    assert (result / "my_prompt.txt").read_text() == "fresh content"


@pytest.mark.asyncio
async def test_fetch_skips_fsspec_instance_cache_for_unpinned_ref(
    tmp_path: Path,
) -> None:
    """Regression test: an unpinned ref must not reuse a stale cached
    GithubFileSystem/DirCache across calls, or new commits on the default
    branch would never be picked up until the process restarts."""
    cache_path = tmp_path / "cache"

    captured_storage_options: dict[str, object] = {}

    class _FakeGithubFilesystem:
        def get(
            self, remote_path: str, local_path: str, recursive: bool = False
        ) -> None:
            del remote_path, recursive
            Path(local_path).mkdir(parents=True, exist_ok=True)

        def ls(self, path: str, detail: bool = False) -> list[str]:
            del path, detail
            return []

    def _fake_filesystem(
        protocol: str, **storage_options: object
    ) -> _FakeGithubFilesystem:
        assert protocol == "github"
        captured_storage_options.update(storage_options)
        return _FakeGithubFilesystem()

    with patch(
        "languagemodelcommon.configs.config_reader.github_directory_downloader.fsspec.filesystem",
        side_effect=_fake_filesystem,
    ):
        downloader = GithubDirectoryDownloader()
        await downloader.download(
            source_uri="github://my-org/private-repo/configs",
            github_token=None,
            cache_path=cache_path,
        )

    assert "sha" not in captured_storage_options
    assert captured_storage_options["skip_instance_cache"] is True


@pytest.mark.asyncio
async def test_fetch_passes_skip_instance_cache_for_pinned_ref(
    tmp_path: Path,
) -> None:
    """A pinned '?ref=' must still pass skip_instance_cache alongside sha."""
    cache_path = tmp_path / "cache"

    captured_storage_options: dict[str, object] = {}

    class _FakeGithubFilesystem:
        def get(
            self, remote_path: str, local_path: str, recursive: bool = False
        ) -> None:
            del remote_path, recursive
            Path(local_path).mkdir(parents=True, exist_ok=True)

        def ls(self, path: str, detail: bool = False) -> list[str]:
            del path, detail
            return []

    def _fake_filesystem(
        protocol: str, **storage_options: object
    ) -> _FakeGithubFilesystem:
        assert protocol == "github"
        captured_storage_options.update(storage_options)
        return _FakeGithubFilesystem()

    with patch(
        "languagemodelcommon.configs.config_reader.github_directory_downloader.fsspec.filesystem",
        side_effect=_fake_filesystem,
    ):
        downloader = GithubDirectoryDownloader()
        await downloader.download(
            source_uri="github://my-org/private-repo/configs?ref=main",
            github_token="token-value",
            cache_path=cache_path,
        )

    assert captured_storage_options == {
        "org": "my-org",
        "repo": "private-repo",
        "sha": "main",
        "username": "x-access-token",
        "token": "token-value",
        "skip_instance_cache": True,
    }


class _FakeGithubContentsResponse:
    """Mimics an httpx.Response for the contents-API probe (BAI-941)."""

    def __init__(
        self, *, status_code: int, headers: dict[str, str], body: dict[str, str]
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self._body = body
        self.text = json.dumps(body)

    def json(self) -> dict[str, str]:
        return self._body


class _FakeHttpxClient:
    """Records the probe request and returns a canned 404 response."""

    captured_url: str | None = None
    captured_params: Any = None
    captured_auth: Any = None
    call_count: int = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> "_FakeHttpxClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def get(
        self, url: str, params: Any = None, auth: Any = None
    ) -> _FakeGithubContentsResponse:
        type(self).captured_url = url
        type(self).captured_params = params
        type(self).captured_auth = auth
        type(self).call_count += 1
        return _FakeGithubContentsResponse(
            status_code=404,
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1790000000"},
            body={"message": "Not Found"},
        )


class _FakeGithubFilesystemNotFound:
    """Fails exactly like fsspec's GithubFileSystem on a 404 -- both on
    construction (root ls("")) and on fetching a specific subpath."""

    def get(self, remote_path: str, local_path: str, recursive: bool = False) -> None:
        raise FileNotFoundError(remote_path)

    def ls(self, path: str, detail: bool = False) -> list[str]:
        raise FileNotFoundError(path)


def _fake_filesystem_not_found(protocol: str, **storage_options: object) -> Any:
    assert protocol == "github"
    return _FakeGithubFilesystemNotFound()


@pytest.mark.asyncio
async def test_file_not_found_logs_diagnostic_probe(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """BAI-941: a bare FileNotFoundError from fsspec is ambiguous -- GitHub
    returns the same 404 for a missing ref/subpath and for a private repo the
    caller's credentials can't see. On that failure we must re-probe the
    actual source_path (not just the repo root, which fsspec's own
    construction-time check already covers) outside fsspec and log what
    GitHub actually said, since fsspec discards the real response before
    raising."""
    cache_path = tmp_path / "cache"
    _FakeHttpxClient.call_count = 0

    downloader = GithubDirectoryDownloader()

    with (
        patch(
            "languagemodelcommon.configs.config_reader.github_directory_downloader.fsspec.filesystem",
            side_effect=_fake_filesystem_not_found,
        ),
        patch(
            "languagemodelcommon.configs.config_reader.github_directory_downloader.httpx.Client",
            _FakeHttpxClient,
        ),
        caplog.at_level(logging.ERROR),
        pytest.raises(FileNotFoundError),
    ):
        await downloader.download(
            source_uri="github://icanbwell/baileyai-configuration/mcp-fhir-agent/configs/official?ref=prod",
            github_token="minted-installation-token",
            cache_path=cache_path,
        )

    assert _FakeHttpxClient.captured_url == (
        "https://api.github.com/repos/icanbwell/baileyai-configuration"
        "/contents/mcp-fhir-agent/configs/official"
    )
    assert _FakeHttpxClient.captured_params == {"ref": "prod"}
    assert _FakeHttpxClient.captured_auth == (
        "x-access-token",
        "minted-installation-token",
    )
    assert "status=404" in caplog.text
    assert "token_set=True" in caplog.text
    assert "rate_limit_remaining=0" in caplog.text
    assert "Not Found" in caplog.text


@pytest.mark.asyncio
async def test_file_not_found_probe_omits_ref_for_unpinned_branch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unpinned ref (no branch/tag in the URI) has no fixed sha to probe
    with. GitHub's git/trees API does not accept "HEAD" as a ref, so the
    probe must omit ?ref= entirely and let GitHub resolve the default
    branch -- exactly what fsspec itself does when sha is None -- rather than
    guessing a placeholder that would 404 for an unrelated reason and produce
    a misleading diagnosis."""
    cache_path = tmp_path / "cache"
    _FakeHttpxClient.call_count = 0

    downloader = GithubDirectoryDownloader()

    with (
        patch(
            "languagemodelcommon.configs.config_reader.github_directory_downloader.fsspec.filesystem",
            side_effect=_fake_filesystem_not_found,
        ),
        patch(
            "languagemodelcommon.configs.config_reader.github_directory_downloader.httpx.Client",
            _FakeHttpxClient,
        ),
        caplog.at_level(logging.ERROR),
        pytest.raises(FileNotFoundError),
    ):
        await downloader.download(
            source_uri="github://my-org/private-repo/configs",
            github_token="minted-installation-token",
            cache_path=cache_path,
        )

    assert _FakeHttpxClient.captured_params is None
    assert "ref=(default branch)" in caplog.text


@pytest.mark.asyncio
async def test_file_not_found_probe_is_throttled_per_path(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """During a sustained outage, every failed download would otherwise fire
    its own probe -- doubling GitHub API traffic indefinitely. A second
    failure for the same (owner, repo, sha, path) within the throttle window
    must not re-probe."""
    cache_path = tmp_path / "cache"
    _FakeHttpxClient.call_count = 0

    downloader = GithubDirectoryDownloader()

    with (
        patch(
            "languagemodelcommon.configs.config_reader.github_directory_downloader.fsspec.filesystem",
            side_effect=_fake_filesystem_not_found,
        ),
        patch(
            "languagemodelcommon.configs.config_reader.github_directory_downloader.httpx.Client",
            _FakeHttpxClient,
        ),
        caplog.at_level(logging.ERROR),
    ):
        for _ in range(3):
            with pytest.raises(FileNotFoundError):
                await downloader.download(
                    source_uri="github://icanbwell/baileyai-configuration/mcp-fhir-agent/configs/official?ref=prod",
                    github_token="minted-installation-token",
                    cache_path=cache_path,
                )

    assert _FakeHttpxClient.call_count == 1


@pytest.mark.asyncio
async def test_file_not_found_diagnostic_probe_survives_its_own_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The diagnostic probe is best-effort: if the re-probe request itself
    fails (e.g. network blip), the original FileNotFoundError must still
    propagate rather than being masked by the probe's own exception."""
    cache_path = tmp_path / "cache"

    def _raise_probe_error(*args: object, **kwargs: object) -> None:
        raise RuntimeError("probe network blip")

    downloader = GithubDirectoryDownloader()

    with (
        patch(
            "languagemodelcommon.configs.config_reader.github_directory_downloader.fsspec.filesystem",
            side_effect=_fake_filesystem_not_found,
        ),
        patch(
            "languagemodelcommon.configs.config_reader.github_directory_downloader.httpx.Client",
            side_effect=_raise_probe_error,
        ),
        caplog.at_level(logging.ERROR),
        pytest.raises(FileNotFoundError),
    ):
        await downloader.download(
            source_uri="github://icanbwell/baileyai-configuration/mcp-fhir-agent/configs/official?ref=prod",
            github_token="minted-installation-token",
            cache_path=cache_path,
        )

    assert "diagnostic probe failed" in caplog.text
