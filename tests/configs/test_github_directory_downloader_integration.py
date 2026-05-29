import json
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


def test_resolve_github_path_local(tmp_path: Path) -> None:
    helper = GitHubDirectoryHelper()
    result = helper.resolve_github_path(str(tmp_path))
    assert result == tmp_path


def test_resolve_github_path_github_uri(tmp_path: Path) -> None:
    helper = GitHubDirectoryHelper()
    with patch.object(
        helper,
        "download_github_directory",
        return_value=tmp_path,
    ) as mock_download:
        result = helper.resolve_github_path("github://org/repo/configs?ref=main")

    assert result == tmp_path
    mock_download.assert_called_once_with("github://org/repo/configs?ref=main")


def test_resolve_github_path_https_url(tmp_path: Path) -> None:
    helper = GitHubDirectoryHelper()
    with patch.object(
        helper,
        "download_github_directory",
        return_value=tmp_path,
    ) as mock_download:
        result = helper.resolve_github_path(
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


def test_download_returns_content_subdir(tmp_path: Path) -> None:
    """download() returns the nested content directory, not the cache root."""
    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    downloader = GithubDirectoryDownloader()

    def fake_fetch(
        *, git_location: Any, source_path: str, github_token: Any, target_dir: Path
    ) -> None:
        # Simulate fsspec behavior: creates last component as subdirectory
        subdir = target_dir / Path(source_path).name
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / "system_prompt.txt").write_text("hello")

    with patch.object(downloader, "_fetch_to_directory", side_effect=fake_fetch):
        result = downloader.download(
            source_uri="github://org/repo/bailey/prompts?ref=main",
            github_token="fake-token",
            cache_path=cache_path,
            cache_ttl_seconds=0,
        )

    assert result.name == "prompts"
    assert (result / "system_prompt.txt").read_text() == "hello"


def test_download_cached_returns_content_subdir(tmp_path: Path) -> None:
    """download() returns content subdir even on cache hit."""
    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    downloader = GithubDirectoryDownloader()

    # Pre-populate the cache to simulate a cache hit
    from hashlib import sha256

    key = "org/repo:main:bailey/prompts"
    cache_dir_name = f"org-repo-{sha256(key.encode('utf-8')).hexdigest()[:12]}"
    target_dir = cache_path / cache_dir_name
    target_dir.mkdir()
    prompts_subdir = target_dir / "prompts"
    prompts_subdir.mkdir()
    (prompts_subdir / "my_prompt.txt").write_text("cached content")

    # Write the timestamp file to make cache fresh
    ts_file = target_dir.with_name(target_dir.name + ".ts")
    import time

    ts_file.write_text(str(time.time()))

    result = downloader.download(
        source_uri="github://org/repo/bailey/prompts?ref=main",
        github_token="fake-token",
        cache_path=cache_path,
        cache_ttl_seconds=3600,
    )

    assert result.name == "prompts"
    assert (result / "my_prompt.txt").read_text() == "cached content"
