import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from languagemodelcommon.configs.config_reader.config_reader import ConfigReader
from languagemodelcommon.configs.config_reader.github_directory_helper import (
    join_github_uri_path,
)
from languagemodelcommon.configs.prompt_library.prompt_library_environment_variables import (
    PromptLibraryEnvironmentVariables,
)
from languagemodelcommon.utilities.cache.config_expiring_cache import (
    ConfigExpiringCache,
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

    cache = ConfigExpiringCache(ttl_seconds=0)
    prompt_mgr = _make_prompt_library_manager(tmp_path)
    reader = ConfigReader(cache=cache, prompt_library_manager=prompt_mgr)

    with patch(
        "languagemodelcommon.configs.config_reader.config_reader.download_github_directory",
        return_value=local_dir,
    ) as mock_download:
        models = await reader.read_models_from_path_async(
            "github://org/repo/configs?ref=main"
        )

    assert len(models) == 1
    assert models[0].name == "Model One"
    mock_download.assert_called_once_with("github://org/repo/configs?ref=main")


@pytest.mark.asyncio
async def test_github_uri_resolves_mcp_json(tmp_path: Path, monkeypatch: Any) -> None:
    """github:// download path still resolves .mcp.json references."""
    local_dir = tmp_path / "downloaded"
    local_dir.mkdir()
    _write_json(
        local_dir / "model.json",
        {
            "id": "m1",
            "name": "Model One",
            "tools": [{"name": "drive", "mcp_server": "google-drive"}],
        },
    )
    _write_json(
        local_dir / ".mcp.json",
        {"mcpServers": {"google-drive": {"url": "https://mcp.example.com/drive/"}}},
    )

    cache = ConfigExpiringCache(ttl_seconds=0)
    prompt_mgr = _make_prompt_library_manager(tmp_path)
    reader = ConfigReader(cache=cache, prompt_library_manager=prompt_mgr)

    with patch(
        "languagemodelcommon.configs.config_reader.config_reader.download_github_directory",
        return_value=local_dir,
    ):
        models = await reader.read_models_from_path_async(
            "github://org/repo/configs?ref=main"
        )

    assert len(models) == 1
    assert models[0].tools is not None
    assert models[0].tools[0].url == "https://mcp.example.com/drive/"


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

    cache = ConfigExpiringCache(ttl_seconds=0)
    prompt_mgr = _make_prompt_library_manager(tmp_path)
    reader = ConfigReader(cache=cache, prompt_library_manager=prompt_mgr)

    with patch(
        "languagemodelcommon.configs.config_reader.config_reader.download_github_directory",
        return_value=local_dir,
    ):
        models = await reader.read_model_configs_async()

    assert len(models) == 1
    assert models[0].name == "Model One"


def test_override_config_path_with_github_uri() -> None:
    result = ConfigReader._resolve_override_config_path(
        config_path="github://org/repo/configs?ref=main",
        client_id="client-123",
    )
    assert result == "github://org/repo/configs/clients/client-123?ref=main"


def test_join_path_preserves_github_query_params() -> None:
    result = join_github_uri_path(
        "github://org/repo/configs?ref=main", "clients/client-123"
    )
    assert result == "github://org/repo/configs/clients/client-123?ref=main"


def test_join_path_works_without_query_params() -> None:
    result = join_github_uri_path("github://org/repo/configs", "clients/client-123")
    assert result == "github://org/repo/configs/clients/client-123"
