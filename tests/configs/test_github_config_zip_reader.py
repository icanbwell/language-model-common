import pytest
import tempfile

from typing import Any
from unittest.mock import patch, MagicMock, AsyncMock
from languagemodelcommon.configs.config_reader.github_config_zip_reader import (
    GitHubConfigZipDownloader,
)


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get", new_callable=AsyncMock)
@patch("tempfile.NamedTemporaryFile")
@patch("zipfile.ZipFile")
@patch("os.unlink")
async def test_download_zip_success(
    mock_unlink: AsyncMock,
    mock_zipfile: MagicMock,
    mock_tempfile: MagicMock,
    mock_get: AsyncMock,
) -> None:
    # Mock HTTP response
    response = MagicMock()
    response.status_code = 200
    response.content = b"zipcontent"
    response.raise_for_status = MagicMock()
    mock_get.return_value = response
    # Mock temp file
    mock_temp = MagicMock()
    mock_temp.__enter__.return_value = mock_temp
    mock_temp.name = f"{tempfile.gettempdir()}fake.zip"
    mock_tempfile.return_value = mock_temp
    # Mock zipfile
    mock_zip = MagicMock()
    mock_zip.__enter__.return_value = mock_zip
    mock_zip.namelist.return_value = ["repo-root/file.json"]
    mock_zipfile.return_value = mock_zip

    downloader = GitHubConfigZipDownloader()
    result = await downloader.download_zip("http://fake.zip")
    assert "repo-root" in result


def test_find_json_configs_reads_and_sorts(monkeypatch: Any, tmp_path: Any) -> None:
    # Create fake JSON files
    config1 = tmp_path / "a.json"
    config2 = tmp_path / "b.json"
    config1.write_text('{"name": "B", "id": "b"}')
    config2.write_text('{"name": "A", "id": "a"}')
    # Patch ChatModelConfig to a simple class
    monkeypatch.setattr(
        "languagemodelcommon.configs.config_reader.github_config_zip_reader.ChatModelConfig",
        lambda **kwargs: type("C", (), kwargs)(),
    )
    downloader = GitHubConfigZipDownloader()
    configs = downloader._find_json_configs(str(tmp_path))
    assert [c.name for c in configs] == ["A", "B"]


def test_find_json_configs_blocks_directory_traversal(
    monkeypatch: Any, tmp_path: Any
) -> None:
    outside_dir = tmp_path.parent / "outside-configs"
    outside_dir.mkdir(parents=True, exist_ok=True)
    (outside_dir / "leak.json").write_text('{"name": "LEAK", "id": "leak"}')

    monkeypatch.setattr(
        "languagemodelcommon.configs.config_reader.github_config_zip_reader.ChatModelConfig",
        lambda **kwargs: type("C", (), kwargs)(),
    )

    downloader = GitHubConfigZipDownloader()
    configs = downloader._find_json_configs(
        str(tmp_path), config_dir="../outside-configs"
    )

    assert configs == []


def test_find_json_configs_skips_symlink_escape(
    monkeypatch: Any, tmp_path: Any
) -> None:
    allowed_dir = tmp_path / "configs"
    allowed_dir.mkdir(parents=True, exist_ok=True)
    (allowed_dir / "safe.json").write_text('{"name": "SAFE", "id": "safe"}')

    outside_file = tmp_path.parent / "outside-secret.json"
    outside_file.write_text('{"name": "LEAK", "id": "leak"}')
    (allowed_dir / "linked-secret.json").symlink_to(outside_file)

    monkeypatch.setattr(
        "languagemodelcommon.configs.config_reader.github_config_zip_reader.ChatModelConfig",
        lambda **kwargs: type("C", (), kwargs)(),
    )

    downloader = GitHubConfigZipDownloader()
    configs = downloader._find_json_configs(str(tmp_path), config_dir="configs")

    assert [c.name for c in configs] == ["SAFE"]


@pytest.mark.asyncio
@patch.object(GitHubConfigZipDownloader, "download_zip", new_callable=AsyncMock)
@patch.object(GitHubConfigZipDownloader, "_find_json_configs")
async def test_read_model_configs_success(
    mock_find: AsyncMock, mock_download: Any
) -> None:
    mock_download.return_value = "/repo"
    mock_find.side_effect = [
        [type("C", (), {"name": "A"})()],
        [type("C", (), {"name": "B"})()],
    ]
    # Patch ChatModelConfig for the testing marker
    with patch(
        "languagemodelcommon.configs.config_reader.github_config_zip_reader.ChatModelConfig",
        side_effect=lambda **kwargs: type("C", (), kwargs)(),
    ):
        downloader = GitHubConfigZipDownloader()
        configs = await downloader.read_model_configs(
            github_url="http://fake.zip",
            models_official_path="official",
            models_testing_path="testing",
        )
        assert any(
            getattr(c, "name", None) == "----- Models in Testing -----" for c in configs
        )


@pytest.mark.asyncio
@patch.object(GitHubConfigZipDownloader, "download_zip", side_effect=Exception("fail"))
async def test_read_model_configs_download_error(mock_download: Any) -> None:
    downloader = GitHubConfigZipDownloader()
    configs = await downloader.read_model_configs(
        github_url="http://fake.zip",
        models_official_path="official",
        models_testing_path=None,
    )
    assert configs == []
