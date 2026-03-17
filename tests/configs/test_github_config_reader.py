import httpx
import time
import pytest
from typing import Any
from unittest.mock import AsyncMock, patch, MagicMock
from languagemodelcommon.configs.config_reader.github_config_reader import (
    GitHubConfigReader,
)


def test_parse_github_url_valid() -> None:
    reader = GitHubConfigReader()
    url = "https://github.com/owner/repo/tree/main/configs"
    repo, path, branch = reader.parse_github_url(url)
    assert repo == "owner/repo"
    assert path == "configs"
    assert branch == "main"


@pytest.mark.parametrize(
    "url",
    [
        "https://notgithub.com/owner/repo/tree/main/configs",
        "https://github.com/owner/repo/branch/main/configs",
        "https://github.com/owner/repo/tree",
    ],
)
def test_parse_github_url_invalid(url: str) -> None:
    reader = GitHubConfigReader()
    with pytest.raises(ValueError):
        reader.parse_github_url(url)


@pytest.fixture
def github_reader() -> GitHubConfigReader:
    return GitHubConfigReader()


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_read_model_configs_success(
    mock_get: AsyncMock, github_reader: GitHubConfigReader
) -> None:
    # Mock directory listing
    mock_get.side_effect = [
        MagicMock(
            status_code=200,
            json=MagicMock(
                return_value=[
                    {
                        "type": "file",
                        "name": "model1.json",
                        "download_url": "http://file1",
                    },
                    {
                        "type": "file",
                        "name": "model2.json",
                        "download_url": "http://file2",
                    },
                ]
            ),
        ),
        # Mock file1
        MagicMock(
            status_code=200, json=MagicMock(return_value={"name": "A", "other": "x"})
        ),
        # Mock file2
        MagicMock(
            status_code=200, json=MagicMock(return_value={"name": "B", "other": "y"})
        ),
    ]
    # Patch ChatModelConfig to accept any dict
    with patch(
        "languagemodelcommon.configs.config_reader.github_config_reader.ChatModelConfig",
        side_effect=lambda **kwargs: type("C", (), kwargs)(),
    ):
        configs = await github_reader.read_model_configs(
            github_url="https://github.com/owner/repo/tree/main/configs"
        )
        assert len(configs) == 2
        assert configs[0].name == "A"
        assert configs[1].name == "B"


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_read_model_configs_network_error(
    mock_get: AsyncMock, github_reader: GitHubConfigReader
) -> None:
    mock_get.side_effect = Exception("Network error")
    configs = await github_reader.read_model_configs(
        github_url="https://github.com/owner/repo/tree/main/configs"
    )
    assert configs == []


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_read_model_configs_empty_path(mock_get: AsyncMock) -> None:
    reader = GitHubConfigReader()
    with pytest.raises(ValueError):
        await reader._read_model_configs(
            repo_url="owner/repo", path="", branch="main", github_token=None
        )


@pytest.mark.asyncio
async def test_read_model_configs_non_json_files(
    github_reader: GitHubConfigReader,
) -> None:
    # Only the .json file will be fetched
    with patch(
        "languagemodelcommon.configs.config_reader.github_config_reader.ChatModelConfig",
        side_effect=lambda **kwargs: type("C", (), kwargs)(),
    ):
        # Mock file fetch when Directory contains non-JSON files
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as file_get:
            file_get.side_effect = [
                MagicMock(
                    status_code=200,
                    json=MagicMock(
                        return_value=[
                            {
                                "type": "file",
                                "name": "README.md",
                                "download_url": "http://file1",
                            },
                            {
                                "type": "file",
                                "name": "model.json",
                                "download_url": "http://file2",
                            },
                        ]
                    ),
                ),
                MagicMock(status_code=200, json=MagicMock(return_value={"name": "A"})),
            ]
            configs = await github_reader.read_model_configs(
                github_url="https://github.com/owner/repo/tree/main/configs"
            )
            assert len(configs) == 1
            assert configs[0].name == "A"


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_read_model_configs_json_decode_error(
    mock_get: AsyncMock, github_reader: GitHubConfigReader
) -> None:
    # Directory listing
    mock_get.side_effect = [
        MagicMock(
            status_code=200,
            json=MagicMock(
                return_value=[
                    {"type": "file", "name": "bad.json", "download_url": "http://bad"}
                ]
            ),
        ),
        MagicMock(
            status_code=200, json=MagicMock(side_effect=ValueError("Invalid JSON"))
        ),
    ]
    with patch(
        "languagemodelcommon.configs.config_reader.github_config_reader.ChatModelConfig",
        side_effect=lambda **kwargs: type("C", (), kwargs)(),
    ):
        configs = await github_reader.read_model_configs(
            github_url="https://github.com/owner/repo/tree/main/configs"
        )
        assert configs == []


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_make_request_rate_limit(mock_get: AsyncMock) -> None:
    reader = GitHubConfigReader()
    # Simulate rate limit exceeded
    mock_resp = MagicMock(
        status_code=403,
        text="API rate limit exceeded",
        headers={
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(time.time()) + 1),
        },
    )
    # After waiting, return success
    mock_get.side_effect = [
        mock_resp,
        MagicMock(status_code=200, json=MagicMock(return_value=[])),
    ]
    with patch("asyncio.sleep", new_callable=AsyncMock):
        async with httpx.AsyncClient() as client:
            resp = await reader._make_request(client=client, url="url", headers={})
            assert resp.status_code == 200


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_make_request_http_error(mock_get: AsyncMock) -> None:
    reader = GitHubConfigReader()
    # Simulate 500 error, then success
    mock_get.side_effect = [
        MagicMock(status_code=500, text="error", headers={}),
        MagicMock(status_code=200, json=MagicMock(return_value=[])),
    ]
    with patch("asyncio.sleep", new_callable=AsyncMock):
        async with httpx.AsyncClient() as client:
            resp = await reader._make_request(client=client, url="url", headers={})
            assert resp.status_code == 200


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_read_model_configs_with_token(
    mock_get: AsyncMock, github_reader: GitHubConfigReader, monkeypatch: Any
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token123")
    with patch(
        "languagemodelcommon.configs.config_reader.github_config_reader.ChatModelConfig",
        side_effect=lambda **kwargs: type("C", (), kwargs)(),
    ):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as file_get:
            file_get.side_effect = [
                MagicMock(
                    status_code=200,
                    json=MagicMock(
                        return_value=[
                            {
                                "type": "file",
                                "name": "model.json",
                                "download_url": "http://file",
                            }
                        ]
                    ),
                ),
                MagicMock(status_code=200, json=MagicMock(return_value={"name": "A"})),
            ]
            configs = await github_reader.read_model_configs(
                github_url="https://github.com/owner/repo/tree/main/configs"
            )
            assert configs[0].name == "A"
