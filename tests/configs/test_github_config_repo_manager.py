import io
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from languagemodelcommon.configs.config_reader.github_config_repo_manager import (
    GithubConfigRepoManager,
)


def _make_zipball(files: dict[str, str], prefix: str = "owner-repo-abc123") -> bytes:
    """Create an in-memory zipball with the given files under a prefix directory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, content in files.items():
            zf.writestr(f"{prefix}/{path}", content)
    return buf.getvalue()


@pytest.fixture
def manager_env(tmp_path: Path, monkeypatch: Any) -> Path:
    """Set up env vars for a test manager and return the cache dir."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(
        "GITHUB_CONFIG_REPO_URL",
        "https://api.github.com/repos/org/repo/zipball/main",
    )
    monkeypatch.setenv("GITHUB_CACHE_FOLDER", str(cache_dir))
    monkeypatch.setenv("CONFIG_CACHE_TIMEOUT_SECONDS", "3600")
    return cache_dir


class TestIsEnabled:
    def test_enabled_when_url_set(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("GITHUB_CONFIG_REPO_URL", "https://example.com/zip")
        assert GithubConfigRepoManager().is_enabled is True

    def test_disabled_when_url_not_set(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("GITHUB_CONFIG_REPO_URL", raising=False)
        assert GithubConfigRepoManager().is_enabled is False

    def test_disabled_when_url_blank(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("GITHUB_CONFIG_REPO_URL", "  ")
        assert GithubConfigRepoManager().is_enabled is False


class TestExtractZip:
    def test_extracts_and_returns_prefix_directory(self, tmp_path: Path) -> None:
        zip_bytes = _make_zipball(
            {"configs/model.json": '{"name": "test"}'},
            prefix="owner-repo-abc123",
        )
        repo_root = GithubConfigRepoManager._extract_zip(zip_bytes, tmp_path)
        assert repo_root.name == "owner-repo-abc123"
        assert (repo_root / "configs" / "model.json").exists()

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../../etc/passwd", "malicious")
        with pytest.raises(ValueError, match="Path traversal"):
            GithubConfigRepoManager._extract_zip(buf.getvalue(), tmp_path)


class TestDownloadAndExtract:
    @pytest.mark.asyncio
    async def test_extracts_flattened_to_cache_dir(self, manager_env: Path) -> None:
        """Zipball is extracted with the SHA prefix directory flattened."""
        zip_bytes = _make_zipball(
            {
                "configs/official/model.json": '{"name": "m1"}',
                "configs/mcp/.mcp.json": '{"mcpServers": {}}',
            }
        )

        with patch.object(
            GithubConfigRepoManager,
            "_download_zipball",
            new_callable=AsyncMock,
            return_value=zip_bytes,
        ):
            mgr = GithubConfigRepoManager()
            await mgr._download_and_extract()

        # Files should be directly under cache_dir, not under owner-repo-sha/
        assert (manager_env / "configs" / "official" / "model.json").exists()
        assert (manager_env / "configs" / "mcp" / ".mcp.json").exists()

    @pytest.mark.asyncio
    async def test_atomic_swap_replaces_contents(self, manager_env: Path) -> None:
        """Second download replaces the first cleanly."""
        zip_v1 = _make_zipball(
            {"configs/mcp/.mcp.json": '{"v": 1}'},
            prefix="owner-repo-aaa111",
        )
        zip_v2 = _make_zipball(
            {"configs/mcp/.mcp.json": '{"v": 2}'},
            prefix="owner-repo-bbb222",
        )

        call_count = 0

        async def mock_download(url: str) -> bytes:
            nonlocal call_count
            call_count += 1
            return zip_v1 if call_count == 1 else zip_v2

        with patch.object(
            GithubConfigRepoManager,
            "_download_zipball",
            side_effect=mock_download,
        ):
            mgr = GithubConfigRepoManager()
            await mgr._download_and_extract()
            assert (
                manager_env / "configs" / "mcp" / ".mcp.json"
            ).read_text() == '{"v": 1}'

            await mgr._download_and_extract()
            assert (
                manager_env / "configs" / "mcp" / ".mcp.json"
            ).read_text() == '{"v": 2}'

        # No stale staging directories left behind
        assert not manager_env.with_name(manager_env.name + ".old").exists()
        assert not manager_env.with_name(manager_env.name + ".new").exists()


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_skips_when_disabled(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("GITHUB_CONFIG_REPO_URL", raising=False)
        mgr = GithubConfigRepoManager()
        await mgr.start()
        assert mgr._background_task is None

    @pytest.mark.asyncio
    async def test_stop_cancels_background_task(self, manager_env: Path) -> None:
        zip_bytes = _make_zipball({"configs/mcp/.mcp.json": "{}"})

        with patch.object(
            GithubConfigRepoManager,
            "_download_zipball",
            new_callable=AsyncMock,
            return_value=zip_bytes,
        ):
            mgr = GithubConfigRepoManager()
            await mgr.start()
            assert mgr._background_task is not None
            assert not mgr._background_task.done()

            await mgr.stop()
            assert mgr._background_task.done()
