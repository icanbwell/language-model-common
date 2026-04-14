"""Downloads a GitHub repository zipball and extracts it locally.

When ``GITHUB_CONFIG_REPO_URL`` is set, the manager downloads the entire
repository as a ZIP archive at startup, extracts it to the directory
specified by ``GITHUB_CACHE_FOLDER``, and refreshes on a configurable
interval.  Config-path environment variables in docker-compose should
point at local subdirectories within the cache folder.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import httpx

from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)


class GithubConfigRepoManager:
    """Downloads a GitHub repo zipball and keeps a local cache refreshed.

    Startup flow:
      1. ``start()`` downloads the zipball and extracts it to the
         ``GITHUB_CACHE_FOLDER`` directory.
      2. A background ``asyncio.Task`` re-downloads every
         ``CONFIG_CACHE_TIMEOUT_SECONDS`` (default 120 s).

    The extraction uses an atomic directory swap so that readers using
    the previous extraction are not disrupted mid-request.
    """

    def __init__(self) -> None:
        self._repo_url: str | None = os.environ.get("GITHUB_CONFIG_REPO_URL")
        self._cache_dir = Path(
            os.environ.get(
                "GITHUB_CACHE_FOLDER",
                str(Path(tempfile.gettempdir()) / "github_config_cache"),
            )
        )
        self._refresh_seconds = int(
            os.environ.get("CONFIG_CACHE_TIMEOUT_SECONDS", "120")
        )
        self._github_token: str | None = os.environ.get("GITHUB_TOKEN")
        self._background_task: asyncio.Task[None] | None = None
        self._is_initial_download: bool = True

    @property
    def is_enabled(self) -> bool:
        """Return ``True`` when a repo URL is configured."""
        return bool(self._repo_url and self._repo_url.strip())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Download the repo and start the background refresh loop."""
        if not self.is_enabled:
            logger.info("GITHUB_CONFIG_REPO_URL not set — skipping repo download")
            return
        await self._download_and_extract()
        self._background_task = asyncio.create_task(self._refresh_loop())
        logger.info(
            "Background config refresh scheduled every %d s",
            self._refresh_seconds,
        )

    async def stop(self) -> None:
        """Cancel the background refresh task."""
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                # Expected during shutdown; task was cancelled successfully
                logger.debug("Background task cancellation completed")
            logger.info("Background config refresh stopped")

    # ------------------------------------------------------------------
    # Download + extract
    # ------------------------------------------------------------------

    async def _download_and_extract(self) -> None:
        """Download the zipball and extract with atomic directory swap.

        GitHub zipballs contain a top-level directory named
        ``{owner}-{repo}-{sha}``.  We flatten it so that the cache
        directory structure is stable across refreshes (no SHA in path).

        Multiple Gunicorn workers may call this concurrently on startup.
        The on-disk timestamp check avoids most redundant downloads, and
        the atomic swap ensures readers never see a partial directory.
        A few redundant downloads on cold start are harmless.
        """
        if not self._repo_url:
            raise RuntimeError("Cannot download: GITHUB_CONFIG_REPO_URL is not set")

        parent = self._cache_dir.parent
        parent.mkdir(parents=True, exist_ok=True)

        # If another worker already populated a fresh cache, skip.
        if self._is_initial_download and self._is_cache_fresh():
            logger.info(
                "Cache directory %s already fresh — skipping download",
                self._cache_dir,
            )
            self._is_initial_download = False
            return

        zip_bytes = await self._download_zipball(self._repo_url)

        # Place temporary directories *inside* the cache dir's parent to
        # stay on the same filesystem.  In Kubernetes the cache dir is
        # often a mounted volume; sibling paths created with `with_name()`
        # may land on the container overlay FS, causing EXDEV (errno 18)
        # on rename.  Using shutil.move handles the cross-device case
        # gracefully by falling back to copy+delete.
        extract_dir = parent / (self._cache_dir.name + ".extract")
        staging_dir = parent / (self._cache_dir.name + ".new")
        old_dir = parent / (self._cache_dir.name + ".old")

        # Extract into a temporary directory
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        repo_root = self._extract_zip(zip_bytes, extract_dir)

        # Flatten: move the single top-level {owner-repo-sha}/ directory
        # up so that paths are stable across refreshes
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if repo_root != extract_dir:
            shutil.move(str(repo_root), str(staging_dir))
            shutil.rmtree(extract_dir, ignore_errors=True)
        else:
            shutil.move(str(extract_dir), str(staging_dir))

        # Atomic swap: staging → current, current → old
        if old_dir.exists():
            shutil.rmtree(old_dir)
        if self._cache_dir.exists():
            shutil.move(str(self._cache_dir), str(old_dir))
        shutil.move(str(staging_dir), str(self._cache_dir))

        # Clean up old directory (best-effort)
        if old_dir.exists():
            shutil.rmtree(old_dir, ignore_errors=True)

        self._mark_cache_fresh()
        self._is_initial_download = False
        logger.info("Config repo extracted to %s", self._cache_dir)

    def _is_cache_fresh(self) -> bool:
        """Return True if the cache directory exists and was refreshed recently."""
        ts_file = self._cache_dir.parent / (self._cache_dir.name + ".ts")
        if not ts_file.exists() or not self._cache_dir.is_dir():
            return False
        age = time.time() - ts_file.stat().st_mtime
        return age < self._refresh_seconds

    def _mark_cache_fresh(self) -> None:
        """Write a timestamp marker so other workers know the cache is fresh."""
        ts_file = self._cache_dir.parent / (self._cache_dir.name + ".ts")
        ts_file.write_text(str(time.time()))

    async def _download_zipball(self, url: str) -> bytes:
        """Download a zipball from the GitHub API."""
        headers: dict[str, str] = {
            "X-GitHub-Api-Version": "2022-11-28",
            "Accept": "application/vnd.github+json",
        }
        if self._github_token:
            headers["Authorization"] = f"Bearer {self._github_token}"

        timeout = httpx.Timeout(
            timeout=float(os.environ.get("GITHUB_TIMEOUT", "300")),
        )
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

        logger.info("Downloaded zipball from %s (%d bytes)", url, len(response.content))
        return response.content

    @staticmethod
    def _extract_zip(zip_bytes: bytes, target_dir: Path) -> Path:
        """Extract a zipball and return the path to the repo root inside.

        GitHub zipballs contain a single top-level directory named
        ``{owner}-{repo}-{sha}/``.  This method validates all paths
        for traversal attacks before extraction.
        """
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Validate all paths before extracting
            for member in zf.namelist():
                member_path = (target_dir / member).resolve()
                if not str(member_path).startswith(str(target_dir.resolve())):
                    raise ValueError(f"Path traversal detected in zip member: {member}")
            zf.extractall(target_dir)

        # Find the single top-level directory that GitHub creates
        top_level = [d for d in target_dir.iterdir() if d.is_dir()]
        if len(top_level) == 1:
            return top_level[0]

        # Fallback: if there's no single top-level dir, use target_dir itself
        return target_dir

    # ------------------------------------------------------------------
    # Background refresh
    # ------------------------------------------------------------------

    async def _refresh_loop(self) -> None:
        """Re-download the repo on a fixed interval."""
        while True:
            await asyncio.sleep(self._refresh_seconds)
            try:
                await self._download_and_extract()
                logger.info("Background config refresh completed")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Background config refresh failed", exc_info=True)
