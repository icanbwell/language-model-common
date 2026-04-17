"""Helper for downloading GitHub content and manipulating github:// paths.

This module isolates GitHub-specific download, token handling, tempdir
caching, URL conversion, and URI joining so that :class:`ConfigReader`
and other readers stay focused on reading and parsing configuration files.

All GitHub access uses the fsspec-based ``github://`` URI scheme.
``https://github.com/`` URLs are converted to ``github://`` before download.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS
from languagemodelcommon.utilities.url_parser import UrlParser

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)


class GitHubDirectoryHelper:
    """Encapsulates GitHub directory download, caching, and URI manipulation.

    Environment-dependent configuration (token, cache dir, TTL) is read
    from the injected ``environment_variables`` instance rather than from
    ``os.environ`` directly, keeping this class testable and DI-friendly.

    The in-memory ``_cache`` dict is per-instance (and therefore
    per-worker when registered as a singleton in the container), which
    avoids cross-worker state issues under Gunicorn prefork.
    """

    def __init__(
        self,
        *,
        environment_variables: LanguageModelCommonEnvironmentVariables | None = None,
    ) -> None:
        self._environment_variables = environment_variables
        self._cache: dict[str, tuple[Path, float]] = {}

    # ------------------------------------------------------------------
    # Pure / static helpers — no env vars or instance state needed
    # ------------------------------------------------------------------

    @staticmethod
    def github_url_to_uri(url: str) -> str:
        """Convert an ``https://github.com/`` URL to a ``github://`` URI.

        Accepts URLs in the form::

            https://github.com/owner/repo/tree/branch/path/to/dir

        and returns::

            github://owner/repo/path/to/dir?ref=branch
        """
        parsed = urlsplit(url)
        if parsed.hostname not in ("github.com",) and not (
            parsed.hostname and parsed.hostname.endswith(".github.com")
        ):
            raise ValueError(f"Not a GitHub URL: {url}")

        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 4 or parts[2] != "tree":
            raise ValueError(
                "Invalid GitHub URL format. Expected: "
                "https://github.com/owner/repo/tree/branch/path"
            )

        owner = parts[0]
        repo = parts[1]
        branch = unquote(parts[3])
        path = unquote("/".join(parts[4:])) if len(parts) > 4 else ""

        uri_path = f"/{repo}/{path}" if path else f"/{repo}"
        return f"github://{owner}{uri_path}?ref={branch}"

    @staticmethod
    def is_github_path(path: str) -> bool:
        """Return ``True`` if *path* is a ``github://`` URI or a convertible GitHub tree URL.

        Only matches ``https://github.com/owner/repo/tree/...`` URLs that
        ``github_url_to_uri`` can convert.  Does not match API URLs like
        ``https://api.github.com/...``.
        """
        if path.startswith("github://"):
            return True
        if not UrlParser.is_github_url(path):
            return False
        # Only accept tree URLs that github_url_to_uri can handle
        parsed = urlsplit(path)
        if parsed.hostname != "github.com":
            return False
        parts = [p for p in parsed.path.split("/") if p]
        return len(parts) >= 4 and parts[2] == "tree"

    @staticmethod
    def to_github_uri(path: str) -> str:
        """Normalize a GitHub path to a ``github://`` URI.

        Passes ``github://`` URIs through unchanged and converts
        ``https://github.com/`` tree URLs.  Raises :class:`ValueError` if
        *path* is not a recognized GitHub path.
        """
        if path.startswith("github://"):
            return path
        if GitHubDirectoryHelper.is_github_path(path):
            return GitHubDirectoryHelper.github_url_to_uri(path)
        raise ValueError(f"Not a GitHub path: {path}")

    @staticmethod
    def join_github_uri_path(base_uri: str, suffix: str) -> str:
        """Join a path suffix onto a ``github://`` URI, preserving query params."""
        parts = urlsplit(base_uri)
        new_path = parts.path.rstrip("/") + "/" + suffix.strip("/")
        return urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, ""))

    # ------------------------------------------------------------------
    # Instance methods — use environment variables and instance cache
    # ------------------------------------------------------------------

    def resolve_github_path(self, path: str) -> Path:
        """Resolve a GitHub path to a local directory.

        Accepts ``github://`` URIs, ``https://github.com/`` URLs, or local paths.
        GitHub paths are downloaded via fsspec; local paths are returned as-is.
        """
        if self.is_github_path(path):
            return self.download_github_directory(self.to_github_uri(path))
        return Path(path)

    def download_github_directory(self, github_uri: str) -> Path:
        """Download a ``github://`` URI to a local cache directory using fsspec.

        Results are cached for ``config_cache_timeout_seconds``.
        The cache directory defaults to ``{tempdir}/github_config_cache`` and can
        be overridden with ``GITHUB_CONFIG_CACHE_DIR``.

        Caching is checked at two levels:

        1. **In-memory** (per-instance / per-worker) — avoids filesystem stat
           calls on hot paths within the same Gunicorn worker.
        2. **On-disk timestamp file** — checked by ``GithubDirectoryDownloader``
           after acquiring its per-URI lock, allowing workers that lost the lock
           race to skip redundant downloads.

        The downloader itself handles locking, atomic swap, and retry so this
        method no longer needs its own file lock.
        """
        from langchain_ai_skills_framework.loaders.github_directory_downloader import (
            GithubDirectoryDownloader,
        )

        import os
        import tempfile

        cache_ttl = (
            self._environment_variables.config_cache_timeout_seconds
            if self._environment_variables
            else int(os.environ.get("CONFIG_CACHE_TIMEOUT_SECONDS", "3600"))
        )
        cache_dir = Path(
            self._environment_variables.github_config_cache_dir
            if self._environment_variables
            else os.environ.get(
                "GITHUB_CONFIG_CACHE_DIR",
                str(Path(tempfile.gettempdir()) / "github_config_cache"),
            )
        )

        now = time.monotonic()
        cached = self._cache.get(github_uri)

        if cached is not None:
            cached_path, cached_time = cached
            if (now - cached_time) < cache_ttl and cached_path.is_dir():
                logger.debug(
                    "Using cached GitHub download for %s (age: %.0fs)",
                    github_uri,
                    now - cached_time,
                )
                return cached_path

        cache_dir.mkdir(parents=True, exist_ok=True)

        github_token = (
            self._environment_variables.github_token
            if self._environment_variables
            else os.environ.get("GITHUB_TOKEN")
        )
        downloader = GithubDirectoryDownloader()
        result: Path = downloader.download(
            source_uri=github_uri,
            github_token=github_token,
            cache_path=cache_dir,
            cache_ttl_seconds=cache_ttl,
        )
        self._cache[github_uri] = (result, time.monotonic())
        logger.info("Downloaded and cached GitHub content from %s", github_uri)
        return result
