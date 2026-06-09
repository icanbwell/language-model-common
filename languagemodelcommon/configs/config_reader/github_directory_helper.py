"""Helper for downloading GitHub content and manipulating github:// paths.

This module isolates GitHub-specific download, token handling, tempdir
caching, URL conversion, and URI joining so that :class:`ConfigReader`
and other readers stay focused on reading and parsing configuration files.

All GitHub access uses the fsspec-based ``github://`` URI scheme.
``https://github.com/`` URLs are converted to ``github://`` before download.
"""

import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from key_value.aio.protocols.key_value import AsyncKeyValueProtocol

from languagemodelcommon.github.token_provider import GitHubTokenProvider
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS
from languagemodelcommon.utilities.url_parser import UrlParser

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)


class GitHubDirectoryHelper:
    """Encapsulates GitHub directory download, caching, and URI manipulation.

    Environment-dependent configuration (token, cache dir) is read
    from the injected ``environment_variables`` instance rather than from
    ``os.environ`` directly, keeping this class testable and DI-friendly.

    Caching is handled at the on-disk level by ``GithubDirectoryDownloader``
    (with advisory lock coordination) and at the application level by the
    MongoDB model config cache. There is no in-memory cache layer.
    """

    def __init__(
        self,
        *,
        environment_variables: LanguageModelCommonEnvironmentVariables | None = None,
        store: AsyncKeyValueProtocol | None = None,
        token_provider: GitHubTokenProvider | None = None,
    ) -> None:
        self._environment_variables = environment_variables
        self._store = store
        self._token_provider = token_provider

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
    def join_github_uri_path(*, base_uri: str, suffix: str) -> str:
        """Join a path suffix onto a ``github://`` URI, preserving query params."""
        parts = urlsplit(base_uri)
        new_path = parts.path.rstrip("/") + "/" + suffix.strip("/")
        return urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, ""))

    # ------------------------------------------------------------------
    # Instance methods — use environment variables and instance cache
    # ------------------------------------------------------------------

    async def resolve_github_path(self, path: str) -> Path | None:
        """Resolve a GitHub path to a local directory.

        Accepts ``github://`` URIs, ``https://github.com/`` URLs, or local paths.
        GitHub paths are downloaded via fsspec; local paths are returned as-is.

        Returns None if another worker holds the download lock.
        """
        if self.is_github_path(path):
            return await self.download_github_directory(self.to_github_uri(path))
        return Path(path)

    async def download_github_directory(self, github_uri: str) -> Path | None:
        """Download a ``github://`` URI to a local cache directory using fsspec.

        The cache directory defaults to ``{tempdir}/github_config_cache`` and can
        be overridden with ``GITHUB_CONFIG_CACHE_DIR``.

        Returns None if another worker holds the download lock.
        """
        from languagemodelcommon.configs.config_reader.github_directory_downloader import (
            GithubDirectoryDownloader,
        )

        env = self._environment_variables
        cache_dir = Path(
            env.github_config_cache_dir
            if env
            else os.environ.get(
                "GITHUB_CONFIG_CACHE_DIR",
                str(Path(tempfile.gettempdir()) / "github_config_cache"),
            )
        )

        cache_dir.mkdir(parents=True, exist_ok=True)

        github_token: str | None = None
        if self._token_provider is not None:
            github_token = await self._token_provider.get_token()
        elif env:
            github_token = env.github_token
        else:
            github_token = os.environ.get("GITHUB_TOKEN")

        downloader = GithubDirectoryDownloader()
        result = await downloader.download(
            source_uri=github_uri,
            github_token=github_token,
            cache_path=cache_dir,
            store=self._store,
        )
        return result
