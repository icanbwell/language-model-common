"""Helper for downloading GitHub content and manipulating github:// paths.

This module isolates GitHub-specific download, token handling, tempdir
caching, URL conversion, and URI joining so that :class:`ConfigReader`
and other readers stay focused on reading and parsing configuration files.

All GitHub access uses the fsspec-based ``github://`` URI scheme.
``https://github.com/`` URLs are converted to ``github://`` before download.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx
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
        # In-flight downloads by (uri, flags): concurrent same-process callers
        # share one download. Holds tasks only, removed on completion; not a
        # cache of data. Lives here (a long-lived singleton) because a new
        # GithubDirectoryDownloader is built per call.
        self._inflight_downloads: dict[
            tuple[str, bool, bool], asyncio.Task[Path | None]
        ] = {}

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

    async def resolve_github_path(
        self,
        path: str,
        *,
        log_diagnostics_on_not_found: bool = True,
        raise_on_lock_held: bool = False,
    ) -> Path | None:
        """Resolve a GitHub path to a local directory.

        Accepts ``github://`` URIs, ``https://github.com/`` URLs, or local paths.
        GitHub paths are downloaded via fsspec; local paths are returned as-is.

        ``log_diagnostics_on_not_found`` (BAI-941): pass ``False`` when the
        caller already treats a missing path as expected/benign (e.g. an
        optional override directory or prompts folder), so a routine
        FileNotFoundError doesn't trigger the diagnostic re-probe/ERROR log
        meant for genuine failures.

        ``raise_on_lock_held`` (BAI-1380): raise ``DownloadLockHeldError`` when
        another worker holds the download lock, instead of returning None or a
        possibly stale on-disk copy.

        Returns None if another worker holds the download lock.
        """
        if self.is_github_path(path):
            return await self.download_github_directory(
                self.to_github_uri(path),
                log_diagnostics_on_not_found=log_diagnostics_on_not_found,
                raise_on_lock_held=raise_on_lock_held,
            )
        return Path(path)

    async def download_github_directory(
        self,
        github_uri: str,
        *,
        log_diagnostics_on_not_found: bool = True,
        raise_on_lock_held: bool = False,
    ) -> Path | None:
        """Download a ``github://`` URI to a local cache directory using fsspec.

        The cache directory defaults to ``{tempdir}/github_config_cache`` and can
        be overridden with ``GITHUB_CONFIG_CACHE_DIR``.

        Concurrent calls in this process for the same URI share one download.

        Returns None if another worker holds the download lock.
        """
        key = (github_uri, log_diagnostics_on_not_found, raise_on_lock_held)
        task = self._inflight_downloads.get(key)
        if task is None:
            task = asyncio.ensure_future(
                self._download_once(
                    github_uri=github_uri,
                    log_diagnostics_on_not_found=log_diagnostics_on_not_found,
                    raise_on_lock_held=raise_on_lock_held,
                )
            )
            self._inflight_downloads[key] = task
            task.add_done_callback(lambda _t: self._inflight_downloads.pop(key, None))
        # shield: one cancelled caller must not cancel the shared download.
        return await asyncio.shield(task)

    async def _resolve_github_token(self) -> str | None:
        env = self._environment_variables
        if self._token_provider is not None:
            return await self._token_provider.get_token()
        if env:
            return env.github_token
        return os.environ.get("GITHUB_TOKEN")

    async def _download_once(
        self,
        *,
        github_uri: str,
        log_diagnostics_on_not_found: bool,
        raise_on_lock_held: bool,
    ) -> Path | None:
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
        github_token = await self._resolve_github_token()

        downloader = GithubDirectoryDownloader()
        return await downloader.download(
            source_uri=github_uri,
            github_token=github_token,
            cache_path=cache_dir,
            store=self._store,
            log_diagnostics_on_not_found=log_diagnostics_on_not_found,
            raise_on_lock_held=raise_on_lock_held,
        )

    async def confirm_directory_absent(self, *, github_uri: str) -> bool:
        """Return True only if GitHub positively shows the directory is missing.

        fsspec raises a bare ``FileNotFoundError`` for any non-2xx reply (bad
        token, rate limit, private repo), so that exception alone does not prove
        absence. This lists the parent directory with a status-checked request:
        200 with a listing that lacks the name is real absence. Anything else
        (non-200, error, truncated listing) returns False (unconfirmed).
        """
        from languagemodelcommon.configs.config_reader.github_directory_downloader import (
            GithubDirectoryDownloader,
        )

        location = GithubDirectoryDownloader.parse_github_uri(github_uri)
        path = location.path.strip("/")
        parent, _, name = path.rpartition("/")
        if not name:
            return False
        token = await self._resolve_github_token()
        url = (
            f"https://api.github.com/repos/{location.owner}/{location.repository}"
            f"/contents/{parent}"
        ).rstrip("/")
        params = {"ref": location.branch} if location.branch else None
        auth = (
            (GithubDirectoryDownloader._github_token_username, token) if token else None
        )
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url, params=params, auth=auth)
            if response.status_code != 200:
                return False
            entries = response.json()
        except Exception as e:
            logger.warning(
                "Could not confirm absence of %s: [%s] %s",
                github_uri,
                type(e).__name__,
                e,
            )
            return False
        # The contents API truncates at 1000 entries; a truncated listing
        # cannot prove absence.
        if not isinstance(entries, list) or len(entries) >= 1000:
            return False
        return all(entry.get("name") != name for entry in entries)
