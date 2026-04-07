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

from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS
from languagemodelcommon.utilities.url_parser import UrlParser

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)


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
    branch = parts[3]
    path = unquote("/".join(parts[4:])) if len(parts) > 4 else ""

    uri_path = f"/{repo}/{path}" if path else f"/{repo}"
    return f"github://{owner}{uri_path}?ref={branch}"


def is_github_path(path: str) -> bool:
    """Return ``True`` if *path* is a ``github://`` URI or ``https://github.com/`` URL."""
    return path.startswith("github://") or UrlParser.is_github_url(path)


def to_github_uri(path: str) -> str:
    """Normalize a GitHub path to a ``github://`` URI.

    Passes ``github://`` URIs through unchanged and converts
    ``https://github.com/`` URLs.  Raises :class:`ValueError` if
    *path* is not a recognized GitHub path.
    """
    if path.startswith("github://"):
        return path
    if UrlParser.is_github_url(path):
        return github_url_to_uri(path)
    raise ValueError(f"Not a GitHub path: {path}")


def resolve_github_path(path: str) -> Path:
    """Resolve a GitHub path to a local directory.

    Accepts ``github://`` URIs, ``https://github.com/`` URLs, or local paths.
    GitHub paths are downloaded via fsspec; local paths are returned as-is.
    """
    if is_github_path(path):
        return download_github_directory(to_github_uri(path))
    return Path(path)


def download_github_directory(github_uri: str) -> Path:
    """Download a ``github://`` URI to a local cache directory using fsspec."""
    from langchain_ai_skills_framework.loaders.github_directory_downloader import (
        GithubDirectoryDownloader,
    )

    github_token = os.environ.get("GITHUB_TOKEN")
    cache_path = Path(tempfile.gettempdir()) / "github_config_cache"
    downloader = GithubDirectoryDownloader()
    result: Path = downloader.download(
        source_uri=github_uri,
        github_token=github_token,
        cache_path=cache_path,
    )
    return result


def join_github_uri_path(base_uri: str, suffix: str) -> str:
    """Join a path suffix onto a ``github://`` URI, preserving query params."""
    parts = urlsplit(base_uri)
    new_path = parts.path.rstrip("/") + "/" + suffix.strip("/")
    return urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, ""))
