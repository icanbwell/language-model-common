"""Helper for downloading github:// URIs and manipulating github:// paths.

This module isolates GitHub-specific download, token handling, tempdir
caching, and URI joining so that :class:`ConfigReader` stays focused on
reading and parsing configuration files.
"""

import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)


def download_github_directory(github_uri: str) -> Path:
    """Download a ``github://`` URI to a local cache directory using fsspec."""
    from langchain_ai_skills_framework.loaders.github_directory_downloader import (  # type: ignore[import-not-found]
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
