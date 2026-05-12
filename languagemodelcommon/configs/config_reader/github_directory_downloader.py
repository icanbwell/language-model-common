import logging
import shutil
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import fsspec  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GitLocation:
    repo_url: str
    owner: str
    repository: str
    path: str
    branch: str | None


class GithubDirectoryDownloader:
    """Downloads github:// directories into a local cache path using fsspec."""

    _github_uri_example = "github://my-org/my-repo/path?ref=main"
    _github_token_username = "x-access-token"

    _MAX_RETRIES = 3
    _RETRY_BASE_DELAY = 2.0

    def download(
        self,
        *,
        source_uri: str,
        github_token: str | None,
        cache_path: Path,
        cache_ttl_seconds: int = 0,
    ) -> Path:
        """Download a github:// URI to a local directory.

        Args:
            source_uri: github://owner/repo/path?ref=branch
            github_token: Optional GitHub token for private repos.
            cache_path: Local directory for cached downloads.
            cache_ttl_seconds: If > 0, skip download when existing cache
                is younger than this many seconds.

        Returns:
            Resolved path to the downloaded directory.

        Raises:
            ValueError: If the URI is malformed or download fails.
        """
        git_location = self.parse_github_uri(source_uri)
        source_path = git_location.path.strip("/")
        ref = git_location.branch or "HEAD"

        cache_root = cache_path.expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        key = f"{git_location.owner}/{git_location.repository}:{ref}:{source_path}"
        cache_dir_name = f"{git_location.owner}-{git_location.repository}-{sha256(key.encode('utf-8')).hexdigest()[:12]}"
        target_dir = (cache_root / cache_dir_name).resolve()
        if not str(target_dir).startswith(str(cache_root)):
            raise ValueError(f"Path traversal detected in github:// URI: {source_uri}")

        # If the on-disk cache is fresh, skip the download.  Multiple
        # workers may check this concurrently — that is fine; the worst
        # case is a few redundant downloads whose atomic swaps are harmless.
        if cache_ttl_seconds > 0 and self._is_cache_fresh(
            target_dir, cache_ttl_seconds
        ):
            logger.debug(
                "Cache for %s is fresh — skipping download",
                source_uri,
            )
            return target_dir.resolve()

        try:
            self._download_with_retry(
                git_location=git_location,
                source_path=source_path,
                github_token=github_token,
                target_dir=target_dir,
            )
            self._mark_cache_fresh(target_dir)
        except ValueError:
            # Download failed — fall back to stale cache if it exists.
            if target_dir.is_dir():
                logger.warning(
                    "Download failed for %s — serving stale cache from %s",
                    source_uri,
                    target_dir,
                )
            else:
                raise
        return target_dir.resolve()

    @staticmethod
    def _is_cache_fresh(target_dir: Path, ttl_seconds: int) -> bool:
        """Return True if the cache directory exists and was refreshed recently."""
        ts_file = target_dir.with_name(target_dir.name + ".ts")
        if not ts_file.exists() or not target_dir.is_dir():
            return False
        age = time.time() - ts_file.stat().st_mtime
        return age < ttl_seconds

    @staticmethod
    def _mark_cache_fresh(target_dir: Path) -> None:
        """Write a timestamp marker so other workers know the cache is fresh."""
        ts_file = target_dir.with_name(target_dir.name + ".ts")
        ts_file.write_text(str(time.time()))

    def _download_with_retry(
        self,
        *,
        git_location: GitLocation,
        source_path: str,
        github_token: str | None,
        target_dir: Path,
    ) -> None:
        """Try the download up to ``_MAX_RETRIES`` times with exponential backoff."""
        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                self._fetch_to_directory(
                    git_location=git_location,
                    source_path=source_path,
                    github_token=github_token,
                    target_dir=target_dir,
                )
                return
            except Exception as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES - 1:
                    delay = self._RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "Download attempt %d/%d failed for %s/%s (retrying in %.1fs): %s",
                        attempt + 1,
                        self._MAX_RETRIES,
                        git_location.owner,
                        git_location.repository,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
        raise ValueError(
            f"Download failed after {self._MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    def _fetch_to_directory(
        self,
        *,
        git_location: GitLocation,
        source_path: str,
        github_token: str | None,
        target_dir: Path,
    ) -> None:
        """Download remote content into *target_dir* using atomic swap.

        Downloads into a staging directory first, then swaps it into place.
        If the download fails, the existing *target_dir* is left untouched
        so callers can fall back to stale-but-valid cached data.
        """
        staging_dir = target_dir.with_name(target_dir.name + ".staging")
        old_dir = target_dir.with_name(target_dir.name + ".old")

        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            storage_options: dict[str, object] = {
                "org": git_location.owner,
                "repo": git_location.repository,
            }
            if git_location.branch:
                storage_options["sha"] = git_location.branch
            if github_token:
                storage_options["username"] = self._github_token_username
                storage_options["token"] = github_token

            filesystem = fsspec.filesystem("github", **storage_options)
            if source_path:
                filesystem.get(source_path, str(staging_dir), recursive=True)
            else:
                for remote_item in filesystem.ls("", detail=False):
                    item_path = str(remote_item)
                    if item_path in {".git", ".github"}:
                        continue
                    destination = staging_dir / Path(item_path).name
                    filesystem.get(item_path, str(destination), recursive=True)
        except ValueError:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise ValueError(
                "Unable to download github:// directory into cache"
            ) from exc

        # Atomic swap: staging → target, target → old
        if old_dir.exists():
            shutil.rmtree(old_dir)
        if target_dir.exists():
            shutil.move(str(target_dir), str(old_dir))
        shutil.move(str(staging_dir), str(target_dir))
        if old_dir.exists():
            shutil.rmtree(old_dir, ignore_errors=True)

    @classmethod
    def parse_github_uri(cls, source_uri: str) -> GitLocation:
        """Parse a github:// URI into components.

        Raises:
            ValueError: If the URI is not a valid github:// URI.
        """
        parsed = urlsplit(source_uri)
        if parsed.scheme != "github":
            raise ValueError(
                f"URI must use the github:// scheme, e.g. {cls._github_uri_example}"
            )
        if parsed.fragment:
            raise ValueError("github:// URI must not include a fragment")

        query_values = parse_qs(parsed.query, keep_blank_values=True)
        unsupported_query_params = set(query_values.keys()) - {"ref"}
        if unsupported_query_params:
            unsupported = ", ".join(sorted(unsupported_query_params))
            raise ValueError(
                f"github:// URI supports only '?ref=' query parameter; got: {unsupported}"
            )

        ref_values = query_values.get("ref")
        if ref_values and len(ref_values) > 1:
            raise ValueError("github:// URI must include a single '?ref=' value")
        if ref_values is not None and not ref_values[0].strip():
            raise ValueError("github:// URI '?ref=' value must not be empty")
        branch_from_query = ref_values[0].strip() if ref_values else None

        owner = parsed.netloc.strip()
        path_parts = [part for part in parsed.path.split("/") if part]

        if ":" in owner:
            repository_without_ref, separator, branch = owner.partition("@")
            if ":" not in repository_without_ref:
                raise ValueError(
                    f"github:// URI must include owner and repo, e.g. {cls._github_uri_example}"
                )
            legacy_owner, repo = repository_without_ref.split(":", 1)
            if not legacy_owner or not repo:
                raise ValueError(
                    f"github:// URI must include owner and repo, e.g. {cls._github_uri_example}"
                )
            if (
                branch_from_query is not None
                and separator
                and branch
                and branch_from_query != branch
            ):
                raise ValueError(
                    "github:// URI ref mismatch between legacy '@branch' and '?ref='"
                )
            owner = legacy_owner
            path_value = "/".join(path_parts)
            normalized_branch = (
                branch_from_query
                if branch_from_query is not None
                else (branch if separator and branch else None)
            )
        else:
            if not owner or not path_parts:
                raise ValueError(
                    f"github:// URI must include owner and repo, e.g. {cls._github_uri_example}"
                )
            repo = path_parts[0]
            path_value = "/".join(path_parts[1:])
            normalized_branch = branch_from_query

        if not owner or not repo:
            raise ValueError(
                f"github:// URI must include owner and repo, e.g. {cls._github_uri_example}"
            )

        return GitLocation(
            repo_url=f"https://github.com/{owner}/{repo}.git",
            owner=owner,
            repository=repo,
            path=path_value,
            branch=normalized_branch,
        )
