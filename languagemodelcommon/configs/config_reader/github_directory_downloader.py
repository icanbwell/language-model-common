import logging
import os
import shutil
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import fsspec  # type: ignore[import-untyped]
import httpx
from key_value.aio.protocols.key_value import AsyncKeyValueProtocol

from languagemodelcommon.utilities.cache.advisory_lock import AdvisoryLock

logger = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"


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
    _PROBE_THROTTLE_SECONDS = 60.0

    def __init__(self) -> None:
        self._last_probed_at: dict[str, float] = {}

    async def download(
        self,
        *,
        source_uri: str,
        github_token: str | None,
        cache_path: Path,
        store: AsyncKeyValueProtocol | None = None,
        lock_ttl_seconds: int = 300,
    ) -> Path | None:
        """Download a github:// URI to a local directory.

        Args:
            source_uri: github://owner/repo/path?ref=branch
            github_token: Optional GitHub token for private repos.
            cache_path: Local directory for cached downloads.
            store: Optional key-value store for advisory locking.
                When provided, acquires a distributed lock before
                downloading. Returns None if lock is held by another worker.
            lock_ttl_seconds: TTL for the advisory lock.

        Returns:
            Resolved path to the downloaded content directory,
            or None if another worker holds the download lock.

        Raises:
            ValueError: If the URI is malformed or download fails.
        """
        git_location = self.parse_github_uri(source_uri)
        source_path = git_location.path.strip("/")
        ref = git_location.branch or "HEAD"

        cache_root = cache_path.expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        key = f"{git_location.owner}/{git_location.repository}:{ref}:{source_path}"
        cache_dir_name = (
            f"{git_location.owner}-{git_location.repository}"
            f"-{sha256(key.encode('utf-8')).hexdigest()[:12]}"
        )
        target_dir = (cache_root / cache_dir_name).resolve()
        if not str(target_dir).startswith(str(cache_root)):
            raise ValueError(f"Path traversal detected in github:// URI: {source_uri}")

        if store is not None:
            lock_key = f"github_download:{sha256(key.encode('utf-8')).hexdigest()[:16]}"
            async with AdvisoryLock(
                store, lock_key, ttl_seconds=lock_ttl_seconds
            ) as acquired:
                if not acquired:
                    logger.info(
                        "Download lock held for %s — skipping download",
                        source_uri,
                    )
                    if target_dir.is_dir():
                        return self._resolve_content_dir(
                            target_dir=target_dir, source_path=source_path
                        )
                    return None
                self._do_download(
                    git_location=git_location,
                    source_path=source_path,
                    github_token=github_token,
                    target_dir=target_dir,
                )
        else:
            self._do_download(
                git_location=git_location,
                source_path=source_path,
                github_token=github_token,
                target_dir=target_dir,
            )

        return self._resolve_content_dir(target_dir=target_dir, source_path=source_path)

    def _do_download(
        self,
        *,
        git_location: GitLocation,
        source_path: str,
        github_token: str | None,
        target_dir: Path,
    ) -> None:
        self._download_with_retry(
            git_location=git_location,
            source_path=source_path,
            github_token=github_token,
            target_dir=target_dir,
        )

    @staticmethod
    def _resolve_content_dir(*, target_dir: Path, source_path: str) -> Path:
        """Return the actual content directory within the cache.

        fsspec's ``get()`` preserves the last component of ``source_path``
        as a subdirectory inside ``target_dir``.
        """
        if source_path:
            last_component = Path(source_path).name
            content_dir = target_dir / last_component
            if content_dir.is_dir():
                return content_dir.resolve()
        return target_dir.resolve()

    def _download_with_retry(
        self,
        *,
        git_location: GitLocation,
        source_path: str,
        github_token: str | None,
        target_dir: Path,
    ) -> None:
        """Try the download up to ``_MAX_RETRIES`` times with exponential backoff.

        FileNotFoundError is never retried — it indicates the path does not
        exist in the repository (e.g. a missing client-override directory).
        """
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
            except FileNotFoundError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES - 1:
                    delay = self._RETRY_BASE_DELAY * (2**attempt)
                    exc_type = type(exc).__name__
                    logger.warning(
                        "Download attempt %d/%d failed for %s/%s "
                        "(retrying in %.1fs): [%s] %s",
                        attempt + 1,
                        self._MAX_RETRIES,
                        git_location.owner,
                        git_location.repository,
                        delay,
                        exc_type,
                        exc,
                    )
                    time.sleep(delay)
        source_uri = (
            f"github://{git_location.owner}/{git_location.repository}/{source_path}"
        )
        if git_location.branch:
            source_uri += f"?ref={git_location.branch}"
        token_status = (
            "GITHUB_TOKEN is set" if github_token else "GITHUB_TOKEN is NOT set"
        )
        exc_type = type(last_exc).__name__ if last_exc else "unknown"
        raise ValueError(
            f"Download failed after {self._MAX_RETRIES} attempts for {source_uri} "
            f"({token_status}): [{exc_type}] {last_exc}"
        ) from last_exc

    def _log_fetch_failure_diagnostics(
        self,
        *,
        git_location: GitLocation,
        source_path: str,
        github_token: str | None,
    ) -> None:
        """Best-effort re-probe of the request fsspec just made.

        fsspec's ``GithubFileSystem`` collapses any non-2xx response from
        GitHub into a bare ``FileNotFoundError`` -- with no status code,
        rate-limit headers, or response body attached (BAI-941). GitHub
        returns that same 404 both for a genuinely missing ref/subpath AND
        for a private repo the caller's credentials can't see, so the bare
        exception is ambiguous. This redoes the request outside fsspec purely
        to log what GitHub actually said, since fsspec has already discarded
        it by the time we get here.

        Uses the Contents API against ``source_path`` specifically (not just
        the repo root) -- fsspec's own root-only ``ls("")`` check during
        filesystem construction would report success even when the actual
        failure is a missing subpath deeper in the tree (e.g. a missing
        client-override directory, which ``_download_with_retry`` explicitly
        expects `FileNotFoundError` to also cover).

        Throttled per (owner, repo, sha, path) to avoid doubling GitHub API
        traffic on every retry during a sustained outage.
        """
        sha = git_location.branch
        probe_key = (
            f"{git_location.owner}/{git_location.repository}:{sha}:{source_path}"
        )
        now = time.monotonic()
        last_probed_at = self._last_probed_at.get(probe_key)
        if last_probed_at is not None and (
            now - last_probed_at < self._PROBE_THROTTLE_SECONDS
        ):
            return
        self._last_probed_at[probe_key] = now

        url = (
            f"{_GITHUB_API_BASE}/repos/{git_location.owner}/{git_location.repository}"
            f"/contents/{source_path}"
        )
        params = {"ref": sha} if sha else None
        # Mirrors fsspec.implementations.github.GithubFileSystem.kw: HTTP Basic
        # Auth with the token as the password, not a Bearer header — must match
        # exactly what fsspec just sent, or the probe answers a different question.
        auth = (self._github_token_username, github_token) if github_token else None
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(url, params=params, auth=auth)
        except Exception as probe_exc:
            logger.error(
                "GitHub download diagnostic probe failed for %s: [%s] %s",
                url,
                type(probe_exc).__name__,
                probe_exc,
            )
            return
        try:
            body_message = response.json().get("message")
        except Exception:
            body_message = response.text[:200]
        logger.error(
            "GitHub download hit FileNotFoundError for %s (ref=%s, token_set=%s, "
            "token_len=%s) — probe response: status=%s "
            "rate_limit_remaining=%s rate_limit_reset=%s body_message=%r",
            url,
            sha or "(default branch)",
            bool(github_token),
            len(github_token) if github_token else 0,
            response.status_code,
            response.headers.get("X-RateLimit-Remaining"),
            response.headers.get("X-RateLimit-Reset"),
            body_message,
        )

    def _fetch_to_directory(
        self,
        *,
        git_location: GitLocation,
        source_path: str,
        github_token: str | None,
        target_dir: Path,
    ) -> None:
        """Download remote content into *target_dir* using atomic swap."""
        pid = os.getpid()
        staging_dir = target_dir.with_name(f"{target_dir.name}.staging.{pid}")
        old_dir = target_dir.with_name(f"{target_dir.name}.old.{pid}")

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

            filesystem = fsspec.filesystem(
                "github", skip_instance_cache=True, **storage_options
            )
            if source_path:
                filesystem.get(source_path, str(staging_dir), recursive=True)
            else:
                for remote_item in filesystem.ls("", detail=False):
                    item_path = str(remote_item)
                    if item_path in {".git", ".github"}:
                        continue
                    destination = staging_dir / Path(item_path).name
                    filesystem.get(item_path, str(destination), recursive=True)
        except (ValueError, FileNotFoundError) as exc:
            if isinstance(exc, FileNotFoundError):
                self._log_fetch_failure_diagnostics(
                    git_location=git_location,
                    source_path=source_path,
                    github_token=github_token,
                )
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(staging_dir, ignore_errors=True)
            source_uri = (
                f"github://{git_location.owner}/{git_location.repository}/{source_path}"
            )
            if git_location.branch:
                source_uri += f"?ref={git_location.branch}"
            exc_type = type(exc).__name__
            exc_detail = str(exc) or "(no message)"
            status_code = ""
            if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
                status_code = f" [HTTP {exc.response.status_code}]"
            raise ValueError(
                f"Unable to download github:// directory into cache: {source_uri} — "
                f"{exc_type}{status_code}: {exc_detail}"
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
                f"github:// URI supports only '?ref=' query parameter; "
                f"got: {unsupported}"
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
                    f"github:// URI must include owner and repo, "
                    f"e.g. {cls._github_uri_example}"
                )
            legacy_owner, repo = repository_without_ref.split(":", 1)
            if not legacy_owner or not repo:
                raise ValueError(
                    f"github:// URI must include owner and repo, "
                    f"e.g. {cls._github_uri_example}"
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
                    f"github:// URI must include owner and repo, "
                    f"e.g. {cls._github_uri_example}"
                )
            repo = path_parts[0]
            path_value = "/".join(path_parts[1:])
            normalized_branch = branch_from_query

        if not owner or not repo:
            raise ValueError(
                f"github:// URI must include owner and repo, "
                f"e.g. {cls._github_uri_example}"
            )

        return GitLocation(
            repo_url=f"https://github.com/{owner}/{repo}.git",
            owner=owner,
            repository=repo,
            path=path_value,
            branch=normalized_branch,
        )
