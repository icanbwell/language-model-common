import asyncio
import logging
import re
import time

from hashlib import sha256
from pathlib import Path
from typing import Any, List, Optional, cast
from uuid import UUID, uuid4

from languagemodelcommon.configs.config_reader.file_config_reader import (
    FileConfigReader,
)
from languagemodelcommon.configs.config_reader.github_directory_helper import (
    GitHubDirectoryHelper,
)
from languagemodelcommon.configs.config_reader.override_config_cache import (
    CachedOverride,
    ClientOverrideConfigCache,
)
from languagemodelcommon.configs.config_reader.override_read_result import (
    DownloadLockHeldError,
    OverrideOutcome,
    OverrideReadResult,
)
from languagemodelcommon.configs.config_reader.s3_config_reader import S3ConfigReader
from oidcauthlib.cache_gate.advisory_lock import AdvisoryLock
from languagemodelcommon.configs.schemas.config_schema import (
    ChatModelConfig,
    PromptConfig,
)
from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
    PromptLibraryManager,
)
from key_value.aio.stores.base import BaseStore

from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS
from languagemodelcommon.utilities.ref_hash import compute_short_ref_hash
from languagemodelcommon.configs.config_reader.mcp_json_fetcher import McpJsonFetcher
from languagemodelcommon.configs.config_reader.mcp_json_reader import (
    resolve_mcp_servers_from_plugins,
)

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)

_MAX_CLIENT_ID_LENGTH = 64
_COLD_MISS_TIMEOUT_SECONDS = 180.0
_OVERRIDE_PEER_POLL_INTERVAL_SECONDS = 0.05
_OVERRIDE_PEER_POLL_TIMEOUT_SECONDS = 3.0
_OVERRIDE_REFRESH_LOCK_TTL_SECONDS = 60
_OVERRIDE_REFRESH_RETRY_AFTER_SECONDS = 15.0


class ConfigReader:
    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        prompt_library_manager: PromptLibraryManager,
        environment_variables: LanguageModelCommonEnvironmentVariables,
        mcp_json_fetcher: McpJsonFetcher | None = None,
        github_directory_helper: GitHubDirectoryHelper | None = None,
        model_config_cache_store: BaseStore | None = None,
        file_config_reader: FileConfigReader | None = None,
        s3_config_reader: S3ConfigReader | None = None,
        override_config_cache: ClientOverrideConfigCache | None = None,
    ) -> None:
        self._identifier: UUID = uuid4()
        self._lock: asyncio.Lock = asyncio.Lock()
        self._mcp_resolved_once: bool = False
        self._prompt_library_manager = prompt_library_manager
        self._environment_variables = environment_variables
        self._github_directory_helper = (
            github_directory_helper
            or GitHubDirectoryHelper(environment_variables=self._environment_variables)
        )
        self._mcp_json_fetcher = mcp_json_fetcher
        self._model_config_cache_store = model_config_cache_store
        self._file_config_reader = file_config_reader or FileConfigReader()
        self._s3_config_reader = s3_config_reader or S3ConfigReader()
        self._model_config_cache_collection = (
            self._environment_variables.model_config_cache_collection_name
        )
        self._override_config_cache = override_config_cache
        # In-flight override refreshes by cache key, so concurrent requests in
        # this process share one refresh. Holds tasks only, removed on completion;
        # it is not a cache of config data.
        self._inflight_override_refreshes: dict[
            str, asyncio.Task[OverrideReadResult]
        ] = {}

    async def read_model_configs_async(
        self, *, client_id: str | None = None
    ) -> List[ChatModelConfig]:
        config_path = self._environment_variables.models_official_path
        models_testing_path = self._environment_variables.models_testing_path

        base_models = await self._read_base_models_async(
            config_path=config_path,
            models_testing_path=models_testing_path,
        )

        # Re-resolve MCP servers on first request (cache may be stale)
        # and on subsequent requests if servers remain unresolved.
        if not self._mcp_resolved_once or self._has_unresolved_mcp_servers(base_models):
            await self._retry_mcp_resolution(
                models=base_models,
                config_path=config_path,
            )
            self._mcp_resolved_once = True

        if client_id:
            override_models = await self._read_override_models_async(
                config_path=config_path,
                client_id=client_id,
            )
            if override_models:
                base_models = self._merge_model_configs(
                    base_models=base_models, override_models=override_models
                )

        base_models = [model for model in base_models if not model.disabled]
        await self._resolve_prompt_library_async(
            models=base_models, config_path=config_path
        )
        return base_models

    async def _read_base_models_async(
        self,
        *,
        config_path: str,
        models_testing_path: Optional[str],
    ) -> List[ChatModelConfig]:
        # Read from MongoDB model config cache
        models = await self._read_from_model_config_cache(config_path=config_path)
        if models:
            return models

        logger.info(
            "ConfigReader with id: %s no model config cache, reading from source",
            self._identifier,
        )

        async with self._lock:
            # Double-check after acquiring lock
            models = await self._read_from_model_config_cache(config_path=config_path)
            if models:
                return models

            default_config_path = self._resolve_default_config_path(config_path)
            logger.info(
                "ConfigReader with id: %s reading model configurations from %s",
                self._identifier,
                default_config_path,
            )

            models = await self._read_configs_with_retry(
                config_path=config_path,
                default_config_path=default_config_path,
                models_testing_path=models_testing_path,
            )

            if not models:
                logger.warning(
                    "ConfigReader with id: %s read 0 model configurations "
                    "from %s and no model config cache available",
                    self._identifier,
                    default_config_path,
                )
                return models

            await self._write_to_model_config_cache(
                models=models, config_path=config_path
            )
            return models

    def _model_config_cache_key(self, *, config_path: str, model_name: str) -> str:
        ref_hash = compute_short_ref_hash(config_path)
        return f"v{self.SCHEMA_VERSION}:{ref_hash}:{model_name}"

    async def _read_from_model_config_cache(
        self, *, config_path: str
    ) -> List[ChatModelConfig] | None:
        """Load model configs from the model config cache (one row per model).

        Scans the collection for all entries matching the current schema
        version and config-source prefix.

        Returns ``None`` when the cache has no entries.
        Raises on store or deserialization errors so that a misconfigured
        cache backend surfaces immediately (fail-fast).
        """
        if not self._model_config_cache_store:
            return None
        ref_hash = compute_short_ref_hash(config_path)
        prefix = f"v{self.SCHEMA_VERSION}:{ref_hash}:"
        model_keys = await self._get_cache_keys_by_prefix(prefix=prefix)
        if not model_keys:
            return None
        results = await self._model_config_cache_store.get_many(
            model_keys,
            collection=self._model_config_cache_collection,
        )
        models: List[ChatModelConfig] = []
        for data in results:
            if data is not None:
                models.append(ChatModelConfig.model_validate(data))
        logger.info(
            "ConfigReader with id: %s loaded %d configs from model config cache",
            self._identifier,
            len(models),
        )
        return models if models else None

    async def _get_cache_keys_by_prefix(self, *, prefix: str) -> list[str]:
        """Return all cache keys in the collection that start with *prefix*.

        Uses the underlying MongoDB collection's ``find`` when available,
        falling back to the store's ``keys()`` method for other backends.
        """
        store = self._model_config_cache_store
        if store is None:
            return []

        collection_name = self._model_config_cache_collection
        # MongoDBStore exposes the raw collection via _collections_by_name
        collections_map = getattr(store, "_collections_by_name", None)
        if collections_map and collection_name in collections_map:
            mongo_collection = collections_map[collection_name]
            cursor = mongo_collection.find(
                {"key": {"$regex": f"^{prefix}"}},
                projection={"key": 1, "_id": 0},
            )
            return [doc["key"] async for doc in cursor]

        return []

    async def _write_to_model_config_cache(
        self, *, models: List[ChatModelConfig], config_path: str
    ) -> None:
        """Store each model config as a separate row in the cache.

        Raises on write errors so that a misconfigured cache backend
        surfaces immediately (fail-fast).
        """
        if not self._model_config_cache_store:
            return
        ttl = self._environment_variables.model_config_cache_ttl_seconds
        for model in models:
            key = self._model_config_cache_key(
                config_path=config_path, model_name=model.name
            )
            await self._model_config_cache_store.put(
                key,
                model.model_dump(),
                ttl=ttl,
                collection=self._model_config_cache_collection,
            )
        logger.info(
            "ConfigReader with id: %s wrote %d configs to model config cache",
            self._identifier,
            len(models),
        )

    async def _read_configs_with_retry(
        self,
        *,
        config_path: str,
        default_config_path: str,
        models_testing_path: Optional[str],
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> List[ChatModelConfig]:
        """Read configs from the filesystem, retrying on empty results.

        At startup the config directory may not exist yet (the GitHub
        config repo extraction may still be in progress or another
        Gunicorn worker may be mid-swap).  A short retry loop avoids
        permanently returning 0 configs when the directory appears
        moments later.
        """
        base_exclude_dirs = {"clients", "env"}
        for attempt in range(max_retries + 1):
            try:
                models = await self.read_models_from_path_async(
                    config_path=default_config_path, exclude_dirs=base_exclude_dirs
                )
                if not models and default_config_path != config_path:
                    models = await self.read_models_from_path_async(
                        config_path=config_path, exclude_dirs=base_exclude_dirs
                    )
                if models_testing_path:
                    models_testing = await self.read_models_from_path_async(
                        config_path=models_testing_path, exclude_dirs=base_exclude_dirs
                    )
                    if models_testing and len(models_testing) > 0:
                        models.append(
                            ChatModelConfig(
                                id="testing",
                                name="----- Models in Testing -----",
                                description="",
                            )
                        )
                        models.extend(models_testing)
            except Exception as e:
                logger.warning(
                    "Error reading model configurations (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                models = []

            if models:
                return models

            if attempt < max_retries:
                logger.warning(
                    "ConfigReader with id: %s read 0 configs from %s, "
                    "retrying in %.1fs (attempt %d/%d)",
                    self._identifier,
                    default_config_path,
                    retry_delay,
                    attempt + 1,
                    max_retries + 1,
                )
                await asyncio.sleep(retry_delay)

        return []

    async def _read_override_models_async(
        self,
        *,
        config_path: str,
        client_id: str,
    ) -> List[ChatModelConfig]:
        override_path = self._resolve_override_config_path(
            config_path=config_path, client_id=client_id
        )
        if override_path is None:
            return []
        cache = self._override_config_cache
        if (
            cache is None
            or not self._environment_variables.model_config_override_cache_enabled
            or not self._is_remote_config_path(config_path=config_path)
            or not self._is_cacheable_client_id(client_id=client_id)
        ):
            # Flag off, no store, local path, or an id that must not become a
            # cache key: read live exactly as before the cache existed.
            result = await self._fetch_override_result(
                override_path=override_path, confirm_absent=False
            )
            return list(result.models)
        return await self._read_override_models_cached(
            cache=cache,
            config_path=config_path,
            override_path=override_path,
            client_id=client_id,
        )

    @staticmethod
    def _is_cacheable_client_id(*, client_id: str) -> bool:
        """Stricter than ``_is_valid_client_id``: applies only where a cache key
        is built, so flag-off behaviour for existing ids is unchanged."""
        return len(client_id) <= _MAX_CLIENT_ID_LENGTH and bool(
            re.fullmatch(r"[a-zA-Z0-9_-]+", client_id)
        )

    @staticmethod
    def _is_remote_config_path(*, config_path: str) -> bool:
        return GitHubDirectoryHelper.is_github_path(
            config_path
        ) or config_path.startswith("s3")

    async def _fetch_override_result(
        self, *, override_path: str, confirm_absent: bool
    ) -> OverrideReadResult:
        """Read one override from its source and classify the outcome.

        A bare ``FileNotFoundError`` is ambiguous (fsspec raises it for any
        non-2xx reply, e.g. a bad token or rate limit). With ``confirm_absent``
        it is classified ``ABSENT`` only after GitHub positively shows the
        directory is missing; otherwise ``FAILED``.
        """
        try:
            models = await self.read_models_from_path_async(
                config_path=override_path,
                log_diagnostics_on_not_found=False,
                # Strict only on the cached path; flag-off keeps the pre-cache
                # fallback to the pod-local copy when the lock is held.
                raise_on_lock_held=confirm_absent,
            )
        except DownloadLockHeldError:
            logger.info(
                "ConfigReader with id: %s override download lock held for %s",
                self._identifier,
                override_path,
            )
            return OverrideReadResult(outcome=OverrideOutcome.LOCK_HELD)
        except FileNotFoundError:
            logger.debug(
                "No client overrides found at %s (path does not exist)", override_path
            )
            if confirm_absent and not await self._confirm_override_absent(
                override_path=override_path
            ):
                return OverrideReadResult(outcome=OverrideOutcome.FAILED)
            return OverrideReadResult(outcome=OverrideOutcome.ABSENT)
        except ValueError as e:
            if "FileNotFoundError" in str(e):
                logger.debug(
                    "No client overrides found at %s (path does not exist)",
                    override_path,
                )
            else:
                logger.warning(
                    "Failed to load client overrides from %s: %s", override_path, e
                )
            return OverrideReadResult(outcome=OverrideOutcome.FAILED)
        except Exception as e:
            logger.warning(
                "Failed to load client overrides from %s: %s", override_path, e
            )
            return OverrideReadResult(outcome=OverrideOutcome.FAILED)
        return OverrideReadResult(
            outcome=OverrideOutcome.FOUND,
            models=tuple(models),
            has_unresolved_mcp=self._has_unresolved_mcp_servers(models),
        )

    async def _confirm_override_absent(self, *, override_path: str) -> bool:
        if not GitHubDirectoryHelper.is_github_path(override_path):
            return False
        return await self._github_directory_helper.confirm_directory_absent(
            github_uri=GitHubDirectoryHelper.to_github_uri(override_path)
        )

    async def _read_override_models_cached(
        self,
        *,
        cache: ClientOverrideConfigCache,
        config_path: str,
        override_path: str,
        client_id: str,
    ) -> List[ChatModelConfig]:
        """Read-through, stale-while-revalidate read of one client's override."""
        started = time.monotonic()
        cached = await self._safe_override_cache_get(
            cache=cache, config_path=config_path, client_id=client_id
        )
        if cached is not None:
            if cached.is_fresh:
                outcome = (
                    "hit"
                    if cached.result.outcome == OverrideOutcome.FOUND
                    else "absent_hit"
                )
            else:
                outcome = "stale"
                self._start_override_refresh(
                    cache=cache,
                    config_path=config_path,
                    override_path=override_path,
                    client_id=client_id,
                    stale=cached,
                )
            self._log_override_read(
                outcome=outcome,
                client_id=client_id,
                age=cached.age_seconds,
                started=started,
            )
            return list(cached.result.models)

        task = self._start_override_refresh(
            cache=cache,
            config_path=config_path,
            override_path=override_path,
            client_id=client_id,
            stale=None,
        )
        # shield: a cancelled request must not cancel a refresh others share.
        try:
            result = await asyncio.wait_for(
                asyncio.shield(task), timeout=_COLD_MISS_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.warning(
                "ConfigReader with id: %s override refresh for client %s exceeded %.0fs; "
                "serving base config for this request",
                self._identifier,
                client_id,
                _COLD_MISS_TIMEOUT_SECONDS,
            )
            return []
        self._log_override_read(
            outcome="miss", client_id=client_id, age=0.0, started=started
        )
        return list(result.models)

    def _start_override_refresh(
        self,
        *,
        cache: ClientOverrideConfigCache,
        config_path: str,
        override_path: str,
        client_id: str,
        stale: CachedOverride | None,
    ) -> "asyncio.Task[OverrideReadResult]":
        key = cache.build_key(config_path=config_path, client_id=client_id)
        existing = self._inflight_override_refreshes.get(key)
        if existing is not None:
            return existing
        task = asyncio.ensure_future(
            self._refresh_override(
                cache=cache,
                key=key,
                config_path=config_path,
                override_path=override_path,
                client_id=client_id,
                stale=stale,
            )
        )
        self._inflight_override_refreshes[key] = task
        task.add_done_callback(
            lambda _t: self._inflight_override_refreshes.pop(key, None)
        )
        return task

    async def _refresh_override(
        self,
        *,
        cache: ClientOverrideConfigCache,
        key: str,
        config_path: str,
        override_path: str,
        client_id: str,
        stale: CachedOverride | None,
    ) -> OverrideReadResult:
        """Refresh one override under a cross-worker advisory lock.

        Never raises: a failure keeps stale data (if any) and returns the
        classified result. LOCK_HELD and FAILED are never written to the cache.
        """
        try:
            lock_name = (
                f"override_refresh:{sha256(key.encode('utf-8')).hexdigest()[:16]}"
            )
            async with AdvisoryLock(
                cache.store,
                lock_name,
                ttl_seconds=_OVERRIDE_REFRESH_LOCK_TTL_SECONDS,
            ) as acquired:
                if not acquired:
                    # Another worker is refreshing. Serve stale if we have it;
                    # else wait (bounded) for its row. There is no "fetch
                    # directly" bypass: the downloader has its own lock, so it
                    # would only return lock-held anyway. On timeout, return
                    # LOCK_HELD (no override for this request, nothing cached),
                    # the same outcome the uncached path gives.
                    if stale is not None:
                        return stale.result
                    peer = await self._await_peer_override(
                        cache=cache, config_path=config_path, client_id=client_id
                    )
                    if peer is not None:
                        return peer.result
                    return OverrideReadResult(outcome=OverrideOutcome.LOCK_HELD)
                return await self._fetch_and_store_override(
                    cache=cache,
                    config_path=config_path,
                    override_path=override_path,
                    client_id=client_id,
                    stale=stale,
                )
        except Exception as e:
            logger.warning(
                "override_refresh unexpected error for client %s: [%s] %s",
                client_id,
                type(e).__name__,
                e,
            )
            if stale is not None:
                return stale.result
            return OverrideReadResult(outcome=OverrideOutcome.FAILED)

    async def _await_peer_override(
        self,
        *,
        cache: ClientOverrideConfigCache,
        config_path: str,
        client_id: str,
    ) -> CachedOverride | None:
        """Poll briefly for the lock holder's row; None if it never appears."""
        deadline = time.monotonic() + _OVERRIDE_PEER_POLL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            await asyncio.sleep(_OVERRIDE_PEER_POLL_INTERVAL_SECONDS)
            peer = await self._safe_override_cache_get(
                cache=cache, config_path=config_path, client_id=client_id
            )
            if peer is not None:
                return peer
        return None

    async def _fetch_and_store_override(
        self,
        *,
        cache: ClientOverrideConfigCache,
        config_path: str,
        override_path: str,
        client_id: str,
        stale: CachedOverride | None,
    ) -> OverrideReadResult:
        started = time.monotonic()
        result = await self._fetch_override_result(
            override_path=override_path, confirm_absent=True
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        served_stale = False
        try:
            if result.outcome in (OverrideOutcome.FOUND, OverrideOutcome.ABSENT):
                await cache.put(
                    config_path=config_path, client_id=client_id, result=result
                )
            elif stale is not None:
                served_stale = True
                await cache.defer_refresh(
                    config_path=config_path,
                    client_id=client_id,
                    cached=stale,
                    retry_after_seconds=_OVERRIDE_REFRESH_RETRY_AFTER_SECONDS,
                )
        except Exception as e:
            logger.warning(
                "override_cache write failed for client %s: [%s] %s",
                client_id,
                type(e).__name__,
                e,
            )
        logger.info(
            "override_refresh result=%s download_ms=%d served_stale=%s client_id=%s",
            result.outcome.value,
            duration_ms,
            served_stale,
            client_id,
        )
        if stale is not None and result.outcome in (
            OverrideOutcome.FAILED,
            OverrideOutcome.LOCK_HELD,
        ):
            return stale.result
        return result

    async def _safe_override_cache_get(
        self,
        *,
        cache: ClientOverrideConfigCache,
        config_path: str,
        client_id: str,
    ) -> CachedOverride | None:
        """Cache read that degrades to a live read if the store errors.

        The override cache is an optimisation layer, so a store or row
        deserialisation error is logged and treated as a miss (unlike the base
        model-config cache, which fails fast).
        """
        try:
            return await cache.get(config_path=config_path, client_id=client_id)
        except Exception as e:
            logger.warning(
                "override_cache read failed for client %s, using live read: [%s] %s",
                client_id,
                type(e).__name__,
                e,
            )
            return None

    def _log_override_read(
        self, *, outcome: str, client_id: str, age: float, started: float
    ) -> None:
        logger.info(
            "override_cache outcome=%s age_s=%.1f duration_ms=%d client_id=%s",
            outcome,
            age,
            int((time.monotonic() - started) * 1000),
            client_id,
        )

    async def clear_client_override(self, *, client_id: str) -> None:
        """Drop one client's cached override so the next request re-reads it."""
        cache = self._override_config_cache
        if cache is None or not self._is_valid_client_id(client_id):
            return
        await cache.delete(
            config_path=self._environment_variables.models_official_path,
            client_id=client_id,
        )

    async def read_models_from_path_async(
        self,
        *,
        config_path: str,
        exclude_dirs: set[str] | None = None,
        log_diagnostics_on_not_found: bool = True,
        raise_on_lock_held: bool = False,
    ) -> List[ChatModelConfig]:
        """
        raise_on_lock_held (BAI-1380): when the GitHub download lock is held by
        another worker, raise ``DownloadLockHeldError`` instead of returning
        ``[]``, so callers can tell "nothing read" from "no models".

        log_diagnostics_on_not_found (BAI-941): pass ``False`` when a missing
        ``config_path`` is an expected, benign outcome for this caller (e.g.
        an optional per-client override) rather than a genuine failure worth
        the diagnostic re-probe/ERROR log.
        """
        models: List[ChatModelConfig]
        local_config_path: str = config_path
        if config_path.startswith("s3"):
            models = await self._s3_config_reader.read_model_configs(s3_url=config_path)
            logger.info(
                "ConfigReader with id: %s loaded %s model configurations from S3",
                self._identifier,
                len(models),
            )
        elif GitHubDirectoryHelper.is_github_path(config_path):
            resolved = await self._github_directory_helper.resolve_github_path(
                config_path,
                log_diagnostics_on_not_found=log_diagnostics_on_not_found,
                raise_on_lock_held=raise_on_lock_held,
            )
            if resolved is None:
                if raise_on_lock_held:
                    raise DownloadLockHeldError(config_path)
                logger.info(
                    "ConfigReader with id: %s download lock held for %s — using cache",
                    self._identifier,
                    config_path,
                )
                return []
            local_config_path = str(resolved)
            models = await asyncio.to_thread(
                self._file_config_reader.read_model_configs,
                config_path=local_config_path,
                exclude_dirs=exclude_dirs,
            )
            logger.info(
                "ConfigReader with id: %s loaded %s model configurations from GitHub",
                self._identifier,
                len(models),
            )
        else:
            models = await asyncio.to_thread(
                self._file_config_reader.read_model_configs,
                config_path=config_path,
                exclude_dirs=exclude_dirs,
            )
            logger.info(
                "ConfigReader with id: %s loaded %s model configurations from file system",
                self._identifier,
                len(models),
            )

        # Resolve MCP server references from plugins or local .mcp.json
        await self._resolve_mcp_servers_async(
            models=models,
            config_path=local_config_path,
        )
        return models

    async def _resolve_mcp_servers_async(
        self,
        *,
        models: List[ChatModelConfig],
        config_path: str,
    ) -> None:
        """Resolve ``mcp_server`` references on model tools/agents.

        All MCP server definitions come from marketplace plugins via
        the ``McpJsonFetcher``.  Each model must declare its ``plugins``
        list; servers are resolved only from those declared plugins.
        """
        has_mcp_refs = any(a.mcp_server for m in models for a in m.get_agents())
        if not has_mcp_refs:
            return

        if not self._mcp_json_fetcher:
            logger.warning(
                "Models have mcp_server references but no McpJsonFetcher "
                "is configured — MCP servers will not be resolved"
            )
            return

        all_plugin_names: list[str] = []
        for m in models:
            if m.plugins:
                all_plugin_names.extend(m.plugins)
        unique_plugins = list(dict.fromkeys(all_plugin_names))
        if not unique_plugins:
            models_with_refs = [
                m.name or m.id
                for m in models
                if any(a.mcp_server for a in m.get_agents())
            ]
            logger.warning(
                "Models have mcp_server references but no plugins declared "
                "— MCP servers will not be resolved. Affected models: %s",
                models_with_refs,
            )
            return

        plugin_configs, fetch_errors = await self._mcp_json_fetcher.fetch_plugins_async(
            unique_plugins
        )
        if fetch_errors:
            logger.warning(
                "MCP plugin config fetch errors: %s",
                fetch_errors,
            )
        if plugin_configs:
            resolve_mcp_servers_from_plugins(
                configs=models, plugin_configs=plugin_configs
            )
        else:
            logger.warning(
                "McpJsonFetcher returned no configs for plugins %s from %s "
                "— MCP servers will remain unresolved",
                unique_plugins,
                self._mcp_json_fetcher._url,
            )

    @staticmethod
    def _has_unresolved_mcp_servers(models: List[ChatModelConfig]) -> bool:
        """Check whether any model has agents with ``mcp_server`` set but no ``url``."""
        for model in models:
            for agent in model.get_agents():
                if agent.mcp_server and not agent.url:
                    return True
        return False

    @staticmethod
    def _get_unresolved_mcp_servers(
        models: List[ChatModelConfig],
    ) -> List[str]:
        """Return descriptions of unresolved mcp_server refs for diagnostics."""
        unresolved: List[str] = []
        for model in models:
            for agent in model.get_agents():
                if agent.mcp_server and not agent.url:
                    unresolved.append(
                        f"{model.name or model.id}:{agent.name}(mcp_server={agent.mcp_server})"
                    )
        return unresolved

    async def _retry_mcp_resolution(
        self, *, models: List[ChatModelConfig], config_path: str
    ) -> None:
        """Re-attempt MCP server resolution for models with unresolved refs.

        Called on first request (to pick up .mcp.json changes not in the
        cache) and when cached configs carry ``mcp_server`` references
        without a resolved ``url``.  Always writes back to the model config
        cache so subsequent reads get the resolved data.
        """
        logger.info("Retrying MCP server resolution for models with unresolved refs")
        await self._resolve_mcp_servers_async(models=models, config_path=config_path)
        await self._write_to_model_config_cache(models=models, config_path=config_path)
        unresolved = self._get_unresolved_mcp_servers(models)
        if unresolved:
            fetcher_url = (
                self._mcp_json_fetcher._url if self._mcp_json_fetcher else "N/A"
            )
            logger.warning(
                "MCP server resolution retry did not resolve all refs. "
                "Failed to fetch plugin configs from %s. Unresolved: %s",
                fetcher_url,
                unresolved,
            )
        else:
            logger.info("MCP server resolution retry succeeded")

    @staticmethod
    def _resolve_default_config_path(config_path: str) -> str:
        return config_path

    @staticmethod
    def _resolve_override_config_path(
        *, config_path: str, client_id: str
    ) -> str | None:
        """Resolve the per-client override config path for *client_id*.

        SECURITY: ``client_id`` originates from an untrusted, caller-supplied
        value (e.g. an HTTP header) and is used to build a filesystem/URI
        path. The allowlist check below and the ``resolve()``/``relative_to()``
        containment check further down are both load-bearing — together they
        prevent a path-traversal / local-file-inclusion vulnerability (e.g.
        ``client_id="../../etc/passwd"``). Do not remove either check as
        "unnecessary defensive code": a prior version of this logic (in the
        baileyai repo, before it moved here) was flagged by a security scan
        for exactly this class of bug.
        """
        if not client_id:
            return None
        # Validate client_id to prevent path traversal
        if not ConfigReader._is_valid_client_id(client_id):
            logger.warning("Invalid client_id format: %s", client_id)
            return None
        if GitHubDirectoryHelper.is_github_path(config_path):
            return GitHubDirectoryHelper.join_github_uri_path(
                base_uri=GitHubDirectoryHelper.to_github_uri(config_path),
                suffix=f"clients/{client_id}",
            )
        if config_path.startswith("s3"):
            return ConfigReader._join_path(
                base=config_path, suffix=f"clients/{client_id}"
            )
        config_folder = Path(config_path)
        override_folder = config_folder.joinpath("clients", client_id)
        # Ensure the resolved path is within the config directory. Defense
        # in depth alongside the allowlist check above — see this method's
        # docstring for why both checks must stay.
        try:
            override_folder.resolve().relative_to(config_folder.resolve())
        except ValueError:
            logger.warning(
                "Client config path traversal attempt detected: %s",
                override_folder,
            )
            return None
        if override_folder.exists():
            return str(override_folder)
        return None

    @staticmethod
    def _is_valid_client_id(client_id: str) -> bool:
        """Validate that client_id contains only safe characters.

        ``client_id`` is untrusted input used to build a filesystem/URI
        path (see ``_resolve_override_config_path``). This allowlist blocks
        path-traversal sequences (``..``, ``/``) before the value ever
        reaches a path join.
        """
        import re

        return bool(re.match(r"^[a-zA-Z0-9_-]+$", client_id))

    @staticmethod
    def _join_path(*, base: str, suffix: str) -> str:
        if base.endswith("/"):
            return f"{base}{suffix}"
        return f"{base}/{suffix}"

    @staticmethod
    def _merge_model_configs(
        *,
        base_models: List[ChatModelConfig],
        override_models: List[ChatModelConfig],
    ) -> List[ChatModelConfig]:
        merged_models = list(base_models)
        index_by_key: dict[str, int] = {}
        for idx, model in enumerate(merged_models):
            index_by_key[model.id] = idx
            index_by_key[model.name] = idx

        for override in override_models:
            match_index = None
            if override.id in index_by_key:
                match_index = index_by_key[override.id]
            elif override.name in index_by_key:
                match_index = index_by_key[override.name]

            if match_index is None:
                merged_models.append(override)
                index_by_key[override.id] = len(merged_models) - 1
                index_by_key[override.name] = len(merged_models) - 1
                continue

            base_model = merged_models[match_index]
            merged_payload = cast(
                dict[str, Any],
                ConfigReader._deep_merge(
                    base=base_model.model_dump(),
                    override=override.model_dump(exclude_none=True, exclude_unset=True),
                ),
            )
            merged_models[match_index] = ChatModelConfig(**merged_payload)

        merged_models.sort(key=lambda x: x.name)
        return merged_models

    @staticmethod
    def _deep_merge(*, base: object, override: object) -> object:
        if isinstance(base, dict) and isinstance(override, dict):
            merged = dict(base)
            for key, value in override.items():
                if (
                    key in merged
                    and isinstance(merged[key], dict)
                    and isinstance(value, dict)
                ):
                    merged[key] = ConfigReader._deep_merge(
                        base=merged[key], override=value
                    )
                else:
                    merged[key] = value
            return merged
        if isinstance(base, list) and isinstance(override, list):
            return ConfigReader._merge_list_of_dicts(base=base, override=override)
        return override

    @staticmethod
    def _merge_list_of_dicts(
        *, base: list[object], override: list[object]
    ) -> list[object]:
        if not base:
            return list(override)
        if not override:
            return list(base)

        key_field = None
        if all(isinstance(item, dict) and "name" in item for item in base + override):
            key_field = "name"
        elif all(isinstance(item, dict) and "key" in item for item in base + override):
            key_field = "key"

        if not key_field:
            return list(override)

        merged_list = list(base)
        index_by_key: dict[str, int] = {}
        for idx, item in enumerate(merged_list):
            if isinstance(item, dict):
                index_by_key[str(item[key_field])] = idx

        for override_item in override:
            if not isinstance(override_item, dict):
                merged_list.append(override_item)
                continue
            override_key = str(override_item[key_field])
            if override_key in index_by_key:
                base_item = merged_list[index_by_key[override_key]]
                if isinstance(base_item, dict):
                    merged_list[index_by_key[override_key]] = ConfigReader._deep_merge(
                        base=base_item, override=override_item
                    )
                else:
                    merged_list[index_by_key[override_key]] = override_item
            else:
                index_by_key[override_key] = len(merged_list)
                merged_list.append(override_item)

        return merged_list

    async def _resolve_prompt_library_async(
        self, models: List[ChatModelConfig], *, config_path: str | None = None
    ) -> None:
        # Auto-discover prompts/ folder if no explicit path is configured
        if not self._prompt_library_manager.resolved_path and config_path:
            discovered = await self._discover_prompts_path(config_path)
            if discovered:
                self._prompt_library_manager.resolved_path = discovered

        for model in models:
            await self._resolve_prompt_list_async(model.system_prompts)
            await self._resolve_prompt_list_async(model.example_prompts)

    async def _discover_prompts_path(self, config_path: str) -> str | None:
        """Discover the prompts folder from config_path, supporting GitHub paths."""
        if GitHubDirectoryHelper.is_github_path(config_path):
            from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
                PROMPTS_FOLDER_NAME,
            )

            prompts_uri = GitHubDirectoryHelper.join_github_uri_path(
                base_uri=GitHubDirectoryHelper.to_github_uri(config_path),
                suffix=PROMPTS_FOLDER_NAME,
            )
            try:
                local_path = await self._github_directory_helper.resolve_github_path(
                    prompts_uri, log_diagnostics_on_not_found=False
                )
                if local_path is None:
                    return None
                logger.info("Downloaded prompts from %s to %s", prompts_uri, local_path)
                return str(local_path)
            except Exception as e:
                logger.debug("No prompts folder found at %s: %s", prompts_uri, e)
                return None
        return FileConfigReader.discover_prompts_path(config_path)

    async def _resolve_prompt_list_async(
        self, prompts: List[PromptConfig] | None
    ) -> None:
        if not prompts:
            return
        for prompt in prompts:
            if not prompt.name:
                continue
            try:
                prompt.content = await self._prompt_library_manager.get_prompt_async(
                    prompt.name
                )
            except FileNotFoundError as exc:
                raise ValueError(
                    f"Prompt not found in prompt library: {prompt.name}"
                ) from exc

    async def clear_cache(self) -> None:
        if self._model_config_cache_store:
            prefix = f"v{self.SCHEMA_VERSION}:"
            model_keys = await self._get_cache_keys_by_prefix(prefix=prefix)
            if model_keys:
                await self._model_config_cache_store.delete_many(
                    model_keys,
                    collection=self._model_config_cache_collection,
                )
        logger.info(
            "ConfigReader with id: %s cleared model config cache",
            self._identifier,
        )
