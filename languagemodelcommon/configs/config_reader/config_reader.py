import asyncio
import logging

from pathlib import Path
from typing import Any, List, Optional, cast
from uuid import UUID, uuid4

from languagemodelcommon.configs.config_reader.file_config_reader import (
    FileConfigReader,
)
from languagemodelcommon.configs.config_reader.github_directory_helper import (
    GitHubDirectoryHelper,
)
from languagemodelcommon.configs.config_reader.s3_config_reader import S3ConfigReader
from languagemodelcommon.configs.schemas.config_schema import (
    ChatModelConfig,
    PromptConfig,
)
from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
    PromptLibraryManager,
)
from key_value.aio.stores.base import BaseStore

from languagemodelcommon.utilities.cache.config_expiring_cache import (
    ConfigExpiringCache,
)
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS
from languagemodelcommon.configs.config_reader.mcp_json_fetcher import McpJsonFetcher
from languagemodelcommon.configs.config_reader.mcp_json_reader import (
    resolve_mcp_servers_from_plugins,
)

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)


class ConfigReader:
    def __init__(
        self,
        *,
        cache: ConfigExpiringCache,
        prompt_library_manager: PromptLibraryManager,
        environment_variables: LanguageModelCommonEnvironmentVariables | None = None,
        mcp_json_fetcher: McpJsonFetcher | None = None,
        github_directory_helper: GitHubDirectoryHelper | None = None,
        snapshot_cache_store: BaseStore | None = None,
    ) -> None:
        self._identifier: UUID = uuid4()
        self._lock: asyncio.Lock = asyncio.Lock()
        if cache is None:
            raise ValueError("cache must not be None")
        self._cache: ConfigExpiringCache = cache
        if self._cache is None:
            raise ValueError("self._cache must not be None")
        self._prompt_library_manager = prompt_library_manager
        self._environment_variables = (
            environment_variables or LanguageModelCommonEnvironmentVariables()
        )
        self._github_directory_helper = (
            github_directory_helper
            or GitHubDirectoryHelper(environment_variables=self._environment_variables)
        )
        self._mcp_json_fetcher = mcp_json_fetcher
        self._snapshot_cache_store = snapshot_cache_store
        self._snapshot_cache_collection = (
            self._environment_variables.snapshot_cache_model_configs_collection
        )

    async def read_model_configs_async(
        self, *, client_id: str | None = None
    ) -> List[ChatModelConfig]:
        config_path = self._environment_variables.models_official_path
        models_testing_path = self._environment_variables.models_testing_path

        base_models = await self._read_base_models_async(
            config_path=config_path,
            models_testing_path=models_testing_path,
        )

        # Retry MCP resolution for cached models that have unresolved servers
        # (e.g. the MCP server was unavailable during initial config load)
        if self._has_unresolved_mcp_servers(base_models):
            await self._retry_mcp_resolution(base_models, config_path)

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
        self._resolve_prompt_library(base_models, config_path=config_path)
        return base_models

    async def _read_base_models_async(
        self,
        *,
        config_path: str,
        models_testing_path: Optional[str],
    ) -> List[ChatModelConfig]:
        cached_configs: List[ChatModelConfig] | None = await self._cache.get()
        if cached_configs is not None:
            logger.debug(
                "ConfigReader with id: %s using cached model configurations",
                self._identifier,
            )
            return cached_configs
        logger.info("ConfigReader with id: %s cache is empty", self._identifier)

        async with self._lock:
            cached_configs = await self._cache.get()
            if cached_configs is not None:
                logger.debug(
                    "ConfigReader with id: %s using cached model configurations",
                    self._identifier,
                )
                return cached_configs

            # Check MongoDB snapshot cache before hitting filesystem/GitHub
            models = await self._read_from_snapshot_cache()
            if models:
                await self._cache.set(models)
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
                stale = await self._cache.get_stale()
                if stale:
                    logger.warning(
                        "ConfigReader with id: %s read 0 model configurations "
                        "from %s — returning %d stale cached configs",
                        self._identifier,
                        default_config_path,
                        len(stale),
                    )
                    return stale
                logger.warning(
                    "ConfigReader with id: %s read 0 model configurations "
                    "from %s and no stale cache available",
                    self._identifier,
                    default_config_path,
                )
                return models

            await self._cache.set(models)
            await self._write_to_snapshot_cache(models)
            return models

    _SNAPSHOT_CACHE_KEY = "model_configs"

    async def _read_from_snapshot_cache(self) -> List[ChatModelConfig] | None:
        """Load model configs from the snapshot cache.

        Returns ``None`` when the cache has no entry for the key.
        Raises on store or deserialization errors so that a misconfigured
        cache backend surfaces immediately (fail-fast).
        """
        if not self._snapshot_cache_store:
            return None
        data = await self._snapshot_cache_store.get(
            self._SNAPSHOT_CACHE_KEY,
            collection=self._snapshot_cache_collection,
        )
        if data is None:
            return None
        models_data: list[dict[str, Any]] = data.get("models", [])
        models = [ChatModelConfig.model_validate(d) for d in models_data]
        logger.info(
            "ConfigReader with id: %s loaded %d configs from snapshot cache",
            self._identifier,
            len(models),
        )
        return models if models else None

    async def _write_to_snapshot_cache(self, models: List[ChatModelConfig]) -> None:
        """Store parsed model configs in the snapshot cache.

        Raises on write errors so that a misconfigured cache backend
        surfaces immediately (fail-fast).
        """
        if not self._snapshot_cache_store:
            return
        data = {"models": [m.model_dump() for m in models]}
        ttl = self._environment_variables.snapshot_cache_ttl_seconds
        await self._snapshot_cache_store.put(
            self._SNAPSHOT_CACHE_KEY,
            data,
            ttl=ttl,
            collection=self._snapshot_cache_collection,
        )
        logger.info(
            "ConfigReader with id: %s wrote %d configs to snapshot cache",
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
                    default_config_path, exclude_dirs=base_exclude_dirs
                )
                if not models and default_config_path != config_path:
                    models = await self.read_models_from_path_async(
                        config_path, exclude_dirs=base_exclude_dirs
                    )
                if models_testing_path:
                    models_testing = await self.read_models_from_path_async(
                        models_testing_path, exclude_dirs=base_exclude_dirs
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
        try:
            return await self.read_models_from_path_async(override_path)
        except Exception as e:
            logger.warning(
                "Failed to load client overrides from %s: %s", override_path, e
            )
            return []

    async def read_models_from_path_async(
        self, config_path: str, *, exclude_dirs: set[str] | None = None
    ) -> List[ChatModelConfig]:
        models: List[ChatModelConfig]
        local_config_path: str = config_path
        if config_path.startswith("s3"):
            models = await S3ConfigReader().read_model_configs(s3_url=config_path)
            logger.info(
                "ConfigReader with id: %s loaded %s model configurations from S3",
                self._identifier,
                len(models),
            )
        elif GitHubDirectoryHelper.is_github_path(config_path):
            resolved = self._github_directory_helper.resolve_github_path(config_path)
            local_config_path = str(resolved)
            models = FileConfigReader().read_model_configs(
                config_path=local_config_path,
                exclude_dirs=exclude_dirs,
            )
            logger.info(
                "ConfigReader with id: %s loaded %s model configurations from GitHub",
                self._identifier,
                len(models),
            )
        else:
            models = FileConfigReader().read_model_configs(
                config_path=config_path,
                exclude_dirs=exclude_dirs,
            )
            logger.info(
                "ConfigReader with id: %s loaded %s model configurations from file system",
                self._identifier,
                len(models),
            )

        # Resolve MCP server references from plugins or local .mcp.json
        await self._resolve_mcp_servers_async(models, local_config_path)
        return models

    async def _resolve_mcp_servers_async(
        self,
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
            logger.warning(
                "Models have mcp_server references but no plugins declared "
                "— MCP servers will not be resolved"
            )
            return

        plugin_configs = await self._mcp_json_fetcher.fetch_plugins_async(
            unique_plugins
        )
        if plugin_configs:
            resolve_mcp_servers_from_plugins(models, plugin_configs)

    @staticmethod
    def _has_unresolved_mcp_servers(models: List[ChatModelConfig]) -> bool:
        """Check whether any model has agents with ``mcp_server`` set but no ``url``."""
        for model in models:
            for agent in model.get_agents():
                if agent.mcp_server and not agent.url:
                    return True
        return False

    async def _retry_mcp_resolution(
        self, models: List[ChatModelConfig], config_path: str
    ) -> None:
        """Re-attempt MCP server resolution for models with unresolved refs.

        Called when cached configs still carry ``mcp_server`` references
        without a resolved ``url`` — typically because the MCP server was
        unreachable during the initial config load.  On success the
        in-memory and snapshot caches are updated so subsequent requests
        use the resolved configs.
        """
        logger.info("Retrying MCP server resolution for models with unresolved refs")
        await self._resolve_mcp_servers_async(models, config_path)
        if not self._has_unresolved_mcp_servers(models):
            logger.info("MCP server resolution retry succeeded — updating caches")
            await self._cache.set(models)
            await self._write_to_snapshot_cache(models)
        else:
            logger.warning("MCP server resolution retry did not resolve all refs")

    @staticmethod
    def _resolve_default_config_path(config_path: str) -> str:
        return config_path

    @staticmethod
    def _resolve_override_config_path(
        *, config_path: str, client_id: str
    ) -> str | None:
        if not client_id:
            return None
        # Validate client_id to prevent path traversal
        if not ConfigReader._is_valid_client_id(client_id):
            logger.warning("Invalid client_id format: %s", client_id)
            return None
        if GitHubDirectoryHelper.is_github_path(config_path):
            return GitHubDirectoryHelper.join_github_uri_path(
                GitHubDirectoryHelper.to_github_uri(config_path), f"clients/{client_id}"
            )
        if config_path.startswith("s3"):
            return ConfigReader._join_path(config_path, f"clients/{client_id}")
        config_folder = Path(config_path)
        override_folder = config_folder.joinpath("clients", client_id)
        # Ensure the resolved path is within the config directory
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
        """Validate that client_id contains only safe characters."""
        import re

        return bool(re.match(r"^[a-zA-Z0-9_-]+$", client_id))

    @staticmethod
    def _join_path(base: str, suffix: str) -> str:
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
                    base_model.model_dump(),
                    override.model_dump(exclude_none=True, exclude_unset=True),
                ),
            )
            merged_models[match_index] = ChatModelConfig(**merged_payload)

        merged_models.sort(key=lambda x: x.name)
        return merged_models

    @staticmethod
    def _deep_merge(base: object, override: object) -> object:
        if isinstance(base, dict) and isinstance(override, dict):
            merged = dict(base)
            for key, value in override.items():
                if (
                    key in merged
                    and isinstance(merged[key], dict)
                    and isinstance(value, dict)
                ):
                    merged[key] = ConfigReader._deep_merge(merged[key], value)
                else:
                    merged[key] = value
            return merged
        if isinstance(base, list) and isinstance(override, list):
            return ConfigReader._merge_list_of_dicts(base, override)
        return override

    @staticmethod
    def _merge_list_of_dicts(
        base: list[object], override: list[object]
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
                        base_item, override_item
                    )
                else:
                    merged_list[index_by_key[override_key]] = override_item
            else:
                index_by_key[override_key] = len(merged_list)
                merged_list.append(override_item)

        return merged_list

    def _resolve_prompt_library(
        self, models: List[ChatModelConfig], *, config_path: str | None = None
    ) -> None:
        # Auto-discover prompts/ folder if no explicit path is configured
        if not self._prompt_library_manager.resolved_path and config_path:
            discovered = self._discover_prompts_path(config_path)
            if discovered:
                self._prompt_library_manager.resolved_path = discovered

        for model in models:
            self._resolve_prompt_list(model.system_prompts)
            self._resolve_prompt_list(model.example_prompts)

    def _discover_prompts_path(self, config_path: str) -> str | None:
        """Discover the prompts folder from config_path, supporting GitHub paths."""
        if GitHubDirectoryHelper.is_github_path(config_path):
            from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
                PROMPTS_FOLDER_NAME,
            )

            prompts_uri = GitHubDirectoryHelper.join_github_uri_path(
                GitHubDirectoryHelper.to_github_uri(config_path), PROMPTS_FOLDER_NAME
            )
            try:
                local_path = self._github_directory_helper.resolve_github_path(
                    prompts_uri
                )
                logger.info("Downloaded prompts from %s to %s", prompts_uri, local_path)
                return str(local_path)
            except Exception as e:
                logger.debug("No prompts folder found at %s: %s", prompts_uri, e)
                return None
        return FileConfigReader.discover_prompts_path(config_path)

    def _resolve_prompt_list(self, prompts: List[PromptConfig] | None) -> None:
        if not prompts:
            return
        for prompt in prompts:
            if not prompt.name:
                continue
            try:
                prompt.content = self._prompt_library_manager.get_prompt(prompt.name)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"Prompt not found in prompt library: {prompt.name}"
                ) from exc

    async def clear_cache(self) -> None:
        await self._cache.clear()
        if self._snapshot_cache_store:
            await self._snapshot_cache_store.delete(
                self._SNAPSHOT_CACHE_KEY,
                collection=self._snapshot_cache_collection,
            )
        logger.info(
            "ConfigReader with id: %s cleared in-memory and snapshot caches",
            self._identifier,
        )
