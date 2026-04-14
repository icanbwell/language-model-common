import asyncio
import logging
import os

from pathlib import Path
from typing import Any, List, Optional, cast
from uuid import UUID, uuid4

from languagemodelcommon.configs.config_reader.file_config_reader import (
    FileConfigReader,
)
from languagemodelcommon.configs.config_reader.github_directory_helper import (
    is_github_path,
    join_github_uri_path,
    resolve_github_path,
    to_github_uri,
)
from languagemodelcommon.configs.config_reader.s3_config_reader import S3ConfigReader
from languagemodelcommon.configs.schemas.config_schema import (
    ChatModelConfig,
    PromptConfig,
)
from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
    PromptLibraryManager,
)
from languagemodelcommon.utilities.cache.config_expiring_cache import (
    ConfigExpiringCache,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)


class ConfigReader:
    _identifier: UUID = uuid4()
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(
        self,
        *,
        cache: ConfigExpiringCache,
        prompt_library_manager: PromptLibraryManager,
    ) -> None:
        """
        Initialize the async config reader
        Args:
            cache: Expiring cache for model configurations
        """
        if cache is None:
            raise ValueError("cache must not be None")
        self._cache: ConfigExpiringCache = cache
        if self._cache is None:
            raise ValueError("self._cache must not be None")
        self._prompt_library_manager = prompt_library_manager

    async def read_model_configs_async(
        self, *, client_id: str | None = None
    ) -> List[ChatModelConfig]:
        config_path: str = os.environ.get("MODELS_OFFICIAL_PATH", "")
        if config_path is None or config_path == "":
            raise ValueError("MODELS_OFFICIAL_PATH environment variable is not set")

        base_models = await self._read_base_models_async(
            config_path=config_path,
            models_testing_path=os.environ.get("MODELS_TESTING_PATH"),
        )

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

            default_config_path = self._resolve_default_config_path(config_path)
            logger.info(
                "ConfigReader with id: %s reading model configurations from %s",
                self._identifier,
                default_config_path,
            )

            try:
                base_exclude_dirs = {"clients", "env"}
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
                logger.exception(
                    "Error reading model configurations: %s",
                    e,
                )
                models = []

            if not models:
                logger.warning(
                    "ConfigReader with id: %s read 0 model configurations "
                    "from %s — not caching empty result",
                    self._identifier,
                    default_config_path,
                )
                return models

            await self._cache.set(models)
            return models

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
        if config_path.startswith("s3"):
            models = await S3ConfigReader().read_model_configs(s3_url=config_path)
            logger.info(
                "ConfigReader with id: %s loaded %s model configurations from S3",
                self._identifier,
                len(models),
            )
        elif is_github_path(config_path):
            local_path = resolve_github_path(config_path)
            models = FileConfigReader().read_model_configs(
                config_path=str(local_path), exclude_dirs=exclude_dirs
            )
            logger.info(
                "ConfigReader with id: %s loaded %s model configurations from GitHub",
                self._identifier,
                len(models),
            )
        else:
            models = FileConfigReader().read_model_configs(
                config_path=config_path, exclude_dirs=exclude_dirs
            )
            logger.info(
                "ConfigReader with id: %s loaded %s model configurations from file system",
                self._identifier,
                len(models),
            )
        return models

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
        if is_github_path(config_path):
            return join_github_uri_path(
                to_github_uri(config_path), f"clients/{client_id}"
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
        if is_github_path(config_path):
            from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
                PROMPTS_FOLDER_NAME,
            )

            prompts_uri = join_github_uri_path(
                to_github_uri(config_path), PROMPTS_FOLDER_NAME
            )
            try:
                local_path = resolve_github_path(prompts_uri)
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
        logger.info("ConfigReader with id: %s cleared cache", self._identifier)
