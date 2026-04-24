import json
import logging

from pathlib import Path
from typing import List

from languagemodelcommon.configs.config_reader.mcp_json_reader import (
    McpJsonReader,
    resolve_mcp_servers,
)
from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
    PROMPTS_FOLDER_NAME,
)
from languagemodelcommon.configs.schemas.config_schema import ChatModelConfig
from languagemodelcommon.utilities.config_substitution import substitute_env_vars
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)


class FileConfigReader:
    def __init__(
        self,
        *,
        mcp_json_reader: McpJsonReader | None = None,
    ) -> None:
        self._mcp_json_reader = mcp_json_reader or McpJsonReader()

    # noinspection PyMethodMayBeStatic
    def read_model_configs(
        self,
        *,
        config_path: str,
        exclude_dirs: set[str] | None = None,
    ) -> List[ChatModelConfig]:
        return self._read_model_configs(config_path, exclude_dirs)

    # noinspection PyMethodMayBeStatic
    def _read_model_configs(
        self,
        config_path: str,
        exclude_dirs: set[str] | None = None,
    ) -> List[ChatModelConfig]:
        logger.info("Reading model configurations from %s", config_path)
        config_folder: Path = Path(config_path)
        excluded = exclude_dirs or set()
        # read all the .json files recursively in the config folder
        # for each file, parse the json data into ModelConfig
        configs: List[ChatModelConfig] = []
        # Read all the .json files recursively in the config folder
        for json_file in config_folder.rglob("*.json"):
            if json_file.name == ".mcp.json":
                continue
            if excluded and excluded.intersection(
                json_file.relative_to(config_folder).parts
            ):
                continue
            with open(json_file, "r") as file:
                data = substitute_env_vars(json.load(file))
                configs.append(ChatModelConfig(**data))
        # Resolve mcp_server references from .mcp.json in marketplace plugins
        mcp_config = self._mcp_json_reader.read_mcp_json()
        if mcp_config:
            resolve_mcp_servers(configs, mcp_config)

        # sort the configs by name
        configs.sort(key=lambda x: x.name)
        return configs

    @staticmethod
    def discover_prompts_path(config_path: str) -> str | None:
        """Look for a ``prompts/`` folder alongside the config directory.

        Returns the path as a string if found, ``None`` otherwise.
        """
        prompts_dir = Path(config_path) / PROMPTS_FOLDER_NAME
        if prompts_dir.is_dir():
            logger.info("Discovered prompts folder at %s", prompts_dir)
            return str(prompts_dir)
        return None
