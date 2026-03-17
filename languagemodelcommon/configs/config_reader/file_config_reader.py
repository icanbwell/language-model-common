import json

from pathlib import Path
from typing import List

from languagemodelcommon.configs.schemas.config_schema import ChatModelConfig
from languagemodelcommon.utilities.config_substitution import substitute_env_vars
from languagemodelcommon.utilities.logger.log_levels import logger


class FileConfigReader:
    # noinspection PyMethodMayBeStatic
    def read_model_configs(
        self, *, config_path: str, exclude_dirs: set[str] | None = None
    ) -> List[ChatModelConfig]:
        return self._read_model_configs(config_path, exclude_dirs)

    # noinspection PyMethodMayBeStatic
    def _read_model_configs(
        self, config_path: str, exclude_dirs: set[str] | None = None
    ) -> List[ChatModelConfig]:
        logger.info(f"Reading model configurations from {config_path}")
        config_folder: Path = Path(config_path)
        excluded = exclude_dirs or set()
        # read all the .json files recursively in the config folder
        # for each file, parse the json data into ModelConfig
        configs: List[ChatModelConfig] = []
        # Read all the .json files recursively in the config folder
        for json_file in config_folder.rglob("*.json"):
            if excluded and excluded.intersection(
                json_file.relative_to(config_folder).parts
            ):
                continue
            with open(json_file, "r") as file:
                data = substitute_env_vars(json.load(file))
                configs.append(ChatModelConfig(**data))
        # sort the configs by name
        configs.sort(key=lambda x: x.name)
        return configs
