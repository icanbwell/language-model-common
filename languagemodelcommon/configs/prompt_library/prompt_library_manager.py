from __future__ import annotations

import logging
import re
from pathlib import Path

from languagemodelcommon.configs.prompt_library.prompt_library_environment_variables import (
    PromptLibraryEnvironmentVariables,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)

PROMPTS_FOLDER_NAME = "prompts"

_SUPPORTED_EXTENSIONS = (".md", ".txt")


class PromptLibraryManager:
    _VALID_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

    def __init__(
        self,
        *,
        environment_variables: PromptLibraryEnvironmentVariables,
    ) -> None:
        if not isinstance(environment_variables, PromptLibraryEnvironmentVariables):
            raise TypeError(
                "environment_variables must implement PromptLibraryEnvironmentVariables"
            )

        self._base_path = environment_variables.prompt_library_path
        self._resolved_path: str | None = None
        self._github_resolved: bool = False

    @property
    def resolved_path(self) -> str | None:
        """The effective prompts path after auto-discovery or override."""
        return self._resolved_path or self._base_path

    @resolved_path.setter
    def resolved_path(self, value: str | None) -> None:
        self._resolved_path = value

    def _ensure_local_path(self) -> Path:
        """Return a local filesystem Path, downloading from GitHub if needed."""
        effective_path = self.resolved_path
        if effective_path is None or not str(effective_path).strip():
            raise ValueError("Prompt library path is not configured")

        if not self._github_resolved:
            from languagemodelcommon.configs.config_reader.github_directory_helper import (
                is_github_path,
                resolve_github_path,
            )

            if is_github_path(effective_path):
                local_path = resolve_github_path(effective_path)
                self._resolved_path = str(local_path)
                effective_path = self._resolved_path
            self._github_resolved = True

        return Path(str(effective_path)).expanduser()

    def get_prompt(self, name: str) -> str:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Prompt name must not be empty")
        if not self._VALID_NAME_PATTERN.match(normalized_name):
            raise ValueError(f"Invalid prompt name: {normalized_name}")

        base_path = self._ensure_local_path()

        # If the name already has an extension, try that file directly
        if any(normalized_name.endswith(ext) for ext in _SUPPORTED_EXTENSIONS):
            prompt_path = base_path / normalized_name
            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8")
            raise FileNotFoundError(f"Prompt not found: {normalized_name}")

        # Try each supported extension in order
        for ext in _SUPPORTED_EXTENSIONS:
            prompt_path = base_path / f"{normalized_name}{ext}"
            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8")

        raise FileNotFoundError(f"Prompt not found: {normalized_name}")
