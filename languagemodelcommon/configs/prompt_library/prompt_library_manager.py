from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from languagemodelcommon.configs.config_reader.github_directory_helper import (
    GitHubDirectoryHelper,
)
from languagemodelcommon.configs.prompt_library.prompt_library_environment_variables import (
    PromptLibraryEnvironmentVariables,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

if TYPE_CHECKING:
    from languagemodelcommon.configs.prompt_library.prompt_store import PromptStore

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
        github_directory_helper: GitHubDirectoryHelper | None = None,
        prompt_store: PromptStore | None = None,
    ) -> None:
        if not isinstance(environment_variables, PromptLibraryEnvironmentVariables):
            raise TypeError(
                "environment_variables must implement PromptLibraryEnvironmentVariables"
            )

        self._base_path = environment_variables.prompt_library_path
        self._github_directory_helper = github_directory_helper
        self._prompt_store = prompt_store
        self._resolved_path: str | None = None
        self._github_resolved: bool = False

    @property
    def resolved_path(self) -> str | None:
        """The effective prompts path after auto-discovery or override."""
        return self._resolved_path or self._base_path

    @resolved_path.setter
    def resolved_path(self, value: str | None) -> None:
        self._resolved_path = value
        self._github_resolved = False

    def _ensure_local_path(self) -> Path:
        """Return a local filesystem Path, downloading from GitHub if needed."""
        effective_path = self.resolved_path
        if effective_path is None or not str(effective_path).strip():
            raise ValueError("Prompt library path is not configured")

        if not self._github_resolved:
            if GitHubDirectoryHelper.is_github_path(effective_path):
                if self._github_directory_helper is None:
                    raise RuntimeError(
                        "GitHubDirectoryHelper is required to resolve GitHub paths"
                    )
                local_path = self._github_directory_helper.resolve_github_path(
                    effective_path
                )
                self._resolved_path = str(local_path)
                effective_path = self._resolved_path
            self._github_resolved = True

        return Path(str(effective_path)).expanduser()

    async def get_prompt_async(self, name: str) -> str:
        """Resolve a prompt by name, checking the store first then filesystem."""
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Prompt name must not be empty")
        if not self._VALID_NAME_PATTERN.match(normalized_name):
            raise ValueError(f"Invalid prompt name: {normalized_name}")

        if self._prompt_store is not None:
            content = await self._prompt_store.get_prompt(name=normalized_name)
            if content is not None:
                return content

        return self._get_prompt_from_filesystem(normalized_name=normalized_name)

    def get_prompt(self, name: str) -> str:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Prompt name must not be empty")
        if not self._VALID_NAME_PATTERN.match(normalized_name):
            raise ValueError(f"Invalid prompt name: {normalized_name}")

        return self._get_prompt_from_filesystem(normalized_name=normalized_name)

    def _get_prompt_from_filesystem(self, *, normalized_name: str) -> str:
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

    async def seed_store_from_filesystem(self) -> int:
        """Load all prompts from the configured path into the store.

        Returns the number of prompts seeded. Skips if no store is configured.
        """
        if self._prompt_store is None:
            return 0

        base_path = self._ensure_local_path()
        count = 0
        for ext in _SUPPORTED_EXTENSIONS:
            for prompt_path in base_path.glob(f"*{ext}"):
                prompt_name = prompt_path.stem
                content = prompt_path.read_text(encoding="utf-8")
                await self._prompt_store.put_prompt(name=prompt_name, content=content)
                count += 1
                logger.info("Seeded prompt '%s' into store", prompt_name)
        return count
