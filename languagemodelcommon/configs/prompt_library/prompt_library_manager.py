from __future__ import annotations

import re
from typing import Protocol, runtime_checkable
from pathlib import Path

from languagemodelcommon.utilities.environment.baileyai_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)


@runtime_checkable
class PromptLibraryEnvironmentVariables(Protocol):
    @property
    def prompt_library_path(self) -> str | None: ...


class PromptLibraryManager:
    _VALID_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

    def __init__(
        self,
        *,
        environment_variables: PromptLibraryEnvironmentVariables,
    ) -> None:
        if environment_variables is None:
            environment_variables = LanguageModelCommonEnvironmentVariables()
        if not isinstance(environment_variables, PromptLibraryEnvironmentVariables):
            raise TypeError(
                "environment_variables must implement PromptLibraryEnvironmentVariables"
            )

        self._base_path = environment_variables.prompt_library_path

    def get_prompt(self, name: str) -> str:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Prompt name must not be empty")
        if not self._VALID_NAME_PATTERN.match(normalized_name):
            raise ValueError(f"Invalid prompt name: {normalized_name}")

        file_name = (
            normalized_name
            if normalized_name.endswith(".txt")
            else f"{normalized_name}.txt"
        )
        if self._base_path is None or not str(self._base_path).strip():
            raise ValueError("Prompt library path is not configured")

        # expanduser is used to allow paths like ~/foo
        base_path = Path(str(self._base_path)).expanduser()
        prompt_path = base_path / file_name
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {normalized_name}")

        return prompt_path.read_text(encoding="utf-8")
