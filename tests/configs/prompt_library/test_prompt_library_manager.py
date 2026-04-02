import pytest
from pathlib import Path

from languagemodelcommon.configs.prompt_library.prompt_library_environment_variables import (
    PromptLibraryEnvironmentVariables,
)
from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
    PromptLibraryManager,
)


class _StubPromptLibraryEnv(PromptLibraryEnvironmentVariables):
    def __init__(self, prompt_library_path: str) -> None:
        self._prompt_library_path = prompt_library_path

    @property
    def prompt_library_path(self) -> str | None:
        return self._prompt_library_path


def test_get_prompt_reads_text(tmp_path: Path) -> None:
    prompt_path = tmp_path / "example_prompt.txt"
    prompt_path.write_text("Hello from prompt library.", encoding="utf-8")

    manager = PromptLibraryManager(
        environment_variables=_StubPromptLibraryEnv(str(tmp_path))
    )
    assert manager.get_prompt("example_prompt") == "Hello from prompt library."


def test_get_prompt_missing_raises(tmp_path: Path) -> None:
    manager = PromptLibraryManager(
        environment_variables=_StubPromptLibraryEnv(str(tmp_path))
    )
    with pytest.raises(FileNotFoundError):
        manager.get_prompt("missing_prompt")


def test_get_prompt_invalid_name_raises(tmp_path: Path) -> None:
    manager = PromptLibraryManager(
        environment_variables=_StubPromptLibraryEnv(str(tmp_path))
    )
    with pytest.raises(ValueError):
        manager.get_prompt("../escape")
