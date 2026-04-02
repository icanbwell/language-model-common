import pytest
from pathlib import Path

from languagemodelcommon.configs.prompt_library.prompt_library_environment_variables import (
    PromptLibraryEnvironmentVariables,
)
from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
    PromptLibraryManager,
)


class _StubPromptLibraryEnv(PromptLibraryEnvironmentVariables):
    def __init__(self, prompt_library_path: str | None = None) -> None:
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


def test_get_prompt_reads_markdown(tmp_path: Path) -> None:
    prompt_path = tmp_path / "system_prompt.md"
    prompt_path.write_text("# System Prompt\nYou are helpful.", encoding="utf-8")

    manager = PromptLibraryManager(
        environment_variables=_StubPromptLibraryEnv(str(tmp_path))
    )
    assert manager.get_prompt("system_prompt") == "# System Prompt\nYou are helpful."


def test_md_takes_precedence_over_txt(tmp_path: Path) -> None:
    (tmp_path / "prompt.md").write_text("markdown version", encoding="utf-8")
    (tmp_path / "prompt.txt").write_text("text version", encoding="utf-8")

    manager = PromptLibraryManager(
        environment_variables=_StubPromptLibraryEnv(str(tmp_path))
    )
    assert manager.get_prompt("prompt") == "markdown version"


def test_get_prompt_with_explicit_extension(tmp_path: Path) -> None:
    (tmp_path / "prompt.md").write_text("explicit md", encoding="utf-8")

    manager = PromptLibraryManager(
        environment_variables=_StubPromptLibraryEnv(str(tmp_path))
    )
    assert manager.get_prompt("prompt.md") == "explicit md"


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


def test_resolved_path_overrides_env_path(tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "prompt.txt").write_text("from env", encoding="utf-8")

    override_dir = tmp_path / "override"
    override_dir.mkdir()
    (override_dir / "prompt.md").write_text("from override", encoding="utf-8")

    manager = PromptLibraryManager(
        environment_variables=_StubPromptLibraryEnv(str(env_dir))
    )
    manager.resolved_path = str(override_dir)
    assert manager.get_prompt("prompt") == "from override"


def test_resolved_path_falls_back_to_env(tmp_path: Path) -> None:
    (tmp_path / "prompt.txt").write_text("from env", encoding="utf-8")

    manager = PromptLibraryManager(
        environment_variables=_StubPromptLibraryEnv(str(tmp_path))
    )
    assert manager.resolved_path == str(tmp_path)
    assert manager.get_prompt("prompt") == "from env"
