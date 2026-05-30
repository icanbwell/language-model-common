import pytest
from pathlib import Path

from key_value.aio.stores.memory import MemoryStore

from languagemodelcommon.configs.prompt_library.prompt_store import PromptStore
from languagemodelcommon.configs.prompt_library.prompt_library_environment_variables import (
    PromptLibraryEnvironmentVariables,
)
from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
    PromptLibraryManager,
)


class _StubEnv(PromptLibraryEnvironmentVariables):
    def __init__(self, *, prompt_library_path: str | None = None) -> None:
        self._prompt_library_path = prompt_library_path

    @property
    def prompt_library_path(self) -> str | None:
        return self._prompt_library_path


@pytest.fixture
def memory_store() -> MemoryStore:
    return MemoryStore(default_collection="prompts")


@pytest.fixture
def prompt_store(memory_store: MemoryStore) -> PromptStore:
    return PromptStore(store=memory_store, collection="prompts")


@pytest.mark.asyncio
async def test_put_and_get_prompt(prompt_store: PromptStore) -> None:
    await prompt_store.put_prompt(name="greeting", content="Hello, world!")
    result = await prompt_store.get_prompt(name="greeting")
    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_get_missing_prompt_returns_none(prompt_store: PromptStore) -> None:
    result = await prompt_store.get_prompt(name="nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_delete_prompt(prompt_store: PromptStore) -> None:
    await prompt_store.put_prompt(name="temp", content="temporary")
    await prompt_store.delete_prompt(name="temp")
    result = await prompt_store.get_prompt(name="temp")
    assert result is None


@pytest.mark.asyncio
async def test_get_prompt_async_uses_store(
    tmp_path: Path, prompt_store: PromptStore
) -> None:
    (tmp_path / "fallback.txt").write_text("from filesystem", encoding="utf-8")
    await prompt_store.put_prompt(name="fallback", content="from store")

    manager = PromptLibraryManager(
        environment_variables=_StubEnv(prompt_library_path=str(tmp_path)),
        prompt_store=prompt_store,
    )
    result = await manager.get_prompt_async("fallback")
    assert result == "from store"


@pytest.mark.asyncio
async def test_get_prompt_async_falls_back_to_filesystem(
    tmp_path: Path, prompt_store: PromptStore
) -> None:
    (tmp_path / "local_only.txt").write_text("from disk", encoding="utf-8")

    manager = PromptLibraryManager(
        environment_variables=_StubEnv(prompt_library_path=str(tmp_path)),
        prompt_store=prompt_store,
    )
    result = await manager.get_prompt_async("local_only")
    assert result == "from disk"


@pytest.mark.asyncio
async def test_get_prompt_async_without_store_uses_filesystem(
    tmp_path: Path,
) -> None:
    (tmp_path / "prompt.md").write_text("markdown content", encoding="utf-8")

    manager = PromptLibraryManager(
        environment_variables=_StubEnv(prompt_library_path=str(tmp_path)),
    )
    result = await manager.get_prompt_async("prompt")
    assert result == "markdown content"


@pytest.mark.asyncio
async def test_seed_store_from_filesystem(
    tmp_path: Path, prompt_store: PromptStore
) -> None:
    (tmp_path / "system_prompt.txt").write_text("system content", encoding="utf-8")
    (tmp_path / "skills.md").write_text("skills content", encoding="utf-8")

    manager = PromptLibraryManager(
        environment_variables=_StubEnv(prompt_library_path=str(tmp_path)),
        prompt_store=prompt_store,
    )
    count = await manager.seed_store_from_filesystem()
    assert count == 2

    assert await prompt_store.get_prompt(name="system_prompt") == "system content"
    assert await prompt_store.get_prompt(name="skills") == "skills content"


@pytest.mark.asyncio
async def test_seed_store_returns_zero_when_no_store(tmp_path: Path) -> None:
    (tmp_path / "prompt.txt").write_text("content", encoding="utf-8")

    manager = PromptLibraryManager(
        environment_variables=_StubEnv(prompt_library_path=str(tmp_path)),
    )
    count = await manager.seed_store_from_filesystem()
    assert count == 0
