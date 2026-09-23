from unittest.mock import AsyncMock, MagicMock

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


@pytest.mark.asyncio
async def test_two_source_refs_do_not_collide_on_same_prompt_name(
    memory_store: MemoryStore,
) -> None:
    """BAI-933: without ref-hash scoping, two PromptStore instances pointed
    at different source paths (e.g. two different `?ref=` versions, or two
    unrelated prompt libraries) but sharing the same underlying store and
    collection would silently read/overwrite each other's content for the
    same prompt name."""
    store_a = PromptStore(
        store=memory_store, collection="prompts", source_ref="github://org/repo-a"
    )
    store_b = PromptStore(
        store=memory_store, collection="prompts", source_ref="github://org/repo-b"
    )

    await store_a.put_prompt(name="greeting", content="from repo-a")
    await store_b.put_prompt(name="greeting", content="from repo-b")

    assert await store_a.get_prompt(name="greeting") == "from repo-a"
    assert await store_b.get_prompt(name="greeting") == "from repo-b"


@pytest.mark.asyncio
async def test_no_source_ref_preserves_bare_name_key_backward_compat(
    memory_store: MemoryStore,
) -> None:
    """Without a source_ref (existing callers), keys stay exactly the bare
    prompt name -- unchanged from before this fix, so existing deployments
    that haven't been updated to pass source_ref keep reading their
    already-cached content under the same key."""
    store = PromptStore(store=memory_store, collection="prompts")
    await store.put_prompt(name="greeting", content="hello")

    raw = await memory_store.get("greeting", collection="prompts")
    assert raw is not None
    assert raw["content"] == "hello"


@pytest.mark.asyncio
async def test_put_prompt_passes_ttl_to_underlying_store() -> None:
    """BAI-933: prompt entries must carry a TTL so a new version of a
    prompt pushed upstream is eventually picked up automatically, instead
    of being cached forever until a manual clear()."""
    fake_store = MagicMock()
    fake_store.put = AsyncMock()
    prompt_store = PromptStore(store=fake_store, collection="prompts", ttl_seconds=3600)

    await prompt_store.put_prompt(name="greeting", content="hello")

    fake_store.put.assert_awaited_once()
    _, kwargs = fake_store.put.call_args
    assert kwargs["ttl"] == 3600


@pytest.mark.asyncio
async def test_default_ttl_is_not_none(memory_store: MemoryStore) -> None:
    """A PromptStore constructed with no explicit ttl_seconds must still
    default to a real TTL, not cache forever -- permanent caching was the
    root problem this fix addresses."""
    store = PromptStore(store=memory_store, collection="prompts")
    assert store._ttl_seconds is not None
    assert store._ttl_seconds > 0


@pytest.mark.parametrize("non_positive_ttl", [0, -1, -3600])
@pytest.mark.asyncio
async def test_non_positive_ttl_is_sanitized_to_no_expiry(
    non_positive_ttl: int,
) -> None:
    """PR #108 review finding: PROMPT_STORE_TTL_SECONDS=0 or negative (a
    natural way for an operator to try to disable caching) must not crash
    every prompt write. py-key-value-aio's BaseStore.put raises
    InvalidTTLError for ttl <= 0 -- the sibling McpToolListStore already
    guards the identical put(..., ttl=...) call shape
    (ttl_seconds if ttl_seconds and ttl_seconds > 0 else None); PromptStore
    must apply the same sanitization rather than forwarding a non-positive
    value straight through."""
    fake_store = MagicMock()
    fake_store.put = AsyncMock()
    prompt_store = PromptStore(
        store=fake_store, collection="prompts", ttl_seconds=non_positive_ttl
    )

    await prompt_store.put_prompt(name="greeting", content="hello")

    fake_store.put.assert_awaited_once()
    _, kwargs = fake_store.put.call_args
    assert kwargs["ttl"] is None
