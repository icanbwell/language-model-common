import os
import pytest
import tempfile
from typing import Any
from unittest.mock import AsyncMock, patch
from pathlib import Path

from languagemodelcommon.configs.config_reader.config_reader import ConfigReader
from languagemodelcommon.configs.prompt_library.prompt_library_environment_variables import (
    PromptLibraryEnvironmentVariables,
)
from languagemodelcommon.configs.schemas.config_schema import ChatModelConfig
from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
    PromptLibraryManager,
)


class _StubPromptLibraryEnv(PromptLibraryEnvironmentVariables):
    def __init__(self, prompt_library_path: str | None) -> None:
        self._prompt_library_path = prompt_library_path

    @property
    def prompt_library_path(self) -> str | None:
        return self._prompt_library_path


@pytest.fixture
def cache_mock() -> AsyncMock:
    mock = AsyncMock()
    mock.get.return_value = None
    return mock


@pytest.fixture
def prompt_library_manager(tmp_path: Path) -> PromptLibraryManager:
    return PromptLibraryManager(
        environment_variables=_StubPromptLibraryEnv(str(tmp_path))
    )


@pytest.fixture
def config_reader(
    cache_mock: AsyncMock, prompt_library_manager: PromptLibraryManager
) -> ConfigReader:
    return ConfigReader(cache=cache_mock, prompt_library_manager=prompt_library_manager)


@pytest.mark.asyncio
async def test_cache_hit(
    monkeypatch: Any,
    cache_mock: AsyncMock,
    prompt_library_manager: PromptLibraryManager,
) -> None:
    cache_mock.get.return_value = [ChatModelConfig(id="1", name="Test", description="")]
    os.environ["MODELS_OFFICIAL_PATH"] = tempfile.gettempdir()
    reader = ConfigReader(
        cache=cache_mock, prompt_library_manager=prompt_library_manager
    )
    result = await reader.read_model_configs_async()
    assert result[0].name == "Test"


@pytest.mark.asyncio
async def test_env_var_missing(
    cache_mock: AsyncMock, prompt_library_manager: PromptLibraryManager
) -> None:
    if "MODELS_OFFICIAL_PATH" in os.environ:
        del os.environ["MODELS_OFFICIAL_PATH"]
    reader = ConfigReader(
        cache=cache_mock, prompt_library_manager=prompt_library_manager
    )
    with pytest.raises(ValueError):
        await reader.read_model_configs_async()


@patch("languagemodelcommon.configs.config_reader.config_reader.FileConfigReader")
@pytest.mark.asyncio
async def test_read_from_file(
    FileConfigReaderMock: Any,
    cache_mock: AsyncMock,
    monkeypatch: Any,
    prompt_library_manager: PromptLibraryManager,
) -> None:
    os.environ["MODELS_OFFICIAL_PATH"] = tempfile.gettempdir()
    FileConfigReaderMock.return_value.read_model_configs.return_value = [
        ChatModelConfig(id="1", name="FileModel", description="")
    ]
    reader = ConfigReader(
        cache=cache_mock, prompt_library_manager=prompt_library_manager
    )
    result = await reader.read_model_configs_async()
    assert result[0].name == "FileModel"


@patch("languagemodelcommon.configs.config_reader.config_reader.S3ConfigReader")
@pytest.mark.asyncio
async def test_read_from_s3(
    S3ConfigReaderMock: Any,
    cache_mock: AsyncMock,
    monkeypatch: Any,
    prompt_library_manager: PromptLibraryManager,
) -> None:
    os.environ["MODELS_OFFICIAL_PATH"] = "s3://bucket/models"
    os.environ.pop("MODELS_TESTING_PATH", None)
    S3ConfigReaderMock.return_value.read_model_configs = AsyncMock(
        return_value=[ChatModelConfig(id="2", name="S3Model", description="")]
    )
    reader = ConfigReader(
        cache=cache_mock, prompt_library_manager=prompt_library_manager
    )
    result = await reader.read_model_configs_async()
    assert result[0].name == "S3Model"


@pytest.mark.asyncio
async def test_disabled_models_filtered(
    cache_mock: AsyncMock,
    monkeypatch: Any,
    prompt_library_manager: PromptLibraryManager,
) -> None:
    os.environ["MODELS_OFFICIAL_PATH"] = tempfile.gettempdir()
    cache_mock.get.return_value = None
    with patch(
        "languagemodelcommon.configs.config_reader.config_reader.FileConfigReader"
    ) as FileConfigReaderMock:
        FileConfigReaderMock.return_value.read_model_configs.return_value = [
            ChatModelConfig(id="1", name="Enabled", description="", disabled=False),
            ChatModelConfig(id="2", name="Disabled", description="", disabled=True),
        ]
        reader = ConfigReader(
            cache=cache_mock, prompt_library_manager=prompt_library_manager
        )
        result = await reader.read_model_configs_async()
        assert all(not m.disabled for m in result)


@pytest.mark.asyncio
async def test_client_override_merges_with_default(
    cache_mock: AsyncMock, tmp_path: Any, prompt_library_manager: PromptLibraryManager
) -> None:
    client_dir = tmp_path / "clients" / "client-a"
    client_dir.mkdir(parents=True)

    (tmp_path / "model.json").write_text(
        '{"id": "model-a", "name": "Model A", "description": "base", "model": {"provider": "openai", "model": "gpt-4o-mini"}}',
        encoding="utf-8",
    )
    (client_dir / "model.json").write_text(
        '{"id": "model-a", "name": "Model A", "description": "override", "model": {"provider": "openai", "model": "gpt-4o"}}',
        encoding="utf-8",
    )

    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)
    reader = ConfigReader(
        cache=cache_mock, prompt_library_manager=prompt_library_manager
    )
    configs = await reader.read_model_configs_async(client_id="client-a")

    assert len(configs) == 1
    assert configs[0].description == "override"
    assert configs[0].model is not None
    assert configs[0].model.model == "gpt-4o"


@pytest.mark.asyncio
async def test_prompt_name_resolves_from_library(
    cache_mock: AsyncMock, tmp_path: Any
) -> None:
    prompt_library = tmp_path / "prompt_library" / "prompt_library" / "prompts"
    prompt_library.mkdir(parents=True)
    (prompt_library / "support_prompt.txt").write_text(
        "Use the prompt library when requested.", encoding="utf-8"
    )

    (tmp_path / "model.json").write_text(
        '{"id": "model-a", "name": "Model A", "description": "base", '
        '"model": {"provider": "openai", "model": "gpt-4o-mini"}, '
        '"system_prompts": [{"name": "support_prompt"}]}',
        encoding="utf-8",
    )

    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)
    reader = ConfigReader(
        cache=cache_mock,
        prompt_library_manager=PromptLibraryManager(
            environment_variables=_StubPromptLibraryEnv(str(prompt_library))
        ),
    )
    configs = await reader.read_model_configs_async()

    assert configs[0].system_prompts is not None
    assert (
        configs[0].system_prompts[0].content == "Use the prompt library when requested."
    )


@pytest.mark.asyncio
async def test_override_does_not_clobber_default_fields(
    cache_mock: AsyncMock, tmp_path: Any, prompt_library_manager: PromptLibraryManager
) -> None:
    client_dir = tmp_path / "clients" / "client-b"
    client_dir.mkdir(parents=True)

    (tmp_path / "model.json").write_text(
        '{"id": "model-a", "name": "Model A", "description": "base", "type": "custom", "model": {"provider": "openai", "model": "gpt-4o-mini"}}',
        encoding="utf-8",
    )
    (client_dir / "model.json").write_text(
        '{"id": "model-a", "name": "Model A", "description": "override"}',
        encoding="utf-8",
    )

    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)
    reader = ConfigReader(
        cache=cache_mock, prompt_library_manager=prompt_library_manager
    )
    configs = await reader.read_model_configs_async(client_id="client-b")

    assert len(configs) == 1
    assert configs[0].description == "override"
    assert configs[0].type == "custom"


@pytest.mark.asyncio
async def test_prompt_auto_discovered_from_prompts_folder(
    cache_mock: AsyncMock, tmp_path: Any
) -> None:
    """When no PROMPT_LIBRARY_PATH is set, prompts/ alongside configs is used."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "system_prompt.md").write_text(
        "# System\nYou are helpful.", encoding="utf-8"
    )

    (tmp_path / "model.json").write_text(
        '{"id": "m1", "name": "Model", "description": "test", '
        '"system_prompts": [{"name": "system_prompt"}]}',
        encoding="utf-8",
    )

    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)
    reader = ConfigReader(
        cache=cache_mock,
        prompt_library_manager=PromptLibraryManager(
            environment_variables=_StubPromptLibraryEnv(None)
        ),
    )
    configs = await reader.read_model_configs_async()

    assert configs[0].system_prompts is not None
    assert configs[0].system_prompts[0].content == "# System\nYou are helpful."


@pytest.mark.asyncio
async def test_inline_prompt_content_still_works(
    cache_mock: AsyncMock, tmp_path: Any, prompt_library_manager: PromptLibraryManager
) -> None:
    """Inline content in PromptConfig is preserved (backward compat)."""
    (tmp_path / "model.json").write_text(
        '{"id": "m1", "name": "Model", "description": "test", '
        '"system_prompts": [{"content": "You are a helpful assistant."}]}',
        encoding="utf-8",
    )

    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)
    reader = ConfigReader(
        cache=cache_mock, prompt_library_manager=prompt_library_manager
    )
    configs = await reader.read_model_configs_async()

    assert configs[0].system_prompts is not None
    assert configs[0].system_prompts[0].content == "You are a helpful assistant."


# ── Snapshot cache tests ────────────────────────────────────────────


def _make_snapshot_store_mock() -> AsyncMock:
    """Create an AsyncMock that quacks like key_value BaseStore."""
    store = AsyncMock()
    store.get = AsyncMock(return_value=None)
    store.put = AsyncMock()
    return store


_SAMPLE_MODEL = ChatModelConfig(id="snap-1", name="SnapModel", description="cached")


@pytest.mark.asyncio
async def test_snapshot_cache_hit_short_circuits_disk(
    cache_mock: AsyncMock,
    prompt_library_manager: PromptLibraryManager,
    tmp_path: Path,
) -> None:
    """When the snapshot store returns data, disk/GitHub is never consulted."""
    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)

    snapshot_store = _make_snapshot_store_mock()
    snapshot_store.get.return_value = {
        "models": [_SAMPLE_MODEL.model_dump()],
    }

    reader = ConfigReader(
        cache=cache_mock,
        prompt_library_manager=prompt_library_manager,
        snapshot_cache_store=snapshot_store,
    )

    with patch(
        "languagemodelcommon.configs.config_reader.config_reader.FileConfigReader"
    ) as fc_mock:
        result = await reader.read_model_configs_async()
        fc_mock.assert_not_called()

    assert len(result) == 1
    assert result[0].name == "SnapModel"


@pytest.mark.asyncio
async def test_snapshot_cache_returns_none_falls_through(
    cache_mock: AsyncMock,
    prompt_library_manager: PromptLibraryManager,
    tmp_path: Path,
) -> None:
    """When the snapshot store returns None, configs are read from disk."""
    (tmp_path / "model.json").write_text(
        '{"id": "disk-1", "name": "DiskModel", "description": "from disk"}',
        encoding="utf-8",
    )
    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)

    snapshot_store = _make_snapshot_store_mock()
    snapshot_store.get.return_value = None

    reader = ConfigReader(
        cache=cache_mock,
        prompt_library_manager=prompt_library_manager,
        snapshot_cache_store=snapshot_store,
    )
    result = await reader.read_model_configs_async()
    assert result[0].name == "DiskModel"


@pytest.mark.asyncio
async def test_snapshot_cache_get_error_falls_through(
    cache_mock: AsyncMock,
    prompt_library_manager: PromptLibraryManager,
    tmp_path: Path,
) -> None:
    """If the snapshot store .get() raises, config loading still succeeds."""
    (tmp_path / "model.json").write_text(
        '{"id": "disk-1", "name": "DiskModel", "description": "from disk"}',
        encoding="utf-8",
    )
    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)

    snapshot_store = _make_snapshot_store_mock()
    snapshot_store.get.side_effect = ConnectionError("MongoDB unavailable")

    reader = ConfigReader(
        cache=cache_mock,
        prompt_library_manager=prompt_library_manager,
        snapshot_cache_store=snapshot_store,
    )
    result = await reader.read_model_configs_async()
    assert result[0].name == "DiskModel"


@pytest.mark.asyncio
async def test_snapshot_cache_deserialization_error_falls_through(
    cache_mock: AsyncMock,
    prompt_library_manager: PromptLibraryManager,
    tmp_path: Path,
) -> None:
    """If stored data is corrupt, config loading still succeeds from disk."""
    (tmp_path / "model.json").write_text(
        '{"id": "disk-1", "name": "DiskModel", "description": "from disk"}',
        encoding="utf-8",
    )
    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)

    snapshot_store = _make_snapshot_store_mock()
    snapshot_store.get.return_value = {"models": [{"bad": "data"}]}

    reader = ConfigReader(
        cache=cache_mock,
        prompt_library_manager=prompt_library_manager,
        snapshot_cache_store=snapshot_store,
    )
    result = await reader.read_model_configs_async()
    assert result[0].name == "DiskModel"


@pytest.mark.asyncio
async def test_snapshot_cache_put_error_does_not_break_config_read(
    cache_mock: AsyncMock,
    prompt_library_manager: PromptLibraryManager,
    tmp_path: Path,
) -> None:
    """If snapshot .put() raises, configs are still returned to the caller."""
    (tmp_path / "model.json").write_text(
        '{"id": "disk-1", "name": "DiskModel", "description": "from disk"}',
        encoding="utf-8",
    )
    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)

    snapshot_store = _make_snapshot_store_mock()
    snapshot_store.get.return_value = None  # miss → read from disk
    snapshot_store.put.side_effect = TimeoutError("MongoDB write timeout")

    reader = ConfigReader(
        cache=cache_mock,
        prompt_library_manager=prompt_library_manager,
        snapshot_cache_store=snapshot_store,
    )
    result = await reader.read_model_configs_async()
    assert result[0].name == "DiskModel"
    # put was attempted and failed silently
    snapshot_store.put.assert_called_once()


@pytest.mark.asyncio
async def test_snapshot_cache_none_store_skips_entirely(
    cache_mock: AsyncMock,
    prompt_library_manager: PromptLibraryManager,
    tmp_path: Path,
) -> None:
    """When no snapshot_cache_store is provided, cache logic is a no-op."""
    (tmp_path / "model.json").write_text(
        '{"id": "disk-1", "name": "DiskModel", "description": "from disk"}',
        encoding="utf-8",
    )
    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)

    reader = ConfigReader(
        cache=cache_mock,
        prompt_library_manager=prompt_library_manager,
        snapshot_cache_store=None,
    )
    result = await reader.read_model_configs_async()
    assert result[0].name == "DiskModel"
