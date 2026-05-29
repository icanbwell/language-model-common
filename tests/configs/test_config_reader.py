import os
import pytest
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from languagemodelcommon.configs.config_reader.config_reader import ConfigReader
from languagemodelcommon.configs.config_reader.mcp_json_fetcher import McpJsonFetcher
from languagemodelcommon.configs.prompt_library.prompt_library_environment_variables import (
    PromptLibraryEnvironmentVariables,
)
from languagemodelcommon.configs.schemas.config_schema import (
    AgentConfig,
    ChatModelConfig,
)
from languagemodelcommon.configs.schemas.mcp_json_schema import (
    McpJsonConfig,
    McpServerEntry,
)
from languagemodelcommon.configs.prompt_library.prompt_library_manager import (
    PromptLibraryManager,
)
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)


class _StubPromptLibraryEnv(PromptLibraryEnvironmentVariables):
    def __init__(self, prompt_library_path: str | None) -> None:
        self._prompt_library_path = prompt_library_path

    @property
    def prompt_library_path(self) -> str | None:
        return self._prompt_library_path


def _make_snapshot_store_mock() -> AsyncMock:
    """Create an AsyncMock that quacks like key_value BaseStore."""
    store = AsyncMock()
    store.get = AsyncMock(return_value=None)
    store.put = AsyncMock()
    return store


@pytest.fixture
def prompt_library_manager(tmp_path: Path) -> PromptLibraryManager:
    return PromptLibraryManager(
        environment_variables=_StubPromptLibraryEnv(str(tmp_path))
    )


@pytest.fixture
def environment_variables() -> LanguageModelCommonEnvironmentVariables:
    return LanguageModelCommonEnvironmentVariables()


@pytest.fixture
def config_reader(
    prompt_library_manager: PromptLibraryManager,
    environment_variables: LanguageModelCommonEnvironmentVariables,
) -> ConfigReader:
    return ConfigReader(
        prompt_library_manager=prompt_library_manager,
        environment_variables=environment_variables,
    )


@pytest.mark.asyncio
async def test_snapshot_cache_hit(
    monkeypatch: Any,
    prompt_library_manager: PromptLibraryManager,
) -> None:
    os.environ["MODELS_OFFICIAL_PATH"] = tempfile.gettempdir()
    snapshot_store = _make_snapshot_store_mock()
    snapshot_store.get.return_value = {
        "models": [ChatModelConfig(id="1", name="Test", description="").model_dump()],
    }
    reader = ConfigReader(
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        snapshot_cache_store=snapshot_store,
    )
    result = await reader.read_model_configs_async()
    assert result[0].name == "Test"


@pytest.mark.asyncio
async def test_env_var_missing(prompt_library_manager: PromptLibraryManager) -> None:
    if "MODELS_OFFICIAL_PATH" in os.environ:
        del os.environ["MODELS_OFFICIAL_PATH"]
    reader = ConfigReader(
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
    )
    with pytest.raises(ValueError):
        await reader.read_model_configs_async()


@patch("languagemodelcommon.configs.config_reader.config_reader.FileConfigReader")
@pytest.mark.asyncio
async def test_read_from_file(
    FileConfigReaderMock: Any,
    monkeypatch: Any,
    prompt_library_manager: PromptLibraryManager,
) -> None:
    os.environ["MODELS_OFFICIAL_PATH"] = tempfile.gettempdir()
    FileConfigReaderMock.return_value.read_model_configs.return_value = [
        ChatModelConfig(id="1", name="FileModel", description="")
    ]
    reader = ConfigReader(
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
    )
    result = await reader.read_model_configs_async()
    assert result[0].name == "FileModel"


@patch("languagemodelcommon.configs.config_reader.config_reader.S3ConfigReader")
@pytest.mark.asyncio
async def test_read_from_s3(
    S3ConfigReaderMock: Any,
    monkeypatch: Any,
    prompt_library_manager: PromptLibraryManager,
) -> None:
    os.environ["MODELS_OFFICIAL_PATH"] = "s3://bucket/models"
    os.environ.pop("MODELS_TESTING_PATH", None)
    S3ConfigReaderMock.return_value.read_model_configs = AsyncMock(
        return_value=[ChatModelConfig(id="2", name="S3Model", description="")]
    )
    reader = ConfigReader(
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
    )
    result = await reader.read_model_configs_async()
    assert result[0].name == "S3Model"


@pytest.mark.asyncio
async def test_disabled_models_filtered(
    monkeypatch: Any,
    prompt_library_manager: PromptLibraryManager,
) -> None:
    os.environ["MODELS_OFFICIAL_PATH"] = tempfile.gettempdir()
    with patch(
        "languagemodelcommon.configs.config_reader.config_reader.FileConfigReader"
    ) as FileConfigReaderMock:
        FileConfigReaderMock.return_value.read_model_configs.return_value = [
            ChatModelConfig(id="1", name="Enabled", description="", disabled=False),
            ChatModelConfig(id="2", name="Disabled", description="", disabled=True),
        ]
        reader = ConfigReader(
            prompt_library_manager=prompt_library_manager,
            environment_variables=LanguageModelCommonEnvironmentVariables(),
        )
        result = await reader.read_model_configs_async()
        assert all(not m.disabled for m in result)


@pytest.mark.asyncio
async def test_client_override_merges_with_default(
    tmp_path: Any, prompt_library_manager: PromptLibraryManager
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
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
    )
    configs = await reader.read_model_configs_async(client_id="client-a")

    assert len(configs) == 1
    assert configs[0].description == "override"
    assert configs[0].model is not None
    assert configs[0].model.model == "gpt-4o"


@pytest.mark.asyncio
async def test_prompt_name_resolves_from_library(tmp_path: Any) -> None:
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
        prompt_library_manager=PromptLibraryManager(
            environment_variables=_StubPromptLibraryEnv(str(prompt_library))
        ),
        environment_variables=LanguageModelCommonEnvironmentVariables(),
    )
    configs = await reader.read_model_configs_async()

    assert configs[0].system_prompts is not None
    assert (
        configs[0].system_prompts[0].content == "Use the prompt library when requested."
    )


@pytest.mark.asyncio
async def test_override_does_not_clobber_default_fields(
    tmp_path: Any, prompt_library_manager: PromptLibraryManager
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
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
    )
    configs = await reader.read_model_configs_async(client_id="client-b")

    assert len(configs) == 1
    assert configs[0].description == "override"
    assert configs[0].type == "custom"


@pytest.mark.asyncio
async def test_prompt_auto_discovered_from_prompts_folder(tmp_path: Any) -> None:
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
        prompt_library_manager=PromptLibraryManager(
            environment_variables=_StubPromptLibraryEnv(None)
        ),
        environment_variables=LanguageModelCommonEnvironmentVariables(),
    )
    configs = await reader.read_model_configs_async()

    assert configs[0].system_prompts is not None
    assert configs[0].system_prompts[0].content == "# System\nYou are helpful."


@pytest.mark.asyncio
async def test_inline_prompt_content_still_works(
    tmp_path: Any, prompt_library_manager: PromptLibraryManager
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
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
    )
    configs = await reader.read_model_configs_async()

    assert configs[0].system_prompts is not None
    assert configs[0].system_prompts[0].content == "You are a helpful assistant."


# ── Snapshot cache tests ────────────────────────────────────────────


_SAMPLE_MODEL = ChatModelConfig(id="snap-1", name="SnapModel", description="cached")


@pytest.mark.asyncio
async def test_snapshot_cache_hit_short_circuits_disk(
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
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
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
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        snapshot_cache_store=snapshot_store,
    )
    result = await reader.read_model_configs_async()
    assert result[0].name == "DiskModel"


@pytest.mark.asyncio
async def test_snapshot_cache_get_error_propagates(
    prompt_library_manager: PromptLibraryManager,
    tmp_path: Path,
) -> None:
    """If the snapshot store .get() raises, the error propagates (fail-fast)."""
    (tmp_path / "model.json").write_text(
        '{"id": "disk-1", "name": "DiskModel", "description": "from disk"}',
        encoding="utf-8",
    )
    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)

    snapshot_store = _make_snapshot_store_mock()
    snapshot_store.get.side_effect = ConnectionError("MongoDB unavailable")

    reader = ConfigReader(
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        snapshot_cache_store=snapshot_store,
    )
    with pytest.raises(ConnectionError, match="MongoDB unavailable"):
        await reader.read_model_configs_async()


@pytest.mark.asyncio
async def test_snapshot_cache_deserialization_error_propagates(
    prompt_library_manager: PromptLibraryManager,
    tmp_path: Path,
) -> None:
    """If stored data is corrupt, the error propagates (fail-fast)."""
    (tmp_path / "model.json").write_text(
        '{"id": "disk-1", "name": "DiskModel", "description": "from disk"}',
        encoding="utf-8",
    )
    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)

    snapshot_store = _make_snapshot_store_mock()
    snapshot_store.get.return_value = {"models": [{"bad": "data"}]}

    reader = ConfigReader(
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        snapshot_cache_store=snapshot_store,
    )
    with pytest.raises(Exception):
        await reader.read_model_configs_async()


@pytest.mark.asyncio
async def test_snapshot_cache_put_error_propagates(
    prompt_library_manager: PromptLibraryManager,
    tmp_path: Path,
) -> None:
    """If snapshot .put() raises, the error propagates (fail-fast)."""
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
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        snapshot_cache_store=snapshot_store,
    )
    with pytest.raises(TimeoutError, match="MongoDB write timeout"):
        await reader.read_model_configs_async()


@pytest.mark.asyncio
async def test_snapshot_cache_none_store_skips_entirely(
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
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        snapshot_cache_store=None,
    )
    result = await reader.read_model_configs_async()
    assert result[0].name == "DiskModel"


@pytest.mark.asyncio
async def test_clear_cache_deletes_snapshot_entry(
    prompt_library_manager: PromptLibraryManager,
) -> None:
    """clear_cache() should delete the snapshot cache entry."""
    snapshot_store = _make_snapshot_store_mock()
    snapshot_store.delete = AsyncMock(return_value=True)

    reader = ConfigReader(
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        snapshot_cache_store=snapshot_store,
    )
    await reader.clear_cache()

    snapshot_store.delete.assert_awaited_once_with(
        "model_configs:v1",
        collection=None,
    )


@pytest.mark.asyncio
async def test_clear_cache_without_snapshot_store(
    prompt_library_manager: PromptLibraryManager,
) -> None:
    """clear_cache() with no snapshot store is a no-op."""
    reader = ConfigReader(
        prompt_library_manager=prompt_library_manager,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        snapshot_cache_store=None,
    )
    await reader.clear_cache()


# ── MCP resolution retry tests ─────────────────────────────────────


def _model_with_unresolved_mcp(plugin: str = "all-employees") -> ChatModelConfig:
    """Create a model config with an unresolved mcp_server wildcard."""
    return ChatModelConfig(
        id="test-model",
        name="Test Model",
        description="",
        plugins=[plugin],
        tools=[AgentConfig(name="all_mcp_servers", mcp_server="*")],
    )


def _model_without_mcp() -> ChatModelConfig:
    return ChatModelConfig(id="simple", name="Simple", description="")


def _mcp_json_config() -> McpJsonConfig:
    return McpJsonConfig(
        mcpServers={
            "skills": McpServerEntry(
                url="http://mcp:5000/skills/", description="Skills"
            ),
            "google": McpServerEntry(
                url="http://mcp:5000/google/", description="Google"
            ),
        }
    )


@pytest.mark.asyncio
async def test_retry_resolves_cached_models_with_unresolved_mcp(
    prompt_library_manager: PromptLibraryManager,
    tmp_path: Path,
) -> None:
    """When snapshot returns models with unresolved mcp_server refs, retry resolves them."""
    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)

    unresolved_model = _model_with_unresolved_mcp()
    snapshot_store = _make_snapshot_store_mock()
    snapshot_store.get.return_value = {
        "models": [unresolved_model.model_dump()],
    }

    fetcher = MagicMock(spec=McpJsonFetcher)
    fetcher._url = "http://localhost:5000/plugin-marketplace/"
    fetcher.fetch_plugins_async = AsyncMock(
        return_value=({"all-employees": _mcp_json_config()}, [])
    )

    reader = ConfigReader(
        prompt_library_manager=prompt_library_manager,
        mcp_json_fetcher=fetcher,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        snapshot_cache_store=snapshot_store,
    )
    result = await reader.read_model_configs_async()

    # The wildcard should have been expanded and URLs populated
    agents = result[0].get_agents()
    assert len(agents) == 2
    assert all(a.url for a in agents), "All agents should have resolved URLs"

    # Snapshot cache should have been updated with resolved models
    assert snapshot_store.put.await_count >= 1


@pytest.mark.asyncio
async def test_first_request_always_resolves_mcp(
    prompt_library_manager: PromptLibraryManager,
    tmp_path: Path,
) -> None:
    """First request always re-resolves MCP servers even if URLs are present."""
    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)

    resolved_model = ChatModelConfig(
        id="test",
        name="Test",
        description="",
        plugins=["all-employees"],
        tools=[
            AgentConfig(
                name="skills", mcp_server="skills", url="http://mcp:5000/skills/"
            ),
        ],
    )
    snapshot_store = _make_snapshot_store_mock()
    snapshot_store.get.return_value = {
        "models": [resolved_model.model_dump()],
    }

    fetcher = MagicMock(spec=McpJsonFetcher)
    fetcher._url = "http://localhost:5000/plugin-marketplace/"
    fetcher.fetch_plugins_async = AsyncMock(
        return_value=({"all-employees": _mcp_json_config()}, [])
    )

    reader = ConfigReader(
        prompt_library_manager=prompt_library_manager,
        mcp_json_fetcher=fetcher,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        snapshot_cache_store=snapshot_store,
    )
    await reader.read_model_configs_async()

    # First request always resolves to pick up .mcp.json changes
    fetcher.fetch_plugins_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_retry_still_fails_gracefully(
    prompt_library_manager: PromptLibraryManager,
    tmp_path: Path,
) -> None:
    """When retry also fails, models are returned unresolved without error."""
    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)

    unresolved_model = _model_with_unresolved_mcp()
    snapshot_store = _make_snapshot_store_mock()
    snapshot_store.get.return_value = {
        "models": [unresolved_model.model_dump()],
    }

    fetcher = MagicMock(spec=McpJsonFetcher)
    fetcher._url = "http://localhost:5000/plugin-marketplace/"
    fetcher.fetch_plugins_async = AsyncMock(return_value=({}, []))

    reader = ConfigReader(
        prompt_library_manager=prompt_library_manager,
        mcp_json_fetcher=fetcher,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        snapshot_cache_store=snapshot_store,
    )
    result = await reader.read_model_configs_async()

    # Models returned but mcp_server still unresolved
    agents = result[0].get_agents()
    assert agents[0].mcp_server == "*"
    assert agents[0].url is None


@pytest.mark.asyncio
async def test_no_retry_when_no_mcp_refs(
    prompt_library_manager: PromptLibraryManager,
    tmp_path: Path,
) -> None:
    """Models without mcp_server references skip the retry entirely."""
    os.environ["MODELS_OFFICIAL_PATH"] = str(tmp_path)
    os.environ.pop("MODELS_TESTING_PATH", None)

    snapshot_store = _make_snapshot_store_mock()
    snapshot_store.get.return_value = {
        "models": [_model_without_mcp().model_dump()],
    }

    fetcher = MagicMock(spec=McpJsonFetcher)
    fetcher.fetch_plugins_async = AsyncMock()

    reader = ConfigReader(
        prompt_library_manager=prompt_library_manager,
        mcp_json_fetcher=fetcher,
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        snapshot_cache_store=snapshot_store,
    )
    await reader.read_model_configs_async()

    fetcher.fetch_plugins_async.assert_not_awaited()
