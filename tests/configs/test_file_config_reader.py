import json
from pathlib import Path
from typing import Any

import pytest
from languagemodelcommon.configs.config_reader.file_config_reader import (
    FileConfigReader,
)


def create_json_file(path: Path, data: dict[str, str]) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def test_read_model_configs_reads_all_json(tmp_path: Path) -> None:
    # Arrange
    config1 = {"name": "modelA", "id": "1", "description": "A model", "type": "str"}
    config2 = {"name": "modelB", "id": "2", "description": "B model", "type": "str"}
    create_json_file(tmp_path / "a.json", config1)
    create_json_file(tmp_path / "b.json", config2)
    reader = FileConfigReader()

    # Act
    configs = reader.read_model_configs(config_path=str(tmp_path))

    # Assert
    assert len(configs) == 2
    assert configs[0].name == "modelA"
    assert configs[1].name == "modelB"


def test_read_model_configs_empty_dir(tmp_path: Path) -> None:
    reader = FileConfigReader()
    configs = reader.read_model_configs(config_path=str(tmp_path))
    assert configs == []


def test_read_model_configs_invalid_json(tmp_path: Path) -> None:
    invalid_file = tmp_path / "bad.json"
    invalid_file.write_text("{not: valid json}")
    reader = FileConfigReader()
    with pytest.raises(json.JSONDecodeError):
        reader.read_model_configs(config_path=str(tmp_path))


@pytest.mark.parametrize(
    ("env_value", "placeholder", "expected"),
    [
        ("https://example.test/fhir/", "${MCP_FHIR_URL}", "https://example.test/fhir/"),
    ],
)
def test_env_var_substitution(
    tmp_path: Path,
    monkeypatch: Any,
    env_value: str | None,
    placeholder: str,
    expected: str,
) -> None:
    if env_value is None:
        monkeypatch.delenv("MCP_FHIR_URL", raising=False)
    else:
        monkeypatch.setenv("MCP_FHIR_URL", env_value)

    config_path = tmp_path / "model.json"
    config_path.write_text(
        json.dumps(
            {
                "id": "model-a",
                "name": "Model A",
                "description": "Example",
                "model": {"provider": "openai", "model": "gpt-4o"},
                "tools": [{"name": "fhir_server", "url": placeholder}],
            }
        ),
        encoding="utf-8",
    )

    configs = FileConfigReader().read_model_configs(config_path=str(tmp_path))
    assert configs[0].tools is not None
    assert configs[0].tools[0].url == expected


def test_env_var_substitution_missing_raises(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv("MCP_FHIR_URL", raising=False)

    config_path = tmp_path / "model.json"
    config_path.write_text(
        json.dumps(
            {
                "id": "model-a",
                "name": "Model A",
                "description": "Example",
                "model": {"provider": "openai", "model": "gpt-4o"},
                "tools": [{"name": "fhir_server", "url": "${MCP_FHIR_URL}"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing environment variable"):
        FileConfigReader().read_model_configs(config_path=str(tmp_path))


def test_discover_prompts_path_found(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "system.md").write_text("# System", encoding="utf-8")

    result = FileConfigReader.discover_prompts_path(str(tmp_path))
    assert result == str(prompts_dir)


def test_discover_prompts_path_not_found(tmp_path: Path) -> None:
    result = FileConfigReader.discover_prompts_path(str(tmp_path))
    assert result is None
