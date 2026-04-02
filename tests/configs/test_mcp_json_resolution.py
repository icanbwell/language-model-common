import json
from pathlib import Path
from typing import Any


from languagemodelcommon.configs.config_reader.file_config_reader import (
    FileConfigReader,
)
from languagemodelcommon.configs.config_reader.mcp_json_reader import (
    MCP_JSON_PATH_ENV,
    McpJsonConfig,
    McpServerEntry,
    read_mcp_json,
    resolve_mcp_servers,
)
from languagemodelcommon.configs.schemas.config_schema import (
    AgentConfig,
    ChatModelConfig,
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_model_config(
    tool_name: str, mcp_server: str | None = None, url: str | None = None
) -> dict[str, Any]:
    tool: dict[str, Any] = {"name": tool_name}
    if mcp_server is not None:
        tool["mcp_server"] = mcp_server
    if url is not None:
        tool["url"] = url
    return {
        "id": f"model-{tool_name}",
        "name": f"Model {tool_name}",
        "tools": [tool],
    }


def _make_mcp_json(servers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"mcpServers": servers}


class TestReadMcpJson:
    def test_reads_from_config_dir(self, tmp_path: Path) -> None:
        mcp_data = _make_mcp_json({"my-server": {"url": "https://example.com/mcp/"}})
        _write_json(tmp_path / ".mcp.json", mcp_data)

        result = read_mcp_json(config_dir=str(tmp_path))

        assert result is not None
        assert "my-server" in result.mcpServers
        assert result.mcpServers["my-server"].url == "https://example.com/mcp/"

    def test_returns_none_when_no_file(self, tmp_path: Path) -> None:
        result = read_mcp_json(config_dir=str(tmp_path))
        assert result is None

    def test_env_var_overrides_config_dir(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        env_mcp = tmp_path / "custom" / ".mcp.json"
        env_mcp.parent.mkdir()
        _write_json(
            env_mcp, _make_mcp_json({"env-server": {"url": "https://env.example.com/"}})
        )

        # Also create one in config_dir to prove it's NOT used
        _write_json(
            tmp_path / ".mcp.json",
            _make_mcp_json({"dir-server": {"url": "https://dir.example.com/"}}),
        )

        monkeypatch.setenv(MCP_JSON_PATH_ENV, str(env_mcp))
        result = read_mcp_json(config_dir=str(tmp_path))

        assert result is not None
        assert "env-server" in result.mcpServers
        assert "dir-server" not in result.mcpServers

    def test_returns_none_when_no_config_dir(self) -> None:
        result = read_mcp_json(config_dir=None)
        assert result is None

    def test_env_var_substitution(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setenv("MCP_SERVER_URL", "https://resolved.example.com/")
        mcp_data = _make_mcp_json({"my-server": {"url": "${MCP_SERVER_URL}"}})
        _write_json(tmp_path / ".mcp.json", mcp_data)

        result = read_mcp_json(config_dir=str(tmp_path))

        assert result is not None
        assert result.mcpServers["my-server"].url == "https://resolved.example.com/"

    def test_extra_fields_preserved(self, tmp_path: Path) -> None:
        mcp_data = _make_mcp_json(
            {
                "server": {
                    "type": "http",
                    "url": "https://example.com/",
                    "oauth": {"clientId": "abc123"},
                }
            }
        )
        _write_json(tmp_path / ".mcp.json", mcp_data)

        result = read_mcp_json(config_dir=str(tmp_path))
        assert result is not None


class TestResolveMcpServers:
    def test_resolves_url_from_mcp_server(self) -> None:
        config = ChatModelConfig(
            **_make_model_config("drive", mcp_server="google-drive")
        )
        mcp = McpJsonConfig(
            mcpServers={
                "google-drive": McpServerEntry(url="https://mcp.example.com/drive/")
            }
        )

        resolve_mcp_servers([config], mcp)

        assert config.tools is not None
        assert config.tools[0].url == "https://mcp.example.com/drive/"

    def test_mcp_server_overrides_existing_url(self) -> None:
        config = ChatModelConfig(
            **_make_model_config(
                "drive", mcp_server="google-drive", url="https://old.example.com/"
            )
        )
        mcp = McpJsonConfig(
            mcpServers={
                "google-drive": McpServerEntry(url="https://new.example.com/drive/")
            }
        )

        resolve_mcp_servers([config], mcp)

        assert config.tools is not None
        assert config.tools[0].url == "https://new.example.com/drive/"

    def test_missing_mcp_server_key_falls_back(self) -> None:
        config = ChatModelConfig(
            **_make_model_config(
                "drive", mcp_server="nonexistent", url="https://fallback.example.com/"
            )
        )
        mcp = McpJsonConfig(
            mcpServers={"google-drive": McpServerEntry(url="https://example.com/")}
        )

        resolve_mcp_servers([config], mcp)

        assert config.tools is not None
        assert config.tools[0].url == "https://fallback.example.com/"

    def test_no_mcp_server_field_unchanged(self) -> None:
        config = ChatModelConfig(
            **_make_model_config("drive", url="https://original.example.com/")
        )
        mcp = McpJsonConfig(
            mcpServers={"google-drive": McpServerEntry(url="https://new.example.com/")}
        )

        resolve_mcp_servers([config], mcp)

        assert config.tools is not None
        assert config.tools[0].url == "https://original.example.com/"

    def test_resolves_agents_field(self) -> None:
        config = ChatModelConfig(
            id="model-1",
            name="Model 1",
            agents=[AgentConfig(name="drive", mcp_server="google-drive")],
        )
        mcp = McpJsonConfig(
            mcpServers={
                "google-drive": McpServerEntry(url="https://mcp.example.com/drive/")
            }
        )

        resolve_mcp_servers([config], mcp)

        assert config.agents is not None
        assert config.agents[0].url == "https://mcp.example.com/drive/"


class TestFileConfigReaderMcpJsonIntegration:
    def test_resolves_mcp_server_during_read(self, tmp_path: Path) -> None:
        _write_json(
            tmp_path / "model.json",
            _make_model_config("drive", mcp_server="google-drive"),
        )
        _write_json(
            tmp_path / ".mcp.json",
            _make_mcp_json(
                {
                    "google-drive": {
                        "type": "http",
                        "url": "https://mcp.example.com/drive/",
                    },
                }
            ),
        )

        configs = FileConfigReader().read_model_configs(config_path=str(tmp_path))

        assert len(configs) == 1
        assert configs[0].tools is not None
        assert configs[0].tools[0].url == "https://mcp.example.com/drive/"
        assert configs[0].tools[0].mcp_server == "google-drive"

    def test_no_mcp_json_still_works(self, tmp_path: Path) -> None:
        _write_json(
            tmp_path / "model.json",
            _make_model_config("drive", url="https://direct.example.com/"),
        )

        configs = FileConfigReader().read_model_configs(config_path=str(tmp_path))

        assert len(configs) == 1
        assert configs[0].tools is not None
        assert configs[0].tools[0].url == "https://direct.example.com/"

    def test_env_var_mcp_json_path(self, tmp_path: Path, monkeypatch: Any) -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _write_json(
            model_dir / "model.json", _make_model_config("drive", mcp_server="gd")
        )

        mcp_file = tmp_path / "custom-mcp.json"
        _write_json(
            mcp_file, _make_mcp_json({"gd": {"url": "https://custom.example.com/"}})
        )

        monkeypatch.setenv(MCP_JSON_PATH_ENV, str(mcp_file))
        configs = FileConfigReader().read_model_configs(config_path=str(model_dir))

        assert len(configs) == 1
        assert configs[0].tools is not None
        assert configs[0].tools[0].url == "https://custom.example.com/"
