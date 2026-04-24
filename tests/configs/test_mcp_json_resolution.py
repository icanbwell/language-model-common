import json
from pathlib import Path
from typing import Any

import pytest


from languagemodelcommon.configs.config_reader.file_config_reader import (
    FileConfigReader,
)
from languagemodelcommon.configs.config_reader.mcp_json_reader import (
    McpJsonReader,
    read_mcp_json,
    resolve_mcp_servers,
)
from languagemodelcommon.configs.schemas.mcp_json_schema import (
    McpJsonConfig,
    McpServerEntry,
)
from languagemodelcommon.configs.schemas.config_schema import (
    AgentConfig,
    ChatModelConfig,
    McpOAuthConfig,
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_plugin_mcp_json(
    marketplace_path: Path,
    plugin_name: str,
    data: dict[str, Any],
) -> Path:
    """Create ``{marketplace_path}/plugins/{plugin_name}/.mcp.json``."""
    plugin_dir = marketplace_path / "plugins" / plugin_name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    mcp_path = plugin_dir / ".mcp.json"
    _write_json(mcp_path, data)
    return mcp_path


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
    def test_reads_from_marketplace_plugin(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        mcp_data = _make_mcp_json({"my-server": {"url": "https://example.com/mcp/"}})
        _write_plugin_mcp_json(tmp_path, "test-plugin", mcp_data)
        monkeypatch.setenv("PLUGINS_MARKETPLACE", str(tmp_path))

        result = read_mcp_json()

        assert result is not None
        assert "my-server" in result.mcpServers
        assert result.mcpServers["my-server"].url == "https://example.com/mcp/"

    def test_returns_none_when_no_plugins_dir(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setenv("PLUGINS_MARKETPLACE", str(tmp_path))
        result = read_mcp_json()
        assert result is None

    def test_returns_none_when_no_env_var(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("PLUGINS_MARKETPLACE", raising=False)
        result = read_mcp_json()
        assert result is None

    def test_merges_multiple_plugins(self, tmp_path: Path, monkeypatch: Any) -> None:
        _write_plugin_mcp_json(
            tmp_path,
            "plugin-a",
            _make_mcp_json({"server-a": {"url": "https://a.example.com/"}}),
        )
        _write_plugin_mcp_json(
            tmp_path,
            "plugin-b",
            _make_mcp_json({"server-b": {"url": "https://b.example.com/"}}),
        )
        monkeypatch.setenv("PLUGINS_MARKETPLACE", str(tmp_path))

        result = read_mcp_json()

        assert result is not None
        assert "server-a" in result.mcpServers
        assert "server-b" in result.mcpServers

    def test_env_var_substitution(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setenv("MCP_SERVER_URL", "https://resolved.example.com/")
        mcp_data = _make_mcp_json({"my-server": {"url": "${MCP_SERVER_URL}"}})
        _write_plugin_mcp_json(tmp_path, "test-plugin", mcp_data)
        monkeypatch.setenv("PLUGINS_MARKETPLACE", str(tmp_path))

        result = read_mcp_json()

        assert result is not None
        assert result.mcpServers["my-server"].url == "https://resolved.example.com/"

    def test_extra_fields_preserved(self, tmp_path: Path, monkeypatch: Any) -> None:
        mcp_data = _make_mcp_json(
            {
                "server": {
                    "type": "http",
                    "url": "https://example.com/",
                    "oauth": {
                        "clientId": "abc123",
                        "authServerMetadataUrl": "https://idp.example.com/.well-known/openid-configuration",
                    },
                    "customField": "custom_value",
                }
            }
        )
        _write_plugin_mcp_json(tmp_path, "test-plugin", mcp_data)
        monkeypatch.setenv("PLUGINS_MARKETPLACE", str(tmp_path))

        result = read_mcp_json()
        assert result is not None
        server = result.mcpServers["server"]
        # oauth is now a first-class field
        assert server.oauth is not None
        assert server.oauth.client_id == "abc123"
        # extra fields beyond the known ones are still preserved
        extras = getattr(server, "model_extra", None)
        if extras is None:
            extras = getattr(server, "__pydantic_extra__", None)
        assert extras is not None
        assert "customField" in extras
        assert extras["customField"] == "custom_value"

    def test_reader_uses_environment_variables_object(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        mcp_data = _make_mcp_json({"srv": {"url": "https://srv.example.com/"}})
        _write_plugin_mcp_json(tmp_path, "test-plugin", mcp_data)
        monkeypatch.setenv("PLUGINS_MARKETPLACE", str(tmp_path))

        from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
            LanguageModelCommonEnvironmentVariables,
        )

        reader = McpJsonReader(
            environment_variables=LanguageModelCommonEnvironmentVariables()
        )
        result = reader.read_mcp_json()

        assert result is not None
        assert "srv" in result.mcpServers


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

    def test_resolves_auth_fields_from_mcp_json(self) -> None:
        config = ChatModelConfig(
            **_make_model_config("drive", mcp_server="google-drive")
        )
        mcp = McpJsonConfig(
            mcpServers={
                "google-drive": McpServerEntry(
                    url="https://mcp.example.com/drive/",
                    auth="jwt_token",
                    headers={"X-Client-Id": "test"},
                    auth_providers=["google"],
                    issuers=["https://accounts.google.com"],
                    auth_optional=True,
                )
            }
        )

        resolve_mcp_servers([config], mcp)

        assert config.tools is not None
        tool = config.tools[0]
        assert tool.url == "https://mcp.example.com/drive/"
        assert tool.auth == "jwt_token"
        assert tool.headers == {"X-Client-Id": "test"}
        assert tool.auth_providers == ["google"]
        assert tool.issuers == ["https://accounts.google.com"]
        assert tool.auth_optional is True

    def test_mcp_json_overrides_inline_auth(self) -> None:
        """mcp_server resolution always uses .mcp.json values."""
        config = ChatModelConfig(
            id="model-1",
            name="Model 1",
            tools=[
                AgentConfig(
                    name="drive",
                    mcp_server="google-drive",
                    auth="oauth",
                    headers={"X-Custom": "mine"},
                )
            ],
        )
        mcp = McpJsonConfig(
            mcpServers={
                "google-drive": McpServerEntry(
                    url="https://mcp.example.com/drive/",
                    auth="jwt_token",
                    headers={"X-Client-Id": "test"},
                )
            }
        )

        resolve_mcp_servers([config], mcp)

        assert config.tools is not None
        tool = config.tools[0]
        assert tool.url == "https://mcp.example.com/drive/"
        assert tool.auth == "jwt_token"
        assert tool.headers == {"X-Client-Id": "test"}

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

    def test_resolves_oauth_from_mcp_json(self) -> None:
        config = ChatModelConfig(
            **_make_model_config("drive", mcp_server="google-drive")
        )
        mcp = McpJsonConfig(
            mcpServers={
                "google-drive": McpServerEntry(
                    url="https://mcp.example.com/drive/",
                    oauth=McpOAuthConfig(
                        client_id="abc123",
                        auth_server_metadata_url="https://idp.example.com/.well-known/openid-configuration",
                    ),
                )
            }
        )

        resolve_mcp_servers([config], mcp)

        assert config.tools is not None
        tool = config.tools[0]
        assert tool.url == "https://mcp.example.com/drive/"
        assert tool.oauth is not None
        assert tool.oauth.client_id == "abc123"
        assert (
            tool.oauth.auth_server_metadata_url
            == "https://idp.example.com/.well-known/openid-configuration"
        )
        assert tool.auth == "jwt_token"
        assert tool.auth_providers == ["mcp_oauth_abc123"]

    def test_oauth_overrides_inline_auth_and_providers(self) -> None:
        """When .mcp.json has oauth, it overrides any inline auth/auth_providers."""
        config = ChatModelConfig(
            id="model-1",
            name="Model 1",
            tools=[
                AgentConfig(
                    name="drive",
                    mcp_server="google-drive",
                    auth="oauth",
                    auth_providers=["custom-provider"],
                )
            ],
        )
        mcp = McpJsonConfig(
            mcpServers={
                "google-drive": McpServerEntry(
                    url="https://mcp.example.com/drive/",
                    oauth=McpOAuthConfig(
                        client_id="abc123",
                        auth_server_metadata_url="https://idp.example.com/.well-known/openid-configuration",
                    ),
                )
            }
        )

        resolve_mcp_servers([config], mcp)

        tool = config.tools[0]  # type: ignore[index]
        assert tool.oauth is not None
        assert tool.auth == "jwt_token"
        assert tool.auth_providers == ["mcp_oauth_abc123"]

    def test_oauth_parsed_from_camel_case_json(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        _write_plugin_mcp_json(
            tmp_path,
            "test-plugin",
            _make_mcp_json(
                {
                    "google-drive": {
                        "type": "http",
                        "url": "https://mcp.example.com/drive/",
                        "oauth": {
                            "clientId": "abc123",
                            "authServerMetadataUrl": "https://idp.example.com/.well-known/openid-configuration",
                            "callbackPort": 8086,
                        },
                    }
                }
            ),
        )
        monkeypatch.setenv("PLUGINS_MARKETPLACE", str(tmp_path))

        result = read_mcp_json()

        assert result is not None
        server = result.mcpServers["google-drive"]
        assert server.oauth is not None
        assert server.oauth.client_id == "abc123"
        assert (
            server.oauth.auth_server_metadata_url
            == "https://idp.example.com/.well-known/openid-configuration"
        )
        assert server.oauth.callback_port == 8086

    def test_oauth_with_explicit_endpoints(self) -> None:
        config = ChatModelConfig(
            **_make_model_config("vendor", mcp_server="vendor-api")
        )
        mcp = McpJsonConfig(
            mcpServers={
                "vendor-api": McpServerEntry(
                    url="https://vendor.example.com/mcp",
                    oauth=McpOAuthConfig(
                        client_id="vid",
                        authorization_url="https://vendor.example.com/oauth/authorize",
                        token_url="https://vendor.example.com/oauth/token",
                        client_secret="secret123",  # pragma: allowlist secret
                        scopes=["read", "write"],
                        redirect_uri="http://localhost:9090/callback",
                    ),
                )
            }
        )

        resolve_mcp_servers([config], mcp)

        tool = config.tools[0]  # type: ignore[index]
        assert tool.oauth is not None
        assert tool.oauth.client_id == "vid"
        assert (
            tool.oauth.authorization_url == "https://vendor.example.com/oauth/authorize"
        )
        assert tool.oauth.token_url == "https://vendor.example.com/oauth/token"
        assert tool.oauth.client_secret == "secret123"  # pragma: allowlist secret
        assert tool.oauth.scopes == ["read", "write"]
        assert tool.oauth.scope_string == "read write"
        assert tool.oauth.redirect_uri == "http://localhost:9090/callback"
        assert tool.oauth.auth_server_metadata_url is None
        assert tool.auth == "jwt_token"
        assert tool.auth_providers == ["mcp_oauth_vid"]

    def test_oauth_explicit_endpoints_from_json(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        _write_plugin_mcp_json(
            tmp_path,
            "test-plugin",
            _make_mcp_json(
                {
                    "vendor": {
                        "type": "http",
                        "url": "https://vendor.example.com/mcp",
                        "oauth": {
                            "authorizationUrl": "https://vendor.example.com/oauth/authorize",
                            "tokenUrl": "https://vendor.example.com/oauth/token",
                            "clientId": "vid",
                            "clientSecret": "secret",  # pragma: allowlist secret
                            "scopes": ["scope1", "scope2"],
                            "redirectUri": "http://localhost:8080/callback",
                        },
                    }
                }
            ),
        )
        monkeypatch.setenv("PLUGINS_MARKETPLACE", str(tmp_path))

        result = read_mcp_json()

        assert result is not None
        server = result.mcpServers["vendor"]
        assert server.oauth is not None
        assert server.oauth.client_id == "vid"
        assert server.oauth.client_secret == "secret"  # pragma: allowlist secret
        assert (
            server.oauth.authorization_url
            == "https://vendor.example.com/oauth/authorize"
        )
        assert server.oauth.token_url == "https://vendor.example.com/oauth/token"
        assert server.oauth.scopes == ["scope1", "scope2"]
        assert server.oauth.redirect_uri == "http://localhost:8080/callback"
        assert server.oauth.auth_server_metadata_url is None

    def test_resolves_display_name_from_mcp_json(self) -> None:
        config = ChatModelConfig(
            **_make_model_config("drive", mcp_server="google-drive")
        )
        mcp = McpJsonConfig(
            mcpServers={
                "google-drive": McpServerEntry(
                    url="https://mcp.example.com/drive/",
                    display_name="Google Drive",
                )
            }
        )

        resolve_mcp_servers([config], mcp)

        assert config.tools is not None
        tool = config.tools[0]
        assert tool.url == "https://mcp.example.com/drive/"
        assert tool.display_name == "Google Drive"

    def test_display_name_parsed_from_camel_case_json(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        _write_plugin_mcp_json(
            tmp_path,
            "test-plugin",
            _make_mcp_json(
                {
                    "google-drive": {
                        "type": "http",
                        "url": "https://mcp.example.com/drive/",
                        "displayName": "Google Drive",
                    }
                }
            ),
        )
        monkeypatch.setenv("PLUGINS_MARKETPLACE", str(tmp_path))

        result = read_mcp_json()

        assert result is not None
        server = result.mcpServers["google-drive"]
        assert server.display_name == "Google Drive"

    def test_display_name_not_set_when_absent(self) -> None:
        config = ChatModelConfig(
            **_make_model_config("drive", mcp_server="google-drive")
        )
        mcp = McpJsonConfig(
            mcpServers={
                "google-drive": McpServerEntry(
                    url="https://mcp.example.com/drive/",
                )
            }
        )

        resolve_mcp_servers([config], mcp)

        assert config.tools is not None
        tool = config.tools[0]
        assert tool.display_name is None

    def test_resolves_headers_auth_from_mcp_json(self) -> None:
        """Groundcover-style config: headers with Authorization, no oauth."""
        config = ChatModelConfig(
            **_make_model_config("groundcover", mcp_server="groundcover")
        )
        mcp = McpJsonConfig(
            mcpServers={
                "groundcover": McpServerEntry(
                    url="https://mcp.groundcover.com/api/mcp",
                    headers={
                        "Authorization": "Bearer fake-api-key",
                        "X-Backend-Id": "groundcover",
                        "X-Tenant-UUID": "6310efb6-d0c9-4751-8cec-70bc0f0a599f",
                    },
                )
            }
        )

        resolve_mcp_servers([config], mcp)

        assert config.tools is not None
        tool = config.tools[0]
        assert tool.url == "https://mcp.groundcover.com/api/mcp"
        assert tool.headers == {
            "Authorization": "Bearer fake-api-key",
            "X-Backend-Id": "groundcover",
            "X-Tenant-UUID": "6310efb6-d0c9-4751-8cec-70bc0f0a599f",
        }
        assert tool.auth == "headers"
        assert tool.oauth is None


class TestFileConfigReaderMcpJsonIntegration:
    def test_resolves_mcp_server_during_read(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _write_json(
            model_dir / "model.json",
            _make_model_config("drive", mcp_server="google-drive"),
        )

        marketplace = tmp_path / "marketplace"
        _write_plugin_mcp_json(
            marketplace,
            "all-employees",
            _make_mcp_json(
                {
                    "google-drive": {
                        "type": "http",
                        "url": "https://mcp.example.com/drive/",
                    },
                }
            ),
        )
        monkeypatch.setenv("PLUGINS_MARKETPLACE", str(marketplace))

        reader = McpJsonReader()
        configs = FileConfigReader(mcp_json_reader=reader).read_model_configs(
            config_path=str(model_dir)
        )

        assert len(configs) == 1
        assert configs[0].tools is not None
        assert configs[0].tools[0].url == "https://mcp.example.com/drive/"
        assert configs[0].tools[0].mcp_server == "google-drive"

    def test_no_mcp_json_still_works(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.delenv("PLUGINS_MARKETPLACE", raising=False)
        _write_json(
            tmp_path / "model.json",
            _make_model_config("drive", url="https://direct.example.com/"),
        )

        configs = FileConfigReader().read_model_configs(config_path=str(tmp_path))

        assert len(configs) == 1
        assert configs[0].tools is not None
        assert configs[0].tools[0].url == "https://direct.example.com/"

    def test_marketplace_plugin_mcp_json(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        _write_json(
            model_dir / "model.json",
            _make_model_config("drive", mcp_server="gd"),
        )

        marketplace = tmp_path / "marketplace"
        _write_plugin_mcp_json(
            marketplace,
            "my-plugin",
            _make_mcp_json({"gd": {"url": "https://custom.example.com/"}}),
        )
        monkeypatch.setenv("PLUGINS_MARKETPLACE", str(marketplace))

        reader = McpJsonReader()
        configs = FileConfigReader(mcp_json_reader=reader).read_model_configs(
            config_path=str(model_dir)
        )

        assert len(configs) == 1
        assert configs[0].tools is not None
        assert configs[0].tools[0].url == "https://custom.example.com/"

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_resolves_description_from_mcp_json(self) -> None:
        config = ChatModelConfig(
            **_make_model_config("drive", mcp_server="google-drive")
        )
        mcp = McpJsonConfig(
            mcpServers={
                "google-drive": McpServerEntry(
                    url="https://mcp.example.com/drive/",
                    description="Google Drive file management - search, read, and download files",
                )
            }
        )

        resolve_mcp_servers([config], mcp)

        assert config.tools is not None
        tool = config.tools[0]
        assert (
            tool.description
            == "Google Drive file management - search, read, and download files"
        )

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_mcp_json_description_overrides_model_config_description(self) -> None:
        """Description from .mcp.json takes precedence over model config."""
        config = ChatModelConfig(
            id="model-1",
            name="Model 1",
            tools=[
                AgentConfig(
                    name="drive",
                    mcp_server="google-drive",
                    description="Old description from model config",
                )
            ],
        )
        mcp = McpJsonConfig(
            mcpServers={
                "google-drive": McpServerEntry(
                    url="https://mcp.example.com/drive/",
                    description="New description from mcp.json",
                )
            }
        )

        resolve_mcp_servers([config], mcp)

        tool = config.tools[0]  # type: ignore[index]
        assert tool.description == "New description from mcp.json"

    def test_wildcard_mcp_server_expands_all_entries(self) -> None:
        config = ChatModelConfig(
            id="model-wildcard",
            name="Wildcard Model",
            tools=[AgentConfig(name="all_mcp_servers", mcp_server="*")],
        )
        mcp = McpJsonConfig(
            mcpServers={
                "google-drive": McpServerEntry(url="https://mcp.example.com/drive/"),
                "github": McpServerEntry(url="https://mcp.example.com/github/"),
            }
        )

        resolve_mcp_servers([config], mcp)

        assert config.tools is not None
        assert len(config.tools) == 2
        names = {t.name for t in config.tools}
        assert names == {"google-drive", "github"}
        assert config.tools[0].url == "https://mcp.example.com/drive/"
        assert config.tools[1].url == "https://mcp.example.com/github/"

    def test_wildcard_preserves_non_mcp_tools(self) -> None:
        config = ChatModelConfig(
            id="model-mixed",
            name="Mixed Model",
            tools=[
                AgentConfig(name="custom_tool", description="A custom tool"),
                AgentConfig(name="all_mcp", mcp_server="*"),
            ],
        )
        mcp = McpJsonConfig(
            mcpServers={
                "google-drive": McpServerEntry(url="https://mcp.example.com/drive/"),
            }
        )

        resolve_mcp_servers([config], mcp)

        assert config.tools is not None
        assert len(config.tools) == 2
        assert config.tools[0].name == "custom_tool"
        assert config.tools[0].mcp_server is None
        assert config.tools[1].name == "google-drive"
        assert config.tools[1].url == "https://mcp.example.com/drive/"
