import json
from pathlib import Path
from typing import Any


from languagemodelcommon.configs.config_reader.file_config_reader import (
    FileConfigReader,
)
from languagemodelcommon.configs.config_reader.mcp_json_reader import (
    MCP_JSON_PATH_ENV,
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
                    "oauth": {
                        "clientId": "abc123",
                        "authServerMetadataUrl": "https://idp.example.com/.well-known/openid-configuration",
                    },
                    "customField": "custom_value",
                }
            }
        )
        _write_json(tmp_path / ".mcp.json", mcp_data)

        result = read_mcp_json(config_dir=str(tmp_path))
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

    def test_agent_auth_takes_precedence_over_mcp_json(self) -> None:
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
        assert tool.auth == "oauth"
        assert tool.headers == {"X-Custom": "mine"}

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

    def test_explicit_auth_not_overridden_by_oauth(self) -> None:
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
        assert tool.auth == "oauth"
        assert tool.auth_providers == ["custom-provider"]

    def test_oauth_parsed_from_camel_case_json(self, tmp_path: Path) -> None:
        _write_json(
            tmp_path / ".mcp.json",
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

        result = read_mcp_json(config_dir=str(tmp_path))

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

    def test_oauth_explicit_endpoints_from_json(self, tmp_path: Path) -> None:
        _write_json(
            tmp_path / ".mcp.json",
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

        result = read_mcp_json(config_dir=str(tmp_path))

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
