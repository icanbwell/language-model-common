from languagemodelcommon.configs.schemas.config_schema import (
    McpOAuthConfig,
    McpOAuthClientMetadata,
)


class TestMcpOAuthClientMetadata:
    def test_create_with_all_fields(self) -> None:
        metadata = McpOAuthClientMetadata(
            client_name="My MCP Client",
            client_uri="https://myapp.com",
            logo_uri="https://myapp.com/logo.png",
            contacts=["admin@myapp.com"],
        )
        assert metadata.client_name == "My MCP Client"
        assert metadata.client_uri == "https://myapp.com"

    def test_create_empty(self) -> None:
        metadata = McpOAuthClientMetadata()
        assert metadata.client_name is None


class TestMcpOAuthConfigNewFields:
    def test_client_id_optional(self) -> None:
        config = McpOAuthConfig(
            registration_url="https://auth.example.com/register",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
        )
        assert config.client_id is None
        assert config.registration_url == "https://auth.example.com/register"

    def test_client_id_provided(self) -> None:
        config = McpOAuthConfig(client_id="my-client")
        assert config.client_id == "my-client"

    def test_pkce_defaults(self) -> None:
        config = McpOAuthConfig()
        assert config.use_pkce is True
        assert config.pkce_method == "S256"

    def test_pkce_disabled(self) -> None:
        config = McpOAuthConfig.model_validate({"usePKCE": False})
        assert config.use_pkce is False

    def test_pkce_plain(self) -> None:
        config = McpOAuthConfig.model_validate({"pkceMethod": "plain"})
        assert config.pkce_method == "plain"

    def test_registration_url_alias(self) -> None:
        config = McpOAuthConfig.model_validate(
            {"registrationUrl": "https://auth.example.com/register"}
        )
        assert config.registration_url == "https://auth.example.com/register"

    def test_client_metadata(self) -> None:
        config = McpOAuthConfig.model_validate(
            {
                "clientMetadata": {
                    "client_name": "My Client",
                    "contacts": ["a@b.com"],
                }
            }
        )
        assert config.client_metadata is not None
        assert config.client_metadata.client_name == "My Client"

    def test_full_oauth21_config(self) -> None:
        config = McpOAuthConfig.model_validate(
            {
                "registrationUrl": "https://auth.example.com/register",
                "authorizationUrl": "https://auth.example.com/authorize",
                "tokenUrl": "https://auth.example.com/token",
                "scopes": ["mcp:read", "mcp:write"],
                "usePKCE": True,
                "pkceMethod": "S256",
                "clientMetadata": {
                    "client_name": "My MCP Client",
                    "client_uri": "https://myapp.com",
                },
            }
        )
        assert config.client_id is None
        assert config.scope_string == "mcp:read mcp:write"
        assert config.client_metadata is not None
        assert config.client_metadata.client_name == "My MCP Client"
