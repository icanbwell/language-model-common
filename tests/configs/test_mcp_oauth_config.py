from languagemodelcommon.configs.schemas.config_schema import (
    McpOAuthConfig,
    McpOAuthClientMetadata,
)


class TestMcpOAuthClientMetadata:
    def test_create_with_all_fields(self) -> None:
        metadata = McpOAuthClientMetadata(
            client_name="My MCP Client",
            redirect_uris=["http://localhost:8080/callback"],
            grant_types=["authorization_code"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            client_uri="https://myapp.com",
            logo_uri="https://myapp.com/logo.png",
            contacts=["admin@myapp.com"],
        )
        assert metadata.client_name == "My MCP Client"
        assert metadata.redirect_uris == ["http://localhost:8080/callback"]
        assert metadata.grant_types == ["authorization_code"]
        assert metadata.response_types == ["code"]
        assert metadata.token_endpoint_auth_method == "none"
        assert metadata.client_uri == "https://myapp.com"

    def test_create_empty(self) -> None:
        metadata = McpOAuthClientMetadata()
        assert metadata.client_name is None
        assert metadata.redirect_uris is None
        assert metadata.grant_types is None

    def test_from_camel_case_json(self) -> None:
        metadata = McpOAuthClientMetadata.model_validate(
            {
                "clientName": "Test",
                "redirectUris": ["http://localhost/cb"],
                "grantTypes": ["authorization_code"],
                "responseTypes": ["code"],
                "tokenEndpointAuthMethod": "none",
            }
        )
        assert metadata.client_name == "Test"
        assert metadata.redirect_uris == ["http://localhost/cb"]
        assert metadata.token_endpoint_auth_method == "none"


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
                    "clientName": "My Client",
                    "contacts": ["a@b.com"],
                }
            }
        )
        assert config.client_metadata is not None
        assert config.client_metadata.client_name == "My Client"

    def test_is_dcr_true_when_no_client_id_with_registration_url(self) -> None:
        config = McpOAuthConfig(
            registration_url="https://auth.example.com/register",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
        )
        assert config.is_dcr is True

    def test_is_dcr_false_when_client_id_present(self) -> None:
        config = McpOAuthConfig(client_id="my-client")
        assert config.is_dcr is False

    def test_is_dcr_false_when_no_registration_url(self) -> None:
        config = McpOAuthConfig(
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
        )
        assert config.is_dcr is False

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
                    "clientName": "My MCP Client",
                    "clientUri": "https://myapp.com",
                    "redirectUris": ["http://localhost:8080/callback"],
                    "grantTypes": ["authorization_code"],
                    "responseTypes": ["code"],
                    "tokenEndpointAuthMethod": "none",
                },
            }
        )
        assert config.client_id is None
        assert config.is_dcr is True
        assert config.scope_string == "mcp:read mcp:write"
        assert config.client_metadata is not None
        assert config.client_metadata.client_name == "My MCP Client"
        assert config.client_metadata.redirect_uris == [
            "http://localhost:8080/callback"
        ]
        assert config.client_metadata.grant_types == ["authorization_code"]
        assert config.client_metadata.token_endpoint_auth_method == "none"
