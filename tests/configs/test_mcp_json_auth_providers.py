"""Tests for ``build_auth_configs_from_mcp_json`` in ``mcp_json_reader``."""

from oidcauthlib.auth.config.auth_config import AuthConfig

from languagemodelcommon.configs.config_reader.mcp_json_reader import (
    McpJsonConfig,
    build_auth_configs_from_mcp_json,
)


def _make_config(auth_providers: dict[str, dict[str, object]]) -> McpJsonConfig:
    return McpJsonConfig.model_validate({"authProviders": auth_providers})


class TestBuildAuthConfigsFromMcpJson:
    """Tests for build_auth_configs_from_mcp_json."""

    def test_basic_parsing(self) -> None:
        """Parse a single auth provider with core fields."""
        mcp = _make_config(
            {
                "my-idp": {
                    "issuer": "https://idp.example.com",
                    "audience": "https://api.example.com",
                    "clientId": "client-123",
                    "wellKnownUri": "https://idp.example.com/.well-known/openid-configuration",
                    "scope": "openid profile",
                    "friendlyName": "My IdP",
                }
            }
        )

        result = build_auth_configs_from_mcp_json(mcp)

        assert len(result) == 1
        cfg = result[0]
        assert isinstance(cfg, AuthConfig)
        assert cfg.auth_provider == "my-idp"
        assert cfg.friendly_name == "My IdP"
        assert cfg.audience == "https://api.example.com"
        assert cfg.issuer == "https://idp.example.com"
        assert cfg.client_id == "client-123"
        assert (
            cfg.well_known_uri
            == "https://idp.example.com/.well-known/openid-configuration"
        )
        assert cfg.scope == "openid profile"

    def test_with_client_secret_and_extra_info(self) -> None:
        """Parse provider that includes clientSecret and extraInfo."""
        mcp = _make_config(
            {
                "secure-idp": {
                    "audience": "https://api.example.com",
                    "clientId": "client-456",
                    "clientSecret": "super-secret",  # pragma: allowlist secret
                    "extraInfo": {"tenant": "abc", "region": "us-east-1"},
                }
            }
        )

        result = build_auth_configs_from_mcp_json(mcp)

        assert len(result) == 1
        cfg = result[0]
        assert cfg.client_secret == "super-secret"  # pragma: allowlist secret
        assert cfg.extra_info == {"tenant": "abc", "region": "us-east-1"}

    def test_defaults_friendly_name_and_scope(self) -> None:
        """friendlyName defaults to key name, scope defaults to 'openid profile email'."""
        mcp = _make_config(
            {
                "default-provider": {
                    "audience": "https://api.example.com",
                    "clientId": "client-789",
                }
            }
        )

        result = build_auth_configs_from_mcp_json(mcp)

        assert len(result) == 1
        cfg = result[0]
        assert cfg.friendly_name == "default-provider"
        assert cfg.scope == "openid profile email"

    def test_empty_auth_providers_returns_empty_list(self) -> None:
        """An empty authProviders dict yields an empty list."""
        mcp = _make_config({})

        result = build_auth_configs_from_mcp_json(mcp)

        assert result == []

    def test_missing_auth_providers_returns_empty_list(self) -> None:
        """McpJsonConfig with no authProviders (default) yields an empty list."""
        mcp = McpJsonConfig()

        result = build_auth_configs_from_mcp_json(mcp)

        assert result == []

    def test_multiple_providers(self) -> None:
        """Multiple auth providers are all parsed correctly."""
        mcp = _make_config(
            {
                "idp-alpha": {
                    "audience": "https://alpha.example.com",
                    "clientId": "alpha-client",
                    "scope": "openid",
                    "friendlyName": "Alpha IdP",
                },
                "idp-beta": {
                    "audience": "https://beta.example.com",
                    "clientId": "beta-client",
                    "friendlyName": "Beta IdP",
                    "issuer": "https://beta-issuer.example.com",
                },
            }
        )

        result = build_auth_configs_from_mcp_json(mcp)

        assert len(result) == 2
        by_name = {c.auth_provider: c for c in result}

        alpha = by_name["idp-alpha"]
        assert alpha.audience == "https://alpha.example.com"
        assert alpha.client_id == "alpha-client"
        assert alpha.scope == "openid"
        assert alpha.friendly_name == "Alpha IdP"

        beta = by_name["idp-beta"]
        assert beta.audience == "https://beta.example.com"
        assert beta.client_id == "beta-client"
        assert beta.issuer == "https://beta-issuer.example.com"
        assert beta.friendly_name == "Beta IdP"
        assert beta.scope == "openid profile email"  # default

    def test_authorization_and_token_endpoints(self) -> None:
        """Parse authorizationEndpoint and tokenEndpoint fields."""
        mcp = _make_config(
            {
                "explicit-endpoints": {
                    "audience": "https://api.example.com",
                    "clientId": "client-ep",
                    "authorizationEndpoint": "https://idp.example.com/authorize",
                    "tokenEndpoint": "https://idp.example.com/token",
                }
            }
        )

        result = build_auth_configs_from_mcp_json(mcp)

        assert len(result) == 1
        cfg = result[0]
        assert cfg.authorization_endpoint == "https://idp.example.com/authorize"
        assert cfg.token_endpoint == "https://idp.example.com/token"

    def test_registration_url(self) -> None:
        """Parse registrationUrl for DCR-based providers (no clientId needed)."""
        mcp = _make_config(
            {
                "dcr-provider": {
                    "audience": "https://api.example.com",
                    "registrationUrl": "https://idp.example.com/register",
                }
            }
        )

        result = build_auth_configs_from_mcp_json(mcp)

        assert len(result) == 1
        cfg = result[0]
        assert cfg.registration_url == "https://idp.example.com/register"
        assert cfg.client_id is None
