"""Tests for _ensure_oauth_provider_registered discovery fallback.

When an McpOAuthConfig has neither client_id nor registration_url,
OAuthProviderRegistrar should attempt RFC 8414 / OIDC Discovery
from the server URL to discover the registration endpoint before
calling DcrManager.
"""

from typing import Any

from unittest.mock import AsyncMock, MagicMock

import pytest

from languagemodelcommon.auth.oauth_provider_registrar import OAuthProviderRegistrar
from languagemodelcommon.auth.pass_through_token_manager import (
    PassThroughTokenManager,
)
from languagemodelcommon.configs.schemas.config_schema import McpOAuthConfig
from oidcauthlib.auth.auth_manager import AuthManager
from oidcauthlib.auth.config.auth_config_reader import AuthConfigReader
from oidcauthlib.auth.dcr.dcr_manager import DcrManager


def _make_manager(
    *,
    discovery_result: McpOAuthConfig | None = None,
    dcr_result: dict[str, Any] | None = None,
    existing_provider: bool = False,
    discovery_instance: Any = "default",
) -> PassThroughTokenManager:
    """Create a PassThroughTokenManager with mocked dependencies."""
    manager = object.__new__(PassThroughTokenManager)

    mock_auth_config_reader = MagicMock(spec=AuthConfigReader)
    if existing_provider:
        mock_config = MagicMock()
        mock_config.client_id = "existing-id"
        mock_auth_config_reader.get_config_for_auth_provider.return_value = mock_config
    else:
        mock_auth_config_reader.get_config_for_auth_provider.return_value = None
        mock_auth_config_reader.get_auth_configs_for_all_auth_providers.return_value = []

    manager.auth_config_reader = mock_auth_config_reader
    manager.auth_manager = MagicMock(spec=AuthManager)
    manager.auth_manager.register_dynamic_provider = AsyncMock()
    manager.tool_auth_manager = MagicMock()
    manager.environment_variables = MagicMock()

    mock_dcr = MagicMock(spec=DcrManager)
    if dcr_result is not None:
        dcr_registration = MagicMock()
        dcr_registration.client_id = dcr_result["client_id"]
        dcr_registration.client_secret = dcr_result.get("client_secret")
        mock_dcr.resolve_dcr_credentials = AsyncMock(return_value=dcr_registration)
    else:
        mock_dcr.resolve_dcr_credentials = AsyncMock(return_value=None)

    if discovery_instance == "default":
        mock_discovery = MagicMock()
        mock_discovery.discover = AsyncMock(return_value=discovery_result)
    else:
        mock_discovery = discovery_instance

    registrar = OAuthProviderRegistrar(
        dcr_manager=mock_dcr,
        auth_config_reader=mock_auth_config_reader,
        auth_server_metadata_discovery=mock_discovery,
    )
    manager._oauth_provider_registrar = registrar

    # Expose mocks for test assertions
    manager.dcr_manager = mock_dcr  # type: ignore[attr-defined]
    manager.auth_server_metadata_discovery = mock_discovery  # type: ignore[attr-defined]

    return manager


@pytest.mark.asyncio
async def test_discovery_populates_registration_url() -> None:
    """When oauth has no client_id or registration_url, discovery fills them in."""
    discovered = McpOAuthConfig.model_validate(
        {
            "authorizationUrl": "https://auth.atlassian.com/authorize",
            "tokenUrl": "https://auth.atlassian.com/token",
            "registrationUrl": "https://auth.atlassian.com/register",
            "issuer": "https://auth.atlassian.com",
            "scopes": ["read:jira-work"],
        }
    )
    manager = _make_manager(
        discovery_result=discovered,
        dcr_result={"client_id": "dcr-client-123", "client_secret": "dcr-secret"},
    )

    oauth = McpOAuthConfig.model_validate(
        {
            "clientMetadata": {
                "clientName": "b.well Aiden",
                "clientUri": "https://www.icanbwell.com",
            }
        }
    )

    await manager._ensure_oauth_provider_registered(
        auth_provider="atlassian",
        oauth=oauth,
        server_url="https://mcp.atlassian.com/v1/mcp",
    )

    # Discovery was called with the server URL
    manager.auth_server_metadata_discovery.discover.assert_awaited_once_with(  # type: ignore[union-attr]
        mcp_server_url="https://mcp.atlassian.com/v1/mcp"
    )

    # OAuth config was populated from discovery
    assert oauth.registration_url == "https://auth.atlassian.com/register"
    assert oauth.authorization_url == "https://auth.atlassian.com/authorize"
    assert oauth.token_url == "https://auth.atlassian.com/token"
    assert oauth.issuer == "https://auth.atlassian.com"

    # DCR was called with the discovered registration_url
    manager.dcr_manager.resolve_dcr_credentials.assert_awaited_once()  # type: ignore[attr-defined]
    call_kwargs = manager.dcr_manager.resolve_dcr_credentials.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["registration_url"] == "https://auth.atlassian.com/register"


@pytest.mark.asyncio
async def test_discovery_skipped_when_client_id_present() -> None:
    """Discovery is not attempted when client_id is already configured."""
    manager = _make_manager(discovery_result=None)

    oauth = McpOAuthConfig.model_validate(
        {
            "clientId": "pre-registered-id",
            "authorizationUrl": "https://auth.example.com/authorize",
            "tokenUrl": "https://auth.example.com/token",
        }
    )

    await manager._ensure_oauth_provider_registered(
        auth_provider="test",
        oauth=oauth,
        server_url="https://mcp.example.com/v1/mcp",
    )

    manager.auth_server_metadata_discovery.discover.assert_not_awaited()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_discovery_skipped_when_registration_url_present() -> None:
    """Discovery is not attempted when registration_url is already configured."""
    manager = _make_manager(
        dcr_result={"client_id": "dcr-id", "client_secret": "secret"},
    )

    oauth = McpOAuthConfig.model_validate(
        {
            "registrationUrl": "https://auth.example.com/register",
            "authorizationUrl": "https://auth.example.com/authorize",
            "tokenUrl": "https://auth.example.com/token",
        }
    )

    await manager._ensure_oauth_provider_registered(
        auth_provider="test",
        oauth=oauth,
        server_url="https://mcp.example.com/v1/mcp",
    )

    manager.auth_server_metadata_discovery.discover.assert_not_awaited()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_discovery_skipped_when_no_server_url() -> None:
    """Discovery is not attempted when server_url is not provided."""
    manager = _make_manager(discovery_result=None)

    oauth = McpOAuthConfig.model_validate(
        {
            "clientMetadata": {"clientName": "test"},
        }
    )

    with pytest.raises(ValueError, match="Could not resolve client_id"):
        await manager._ensure_oauth_provider_registered(
            auth_provider="test",
            oauth=oauth,
        )

    manager.auth_server_metadata_discovery.discover.assert_not_awaited()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_discovery_skipped_when_no_discovery_instance() -> None:
    """Discovery is not attempted when auth_server_metadata_discovery is None."""
    manager = _make_manager(discovery_instance=None)

    oauth = McpOAuthConfig.model_validate(
        {
            "clientMetadata": {"clientName": "test"},
        }
    )

    with pytest.raises(ValueError, match="Could not resolve client_id"):
        await manager._ensure_oauth_provider_registered(
            auth_provider="test",
            oauth=oauth,
            server_url="https://mcp.example.com/v1/mcp",
        )


@pytest.mark.asyncio
async def test_discovery_returns_none_falls_through_to_error() -> None:
    """When discovery returns None, registration still fails for missing credentials."""
    manager = _make_manager(discovery_result=None)

    oauth = McpOAuthConfig.model_validate(
        {
            "clientMetadata": {"clientName": "test"},
        }
    )

    with pytest.raises(ValueError, match="Could not resolve client_id"):
        await manager._ensure_oauth_provider_registered(
            auth_provider="test",
            oauth=oauth,
            server_url="https://mcp.example.com/v1/mcp",
        )

    manager.auth_server_metadata_discovery.discover.assert_awaited_once()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_discovery_does_not_overwrite_explicit_fields() -> None:
    """Discovery does not overwrite fields already set on the oauth config."""
    discovered = McpOAuthConfig.model_validate(
        {
            "authorizationUrl": "https://discovered.example.com/authorize",
            "tokenUrl": "https://discovered.example.com/token",
            "registrationUrl": "https://discovered.example.com/register",
            "issuer": "https://discovered.example.com",
            "scopes": ["discovered-scope"],
        }
    )
    manager = _make_manager(
        discovery_result=discovered,
        dcr_result={"client_id": "dcr-id"},
    )

    oauth = McpOAuthConfig.model_validate(
        {
            "authorizationUrl": "https://explicit.example.com/authorize",
            "tokenUrl": "https://explicit.example.com/token",
            "issuer": "https://explicit.example.com",
            "scopes": ["explicit-scope"],
        }
    )

    await manager._ensure_oauth_provider_registered(
        auth_provider="test",
        oauth=oauth,
        server_url="https://mcp.example.com/v1/mcp",
    )

    # registration_url was discovered (was None)
    assert oauth.registration_url == "https://discovered.example.com/register"
    # Explicit fields were NOT overwritten
    assert oauth.authorization_url == "https://explicit.example.com/authorize"
    assert oauth.token_url == "https://explicit.example.com/token"
    assert oauth.issuer == "https://explicit.example.com"
    assert oauth.scopes == ["explicit-scope"]
