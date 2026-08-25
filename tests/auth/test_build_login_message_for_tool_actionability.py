"""Tests that build_login_message_for_tool only produces a "please log in"
message when a real login step exists for the tool.

Regression coverage for the bug where a pass-through-only MCP tool (no
``oauth``/app-login/token-save config, e.g. the skills-library catalog)
would surface "I found a tool X ... This tool requires you to log in
below." with no link actually rendered below it — a dead end for the
user, who has no way to act on it.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from oidcauthlib.auth.auth_manager import AuthManager
from oidcauthlib.auth.config.auth_config import AuthConfig
from oidcauthlib.auth.config.auth_config_reader import AuthConfigReader
from oidcauthlib.auth.models.auth import AuthInformation

from languagemodelcommon.auth.pass_through_token_manager import (
    PassThroughTokenManager,
)
from languagemodelcommon.configs.schemas.config_schema import (
    AgentConfig,
    McpOAuthConfig,
)


def _make_manager(
    *, auth_config_for_provider: AuthConfig | None = None
) -> PassThroughTokenManager:
    """Create a PassThroughTokenManager with mocked dependencies."""
    mock_auth_config_reader = MagicMock(spec=AuthConfigReader)
    mock_auth_config_reader.get_config_for_auth_provider.return_value = (
        auth_config_for_provider
    )

    manager = object.__new__(PassThroughTokenManager)
    manager.auth_config_reader = mock_auth_config_reader
    manager.auth_manager = MagicMock(spec=AuthManager)
    manager.auth_manager.create_authorization_url = AsyncMock(
        return_value="https://auth.example.com/authorize?state=abc"
    )
    manager.tool_auth_manager = MagicMock()
    manager.environment_variables = MagicMock()
    manager.environment_variables.app_login_uri = None
    manager.environment_variables.app_token_save_uri = None
    return manager


@pytest.mark.asyncio
async def test_pass_through_only_tool_returns_none() -> None:
    """A tool with no oauth/app-login/token-save config has no login step —
    build_login_message_for_tool must return None, not a bare "log in
    below" preamble with nothing to click."""
    manager = _make_manager()
    tool_config = AgentConfig(
        name="skills-library",
        display_name="Skills Library",
        url="https://gateway.example.com/skills-library/",
        auth="None",
    )

    message = await manager.build_login_message_for_tool(
        auth_information=AuthInformation(redirect_uri=""),
        authentication_config=tool_config,
    )

    assert message is None


@pytest.mark.asyncio
async def test_oauth_configured_tool_returns_actionable_message() -> None:
    """A tool with a real oauth config still gets the login message with an
    actual clickable link — this must keep working."""
    manager = _make_manager(
        auth_config_for_provider=AuthConfig(
            auth_provider="oauth_provider_1",
            friendly_name="Google Drive",
            audience="client-1",
            scope="openid",
            client_id="client-1",
        )
    )
    tool_config = AgentConfig(
        name="google-drive",
        display_name="Google Drive",
        url="https://gateway.example.com/google-drive/",
        auth="jwt_token",
        auth_providers=["oauth_provider_1"],
        oauth=McpOAuthConfig(client_id="client-1"),
    )

    message = await manager.build_login_message_for_tool(
        auth_information=AuthInformation(
            redirect_uri="https://app.example.com/callback",
            email="user@example.com",
            subject="user-123",
        ),
        authentication_config=tool_config,
        tool_auth_provider="oauth_provider_1",
    )

    assert message is not None
    assert "Google Drive" in message
    assert "https://auth.example.com/authorize?state=abc" in message
