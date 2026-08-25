"""Tests that _list_mcp_tools_for_config degrades gracefully (returns []
rather than raising) when an MCP tool's auth failure has no actionable
login step.

Regression coverage for the bug where a pass-through-only tool (e.g. the
skills-library catalog, which has no ``oauth`` config) surfaced a "This
tool requires you to log in below." message with no login link ever
rendered below it. This is the code path the LLM actually hits when it
calls ``search_tools``/``list_skills`` (see ``prompts/skills.md`` in
baileyai) and the resolver lazily discovers the skills-library server's
tools.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from languagemodelcommon.auth.exceptions.authorization_mcp_tool_token_invalid_exception import (
    AuthorizationMcpToolTokenInvalidException,
)
from languagemodelcommon.configs.schemas.config_schema import AgentConfig
from languagemodelcommon.mcp.interceptors.auth import AuthMcpCallInterceptor
from languagemodelcommon.mcp.mcp_client.server_card_discovery import (
    ServerCardDiscovery,
)
from languagemodelcommon.mcp.mcp_client.tool_list_cache import ToolListCache
from languagemodelcommon.mcp.mcp_tool_provider import MCPToolProvider


def _make_401_exception_group() -> BaseExceptionGroup:
    response = httpx.Response(
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
        request=httpx.Request("POST", "https://mcp.example.com/skills-library/"),
    )
    http_error = httpx.HTTPStatusError(
        "401 Unauthorized", request=response.request, response=response
    )
    return BaseExceptionGroup("mcp errors", [http_error])


def _make_provider() -> MCPToolProvider:
    """Create an MCPToolProvider whose (mocked) server-card lookup misses
    and whose MCP session raises a 401, mirroring
    test_mcp_tool_provider_auth_discovery.py's ``_make_provider`` fixture."""
    provider = object.__new__(MCPToolProvider)
    provider.environment_variables = MagicMock()
    provider.environment_variables.tool_call_timeout_seconds = 30
    provider._default_headers = {}
    provider.pass_through_token_manager = MagicMock()
    provider.tool_list_cache = ToolListCache(ttl_seconds=300.0)

    mock_discovery = MagicMock()
    mock_discovery.discover = AsyncMock(return_value=None)  # no OAuth discoverable
    provider.auth_server_metadata_discovery = mock_discovery

    mock_server_card = MagicMock(spec=ServerCardDiscovery)
    mock_server_card.fetch_tools_from_server_card = AsyncMock(return_value=None)
    provider._server_card_discovery = mock_server_card
    return provider


def _make_tool_config() -> AgentConfig:
    return AgentConfig(
        name="skills-library",
        display_name="Skills Library",
        url="https://mcp.example.com/skills-library/",
        auth="None",  # pass-through, no oauth block
    )


@pytest.mark.asyncio
async def test_no_actionable_login_returns_empty_list_not_dead_end_prompt() -> None:
    """When there's no login link to offer, skip the tool instead of
    telling the user to log in "below" with nothing rendered below."""
    provider = _make_provider()
    auth_interceptor = MagicMock(spec=AuthMcpCallInterceptor)
    auth_interceptor.build_login_message_for_tool = AsyncMock(return_value=None)

    with patch(
        "languagemodelcommon.mcp.mcp_tool_provider.create_mcp_session",
    ) as mock_session_ctx:
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(side_effect=_make_401_exception_group())
        mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await provider._list_mcp_tools_for_config(
            tool_config=_make_tool_config(),
            headers={},
            auth_interceptor=auth_interceptor,
        )

    assert result == []


@pytest.mark.asyncio
async def test_actionable_login_still_raises_with_message() -> None:
    """When a real login link IS available, still raise so the user sees
    it — this must keep working."""
    provider = _make_provider()
    auth_interceptor = MagicMock(spec=AuthMcpCallInterceptor)
    auth_interceptor.build_login_message_for_tool = AsyncMock(
        return_value="I found a tool **Skills Library**...\n"
        "Click here to [Login to Skills Library](https://auth.example.com/authorize)."
    )

    with patch(
        "languagemodelcommon.mcp.mcp_tool_provider.create_mcp_session",
    ) as mock_session_ctx:
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(side_effect=_make_401_exception_group())
        mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

        with pytest.raises(AuthorizationMcpToolTokenInvalidException) as exc_info:
            await provider._list_mcp_tools_for_config(
                tool_config=_make_tool_config(),
                headers={},
                auth_interceptor=auth_interceptor,
            )

    assert "Login to Skills Library" in exc_info.value.message
