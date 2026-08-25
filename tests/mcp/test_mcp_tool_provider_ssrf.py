"""Tests that MCPToolProvider._build_connection_config rejects tool URLs
pointing at internal/loopback/metadata hosts (SSRF).

Regression test for a Gecko finding: the config-driven MCP tool path (e.g.
the FHIR MCP tool registered via .mcp.json with url="${MCP_FHIR_URL}") built
its connection straight from tool_config.url with no validation, unlike the
already-validated per-request MCP tool URL path in baileyai's
AgentServiceFactory._resolve_request_tools.
"""

from unittest.mock import MagicMock

import pytest

from languagemodelcommon.configs.schemas.config_schema import AgentConfig
from languagemodelcommon.mcp.mcp_tool_provider import MCPToolProvider


def _make_provider() -> MCPToolProvider:
    """Create an MCPToolProvider with just enough state for
    _build_connection_config (a pure/sync method) to run."""
    provider = object.__new__(MCPToolProvider)
    provider.environment_variables = MagicMock()
    provider.environment_variables.tool_call_timeout_seconds = 30
    provider._default_headers = {}
    provider.get_httpx_async_client = MagicMock()
    return provider


@pytest.mark.parametrize(
    "blocked_url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://127.0.0.1:8080/",  # loopback
        "http://localhost:8080/",  # localhost by name
        "http://10.0.0.5/internal",  # private range
        "http://0x7f000001/",  # obfuscated hex loopback
        "ftp://mcp.example.com/",  # disallowed scheme
    ],
)
def test_build_connection_config_rejects_blocked_hosts(blocked_url: str) -> None:
    provider = _make_provider()
    tool_config = AgentConfig(name="fhir-server", url=blocked_url)

    with pytest.raises(ValueError, match="SSRF"):
        provider._build_connection_config(tool_config)


def test_build_connection_config_allows_ordinary_https_host() -> None:
    provider = _make_provider()
    tool_config = AgentConfig(
        name="fhir-server", url="https://mcpfhiragent.dev.icanbwell.com/fhir/"
    )

    config = provider._build_connection_config(tool_config)

    assert config["url"] == "https://mcpfhiragent.dev.icanbwell.com/fhir/"
