from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from languagemodelcommon.configs.schemas.config_schema import AgentConfig


@runtime_checkable
class PluginMcpConfigProvider(Protocol):
    """Provides MCP server configurations discovered from marketplace plugins.

    Implementations bridge the skill loader's plugin MCP discovery
    (which refreshes on its own TTL) into the ``AgentConfig`` format
    that ``MCPToolProvider`` already understands.

    This is called per-request in the chat completions provider so
    that newly discovered plugin MCP servers appear without worker
    restart.
    """

    def get_mcp_server_configs(self) -> Sequence[AgentConfig]:
        """Return MCP server configs from all enabled marketplace plugins.

        Each returned AgentConfig should have at minimum:
        - ``name``: namespaced as ``plugin_name__server_key``
        - ``url``: the HTTP endpoint for the MCP server

        Entries without a ``url`` (stdio-only servers) should be excluded
        since the gateway cannot manage subprocess-based MCP servers.
        """
        ...
