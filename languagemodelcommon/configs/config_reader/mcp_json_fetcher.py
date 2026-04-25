"""Fetch per-plugin MCP server config via the ``get_mcp_servers_config`` tool."""

import json
import logging
from typing import Any

from languagemodelcommon.configs.schemas.mcp_json_schema import McpJsonConfig
from languagemodelcommon.mcp.mcp_client.session import (
    MCPConnectionConfig,
    create_mcp_session,
)
from languagemodelcommon.utilities.config_substitution import substitute_env_vars
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)

TOOL_NAME = "get_mcp_servers_config"


class McpJsonFetcher:
    """Fetches per-plugin MCP server definitions by calling an MCP tool.

    Connects to the MCP server at *plugins_mcp_server_url* and calls
    the ``get_mcp_servers_config`` tool once per plugin, returning
    the ``mcpServers`` dict for that plugin.
    """

    def __init__(self, *, plugins_mcp_server_url: str) -> None:
        self._url = plugins_mcp_server_url

    async def fetch_plugin_async(self, plugin_name: str) -> McpJsonConfig | None:
        """Fetch MCP server config for a single plugin.

        Calls the ``get_mcp_servers_config`` tool with the given
        *plugin_name* and returns the resulting ``McpJsonConfig``,
        or ``None`` if the fetch fails or returns no servers.
        Environment variable substitution is applied so placeholders
        like ``${MCP_SERVER_GATEWAY_URL}`` are resolved in the caller's
        environment.
        """
        config: MCPConnectionConfig = {
            "url": self._url,
            "transport": "streamable_http",
        }
        try:
            async with create_mcp_session(config) as session:
                await session.initialize()
                result = await session.call_tool(
                    TOOL_NAME, {"plugin_name": plugin_name}
                )
        except Exception:
            logger.exception(
                "Failed to fetch MCP config for plugin '%s' from %s",
                plugin_name,
                self._url,
            )
            return None

        # Extract text content from the tool result
        text_parts = [
            block.text for block in (result.content or []) if hasattr(block, "text")
        ]
        if not text_parts:
            logger.warning(
                "get_mcp_servers_config returned no text for plugin '%s' from %s",
                plugin_name,
                self._url,
            )
            return None

        raw_json = text_parts[0]
        try:
            data: Any = substitute_env_vars(json.loads(raw_json))
        except (json.JSONDecodeError, TypeError):
            logger.exception(
                "Failed to parse MCP config JSON for plugin '%s' from %s",
                plugin_name,
                self._url,
            )
            return None

        servers = data.get("mcpServers", {})
        if not isinstance(servers, dict) or not servers:
            return None

        logger.info(
            "Fetched %d MCP server definition(s) for plugin '%s' via %s",
            len(servers),
            plugin_name,
            self._url,
        )
        return McpJsonConfig(mcpServers=servers)

    async def fetch_plugins_async(
        self, plugin_names: list[str]
    ) -> dict[str, McpJsonConfig]:
        """Fetch MCP server config for multiple plugins.

        Calls ``fetch_plugin_async`` for each name and returns a dict
        mapping plugin name to its ``McpJsonConfig``.  Plugins that
        fail to fetch or return no servers are omitted from the result.
        """
        result: dict[str, McpJsonConfig] = {}
        for name in plugin_names:
            mcp_config = await self.fetch_plugin_async(name)
            if mcp_config:
                result[name] = mcp_config
        return result
