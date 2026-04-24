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
    the ``get_mcp_servers_config`` tool, which returns ``.mcp.json``
    content keyed by plugin name.
    """

    def __init__(self, *, plugins_mcp_server_url: str) -> None:
        self._url = plugins_mcp_server_url

    async def fetch_async(self) -> dict[str, McpJsonConfig] | None:
        """Call the remote tool and return per-plugin ``McpJsonConfig``.

        Returns a dict mapping plugin name to its ``McpJsonConfig``,
        or ``None`` if the fetch fails or returns empty data.
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
                result = await session.call_tool(TOOL_NAME, {})
        except Exception:
            logger.exception("Failed to fetch MCP server config from %s", self._url)
            return None

        # Extract text content from the tool result
        text_parts = [
            block.text for block in (result.content or []) if hasattr(block, "text")
        ]
        if not text_parts:
            logger.warning(
                "get_mcp_servers_config returned no text content from %s",
                self._url,
            )
            return None

        raw_json = text_parts[0]
        try:
            data: Any = substitute_env_vars(json.loads(raw_json))
        except (json.JSONDecodeError, TypeError):
            logger.exception("Failed to parse MCP config JSON from %s", self._url)
            return None

        plugins_data = data.get("plugins", {})
        if not isinstance(plugins_data, dict) or not plugins_data:
            return None

        result_map: dict[str, McpJsonConfig] = {}
        for plugin_name, plugin_data in plugins_data.items():
            servers = plugin_data.get("mcpServers", {})
            if isinstance(servers, dict) and servers:
                result_map[plugin_name] = McpJsonConfig(mcpServers=servers)

        if not result_map:
            return None

        total_servers = sum(len(c.mcpServers) for c in result_map.values())
        logger.info(
            "Fetched %d MCP server definition(s) from %d plugin(s) via %s",
            total_servers,
            len(result_map),
            self._url,
        )
        return result_map
