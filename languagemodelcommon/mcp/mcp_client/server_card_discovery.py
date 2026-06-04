"""MCP Server Card discovery via .well-known/mcp/server-card.json (SEP-1649).

Fetches static tool definitions from an MCP server's discovery endpoint
without establishing a full MCP session. Falls back gracefully when the
endpoint is unavailable or declares tools as dynamic.
"""

import logging
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from mcp.types import Tool as MCPTool

from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)

SERVER_CARD_TIMEOUT_SECONDS = 3.0
WELL_KNOWN_PATH = "/.well-known/mcp/server-card.json"


def derive_well_known_url(*, mcp_server_url: str) -> str:
    """Derive the .well-known server card URL from an MCP endpoint URL.

    Per RFC 8615, .well-known URIs are relative to the site origin,
    not the MCP endpoint path.
    """
    parsed = urlparse(mcp_server_url)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            WELL_KNOWN_PATH,
            "",
            "",
            "",
        )
    )


def derive_path_relative_well_known_url(*, mcp_server_url: str) -> str | None:
    """Derive a path-relative .well-known URL for servers behind a path-based gateway.

    For ``http://dev:5000/skills-library/`` this returns
    ``http://dev:5000/skills-library/.well-known/mcp/server-card.json``.

    Returns None when the MCP server URL has no meaningful path prefix
    (i.e. it is already at the origin root).
    """
    parsed = urlparse(mcp_server_url)
    path = parsed.path.rstrip("/")
    if not path or path == "/":
        return None
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            path + WELL_KNOWN_PATH,
            "",
            "",
            "",
        )
    )


def _parse_tools_from_card(*, card: dict[str, Any]) -> list[MCPTool] | None:
    """Extract tool definitions from a server card response.

    Returns None if tools are declared as dynamic or missing.
    """
    tools_field = card.get("tools")

    if tools_field is None:
        return None

    if tools_field == ["dynamic"] or tools_field == "dynamic":
        logger.info("Server card declares tools as dynamic, skipping")
        return None

    if not isinstance(tools_field, list):
        logger.warning("Server card 'tools' field is not a list: %s", type(tools_field))
        return None

    parsed_tools: list[MCPTool] = []
    for tool_data in tools_field:
        if not isinstance(tool_data, dict):
            continue
        try:
            parsed_tools.append(MCPTool.model_validate(tool_data))
        except Exception as e:
            logger.warning(
                "Failed to parse tool from server card: %s — %s",
                tool_data.get("name", "unknown"),
                e,
            )

    return parsed_tools if parsed_tools else None


class ServerCardDiscovery:
    """Fetches MCP Server Cards from .well-known endpoints."""

    def __init__(
        self,
        *,
        timeout_seconds: float = SERVER_CARD_TIMEOUT_SECONDS,
        httpx_client_factory: Any | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._httpx_client_factory = httpx_client_factory

    async def fetch_tools_from_server_card(
        self,
        *,
        mcp_server_url: str,
        headers: dict[str, str] | None = None,
    ) -> list[MCPTool] | None:
        """Attempt to fetch static tool definitions from a server card.

        Returns a list of MCPTool objects if the server card endpoint
        exists and provides static tool definitions. Returns None if:
        - The endpoint returns 404 or any non-200 status
        - The request times out
        - The response declares tools as "dynamic"
        - Any parsing error occurs

        Tries the origin-level .well-known URL first (RFC 8615), then
        falls back to a path-relative .well-known URL for servers behind
        a path-based gateway.

        This method never raises — all failures are logged and return None.
        """
        urls_to_try: list[str] = [derive_well_known_url(mcp_server_url=mcp_server_url)]
        path_relative_url = derive_path_relative_well_known_url(
            mcp_server_url=mcp_server_url
        )
        if path_relative_url is not None:
            urls_to_try.append(path_relative_url)

        for well_known_url in urls_to_try:
            tools = await self._try_fetch_server_card(
                well_known_url=well_known_url,
                mcp_server_url=mcp_server_url,
                headers=headers,
            )
            if tools is not None:
                return tools

        return None

    async def _try_fetch_server_card(
        self,
        *,
        well_known_url: str,
        mcp_server_url: str,
        headers: dict[str, str] | None = None,
    ) -> list[MCPTool] | None:
        """Attempt a single server card fetch at the given URL."""
        logger.info(
            "Attempting server card discovery at %s (for MCP server %s)",
            well_known_url,
            mcp_server_url,
        )

        try:
            timeout = httpx.Timeout(self._timeout_seconds)
            if self._httpx_client_factory:
                client = self._httpx_client_factory(
                    headers=headers,
                    timeout=timeout,
                )
            else:
                client = httpx.AsyncClient(
                    headers=headers,
                    timeout=timeout,
                    follow_redirects=True,
                )

            async with client:
                response = await client.get(well_known_url)

            if response.status_code != 200:
                logger.info(
                    "Server card endpoint returned %d for %s",
                    response.status_code,
                    well_known_url,
                )
                return None

            card = response.json()
            tools = _parse_tools_from_card(card=card)

            if tools is not None:
                logger.info(
                    "Server card discovery succeeded for %s: %d tools",
                    mcp_server_url,
                    len(tools),
                )
            else:
                logger.info(
                    "Server card at %s did not provide static tools",
                    well_known_url,
                )

            return tools

        except httpx.TimeoutException:
            logger.info(
                "Server card request timed out for %s (timeout=%.1fs)",
                well_known_url,
                self._timeout_seconds,
            )
            return None
        except Exception as e:
            logger.info(
                "Server card discovery failed for %s: %s",
                well_known_url,
                e,
            )
            return None
