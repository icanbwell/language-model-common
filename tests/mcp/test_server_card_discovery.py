"""Tests for MCP Server Card discovery (SEP-1649)."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch


from languagemodelcommon.mcp.mcp_client.server_card_discovery import (
    ServerCardDiscovery,
    derive_well_known_url,
    _parse_tools_from_card,
)


class TestDeriveWellKnownUrl:
    @pytest.mark.parametrize(
        "mcp_url,expected",
        [
            (
                "https://example.com/mcp",
                "https://example.com/.well-known/mcp/server-card.json",
            ),
            (
                "https://example.com/some/deep/path/",
                "https://example.com/.well-known/mcp/server-card.json",
            ),
            (
                "http://localhost:8080/mcp/",
                "http://localhost:8080/.well-known/mcp/server-card.json",
            ),
            (
                "https://api.example.com:9443/v1/mcp",
                "https://api.example.com:9443/.well-known/mcp/server-card.json",
            ),
        ],
    )
    def test_derives_well_known_from_origin(self, mcp_url: str, expected: str) -> None:
        result = derive_well_known_url(mcp_server_url=mcp_url)
        assert result == expected


class TestParseToolsFromCard:
    def test_static_tools_parsed(self) -> None:
        card = {
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                    },
                }
            ]
        }
        tools = _parse_tools_from_card(card=card)
        assert tools is not None
        assert len(tools) == 1
        assert tools[0].name == "get_weather"

    def test_dynamic_string_returns_none(self) -> None:
        card = {"tools": "dynamic"}
        assert _parse_tools_from_card(card=card) is None

    def test_dynamic_list_returns_none(self) -> None:
        card = {"tools": ["dynamic"]}
        assert _parse_tools_from_card(card=card) is None

    def test_missing_tools_returns_none(self) -> None:
        card = {"serverInfo": {"name": "test"}}
        assert _parse_tools_from_card(card=card) is None

    def test_empty_list_returns_none(self) -> None:
        card: dict[str, list[str]] = {"tools": []}
        assert _parse_tools_from_card(card=card) is None

    def test_invalid_tool_data_skipped(self) -> None:
        card = {
            "tools": [
                {
                    "name": "valid_tool",
                    "description": "works",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                "not a dict",
                {"missing_name": True},
            ]
        }
        tools = _parse_tools_from_card(card=card)
        assert tools is not None
        assert len(tools) == 1
        assert tools[0].name == "valid_tool"


class TestServerCardDiscovery:
    @pytest.mark.asyncio
    async def test_successful_fetch(self) -> None:
        mock_response = httpx.Response(
            status_code=200,
            json={
                "version": "1.0",
                "tools": [
                    {
                        "name": "example_tool",
                        "description": "An example",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ],
            },
        )

        with patch(
            "httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response
        ):
            discovery = ServerCardDiscovery()
            tools = await discovery.fetch_tools_from_server_card(
                mcp_server_url="https://example.com/mcp",
            )

        assert tools is not None
        assert len(tools) == 1
        assert tools[0].name == "example_tool"

    @pytest.mark.asyncio
    async def test_404_returns_none(self) -> None:
        mock_response = httpx.Response(status_code=404)

        with patch(
            "httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response
        ):
            discovery = ServerCardDiscovery()
            tools = await discovery.fetch_tools_from_server_card(
                mcp_server_url="https://example.com/mcp",
            )

        assert tools is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self) -> None:
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timed out"),
        ):
            discovery = ServerCardDiscovery(timeout_seconds=1.0)
            tools = await discovery.fetch_tools_from_server_card(
                mcp_server_url="https://example.com/mcp",
            )

        assert tools is None

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self) -> None:
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            discovery = ServerCardDiscovery()
            tools = await discovery.fetch_tools_from_server_card(
                mcp_server_url="https://example.com/mcp",
            )

        assert tools is None

    @pytest.mark.asyncio
    async def test_dynamic_tools_returns_none(self) -> None:
        mock_response = httpx.Response(
            status_code=200,
            json={
                "version": "1.0",
                "tools": ["dynamic"],
            },
        )

        with patch(
            "httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response
        ):
            discovery = ServerCardDiscovery()
            tools = await discovery.fetch_tools_from_server_card(
                mcp_server_url="https://example.com/mcp",
            )

        assert tools is None
