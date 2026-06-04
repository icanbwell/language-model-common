"""Tests for MCP Server Card discovery (SEP-1649)."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch


from languagemodelcommon.mcp.mcp_client.server_card_discovery import (
    ServerCardDiscovery,
    derive_well_known_url,
    derive_path_relative_well_known_url,
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


class TestDerivePathRelativeWellKnownUrl:
    @pytest.mark.parametrize(
        "mcp_url,expected",
        [
            (
                "http://dev:5000/skills-library/",
                "http://dev:5000/skills-library/.well-known/mcp/server-card.json",
            ),
            (
                "http://dev:5000/skills-library",
                "http://dev:5000/skills-library/.well-known/mcp/server-card.json",
            ),
            (
                "https://gateway.example.com/api/v1/mcp",
                "https://gateway.example.com/api/v1/mcp/.well-known/mcp/server-card.json",
            ),
            (
                "https://example.com/",
                None,
            ),
            (
                "https://example.com",
                None,
            ),
        ],
    )
    def test_derives_path_relative_url(
        self, mcp_url: str, expected: str | None
    ) -> None:
        result = derive_path_relative_well_known_url(mcp_server_url=mcp_url)
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

    @pytest.mark.asyncio
    async def test_falls_back_to_path_relative_url(self) -> None:
        """When origin-level .well-known returns 404, tries path-relative URL."""
        origin_404 = httpx.Response(status_code=404)
        path_relative_200 = httpx.Response(
            status_code=200,
            json={
                "version": "1.0",
                "tools": [
                    {
                        "name": "skill_tool",
                        "description": "A skill",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ],
            },
        )

        async def route_by_url(url: str, **kwargs: object) -> httpx.Response:
            if "/skills-library/.well-known/" in url:
                return path_relative_200
            return origin_404

        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=route_by_url,
        ):
            discovery = ServerCardDiscovery()
            tools = await discovery.fetch_tools_from_server_card(
                mcp_server_url="http://dev:5000/skills-library/",
            )

        assert tools is not None
        assert len(tools) == 1
        assert tools[0].name == "skill_tool"

    @pytest.mark.asyncio
    async def test_origin_level_success_skips_path_relative(self) -> None:
        """When origin-level .well-known succeeds, does not try path-relative."""
        call_urls: list[str] = []

        origin_200 = httpx.Response(
            status_code=200,
            json={
                "version": "1.0",
                "tools": [
                    {
                        "name": "origin_tool",
                        "description": "From origin",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ],
            },
        )

        async def track_calls(url: str, **kwargs: object) -> httpx.Response:
            call_urls.append(url)
            return origin_200

        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=track_calls,
        ):
            discovery = ServerCardDiscovery()
            tools = await discovery.fetch_tools_from_server_card(
                mcp_server_url="http://dev:5000/skills-library/",
            )

        assert tools is not None
        assert tools[0].name == "origin_tool"
        assert len(call_urls) == 1
        assert "/skills-library/.well-known/" not in call_urls[0]

    @pytest.mark.asyncio
    async def test_no_path_prefix_does_not_try_path_relative(self) -> None:
        """When MCP server URL is at origin root, only origin-level URL is tried."""
        call_urls: list[str] = []
        mock_404 = httpx.Response(status_code=404)

        async def track_calls(url: str, **kwargs: object) -> httpx.Response:
            call_urls.append(url)
            return mock_404

        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=track_calls,
        ):
            discovery = ServerCardDiscovery()
            tools = await discovery.fetch_tools_from_server_card(
                mcp_server_url="https://example.com/",
            )

        assert tools is None
        assert len(call_urls) == 1
