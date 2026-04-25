"""Tests for McpJsonFetcher per-plugin MCP config fetching."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult, TextContent

from languagemodelcommon.configs.config_reader.mcp_json_fetcher import (
    McpJsonFetcher,
    TOOL_NAME,
)


def _text_result(data: dict[str, Any]) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(data))])


def _make_session(tool_result: CallToolResult) -> AsyncMock:
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.call_tool = AsyncMock(return_value=tool_result)
    return session


@asynccontextmanager
async def _mock_session_ctx(session: AsyncMock):  # type: ignore[no-untyped-def]
    yield session


class TestFetchPluginAsync:
    @pytest.mark.asyncio
    async def test_returns_mcp_config_for_plugin(self) -> None:
        session = _make_session(
            _text_result(
                {
                    "mcpServers": {
                        "google-drive": {"url": "https://mcp.example.com/drive/"}
                    }
                }
            )
        )
        fetcher = McpJsonFetcher(plugins_mcp_server_url="http://localhost:5000/skills/")

        with patch(
            "languagemodelcommon.configs.config_reader.mcp_json_fetcher.create_mcp_session",
            return_value=_mock_session_ctx(session),
        ):
            result = await fetcher.fetch_plugin_async("all-employees")

        assert result is not None
        assert "google-drive" in result.mcpServers
        assert result.mcpServers["google-drive"].url == "https://mcp.example.com/drive/"
        session.call_tool.assert_awaited_once_with(
            TOOL_NAME, {"plugin_name": "all-employees"}
        )

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_servers(self) -> None:
        session = _make_session(_text_result({"mcpServers": {}}))
        fetcher = McpJsonFetcher(plugins_mcp_server_url="http://localhost:5000/skills/")

        with patch(
            "languagemodelcommon.configs.config_reader.mcp_json_fetcher.create_mcp_session",
            return_value=_mock_session_ctx(session),
        ):
            result = await fetcher.fetch_plugin_async("empty-plugin")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_connection_error(self) -> None:
        fetcher = McpJsonFetcher(plugins_mcp_server_url="http://localhost:5000/skills/")

        @asynccontextmanager
        async def _failing_session(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            raise ConnectionError("refused")
            yield  # pragma: no cover

        with patch(
            "languagemodelcommon.configs.config_reader.mcp_json_fetcher.create_mcp_session",
            return_value=_failing_session(),
        ):
            result = await fetcher.fetch_plugin_async("broken")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self) -> None:
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="not json")]
            )
        )
        fetcher = McpJsonFetcher(plugins_mcp_server_url="http://localhost:5000/skills/")

        with patch(
            "languagemodelcommon.configs.config_reader.mcp_json_fetcher.create_mcp_session",
            return_value=_mock_session_ctx(session),
        ):
            result = await fetcher.fetch_plugin_async("bad-json")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_no_text_content(self) -> None:
        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(return_value=CallToolResult(content=[]))
        fetcher = McpJsonFetcher(plugins_mcp_server_url="http://localhost:5000/skills/")

        with patch(
            "languagemodelcommon.configs.config_reader.mcp_json_fetcher.create_mcp_session",
            return_value=_mock_session_ctx(session),
        ):
            result = await fetcher.fetch_plugin_async("no-content")

        assert result is None

    @pytest.mark.asyncio
    async def test_applies_env_var_substitution(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("DRIVE_URL", "https://resolved.example.com/drive/")
        session = _make_session(
            _text_result({"mcpServers": {"google-drive": {"url": "${DRIVE_URL}"}}})
        )
        fetcher = McpJsonFetcher(plugins_mcp_server_url="http://localhost:5000/skills/")

        with patch(
            "languagemodelcommon.configs.config_reader.mcp_json_fetcher.create_mcp_session",
            return_value=_mock_session_ctx(session),
        ):
            result = await fetcher.fetch_plugin_async("all-employees")

        assert result is not None
        assert (
            result.mcpServers["google-drive"].url
            == "https://resolved.example.com/drive/"
        )


class TestFetchPluginsAsync:
    @pytest.mark.asyncio
    async def test_fetches_multiple_plugins(self) -> None:
        def _make_result(name: str, args: dict[str, Any]) -> CallToolResult:
            plugin = args["plugin_name"]
            return _text_result(
                {
                    "mcpServers": {
                        f"server-{plugin}": {"url": f"https://{plugin}.example.com/"}
                    }
                }
            )

        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(side_effect=_make_result)

        fetcher = McpJsonFetcher(plugins_mcp_server_url="http://localhost:5000/skills/")

        with patch(
            "languagemodelcommon.configs.config_reader.mcp_json_fetcher.create_mcp_session",
            side_effect=lambda _: _mock_session_ctx(session),
        ):
            result = await fetcher.fetch_plugins_async(["plugin-a", "plugin-b"])

        assert len(result) == 2
        assert "plugin-a" in result
        assert "plugin-b" in result
        assert (
            result["plugin-a"].mcpServers["server-plugin-a"].url
            == "https://plugin-a.example.com/"
        )

    @pytest.mark.asyncio
    async def test_omits_failed_plugins(self) -> None:
        def _make_result(name: str, args: dict[str, Any]) -> CallToolResult:
            plugin = args["plugin_name"]
            if plugin == "bad":
                return _text_result({"mcpServers": {}})
            return _text_result(
                {"mcpServers": {"server": {"url": "https://good.example.com/"}}}
            )

        session = AsyncMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(side_effect=_make_result)

        fetcher = McpJsonFetcher(plugins_mcp_server_url="http://localhost:5000/skills/")

        with patch(
            "languagemodelcommon.configs.config_reader.mcp_json_fetcher.create_mcp_session",
            side_effect=lambda _: _mock_session_ctx(session),
        ):
            result = await fetcher.fetch_plugins_async(["good", "bad"])

        assert len(result) == 1
        assert "good" in result
        assert "bad" not in result

    @pytest.mark.asyncio
    async def test_returns_empty_dict_for_empty_list(self) -> None:
        fetcher = McpJsonFetcher(plugins_mcp_server_url="http://localhost:5000/skills/")

        result = await fetcher.fetch_plugins_async([])

        assert result == {}
