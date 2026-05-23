"""Tests for MCP App UI resource helpers."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from languagemodelcommon.mcp.mcp_client.ui_resource import (
    extract_ui_resource_uri,
    fetch_ui_resource,
    inject_tool_data_into_html,
    is_tool_visible_to_model,
    _extract_ui_meta_from_content,
)


class TestExtractUiResourceUri:
    def test_no_meta(self) -> None:
        tool = MagicMock()
        tool.meta = None
        assert extract_ui_resource_uri(tool) is None

    def test_nested_ui_resource_uri(self) -> None:
        tool = MagicMock()
        tool.meta = {"ui": {"resourceUri": "ui://my-server/chart"}}
        assert extract_ui_resource_uri(tool) == "ui://my-server/chart"

    def test_flat_ui_resource_uri(self) -> None:
        tool = MagicMock()
        tool.meta = {"ui/resourceUri": "ui://my-server/panel"}
        assert extract_ui_resource_uri(tool) == "ui://my-server/panel"

    def test_non_ui_uri_ignored(self) -> None:
        tool = MagicMock()
        tool.meta = {"ui": {"resourceUri": "https://example.com"}}
        assert extract_ui_resource_uri(tool) is None


class TestIsToolVisibleToModel:
    def test_no_meta_visible(self) -> None:
        tool = MagicMock()
        tool.meta = None
        assert is_tool_visible_to_model(tool) is True

    def test_no_visibility_visible(self) -> None:
        tool = MagicMock()
        tool.meta = {"ui": {"resourceUri": "ui://x"}}
        assert is_tool_visible_to_model(tool) is True

    def test_model_and_app_visible(self) -> None:
        tool = MagicMock()
        tool.meta = {"ui": {"visibility": ["model", "app"]}}
        assert is_tool_visible_to_model(tool) is True

    def test_app_only_not_visible(self) -> None:
        tool = MagicMock()
        tool.meta = {"ui": {"visibility": ["app"]}}
        assert is_tool_visible_to_model(tool) is False


class TestExtractUiMetaFromContent:
    def test_no_meta(self) -> None:
        item = MagicMock()
        item._meta = None
        item.meta = None
        assert _extract_ui_meta_from_content(item) is None

    def test_with_csp_and_permissions(self) -> None:
        item = MagicMock()
        item._meta = {
            "ui": {
                "csp": {"default-src": "'self'"},
                "permissions": {"camera": "none"},
                "prefersBorder": True,
                "displayMode": "fullscreen",
            }
        }
        item.meta = None
        result = _extract_ui_meta_from_content(item)
        assert result is not None
        assert result.csp == {"default-src": "'self'"}
        assert result.permissions == {"camera": "none"}
        assert result.prefers_border is True
        assert result.display_mode == "fullscreen"


class TestInjectToolDataIntoHtml:
    def test_injects_into_head(self) -> None:
        html = "<html><head><title>App</title></head><body></body></html>"
        result = inject_tool_data_into_html(
            html,
            tool_name="my_tool",
            tool_args={"query": "test"},
            tool_result_text='{"status": "ok"}',
        )
        assert "window.__MCP_TOOL_RESULT__" in result
        assert "window.__MCP_TOOL_ARGS__" in result
        assert "window.__MCP_TOOL_NAME__" in result
        assert "window.__MCP_BRIDGE_CONFIG__" in result
        assert "ui/initialize" in result
        assert "ui/notifications/tool-input" in result
        assert "ui/notifications/tool-result" in result

    def test_injects_without_head_tag(self) -> None:
        html = "<div>No head</div>"
        result = inject_tool_data_into_html(
            html,
            tool_name="tool",
            tool_args={},
            tool_result_text="hello",
        )
        assert "<head>" in result
        assert "window.__MCP_BRIDGE_CONFIG__" in result

    def test_proxy_base_url_included(self) -> None:
        html = "<html><head></head><body></body></html>"
        result = inject_tool_data_into_html(
            html,
            tool_name="tool",
            tool_args={},
            tool_result_text="result",
            proxy_base_url="http://localhost:5000/api/v1",
            session_token="abc123",
        )
        assert "http://localhost:5000/api/v1" in result
        assert "abc123" in result
        assert "mcp-proxy/tools/call" in result
        assert "mcp-proxy/resources/read" in result

    def test_handles_special_characters_in_args(self) -> None:
        html = "<html><head></head><body></body></html>"
        result = inject_tool_data_into_html(
            html,
            tool_name="tool",
            tool_args={"query": 'test "quotes" & <brackets>'},
            tool_result_text='{"key": "value with \\"escaped\\" quotes"}',
        )
        assert "window.__MCP_TOOL_ARGS__" in result

    def test_display_mode_support(self) -> None:
        html = "<html><head></head><body></body></html>"
        result = inject_tool_data_into_html(
            html,
            tool_name="tool",
            tool_args={},
            tool_result_text="result",
        )
        assert "ui/request-display-mode" in result
        assert "inline" in result
        assert "fullscreen" in result

    def test_open_link_support(self) -> None:
        html = "<html><head></head><body></body></html>"
        result = inject_tool_data_into_html(
            html,
            tool_name="tool",
            tool_args={},
            tool_result_text="result",
        )
        assert "ui/open-link" in result
        assert "window.open" in result

    def test_height_reporter(self) -> None:
        html = "<html><head></head><body></body></html>"
        result = inject_tool_data_into_html(
            html,
            tool_name="tool",
            tool_args={},
            tool_result_text="result",
        )
        assert "iframe:height" in result
        assert "MutationObserver" in result


@pytest.mark.asyncio
class TestFetchUiResource:
    async def test_returns_none_on_exception(self) -> None:
        session = AsyncMock()
        session.read_resource.side_effect = RuntimeError("connection failed")
        result = await fetch_ui_resource(session, "ui://server/app")
        assert result is None

    async def test_returns_html_and_meta(self) -> None:
        content_item = MagicMock()
        content_item.text = "<html><body>Hello</body></html>"
        content_item._meta = {
            "ui": {"csp": {"script-src": "'self'"}, "prefersBorder": False}
        }

        read_result = MagicMock()
        read_result.contents = [content_item]

        session = AsyncMock()
        session.read_resource.return_value = read_result

        result = await fetch_ui_resource(session, "ui://server/app")
        assert result is not None
        assert result.html == "<html><body>Hello</body></html>"
        assert result.ui_meta is not None
        assert result.ui_meta.csp == {"script-src": "'self'"}
        assert result.ui_meta.prefers_border is False
