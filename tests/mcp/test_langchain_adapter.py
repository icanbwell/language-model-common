"""Tests for mcp_client.langchain_adapter — MCP metadata propagation."""

import pytest
from mcp.types import Tool as MCPTool, ToolAnnotations

from languagemodelcommon.mcp.mcp_client.langchain_adapter import (
    _resolve_mcp_title,
    _sanitize_tool_name,
    mcp_tool_to_langchain_tool,
)
from languagemodelcommon.mcp.mcp_client.session import MCPConnectionConfig


def _make_connection_config() -> MCPConnectionConfig:
    return {"url": "http://localhost:8080/mcp", "transport": "streamable_http"}


class TestResolveMcpTitle:
    def test_prefers_top_level_title(self) -> None:
        tool = MCPTool(
            name="get_weather",
            title="Weather Info",
            description="Get weather",
            inputSchema={"type": "object"},
            annotations=ToolAnnotations(title="Annotated Weather"),
        )
        assert _resolve_mcp_title(tool) == "Weather Info"

    def test_falls_back_to_annotations_title(self) -> None:
        tool = MCPTool(
            name="get_weather",
            description="Get weather",
            inputSchema={"type": "object"},
            annotations=ToolAnnotations(title="Annotated Weather"),
        )
        assert _resolve_mcp_title(tool) == "Annotated Weather"

    def test_returns_none_when_no_title(self) -> None:
        tool = MCPTool(
            name="get_weather",
            description="Get weather",
            inputSchema={"type": "object"},
        )
        assert _resolve_mcp_title(tool) is None

    def test_returns_none_for_empty_title(self) -> None:
        tool = MCPTool(
            name="get_weather",
            title="",
            description="Get weather",
            inputSchema={"type": "object"},
        )
        assert _resolve_mcp_title(tool) is None

    def test_ignores_annotations_when_no_title_field(self) -> None:
        tool = MCPTool(
            name="get_weather",
            description="Get weather",
            inputSchema={"type": "object"},
            annotations=ToolAnnotations(),
        )
        assert _resolve_mcp_title(tool) is None


class TestSanitizeToolName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("get_weather", "get_weather"),
            ("search-code", "search-code"),
            ("github.search_code", "github_search_code"),
            ("namespace:operation", "namespace_operation"),
            ("tool/with/slashes", "tool_with_slashes"),
            ("tool with spaces", "tool_with_spaces"),
            ("complex.name:v2/query", "complex_name_v2_query"),
        ],
    )
    def test_sanitizes_invalid_characters(self, raw: str, expected: str) -> None:
        assert _sanitize_tool_name(raw) == expected

    def test_sanitized_name_used_in_langchain_tool(self) -> None:
        mcp_tool = MCPTool(
            name="github.search_code",
            description="Search code",
            inputSchema={"type": "object"},
        )
        lc_tool = mcp_tool_to_langchain_tool(
            mcp_tool, connection=_make_connection_config()
        )
        assert lc_tool.name == "github_search_code"


class TestMcpToolToLangchainToolMetadata:
    def test_metadata_includes_mcp_title(self) -> None:
        mcp_tool = MCPTool(
            name="get_weather",
            title="Weather Information Provider",
            description="Get weather data",
            inputSchema={"type": "object"},
        )
        lc_tool = mcp_tool_to_langchain_tool(
            mcp_tool, connection=_make_connection_config()
        )
        assert lc_tool.metadata is not None
        assert lc_tool.metadata["mcp_title"] == "Weather Information Provider"

    def test_metadata_includes_mcp_description(self) -> None:
        mcp_tool = MCPTool(
            name="get_weather",
            description="Get current weather data",
            inputSchema={"type": "object"},
        )
        lc_tool = mcp_tool_to_langchain_tool(
            mcp_tool, connection=_make_connection_config()
        )
        assert lc_tool.metadata is not None
        assert lc_tool.metadata["mcp_description"] == "Get current weather data"

    def test_metadata_is_none_when_no_mcp_metadata(self) -> None:
        mcp_tool = MCPTool(
            name="get_weather",
            inputSchema={"type": "object"},
        )
        lc_tool = mcp_tool_to_langchain_tool(
            mcp_tool, connection=_make_connection_config()
        )
        assert lc_tool.metadata is None

    def test_annotations_title_flows_to_metadata(self) -> None:
        mcp_tool = MCPTool(
            name="get_weather",
            description="desc",
            inputSchema={"type": "object"},
            annotations=ToolAnnotations(title="Annotated Title"),
        )
        lc_tool = mcp_tool_to_langchain_tool(
            mcp_tool, connection=_make_connection_config()
        )
        assert lc_tool.metadata is not None
        assert lc_tool.metadata["mcp_title"] == "Annotated Title"
