"""Tests for mcp_client.langchain_adapter — MCP metadata propagation."""

import pytest
from langchain_core.tools import StructuredTool
from mcp.types import (
    CallToolResult,
    ElicitRequest,
    ElicitRequestFormParams,
    InputRequiredResult,
    TextContent,
)
from mcp.types import Tool as MCPTool, ToolAnnotations

from languagemodelcommon.mcp.interceptors.types import MCPToolCallRequest
from languagemodelcommon.mcp.mcp_client.langchain_adapter import (
    MCPInputRequiredError,
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
            input_schema={"type": "object"},
            annotations=ToolAnnotations(title="Annotated Weather"),
        )
        assert _resolve_mcp_title(tool) == "Weather Info"

    def test_falls_back_to_annotations_title(self) -> None:
        tool = MCPTool(
            name="get_weather",
            description="Get weather",
            input_schema={"type": "object"},
            annotations=ToolAnnotations(title="Annotated Weather"),
        )
        assert _resolve_mcp_title(tool) == "Annotated Weather"

    def test_returns_none_when_no_title(self) -> None:
        tool = MCPTool(
            name="get_weather",
            description="Get weather",
            input_schema={"type": "object"},
        )
        assert _resolve_mcp_title(tool) is None

    def test_returns_none_for_empty_title(self) -> None:
        tool = MCPTool(
            name="get_weather",
            title="",
            description="Get weather",
            input_schema={"type": "object"},
        )
        assert _resolve_mcp_title(tool) is None

    def test_ignores_annotations_when_no_title_field(self) -> None:
        tool = MCPTool(
            name="get_weather",
            description="Get weather",
            input_schema={"type": "object"},
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
            input_schema={"type": "object"},
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
            input_schema={"type": "object"},
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
            input_schema={"type": "object"},
        )
        lc_tool = mcp_tool_to_langchain_tool(
            mcp_tool, connection=_make_connection_config()
        )
        assert lc_tool.metadata is not None
        assert lc_tool.metadata["mcp_description"] == "Get current weather data"

    def test_metadata_is_none_when_no_mcp_metadata(self) -> None:
        mcp_tool = MCPTool(
            name="get_weather",
            input_schema={"type": "object"},
        )
        lc_tool = mcp_tool_to_langchain_tool(
            mcp_tool, connection=_make_connection_config()
        )
        assert lc_tool.metadata is None

    def test_annotations_title_flows_to_metadata(self) -> None:
        mcp_tool = MCPTool(
            name="get_weather",
            description="desc",
            input_schema={"type": "object"},
            annotations=ToolAnnotations(title="Annotated Title"),
        )
        lc_tool = mcp_tool_to_langchain_tool(
            mcp_tool, connection=_make_connection_config()
        )
        assert lc_tool.metadata is not None
        assert lc_tool.metadata["mcp_title"] == "Annotated Title"


class TestMcpToolToLangchainToolInputRequired:
    @pytest.mark.asyncio
    async def test_raises_mcp_input_required_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An InputRequiredResult from the handler chain surfaces as
        MCPInputRequiredError, not silently stringified tool output."""
        input_required = InputRequiredResult(
            result_type="input_required",
            input_requests={
                "confirm": ElicitRequest(
                    method="elicitation/create",
                    params=ElicitRequestFormParams(
                        message="About to save a Patient. Proceed?",
                        requested_schema={
                            "type": "object",
                            "properties": {"confirm": {"type": "boolean"}},
                            "required": ["confirm"],
                        },
                    ),
                )
            },
            request_state="opaque-state-123",
        )

        async def fake_execute_tool(request: MCPToolCallRequest) -> InputRequiredResult:
            return input_required

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.langchain_adapter._make_execute_tool",
            lambda **kwargs: fake_execute_tool,
        )

        mcp_tool = MCPTool(
            name="save_fhir_resource",
            description="Save a FHIR resource",
            input_schema={"type": "object", "properties": {}},
        )
        connection = MCPConnectionConfig(url="https://mcp-fhir-agent.test/mcp")
        tool = mcp_tool_to_langchain_tool(
            tool=mcp_tool, connection=connection, server_name="mcp-fhir-agent"
        )
        assert isinstance(tool, StructuredTool)
        assert tool.coroutine is not None

        with pytest.raises(MCPInputRequiredError) as exc_info:
            await tool.coroutine(resource={"resourceType": "Patient"})

        err = exc_info.value
        assert err.tool_name == "save_fhir_resource"
        assert err.arguments == {"resource": {"resourceType": "Patient"}}
        assert err.server_name == "mcp-fhir-agent"
        assert err.request_state == "opaque-state-123"
        assert "confirm" in err.input_requests


class TestCallToolReturnsStructuredContent:
    """BAI-882: call_tool's coroutine must forward the MCP CallToolResult's
    own structuredContent as the LangChain tool's artifact -- previously
    this was hardcoded to None (content, None), silently discarding it."""

    @pytest.mark.asyncio
    async def test_returns_structured_content_as_artifact(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call_tool_result = CallToolResult(
            content=[TextContent(type="text", text="Found 2 connections.")],
            is_error=False,
            structured_content={"count": 2, "connections": ["a", "b"]},
        )

        async def fake_execute_tool(request: MCPToolCallRequest) -> CallToolResult:
            return call_tool_result

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.langchain_adapter._make_execute_tool",
            lambda **kwargs: fake_execute_tool,
        )

        mcp_tool = MCPTool(
            name="search_connections",
            description="Search connections",
            input_schema={"type": "object", "properties": {}},
        )
        tool = mcp_tool_to_langchain_tool(
            tool=mcp_tool,
            connection=_make_connection_config(),
            server_name="test-server",
        )
        assert isinstance(tool, StructuredTool)
        assert tool.coroutine is not None

        content, artifact = await tool.coroutine(query="labs")

        assert artifact == {"count": 2, "connections": ["a", "b"]}
        assert len(content) == 1

    @pytest.mark.asyncio
    async def test_returns_none_artifact_when_no_structured_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tool that never returns MCP structuredContent must still forward
        None (not an empty dict or the text content), matching the pre-BAI-882
        artifact-absent behavior for tools with no structured payload."""
        call_tool_result = CallToolResult(
            content=[TextContent(type="text", text="Done.")],
            is_error=False,
            structured_content=None,
        )

        async def fake_execute_tool(request: MCPToolCallRequest) -> CallToolResult:
            return call_tool_result

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.langchain_adapter._make_execute_tool",
            lambda **kwargs: fake_execute_tool,
        )

        mcp_tool = MCPTool(
            name="do_thing",
            description="Do a thing",
            input_schema={"type": "object", "properties": {}},
        )
        tool = mcp_tool_to_langchain_tool(
            tool=mcp_tool,
            connection=_make_connection_config(),
            server_name="test-server",
        )
        assert isinstance(tool, StructuredTool)
        assert tool.coroutine is not None

        _, artifact = await tool.coroutine()

        assert artifact is None
