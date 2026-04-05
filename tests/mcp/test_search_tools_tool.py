"""Tests for SearchToolsTool — meta-tool for BM25-ranked tool search."""

import json

import pytest
from mcp.types import Tool as MCPTool

from languagemodelcommon.configs.schemas.config_schema import AgentConfig
from languagemodelcommon.mcp.search_tools_tool import SearchToolsTool
from languagemodelcommon.mcp.tool_catalog import ToolCatalog


def _agent_config() -> AgentConfig:
    return AgentConfig(name="server1", url="https://example.com/mcp")


def _build_catalog() -> ToolCatalog:
    catalog = ToolCatalog()
    catalog.add_tools(
        server_name="fhir",
        category="Healthcare",
        tools=[
            MCPTool(
                name="search_patients",
                description="Search for patients by name",
                inputSchema={"type": "object"},
            ),
            MCPTool(
                name="get_observation",
                description="Get lab observations for a patient",
                inputSchema={"type": "object"},
            ),
        ],
        agent_config=_agent_config(),
    )
    catalog.add_tools(
        server_name="billing",
        category="Financial",
        tools=[
            MCPTool(
                name="create_invoice",
                description="Create a billing invoice",
                inputSchema={"type": "object"},
            ),
        ],
        agent_config=_agent_config(),
    )
    return catalog


class TestSearchToolsTool:
    def test_search_returns_json(self) -> None:
        catalog = _build_catalog()
        tool = SearchToolsTool(catalog=catalog)
        result = tool._run(query="patient")
        parsed = json.loads(result)
        assert len(parsed) > 0
        assert "name" in parsed[0]

    def test_no_results_message(self) -> None:
        catalog = ToolCatalog()
        tool = SearchToolsTool(catalog=catalog)
        result = tool._run(query="anything")
        assert "No tools found" in result

    def test_category_filter(self) -> None:
        catalog = _build_catalog()
        tool = SearchToolsTool(catalog=catalog)
        result = tool._run(query="search", category="Healthcare")
        parsed = json.loads(result)
        assert all(r["server_name"] == "fhir" for r in parsed)

    @pytest.mark.asyncio
    async def test_async_delegates_to_sync(self) -> None:
        catalog = _build_catalog()
        tool = SearchToolsTool(catalog=catalog)
        result = await tool._arun(query="patient")
        parsed = json.loads(result)
        assert len(parsed) > 0

    def test_max_results_respected(self) -> None:
        catalog = _build_catalog()
        tool = SearchToolsTool(catalog=catalog, max_results=1)
        result = tool._run(query="patient")
        parsed = json.loads(result)
        assert len(parsed) <= 1
