"""Tests for SearchToolsTool — meta-tool for BM25-ranked tool search."""

import json
from unittest.mock import AsyncMock

import pytest
from mcp.types import Tool as MCPTool

from languagemodelcommon.configs.schemas.config_schema import AgentConfig
from languagemodelcommon.tools.mcp.search_tools_tool import SearchToolsTool
from languagemodelcommon.mcp.tool_catalog import ToolCatalog


def _agent_config(name: str = "server1") -> AgentConfig:
    return AgentConfig(name=name, url="https://example.com/mcp")


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
    def test_sync_run_raises(self) -> None:
        catalog = _build_catalog()
        tool = SearchToolsTool(catalog=catalog)
        with pytest.raises(NotImplementedError):
            tool._run(query="patient", category="Healthcare")

    @pytest.mark.asyncio
    async def test_search_returns_json(self) -> None:
        catalog = _build_catalog()
        tool = SearchToolsTool(catalog=catalog)
        content, artifact = await tool._arun(query="patient", category="Healthcare")
        parsed = json.loads(content)
        assert len(parsed) > 0
        assert "name" in parsed[0]
        # Artifact includes additional scoring info
        artifact_parsed = json.loads(artifact)
        assert len(artifact_parsed["results"]) > 0
        assert "score" in artifact_parsed["results"][0]
        assert "matched_terms" in artifact_parsed["results"][0]

    @pytest.mark.asyncio
    async def test_no_results_message(self) -> None:
        catalog = ToolCatalog()
        tool = SearchToolsTool(catalog=catalog)
        content, artifact = await tool._arun(query="anything", category="Nonexistent")
        assert "No tools found" in content
        assert "No tools found" in artifact

    @pytest.mark.asyncio
    async def test_category_filter(self) -> None:
        catalog = _build_catalog()
        tool = SearchToolsTool(catalog=catalog)
        content, _artifact = await tool._arun(query="search", category="Healthcare")
        parsed = json.loads(content)
        assert all(r["server_name"] == "fhir" for r in parsed)

    @pytest.mark.asyncio
    async def test_max_results_respected(self) -> None:
        catalog = _build_catalog()
        tool = SearchToolsTool(catalog=catalog, max_results=1)
        content, _artifact = await tool._arun(query="patient", category="Healthcare")
        parsed = json.loads(content)
        assert len(parsed) <= 1

    @pytest.mark.asyncio
    async def test_lazy_resolution_on_search(self) -> None:
        """When a resolver is provided and a category matches an unresolved
        server, the server's tools are fetched on demand."""
        catalog = ToolCatalog()
        config = _agent_config("fhir")
        catalog.register_server(
            server_name="fhir",
            category="Healthcare",
            agent_config=config,
        )

        resolver = AsyncMock()
        resolver.resolve_tools = AsyncMock(
            return_value=[
                MCPTool(
                    name="search_patients",
                    description="Search for a patient by name",
                    inputSchema={"type": "object"},
                ),
            ]
        )

        tool = SearchToolsTool(catalog=catalog, resolver=resolver)
        content, _artifact = await tool._arun(query="patient", category="Healthcare")
        parsed = json.loads(content)
        assert len(parsed) > 0
        assert parsed[0]["name"] == "search_patients"
        resolver.resolve_tools.assert_awaited_once_with(agent_config=config)

    @pytest.mark.asyncio
    async def test_no_resolution_without_resolver(self) -> None:
        """Without a resolver, unresolved servers are simply ignored."""
        catalog = ToolCatalog()
        catalog.register_server(
            server_name="fhir",
            category="Healthcare",
            agent_config=_agent_config("fhir"),
        )

        tool = SearchToolsTool(catalog=catalog)
        content, _artifact = await tool._arun(query="patient", category="Healthcare")
        assert "No tools found" in content

    @pytest.mark.asyncio
    async def test_resolver_failure_does_not_crash(self) -> None:
        """If a resolver raises, the search still returns gracefully."""
        catalog = ToolCatalog()
        catalog.register_server(
            server_name="fhir",
            category="Healthcare",
            agent_config=_agent_config("fhir"),
        )

        resolver = AsyncMock()
        resolver.resolve_tools = AsyncMock(side_effect=ConnectionError("unreachable"))

        tool = SearchToolsTool(catalog=catalog, resolver=resolver)
        content, _artifact = await tool._arun(query="patient", category="Healthcare")
        assert "No tools found" in content

    @pytest.mark.asyncio
    async def test_auth_required_server_skipped_gracefully(self) -> None:
        """Auth exceptions during resolution are skipped so the search
        can still return results from other servers.  The auth prompt
        surfaces later when call_tool targets the protected server."""
        from oidcauthlib.auth.exceptions.authorization_needed_exception import (
            AuthorizationNeededException,
        )

        catalog = ToolCatalog()
        catalog.register_server(
            server_name="google_drive",
            category="Google Drive",
            agent_config=_agent_config("google_drive"),
        )

        async def _resolve(agent_config: AgentConfig) -> list[MCPTool]:
            raise AuthorizationNeededException(message="Login required")

        resolver = AsyncMock()
        resolver.resolve_tools = AsyncMock(side_effect=_resolve)

        tool = SearchToolsTool(catalog=catalog, resolver=resolver)
        content, _artifact = await tool._arun(query="files", category="Google Drive")
        assert "requires authentication" in content or "No tools found" in content
