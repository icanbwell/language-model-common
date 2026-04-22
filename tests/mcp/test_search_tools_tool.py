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
    async def test_oauth_server_skipped_when_query_does_not_match(self) -> None:
        """OAuth servers are not resolved at all when the search query
        does not match their metadata.  Resolution only happens when
        the query specifically targets the server."""
        from languagemodelcommon.configs.schemas.config_schema import McpOAuthConfig

        catalog = ToolCatalog()
        oauth_config = AgentConfig(
            name="google_drive",
            url="https://example.com/mcp",
            oauth=McpOAuthConfig(client_id="test-client-id"),
        )
        catalog.register_server(
            server_name="google_drive",
            category="Google Drive",
            agent_config=oauth_config,
        )

        resolver = AsyncMock()
        resolver.resolve_tools = AsyncMock(return_value=[])

        tool = SearchToolsTool(catalog=catalog, resolver=resolver)
        # "spreadsheet formulas" has no word overlap with "google_drive" / "Google Drive"
        content, _artifact = await tool._arun(
            query="spreadsheet formulas", category="Google Drive"
        )
        # Server should not have been resolved at all
        resolver.resolve_tools.assert_not_awaited()
        assert "No tools found" in content

    @pytest.mark.asyncio
    async def test_auth_exception_raised_when_query_matches(self) -> None:
        """When a search query matches an OAuth server that requires auth,
        the AuthorizationNeededException is re-raised so the gateway
        renders the login prompt — this is when the user has expressed
        intent to use that server's tools."""
        from languagemodelcommon.configs.schemas.config_schema import McpOAuthConfig
        from oidcauthlib.auth.exceptions.authorization_needed_exception import (
            AuthorizationNeededException,
        )

        catalog = ToolCatalog()
        oauth_config = AgentConfig(
            name="google_drive",
            display_name="Google Drive",
            url="https://example.com/mcp",
            oauth=McpOAuthConfig(client_id="test-client-id"),
        )
        catalog.register_server(
            server_name="google_drive",
            category="Google Drive",
            agent_config=oauth_config,
        )

        login_url = "https://accounts.google.com/o/oauth2/auth?client_id=123"
        login_message = f"[Login to Google Drive]({login_url})"

        async def _resolve(agent_config: AgentConfig) -> list[MCPTool]:
            raise AuthorizationNeededException(message=login_message)

        resolver = AsyncMock()
        resolver.resolve_tools = AsyncMock(side_effect=_resolve)

        tool = SearchToolsTool(catalog=catalog, resolver=resolver)
        # "google" in query matches server_name "google_drive"
        with pytest.raises(AuthorizationNeededException) as exc_info:
            await tool._arun(
                query="read google document content", category="Google Drive"
            )
        assert login_url in exc_info.value.message


class TestOAuthSearchScenarios:
    """End-to-end scenarios for OAuth-aware search gating.

    Uses a realistic catalog with both OAuth and non-OAuth servers
    registered simultaneously.  Covers the full decision matrix:

    ┌──────────────────────────────────────────────────────────────────┐
    │ Server type │ Query matches? │ Auth state │ Expected behaviour   │
    ├─────────────┼────────────────┼────────────┼──────────────────────┤
    │ No OAuth    │ yes            │ n/a        │ Resolve → tools      │
    │ OAuth       │ no             │ n/a        │ Skip — no resolution │
    │ OAuth       │ yes            │ not authed │ Raise auth exception │
    │ OAuth       │ yes            │ authed     │ Resolve → tools      │
    └──────────────────────────────────────────────────────────────────┘
    """

    @staticmethod
    def _build_mixed_catalog() -> tuple[
        ToolCatalog,
        AgentConfig,
        AgentConfig,
    ]:
        """Register two servers: google-search (no OAuth) and github (OAuth)."""
        from languagemodelcommon.configs.schemas.config_schema import McpOAuthConfig

        catalog = ToolCatalog()

        search_config = AgentConfig(
            name="google-search",
            url="https://example.com/google-search",
        )
        catalog.register_server(
            server_name="google-search",
            category="Web search",
            agent_config=search_config,
        )

        github_config = AgentConfig(
            name="github",
            display_name="GitHub",
            url="https://api.githubcopilot.com/mcp/",
            oauth=McpOAuthConfig(client_id="Iv23liP9XLkcIxslopoA"),
        )
        catalog.register_server(
            server_name="github",
            category="GitHub repositories",
            agent_config=github_config,
        )

        return catalog, search_config, github_config

    @pytest.mark.asyncio
    async def test_non_oauth_server_resolves_normally(self) -> None:
        """Scenario 1: Non-OAuth server resolves and returns tools when
        the query matches its category."""
        catalog, search_config, _github_config = self._build_mixed_catalog()

        async def _resolve(agent_config: AgentConfig) -> list[MCPTool]:
            if agent_config.name == "google-search":
                return [
                    MCPTool(
                        name="web_search",
                        description="Search the web for information",
                        inputSchema={"type": "object"},
                    ),
                ]
            return []

        resolver = AsyncMock()
        resolver.resolve_tools = AsyncMock(side_effect=_resolve)

        tool = SearchToolsTool(catalog=catalog, resolver=resolver)
        content, _artifact = await tool._arun(
            query="search web content", category="Web search"
        )

        parsed = json.loads(content)
        assert len(parsed) > 0
        assert parsed[0]["name"] == "web_search"
        # Only google-search should have been resolved, not github
        resolver.resolve_tools.assert_awaited_once_with(agent_config=search_config)

    @pytest.mark.asyncio
    async def test_oauth_server_skipped_when_query_does_not_match(self) -> None:
        """Scenario 2: OAuth server is not resolved when the search query
        has no token overlap with its metadata.  The non-OAuth server in
        the same catalog is still resolved normally."""
        catalog, _search_config, _github_config = self._build_mixed_catalog()

        resolver = AsyncMock()
        resolver.resolve_tools = AsyncMock(return_value=[])

        tool = SearchToolsTool(catalog=catalog, resolver=resolver)
        content, _artifact = await tool._arun(
            query="web content", category="GitHub repositories"
        )

        # github is OAuth and "web content" has no overlap with
        # "github" / "GitHub" / "GitHub repositories" → skipped
        resolver.resolve_tools.assert_not_awaited()
        assert "No tools found" in content

    @pytest.mark.asyncio
    async def test_oauth_server_raises_auth_when_query_matches(self) -> None:
        """Scenario 3: OAuth server whose metadata matches the query
        triggers AuthorizationNeededException so the gateway can render
        the login prompt."""
        from oidcauthlib.auth.exceptions.authorization_needed_exception import (
            AuthorizationNeededException,
        )

        catalog, _search_config, github_config = self._build_mixed_catalog()

        login_url = "https://github.com/login/oauth/authorize?client_id=abc"
        login_message = f"[Login to GitHub]({login_url})"

        async def _resolve(agent_config: AgentConfig) -> list[MCPTool]:
            if agent_config.name == "github":
                raise AuthorizationNeededException(message=login_message)
            return []

        resolver = AsyncMock()
        resolver.resolve_tools = AsyncMock(side_effect=_resolve)

        tool = SearchToolsTool(catalog=catalog, resolver=resolver)
        # "github" in query matches server_name "github"
        with pytest.raises(AuthorizationNeededException) as exc_info:
            await tool._arun(
                query="list github repositories", category="GitHub repositories"
            )
        assert login_url in exc_info.value.message
        resolver.resolve_tools.assert_awaited_once_with(agent_config=github_config)

    @pytest.mark.asyncio
    async def test_oauth_server_resolves_when_already_authenticated(self) -> None:
        """Scenario 4: OAuth server that has already been authenticated
        resolves normally and returns tools — no auth exception."""
        catalog, _search_config, github_config = self._build_mixed_catalog()

        async def _resolve(agent_config: AgentConfig) -> list[MCPTool]:
            if agent_config.name == "github":
                return [
                    MCPTool(
                        name="list_repos",
                        description="List GitHub repositories for the authenticated user",
                        inputSchema={"type": "object"},
                    ),
                    MCPTool(
                        name="create_issue",
                        description="Create an issue in a GitHub repository",
                        inputSchema={"type": "object"},
                    ),
                ]
            return []

        resolver = AsyncMock()
        resolver.resolve_tools = AsyncMock(side_effect=_resolve)

        tool = SearchToolsTool(catalog=catalog, resolver=resolver)
        content, _artifact = await tool._arun(
            query="list github repositories", category="GitHub repositories"
        )

        parsed = json.loads(content)
        assert len(parsed) > 0
        assert any(r["name"] == "list_repos" for r in parsed)
        resolver.resolve_tools.assert_awaited_once_with(agent_config=github_config)

    @pytest.mark.asyncio
    async def test_mixed_catalog_isolates_resolution(self) -> None:
        """When both OAuth and non-OAuth servers exist, searching one
        category does not trigger resolution of the other."""
        catalog, search_config, _github_config = self._build_mixed_catalog()

        resolve_calls: list[str] = []

        async def _resolve(agent_config: AgentConfig) -> list[MCPTool]:
            resolve_calls.append(agent_config.name)
            if agent_config.name == "google-search":
                return [
                    MCPTool(
                        name="web_search",
                        description="Search the web",
                        inputSchema={"type": "object"},
                    ),
                ]
            return []

        resolver = AsyncMock()
        resolver.resolve_tools = AsyncMock(side_effect=_resolve)

        tool = SearchToolsTool(catalog=catalog, resolver=resolver)
        content, _artifact = await tool._arun(query="search web", category="Web search")

        parsed = json.loads(content)
        assert len(parsed) > 0
        # Only google-search should have been resolved
        assert resolve_calls == ["google-search"]
