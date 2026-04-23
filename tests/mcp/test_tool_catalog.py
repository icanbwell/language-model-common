"""Tests for ToolCatalog — BM25 Okapi search over MCP tool metadata."""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from mcp.types import Tool as MCPTool

from languagemodelcommon.configs.schemas.config_schema import (
    AgentConfig,
    McpOAuthConfig,
)
from languagemodelcommon.mcp.tool_catalog import (
    ToolCatalog,
    _BM25Index,
    _tokenize,
    _tokenize_with_stems,
    _stem,
    _build_tool_document,
    _format_tool_schema,
)


def _mcp_tool(
    name: str,
    description: str = "",
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> MCPTool:
    schema: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = required
    return MCPTool(name=name, description=description, inputSchema=schema)


def _agent_config(name: str = "server1") -> AgentConfig:
    return AgentConfig(name=name, url="https://example.com/mcp")


class TestTokenize:
    def test_splits_on_underscores(self) -> None:
        assert _tokenize("get_user_info") == ["get", "user", "info"]

    def test_splits_on_hyphens(self) -> None:
        assert _tokenize("get-user-info") == ["get", "user", "info"]

    def test_lowercases(self) -> None:
        assert _tokenize("GetUser") == ["getuser"]

    def test_splits_on_whitespace(self) -> None:
        assert _tokenize("get user info") == ["get", "user", "info"]

    def test_empty_string(self) -> None:
        assert _tokenize("") == []


class TestStem:
    def test_strips_ing(self) -> None:
        assert _stem("scraping") == "scrap"

    def test_strips_tion(self) -> None:
        assert _stem("extraction") == "extrac"

    def test_strips_plural_s(self) -> None:
        assert _stem("patients") == "patient"

    def test_strips_plural_es(self) -> None:
        assert _stem("pages") == "pag"

    def test_does_not_strip_short_words(self) -> None:
        assert _stem("is") == "is"
        assert _stem("as") == "as"

    def test_does_not_strip_ss(self) -> None:
        assert _stem("pass") == "pass"

    def test_preserves_short_stems(self) -> None:
        # "wing" - stem "w" would be < 3 chars, so no strip
        assert _stem("wing") == "wing"


class TestTokenizeWithStems:
    def test_includes_raw_and_stemmed(self) -> None:
        tokens = _tokenize_with_stems("scraping pages")
        assert "scraping" in tokens
        assert "scrap" in tokens
        assert "pages" in tokens
        assert "pag" in tokens

    def test_no_duplicate_when_no_stem(self) -> None:
        tokens = _tokenize_with_stems("url")
        assert tokens.count("url") == 1


class TestBuildToolDocument:
    def test_includes_name_tokens(self) -> None:
        tool = _mcp_tool("search_patients")
        tokens = _build_tool_document(tool)
        assert "search" in tokens
        assert "patients" in tokens

    def test_includes_description_tokens(self) -> None:
        tool = _mcp_tool("search", description="Find patients by name")
        tokens = _build_tool_document(tool)
        assert "find" in tokens
        assert "patients" in tokens

    def test_includes_parameter_tokens(self) -> None:
        tool = _mcp_tool(
            "query",
            properties={
                "patient_id": {
                    "type": "string",
                    "description": "The patient identifier",
                }
            },
        )
        tokens = _build_tool_document(tool)
        assert "patient" in tokens
        assert "identifier" in tokens

    def test_includes_category_tokens(self) -> None:
        tool = _mcp_tool("convert", description="Convert HTML to markdown")
        tokens = _build_tool_document(
            tool, category="Extract and convert web page content to markdown format"
        )
        assert "extract" in tokens
        assert "web" in tokens
        assert "content" in tokens

    def test_includes_stemmed_tokens(self) -> None:
        tool = _mcp_tool("scrape_url", description="Scraping webpages")
        tokens = _build_tool_document(tool)
        assert "scraping" in tokens
        assert "scrap" in tokens


class TestFormatToolSchema:
    def test_basic_format(self) -> None:
        tool = _mcp_tool("test_tool", description="A test tool")
        result = _format_tool_schema(tool)
        assert result["name"] == "test_tool"
        assert result["description"] == "A test tool"

    def test_includes_parameters(self) -> None:
        tool = _mcp_tool(
            "query",
            properties={"name": {"type": "string", "description": "Patient name"}},
            required=["name"],
        )
        result = _format_tool_schema(tool)
        assert "parameters" in result
        assert result["parameters"]["name"]["required"] is True

    def test_empty_description(self) -> None:
        tool = _mcp_tool("test_tool")
        result = _format_tool_schema(tool)
        assert result["description"] == ""


class TestBM25Index:
    def test_search_returns_relevant_doc(self) -> None:
        index = _BM25Index()
        index.build(
            [
                ["patient", "search", "fhir"],
                ["billing", "invoice", "payment"],
                ["patient", "record", "history"],
            ]
        )
        results = index.search(["patient"])
        assert len(results) >= 2
        # Both patient-related docs should rank higher than billing
        doc_indices = [idx for idx, _ in results]
        assert 0 in doc_indices
        assert 2 in doc_indices

    def test_empty_corpus(self) -> None:
        index = _BM25Index()
        index.build([])
        assert index.search(["anything"]) == []

    def test_no_matching_terms(self) -> None:
        index = _BM25Index()
        index.build([["alpha", "beta"]])
        assert index.search(["gamma"]) == []

    def test_respects_top_k(self) -> None:
        index = _BM25Index()
        corpus = [[f"term{i}"] for i in range(20)]
        corpus[0].append("common")
        corpus[1].append("common")
        corpus[2].append("common")
        index.build(corpus)
        results = index.search(["common"], top_k=2)
        assert len(results) == 2


class TestToolCatalog:
    def test_add_and_search(self) -> None:
        catalog = ToolCatalog()
        catalog.add_tools(
            server_name="fhir",
            category="Healthcare",
            tools=[
                _mcp_tool("search_patients", "Search for patients by name"),
                _mcp_tool("get_observation", "Get lab observations"),
            ],
            agent_config=_agent_config(),
        )
        # Use "patients" (matching the exact token in the tool name/description)
        results = catalog.search("patients")
        assert len(results) > 0
        assert results[0]["name"] == "search_patients"

    def test_get_tool_by_name(self) -> None:
        catalog = ToolCatalog()
        catalog.add_tools(
            server_name="fhir",
            category=None,
            tools=[_mcp_tool("search_patients")],
            agent_config=_agent_config(),
        )
        entry = catalog.get_tool("search_patients")
        assert entry is not None
        assert entry.server_name == "fhir"

    def test_get_tool_not_found(self) -> None:
        catalog = ToolCatalog()
        assert catalog.get_tool("nonexistent") is None

    def test_search_empty_catalog(self) -> None:
        catalog = ToolCatalog()
        assert catalog.search("anything") == []

    def test_category_filter(self) -> None:
        catalog = ToolCatalog()
        catalog.add_tools(
            server_name="fhir",
            category="Healthcare",
            tools=[_mcp_tool("search_patients", "Search patients")],
            agent_config=_agent_config("fhir"),
        )
        catalog.add_tools(
            server_name="billing",
            category="Financial",
            tools=[_mcp_tool("create_invoice", "Create an invoice")],
            agent_config=_agent_config("billing"),
        )
        # Filter by category
        results = catalog.search("search", category="Healthcare")
        assert all(r["server_name"] == "fhir" for r in results)

    def test_category_filter_no_match(self) -> None:
        catalog = ToolCatalog()
        catalog.add_tools(
            server_name="fhir",
            category="Healthcare",
            tools=[_mcp_tool("search_patients")],
            agent_config=_agent_config(),
        )
        results = catalog.search("search", category="Nonexistent")
        assert results == []

    def test_get_categories(self) -> None:
        catalog = ToolCatalog()
        catalog.add_tools(
            server_name="fhir",
            category="Healthcare",
            tools=[_mcp_tool("t1"), _mcp_tool("t2")],
            agent_config=_agent_config(),
        )
        categories = catalog.get_categories()
        assert len(categories) == 1
        assert categories[0]["name"] == "fhir"
        assert categories[0]["tool_count"] == 2

    def test_tool_count(self) -> None:
        catalog = ToolCatalog()
        assert catalog.tool_count == 0
        catalog.add_tools(
            server_name="s1",
            category=None,
            tools=[_mcp_tool("t1"), _mcp_tool("t2")],
            agent_config=_agent_config(),
        )
        assert catalog.tool_count == 2

    def test_url_to_markdown_ranks_above_google_drive(self) -> None:
        """Regression: query 'web scraping url content extract markdown' should
        rank url_to_markdown above google_drive tools."""
        catalog = ToolCatalog()
        catalog.add_tools(
            server_name="google_drive",
            category="Google Drive file management - search, read, and download files from Google Drive",
            tools=[
                _mcp_tool(
                    "search_drive",
                    description="Search for files in Google Drive by name or content",
                    properties={
                        "query": {
                            "type": "string",
                            "description": "Search query to find files",
                        }
                    },
                ),
                _mcp_tool(
                    "read_file",
                    description="Read the content of a file from Google Drive",
                    properties={
                        "file_id": {
                            "type": "string",
                            "description": "The ID of the file to read",
                        }
                    },
                ),
            ],
            agent_config=_agent_config("google_drive"),
        )
        catalog.add_tools(
            server_name="url_to_markdown",
            category="Extract and convert web page content to markdown format",
            tools=[
                _mcp_tool(
                    "url_to_markdown",
                    description=(
                        "Fetches the content of a webpage from a given URL "
                        "and converts it to Markdown format"
                    ),
                    properties={
                        "url": {
                            "type": "string",
                            "description": "url of the webpage to scrape",
                        }
                    },
                ),
            ],
            agent_config=_agent_config("url_to_markdown"),
        )

        results = catalog.search("web scraping url content extract markdown")
        assert len(results) > 0
        assert results[0]["name"] == "url_to_markdown"

    def test_index_invalidated_on_add(self) -> None:
        catalog = ToolCatalog()
        catalog.add_tools(
            server_name="s1",
            category=None,
            tools=[_mcp_tool("alpha_tool", "Alpha tool")],
            agent_config=_agent_config(),
        )
        # Trigger index build
        catalog.search("alpha")
        assert catalog._index is not None

        # Adding more tools should invalidate the index
        catalog.add_tools(
            server_name="s2",
            category=None,
            tools=[_mcp_tool("beta_tool", "Beta tool")],
            agent_config=_agent_config(),
        )
        assert catalog._index is None


class TestServerRegistration:
    def test_register_server(self) -> None:
        catalog = ToolCatalog()
        config = _agent_config("fhir")
        catalog.register_server(
            server_name="fhir",
            category="Healthcare",
            agent_config=config,
        )
        unresolved = catalog.get_unresolved_servers()
        assert len(unresolved) == 1
        assert unresolved[0].server_name == "fhir"
        assert unresolved[0].category == "Healthcare"
        assert unresolved[0].resolved is False

    def test_get_unresolved_servers_by_category(self) -> None:
        catalog = ToolCatalog()
        catalog.register_server(
            server_name="fhir",
            category="Healthcare tools",
            agent_config=_agent_config("fhir"),
        )
        catalog.register_server(
            server_name="billing",
            category="Financial tools",
            agent_config=_agent_config("billing"),
        )
        healthcare = catalog.get_unresolved_servers("Healthcare")
        assert len(healthcare) == 1
        assert healthcare[0].server_name == "fhir"

    def test_get_unresolved_servers_by_server_name(self) -> None:
        catalog = ToolCatalog()
        catalog.register_server(
            server_name="fhir",
            category=None,
            agent_config=_agent_config("fhir"),
        )
        matched = catalog.get_unresolved_servers("fhir")
        assert len(matched) == 1

    @pytest.mark.asyncio
    async def test_resolve_server(self) -> None:
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
                _mcp_tool("search_patients", "Search for patients"),
            ]
        )

        await catalog.resolve_server("fhir", resolver)

        assert catalog.get_unresolved_servers() == []
        assert catalog.tool_count == 1
        entry = catalog.get_tool("search_patients")
        assert entry is not None
        assert entry.server_name == "fhir"
        resolver.resolve_tools.assert_awaited_once_with(agent_config=config)

    @pytest.mark.asyncio
    async def test_resolve_server_idempotent(self) -> None:
        """Resolving the same server twice only calls the resolver once."""
        catalog = ToolCatalog()
        catalog.register_server(
            server_name="fhir",
            category="Healthcare",
            agent_config=_agent_config("fhir"),
        )

        resolver = AsyncMock()
        resolver.resolve_tools = AsyncMock(return_value=[])

        await catalog.resolve_server("fhir", resolver)
        await catalog.resolve_server("fhir", resolver)

        assert resolver.resolve_tools.await_count == 1

    @pytest.mark.asyncio
    async def test_resolve_unknown_server_is_noop(self) -> None:
        catalog = ToolCatalog()
        resolver = AsyncMock()
        await catalog.resolve_server("nonexistent", resolver)
        resolver.resolve_tools.assert_not_awaited()

    def test_get_categories_includes_unresolved(self) -> None:
        catalog = ToolCatalog()
        catalog.register_server(
            server_name="fhir",
            category="Healthcare",
            agent_config=_agent_config("fhir"),
        )
        categories = catalog.get_categories()
        assert len(categories) == 1
        assert categories[0]["name"] == "fhir"
        assert categories[0]["description"] == "Healthcare"
        assert categories[0]["resolved"] is False
        assert categories[0]["tool_count"] == 0

    def test_get_categories_marks_oauth_servers(self) -> None:
        """Categories include requires_auth flag for OAuth servers."""
        catalog = ToolCatalog()
        catalog.register_server(
            server_name="google-search",
            category="Web search",
            agent_config=_agent_config("google-search"),
        )
        oauth_config = AgentConfig(
            name="github",
            url="https://api.githubcopilot.com/mcp/",
            oauth=McpOAuthConfig(client_id="Iv23liP9XLkcIxslopoA"),
        )
        catalog.register_server(
            server_name="github",
            category="GitHub repositories",
            agent_config=oauth_config,
        )
        categories = catalog.get_categories()
        by_name = {c["name"]: c for c in categories}
        assert len(by_name) == 2
        assert by_name["google-search"]["requires_auth"] is False
        assert by_name["github"]["requires_auth"] is True

    @pytest.mark.asyncio
    async def test_get_categories_after_resolution(self) -> None:
        catalog = ToolCatalog()
        catalog.register_server(
            server_name="fhir",
            category="Healthcare",
            agent_config=_agent_config("fhir"),
        )

        resolver = AsyncMock()
        resolver.resolve_tools = AsyncMock(
            return_value=[_mcp_tool("t1"), _mcp_tool("t2")]
        )
        await catalog.resolve_server("fhir", resolver)

        categories = catalog.get_categories()
        assert len(categories) == 1
        assert categories[0]["resolved"] is True
        assert categories[0]["tool_count"] == 2
