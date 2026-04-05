"""Tests for ToolCatalog — BM25 Okapi search over MCP tool metadata."""

from typing import Any

from mcp.types import Tool as MCPTool

from languagemodelcommon.configs.schemas.config_schema import AgentConfig
from languagemodelcommon.mcp.tool_catalog import (
    ToolCatalog,
    _BM25Index,
    _tokenize,
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
        results = catalog.search("patient")
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
