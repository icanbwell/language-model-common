"""Tool catalog with BM25 Okapi search for MCP tool discovery.

Provides a searchable index of MCP tools that supports ranked retrieval
by keyword relevance. Used by the meta-discovery tools (search_tools, call_tool).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from mcp.types import Tool as MCPTool

from languagemodelcommon.configs.schemas.config_schema import AgentConfig
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


@dataclass
class ToolCatalogEntry:
    """A single tool in the catalog."""

    server_name: str
    tool: MCPTool
    category: str | None
    agent_config: AgentConfig


@dataclass
class _BM25Index:
    """In-memory BM25 Okapi index over tool documents."""

    k1: float = 1.5
    b: float = 0.75
    corpus: list[list[str]] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    avgdl: float = 0.0
    n_docs: int = 0
    # term -> list of (doc_index, term_freq) pairs
    inverted_index: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # term -> idf score
    idf: dict[str, float] = field(default_factory=dict)

    def build(self, corpus: list[list[str]]) -> None:
        self.corpus = corpus
        self.n_docs = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.n_docs if self.n_docs > 0 else 0.0

        # Build inverted index
        self.inverted_index = {}
        for doc_idx, doc in enumerate(corpus):
            term_freqs: dict[str, int] = {}
            for term in doc:
                term_freqs[term] = term_freqs.get(term, 0) + 1
            for term, freq in term_freqs.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, freq))

        # Compute IDF for each term
        self.idf = {}
        for term, postings in self.inverted_index.items():
            df = len(postings)
            # Standard BM25 IDF formula
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(
        self, query_tokens: list[str], top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Return (doc_index, score) pairs sorted by descending score."""
        scores: dict[int, float] = {}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            idf = self.idf[token]
            for doc_idx, tf in self.inverted_index[token]:
                dl = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score = idf * numerator / denominator
                scores[doc_idx] = scores.get(doc_idx, 0.0) + score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


_TOKENIZE_RE = re.compile(r"[_\-\s/.,;:]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace, underscores, hyphens, and punctuation."""
    return [t for t in _TOKENIZE_RE.split(text.lower()) if t]


def _build_tool_document(tool: MCPTool) -> list[str]:
    """Build a searchable token list from a tool's metadata."""
    parts: list[str] = []

    # Tool name (split on underscores for compound names)
    parts.extend(_tokenize(tool.name))

    # Description
    if tool.description:
        parts.extend(_tokenize(tool.description))

    # Parameter names and descriptions from inputSchema
    schema = tool.inputSchema
    if isinstance(schema, dict):
        properties: dict[str, Any] = schema.get("properties", {})
        for param_name, param_info in properties.items():
            parts.extend(_tokenize(param_name))
            if isinstance(param_info, dict) and "description" in param_info:
                parts.extend(_tokenize(str(param_info["description"])))

    return parts


def _format_tool_schema(tool: MCPTool) -> dict[str, Any]:
    """Format a tool's metadata for search result output."""
    result: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description or "",
    }
    schema = tool.inputSchema
    if isinstance(schema, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if properties:
            params = {}
            for pname, pinfo in properties.items():
                if isinstance(pinfo, dict):
                    params[pname] = {
                        "type": pinfo.get("type", "any"),
                        "description": pinfo.get("description", ""),
                        "required": pname in required,
                    }
                else:
                    params[pname] = {
                        "type": "any",
                        "description": "",
                        "required": pname in required,
                    }
            result["parameters"] = params
    return result


class ToolCatalog:
    """Searchable catalog of MCP tools with BM25 Okapi ranking."""

    def __init__(self) -> None:
        self._entries: list[ToolCatalogEntry] = []
        self._index: _BM25Index | None = None
        self._entries_by_name: dict[str, ToolCatalogEntry] = {}

    def add_tools(
        self,
        *,
        server_name: str,
        category: str | None,
        tools: list[MCPTool],
        agent_config: AgentConfig,
    ) -> None:
        """Add tools from an MCP server to the catalog."""
        for tool in tools:
            entry = ToolCatalogEntry(
                server_name=server_name,
                tool=tool,
                category=category,
                agent_config=agent_config,
            )
            self._entries.append(entry)
            self._entries_by_name[tool.name] = entry
        # Invalidate the index so it gets rebuilt on next search
        self._index = None
        logger.info(
            "Added %d tools from %s to catalog (total: %d)",
            len(tools),
            server_name,
            len(self._entries),
        )

    def _ensure_index(self) -> _BM25Index:
        """Lazily build or return the BM25 index."""
        if self._index is None:
            corpus = [_build_tool_document(entry.tool) for entry in self._entries]
            self._index = _BM25Index()
            self._index.build(corpus)
            logger.info("Built BM25 index over %d tools", len(self._entries))
        return self._index

    def search(
        self,
        query: str,
        category: str | None = None,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for tools matching the query.

        Args:
            query: Natural language search query.
            category: Optional category filter (matches server_name or category description).
            max_results: Maximum number of results to return.

        Returns:
            List of tool descriptions with name, description, parameters, server_name, and category.
        """
        if not self._entries:
            return []

        # If category filter is specified, search only within that category
        if category:
            filtered_entries = [
                e
                for e in self._entries
                if (e.category and category.lower() in e.category.lower())
                or category.lower() in e.server_name.lower()
            ]
            if not filtered_entries:
                return []
            # Build a temporary index for the filtered subset
            corpus = [_build_tool_document(e.tool) for e in filtered_entries]
            index = _BM25Index()
            index.build(corpus)
            query_tokens = _tokenize(query)
            ranked = index.search(query_tokens, top_k=max_results)
            return [
                {
                    **_format_tool_schema(filtered_entries[idx].tool),
                    "server_name": filtered_entries[idx].server_name,
                    "category": filtered_entries[idx].category,
                }
                for idx, _score in ranked
            ]

        # Search across all tools
        index = self._ensure_index()
        query_tokens = _tokenize(query)
        ranked = index.search(query_tokens, top_k=max_results)
        return [
            {
                **_format_tool_schema(self._entries[idx].tool),
                "server_name": self._entries[idx].server_name,
                "category": self._entries[idx].category,
            }
            for idx, _score in ranked
        ]

    def get_tool(self, name: str) -> ToolCatalogEntry | None:
        """Look up a tool by exact name."""
        return self._entries_by_name.get(name)

    def get_categories(self) -> list[dict[str, Any]]:
        """Get a summary of tool categories for the system prompt."""
        categories: dict[str, dict[str, Any]] = {}
        for entry in self._entries:
            key = entry.server_name
            if key not in categories:
                categories[key] = {
                    "name": key,
                    "description": entry.category or entry.server_name,
                    "tool_count": 0,
                }
            categories[key]["tool_count"] += 1
        return list(categories.values())

    @property
    def tool_count(self) -> int:
        return len(self._entries)
