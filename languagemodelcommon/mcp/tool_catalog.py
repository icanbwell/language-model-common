"""Tool catalog with BM25 Okapi search for MCP tool discovery.

Provides a searchable index of MCP tools that supports ranked retrieval
by keyword relevance. Used by the meta-discovery tools (search_tools, call_tool).

Supports lazy resolution: servers can be registered with metadata only,
and their tools are fetched on-demand when a search matches the server's
category.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mcp.types import Tool as MCPTool

from languagemodelcommon.configs.schemas.config_schema import AgentConfig
from languagemodelcommon.mcp.bm25 import BM25Index, tokenize
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


@runtime_checkable
class ToolResolverProtocol(Protocol):
    """Callback to lazily fetch tools from an MCP server."""

    async def resolve_tools(
        self,
        agent_config: AgentConfig,
    ) -> list[MCPTool]: ...


@dataclass
class ServerRegistration:
    """An MCP server registered in the catalog, possibly not yet resolved."""

    server_name: str
    category: str | None
    agent_config: AgentConfig
    resolved: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class ToolCatalogEntry:
    """A single tool in the catalog."""

    server_name: str
    tool: MCPTool
    category: str | None
    agent_config: AgentConfig


def _build_tool_document(tool: MCPTool) -> list[str]:
    """Build a searchable token list from a tool's metadata."""
    parts: list[str] = []

    # Tool name (split on underscores for compound names)
    parts.extend(tokenize(tool.name))

    # Description
    if tool.description:
        parts.extend(tokenize(tool.description))

    # Parameter names and descriptions from inputSchema
    schema = tool.inputSchema
    if isinstance(schema, dict):
        properties: dict[str, Any] = schema.get("properties", {})
        for param_name, param_info in properties.items():
            parts.extend(tokenize(param_name))
            if isinstance(param_info, dict) and "description" in param_info:
                parts.extend(tokenize(str(param_info["description"])))

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
    """Searchable catalog of MCP tools with BM25 Okapi ranking.

    Supports two modes:
    - **Eager:** Call ``add_tools`` to populate immediately.
    - **Lazy:** Call ``register_server`` to record metadata only, then
      ``resolve_server`` (or let ``SearchToolsTool`` do it) to fetch
      tools on-demand when a search matches the server's category.
    """

    def __init__(self) -> None:
        self._entries: list[ToolCatalogEntry] = []
        self._index: BM25Index | None = None
        self._entries_by_name: dict[str, ToolCatalogEntry] = {}
        self._servers: dict[str, ServerRegistration] = {}

    def register_server(
        self,
        *,
        server_name: str,
        category: str | None,
        agent_config: AgentConfig,
    ) -> None:
        """Register a server for lazy resolution (no MCP call yet)."""
        self._servers[server_name] = ServerRegistration(
            server_name=server_name,
            category=category,
            agent_config=agent_config,
        )
        logger.info(
            "Registered server %s (category=%s) for lazy resolution",
            server_name,
            category,
        )

    async def resolve_server(
        self,
        server_name: str,
        resolver: ToolResolverProtocol,
    ) -> None:
        """Resolve a registered server by fetching its tools via the resolver.

        Safe to call concurrently — uses a per-server lock to prevent
        duplicate resolution.
        """
        registration = self._servers.get(server_name)
        if registration is None or registration.resolved:
            return

        async with registration._lock:
            # Double-check after acquiring lock
            if registration.resolved:
                return

            logger.info("Resolving tools for server %s", server_name)
            tools = await resolver.resolve_tools(
                agent_config=registration.agent_config,
            )
            self.add_tools(
                server_name=server_name,
                category=registration.category,
                tools=tools,
                agent_config=registration.agent_config,
            )
            registration.resolved = True
            logger.info("Resolved %d tools for server %s", len(tools), server_name)

    def get_unresolved_servers(
        self, category: str | None = None
    ) -> list[ServerRegistration]:
        """Return unresolved server registrations, optionally filtered by category."""
        unresolved = [s for s in self._servers.values() if not s.resolved]
        if category is None:
            return unresolved
        return [
            s
            for s in unresolved
            if (s.category and category.lower() in s.category.lower())
            or category.lower() in s.server_name.lower()
        ]

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

    def _ensure_index(self) -> BM25Index:
        """Lazily build or return the BM25 index."""
        if self._index is None:
            corpus = [_build_tool_document(entry.tool) for entry in self._entries]
            self._index = BM25Index()
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
            index = BM25Index()
            index.build(corpus)
            query_tokens = tokenize(query)
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
        query_tokens = tokenize(query)
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
        """Get a summary of tool categories for the system prompt.

        Includes both resolved servers (with tool counts) and unresolved
        servers (marked as available but not yet discovered).
        """
        categories: dict[str, dict[str, Any]] = {}

        # Include unresolved servers so the LLM knows they exist
        for reg in self._servers.values():
            if reg.server_name not in categories:
                categories[reg.server_name] = {
                    "name": reg.server_name,
                    "description": reg.category or reg.server_name,
                    "tool_count": 0,
                    "resolved": reg.resolved,
                }

        # Include resolved tool counts
        for entry in self._entries:
            key = entry.server_name
            if key not in categories:
                categories[key] = {
                    "name": key,
                    "description": entry.category or entry.server_name,
                    "tool_count": 0,
                    "resolved": True,
                }
            categories[key]["tool_count"] += 1
            categories[key]["resolved"] = True

        return list(categories.values())

    def list_tools(self, category: str | None = None) -> list[dict[str, Any]]:
        """List all tools, optionally filtered by category."""
        entries = self._entries
        if category:
            entries = [
                e
                for e in entries
                if (e.category and category.lower() in e.category.lower())
                or category.lower() in e.server_name.lower()
            ]
        return [
            {
                **_format_tool_schema(e.tool),
                "server_name": e.server_name,
                "category": e.category,
            }
            for e in entries
        ]

    @property
    def tool_count(self) -> int:
        return len(self._entries)
