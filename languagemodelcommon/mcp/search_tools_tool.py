"""Meta-discovery tool: search_tools.

Provides BM25 Okapi ranked search over MCP tool metadata. The LLM uses
this to discover relevant tools on demand instead of having every tool
schema loaded into the context window.

Supports lazy resolution: when a search matches an unresolved server
category, the tool fetches the server's tools via the resolver before
searching.
"""

from __future__ import annotations

import json
import logging
from typing import Literal, Type

from langchain_core.tools import BaseTool
from oidcauthlib.auth.exceptions.authorization_needed_exception import (
    AuthorizationNeededException,
)
from pydantic import BaseModel, ConfigDict, Field

from languagemodelcommon.mcp.tool_catalog import ToolCatalog, ToolResolverProtocol
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


class SearchToolsInput(BaseModel):
    category: str | None = Field(
        None,
        description="Optional category to filter tools by. Use one of the category names from the system message.",
    )
    query: str = Field(
        ...,
        description="Natural language search query to find relevant tools.",
    )


class SearchToolsTool(BaseTool):
    """Search for available MCP tools by keyword using BM25 ranking.

    When a resolver is provided, unresolved servers matching the search
    category are lazily resolved (their tools fetched from the MCP server)
    before the BM25 search runs.
    """

    name: str = "search_tools"
    description: str = (
        "Search for available tools by keyword. Returns tool names, descriptions, "
        "and parameter schemas ranked by relevance. Use this to discover tools "
        "before calling them with call_tool."
    )
    args_schema: Type[BaseModel] = SearchToolsInput
    response_format: Literal["content", "content_and_artifact"] = "content"

    catalog: ToolCatalog
    resolver: ToolResolverProtocol | None = None
    max_results: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, query: str, category: str | None = None) -> str:
        raise NotImplementedError(
            "search_tools requires async execution for lazy tool resolution. "
            "Use _arun instead."
        )

    async def _arun(self, query: str, category: str | None = None) -> str:
        # Lazily resolve any unresolved servers matching the category
        if self.resolver is not None:
            unresolved = self.catalog.get_unresolved_servers(category)
            for server in unresolved:
                try:
                    await self.catalog.resolve_server(server.server_name, self.resolver)
                except AuthorizationNeededException:
                    # Re-raise auth exceptions so the user sees login links
                    raise
                except Exception as e:
                    logger.warning(
                        "Failed to resolve server %s during search: %s: %s",
                        server.server_name,
                        type(e).__name__,
                        e,
                    )

        results = self.catalog.search(
            query=query,
            category=category,
            max_results=self.max_results,
        )
        if not results:
            return "No tools found matching your query."
        return json.dumps(results, indent=2)
