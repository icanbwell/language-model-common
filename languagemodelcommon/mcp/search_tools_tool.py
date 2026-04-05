"""Meta-discovery tool: search_tools.

Provides BM25 Okapi ranked search over MCP tool metadata. The LLM uses
this to discover relevant tools on demand instead of having every tool
schema loaded into the context window.
"""

from __future__ import annotations

import json
import logging
from typing import Literal, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from languagemodelcommon.mcp.tool_catalog import ToolCatalog
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
    """Search for available MCP tools by keyword using BM25 ranking."""

    name: str = "search_tools"
    description: str = (
        "Search for available tools by keyword. Returns tool names, descriptions, "
        "and parameter schemas ranked by relevance. Use this to discover tools "
        "before calling them with call_tool."
    )
    args_schema: Type[BaseModel] = SearchToolsInput
    response_format: Literal["content", "content_and_artifact"] = "content"

    catalog: ToolCatalog
    max_results: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, query: str, category: str | None = None) -> str:
        results = self.catalog.search(
            query=query,
            category=category,
            max_results=self.max_results,
        )
        if not results:
            return "No tools found matching your query."
        return json.dumps(results, indent=2)

    async def _arun(self, query: str, category: str | None = None) -> str:
        return self._run(query=query, category=category)
