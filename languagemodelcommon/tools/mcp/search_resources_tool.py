"""Meta-discovery tool: search_resources.

Provides BM25 Okapi ranked search over MCP resource metadata. The LLM uses
this to discover available resources on demand instead of having every
resource URI loaded into the context window.

Supports lazy resolution: when a search matches an unresolved server
category, the tool fetches the server's resources via the resolver before
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

from languagemodelcommon.mcp.resource_catalog import (
    ResourceCatalog,
    ResourceResolverProtocol,
)
from languagemodelcommon.utilities.logger.exception_logger import ExceptionLogger
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


class SearchResourcesInput(BaseModel):
    category: str | None = Field(
        None,
        description="Optional category to filter resources by. Use one of the category names from the system message.",
    )
    query: str = Field(
        ...,
        description="Natural language search query to find relevant resources.",
    )


class SearchResourcesTool(BaseTool):
    """Search for available MCP resources by keyword using BM25 ranking.

    When a resolver is provided, unresolved servers matching the search
    category are lazily resolved (their resources fetched from the MCP
    server) before the BM25 search runs.
    """

    name: str = "search_resources"
    description: str = (
        "Search for available resources by keyword. Returns resource names, URIs, "
        "descriptions, and MIME types ranked by relevance. Use this to discover "
        "resources before reading them with read_resource."
    )
    args_schema: Type[BaseModel] = SearchResourcesInput
    response_format: Literal["content", "content_and_artifact"] = "content"

    catalog: ResourceCatalog
    resolver: ResourceResolverProtocol | None = None
    max_results: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, query: str, category: str | None = None) -> str:
        raise NotImplementedError(
            "search_resources requires async execution for lazy resource resolution. "
            "Use _arun instead."
        )

    async def _arun(self, query: str, category: str | None = None) -> str:
        resolution_errors: list[str] = []
        if self.resolver is not None:
            unresolved = self.catalog.get_unresolved_servers(category)
            for server in unresolved:
                try:
                    await self.catalog.resolve_server(server.server_name, self.resolver)
                except AuthorizationNeededException:
                    raise
                except Exception as e:
                    error_msg = (
                        f"Failed to connect to {server.server_name}: "
                        f"{ExceptionLogger.format_exception_message(e)}"
                    )
                    resolution_errors.append(error_msg)
                    logger.warning(
                        "Failed to resolve server %s during resource search: %s: %s",
                        server.server_name,
                        type(e).__name__,
                        e,
                    )

        try:
            results = self.catalog.search(
                query=query,
                category=category,
                max_results=self.max_results,
            )
        except Exception as e:
            logger.error(
                "search_resources catalog search failed: %s: %s",
                type(e).__name__,
                e,
            )
            return f"Error searching resources: {type(e).__name__}: {e}"

        if not results:
            all_resources = self.catalog.list_resources(category=category)
            if all_resources:
                names = [r["name"] for r in all_resources]
                return (
                    f"No resources matched your search query, but the following "
                    f"resources are available in {category} category:\n"
                    f"{', '.join(names)}. "
                    f"Try searching with different keywords."
                )
            if resolution_errors:
                errors_detail = "\n".join(resolution_errors)
                return (
                    f"No resources found in {category}. "
                    f"Resource discovery failed for the following servers:\n"
                    f"{errors_detail}"
                )
            return f"No resources found matching your query in {category}."
        return json.dumps(results, indent=2)