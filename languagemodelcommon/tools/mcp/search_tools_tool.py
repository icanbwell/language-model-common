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
import re
from typing import Literal, Tuple, Type

from langchain_core.tools import BaseTool
from oidcauthlib.auth.exceptions.authorization_needed_exception import (
    AuthorizationNeededException,
)
from pydantic import BaseModel, ConfigDict, Field

from languagemodelcommon.mcp.tool_catalog import (
    ToolCatalog,
    ToolResolverProtocol,
    ServerRegistration,
)
from languagemodelcommon.utilities.logger.exception_logger import ExceptionLogger
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

_WORD_RE = re.compile(r"[a-z0-9]+")

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


class SearchToolsInput(BaseModel):
    category: str = Field(
        ...,
        description="Category to search within. Use one of the category names from the system message.",
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
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    catalog: ToolCatalog
    resolver: ToolResolverProtocol | None = None
    max_results: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _query_matches_server(query: str, server: ServerRegistration) -> bool:
        """Check whether query terms overlap with the server's metadata.

        Used to decide whether to surface a login link for a server that
        requires authentication.  Without tool schemas loaded we can only
        match against the server name, category, and description.
        """
        query_tokens = set(_WORD_RE.findall(query.lower()))
        if not query_tokens:
            return True  # empty query matches everything

        corpus_parts: list[str] = [server.server_name]
        if server.category:
            corpus_parts.append(server.category)
        if server.agent_config:
            if server.agent_config.description:
                corpus_parts.append(server.agent_config.description)
            if server.agent_config.name:
                corpus_parts.append(server.agent_config.name)
            if server.agent_config.display_name:
                corpus_parts.append(server.agent_config.display_name)

        corpus_tokens = set(_WORD_RE.findall(" ".join(corpus_parts).lower()))
        return bool(query_tokens & corpus_tokens)

    def _run(self, query: str, category: str) -> str:
        raise NotImplementedError(
            "search_tools requires async execution for lazy tool resolution. "
            "Use _arun instead."
        )

    async def _arun(self, query: str, category: str) -> Tuple[str, str]:
        # Lazily resolve any unresolved servers matching the category
        resolution_errors: list[str] = []
        # Track auth exceptions for query-matching servers so we can
        # re-raise them when no tools are found, letting the gateway
        # render clickable login links directly to the user.
        auth_exceptions: list[AuthorizationNeededException] = []
        if self.resolver is not None:
            unresolved = self.catalog.get_unresolved_servers(category)
            for server in unresolved:
                try:
                    await self.catalog.resolve_server(server.server_name, self.resolver)
                except AuthorizationNeededException as e:
                    server_url = (
                        server.agent_config.url if server.agent_config else "unknown"
                    )
                    if self._query_matches_server(query, server):
                        auth_exceptions.append(e)
                    logger.info(
                        "Server %s at %s requires auth during search, skipping: %s",
                        server.server_name,
                        server_url,
                        ExceptionLogger.format_exception_message(e),
                    )
                except Exception as e:
                    server_url = (
                        server.agent_config.url if server.agent_config else "unknown"
                    )
                    error_msg = (
                        f"Failed to connect to {server.server_name} "
                        f"(url: {server_url}): "
                        f"{ExceptionLogger.format_exception_message(e)}"
                    )
                    resolution_errors.append(error_msg)
                    logger.warning(
                        "Failed to resolve server %s at %s during search: %s",
                        server.server_name,
                        server_url,
                        ExceptionLogger.format_exception_message(e),
                    )

        try:
            results = self.catalog.search(
                query=query,
                category=category,
                max_results=self.max_results,
            )
        except Exception as e:
            msg = ExceptionLogger.format_exception_message(e)
            logger.error("search_tools catalog search failed: %s", msg)
            error_text = f"Error searching tools: {msg}"
            return error_text, error_text

        if not results:
            # List all tools in the category so the LLM knows what's available
            all_tools = self.catalog.list_tools(category=category)
            if all_tools:
                tool_names = [t["name"] for t in all_tools]
                no_match_text = (
                    f"No tools matched your search query, but the following "
                    f"tools are available in {category} category:\n"
                    f"{', '.join(tool_names)}. "
                    f"Try searching with different keywords."
                )
                return no_match_text, no_match_text
            # When a query-matching server needs authentication and we
            # have no tools to show, re-raise the exception so the
            # gateway renders clickable login links directly to the
            # user instead of passing them through the LLM (which
            # strips the URLs).
            if auth_exceptions:
                raise auth_exceptions[0]
            if resolution_errors:
                errors_detail = "\n".join(resolution_errors)
                error_text = (
                    f"No tools found in {category}. "
                    f"Tool discovery failed for the following servers:\n"
                    f"{errors_detail}"
                )
                return error_text, error_text
            not_found_text = f"No tools found matching your query in {category}."
            return not_found_text, not_found_text

        # Content for the LLM (no scoring details)
        content_json = json.dumps(results, indent=2)

        # Artifact includes scores and matched terms for debugging/display
        scored_results = self.catalog.search_with_scores(
            query=query,
            category=category,
            max_results=self.max_results,
        )
        artifact_json = json.dumps({"results": scored_results}, indent=2)
        return content_json, artifact_json
