"""Meta-discovery tool: read_resource.

Reads an MCP resource by URI and returns its content. The LLM calls this
after discovering resources via search_resources.
"""

from __future__ import annotations

import logging
from typing import Literal, Type

from langchain_core.tools import BaseTool
from mcp.types import ReadResourceResult
from pydantic import BaseModel, ConfigDict, Field

from oidcauthlib.auth.exceptions.authorization_needed_exception import (
    AuthorizationNeededException,
)

from languagemodelcommon.mcp.interceptors.auth import AuthMcpCallInterceptor
from languagemodelcommon.mcp.mcp_client import convert_resource_contents_to_text
from languagemodelcommon.mcp.mcp_tool_provider import MCPToolProvider
from languagemodelcommon.mcp.resource_catalog import ResourceCatalog
from languagemodelcommon.utilities.logger.exception_logger import ExceptionLogger
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


class ReadResourceInput(BaseModel):
    uri: str = Field(
        ...,
        description="The URI of the resource to read (from search_resources results).",
    )


class ReadResourceTool(BaseTool):
    """Read an MCP resource by URI and return its content."""

    name: str = "read_resource"
    description: str = (
        "Read a resource by its URI. Use search_resources first to find "
        "the resource URI. Returns the resource content as text."
    )
    args_schema: Type[BaseModel] = ReadResourceInput
    response_format: Literal["content", "content_and_artifact"] = "content"

    catalog: ResourceCatalog
    mcp_tool_provider: MCPToolProvider
    auth_interceptor: AuthMcpCallInterceptor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, uri: str) -> str:
        raise NotImplementedError("Use async version")

    async def _arun(self, uri: str) -> str:
        agent_config = self.catalog.get_resource_agent_config(uri)
        if agent_config is None:
            return (
                f"Resource '{uri}' not found in any registered server. "
                f"Use search_resources to find available resources."
            )

        try:
            result: ReadResourceResult = (
                await self.mcp_tool_provider.execute_mcp_resource_read(
                    uri=uri,
                    agent_config=agent_config,
                    auth_interceptor=self.auth_interceptor,
                )
            )
            return convert_resource_contents_to_text(result)
        except AuthorizationNeededException:
            raise
        except Exception as e:
            error_detail = ExceptionLogger.format_exception_message(e)
            logger.error(
                "read_resource failed for %s: %s",
                uri,
                error_detail,
            )
            return f"Error reading resource '{uri}': {error_detail}"
