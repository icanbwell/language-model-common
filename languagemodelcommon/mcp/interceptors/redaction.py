import logging
from typing import Awaitable, Callable

from languagemodelcommon.mcp.interceptors.types import (
    MCPToolCallRequest,
    MCPToolCallResult,
    ToolCallInterceptor,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS
from languagemodelcommon.utilities.logger.summarization import summarize_for_logging

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


class RedactionMcpCallInterceptor:
    """Interceptor that logs a redaction-safe summary of tool-call arguments.

    BAI-955: sits alongside TracingMcpCallInterceptor/TruncationMcpCallInterceptor
    on the ToolCallInterceptor protocol (types.py). Built on the shared
    `summarize_for_logging` primitive, modeled on
    `baileyai-skills-service`'s `_safe_summarize_args()`. Never logs or
    forwards a raw argument value - only wraps the call, logging the redacted
    summary before delegating to the handler unchanged.
    """

    def get_tool_interceptor_redaction(self) -> ToolCallInterceptor:
        """Interceptor that logs redacted tool-call arguments before invocation."""

        async def tool_interceptor_redaction(
            request: MCPToolCallRequest,
            handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
        ) -> MCPToolCallResult:
            logger.debug(
                "MCP tool call %s args (redacted): %s",
                request.name,
                summarize_for_logging(request.args),
            )
            return await handler(request)

        return tool_interceptor_redaction
