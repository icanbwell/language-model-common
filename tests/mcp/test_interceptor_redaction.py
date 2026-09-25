"""Tests for RedactionMcpCallInterceptor.

BAI-955: interceptor-shaped wrapper around `summarize_for_logging`, sitting
alongside TracingMcpCallInterceptor/TruncationMcpCallInterceptor
(languagemodelcommon/mcp/interceptors/), so any repo wiring MCP interceptors
gets safe argument logging without copying a function.
"""

import logging

import pytest
from mcp.types import CallToolResult, TextContent

from languagemodelcommon.mcp.interceptors.redaction import (
    RedactionMcpCallInterceptor,
)
from languagemodelcommon.mcp.interceptors.types import (
    MCPToolCallRequest,
    MCPToolCallResult,
)


class TestRedactionMcpCallInterceptor:
    @pytest.mark.asyncio
    async def test_forwards_request_unchanged_to_handler(self) -> None:
        interceptor = RedactionMcpCallInterceptor().get_tool_interceptor_redaction()
        request = MCPToolCallRequest(
            name="get_patient", args={"patient_id": "abc-123"}, server_name="fhir"
        )
        seen_requests: list[MCPToolCallRequest] = []

        async def handler(req: MCPToolCallRequest) -> MCPToolCallResult:
            seen_requests.append(req)
            return CallToolResult(content=[TextContent(type="text", text="ok")])

        result = await interceptor(request, handler)

        assert seen_requests == [request]
        assert isinstance(result, CallToolResult)

    @pytest.mark.asyncio
    async def test_logged_args_never_contain_marker_string(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        interceptor = RedactionMcpCallInterceptor().get_tool_interceptor_redaction()
        marker = "MARKER-DO-NOT-LEAK-55555"
        request = MCPToolCallRequest(
            name="search_notes", args={"query": marker}, server_name="fhir"
        )

        async def handler(req: MCPToolCallRequest) -> MCPToolCallResult:
            return CallToolResult(content=[TextContent(type="text", text="ok")])

        with caplog.at_level(logging.DEBUG):
            await interceptor(request, handler)

        assert marker not in caplog.text
