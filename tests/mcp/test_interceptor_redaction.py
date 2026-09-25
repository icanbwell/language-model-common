"""Tests for RedactionMcpCallInterceptor.

BAI-955: interceptor-shaped wrapper around `summarize_for_logging`, sitting
alongside TracingMcpCallInterceptor/TruncationMcpCallInterceptor
(languagemodelcommon/mcp/interceptors/), so any repo wiring MCP interceptors
gets safe argument logging without copying a function.
"""

import io
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

_REDACTION_LOGGER_NAME = "languagemodelcommon.mcp.interceptors.redaction"


def _capture_rendered_log_output() -> tuple[logging.Logger, io.StringIO]:
    """Attach a real logging.Handler + Formatter to capture rendered DEBUG text.

    caplog record attributes alone don't prove what a real handler would
    render - this exercises the actual formatting path.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))

    target_logger = logging.getLogger(_REDACTION_LOGGER_NAME)
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.DEBUG)
    return target_logger, stream


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

    @pytest.mark.asyncio
    async def test_rendered_output_shows_identifier_shaped_arg_names_never_values(
        self,
    ) -> None:
        interceptor = RedactionMcpCallInterceptor().get_tool_interceptor_redaction()
        marker = "MARKER-DO-NOT-LEAK-77777"
        request = MCPToolCallRequest(
            name="get_patient",
            args={"patient_id": marker, "12345": "also-a-secret"},
            server_name="fhir",
        )

        async def noop_handler(req: MCPToolCallRequest) -> MCPToolCallResult:
            return CallToolResult(content=[TextContent(type="text", text="ok")])

        target_logger, stream = _capture_rendered_log_output()
        try:
            await interceptor(request, noop_handler)
        finally:
            target_logger.handlers.clear()

        rendered = stream.getvalue()
        assert "patient_id" in rendered
        assert marker not in rendered
        assert "also-a-secret" not in rendered
        assert "12345" not in rendered
