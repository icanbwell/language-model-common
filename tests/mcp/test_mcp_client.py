"""Tests for mcp_client — session management, interceptor chain, content conversion."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import ToolException
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl

from languagemodelcommon.mcp.mcp_client import (
    build_interceptor_chain,
    convert_call_tool_result,
    convert_mcp_content_to_lc_block,
)
from languagemodelcommon.mcp.interceptors.types import (
    MCPToolCallRequest,
    MCPToolCallResult,
)


class TestBuildInterceptorChain:
    @pytest.mark.asyncio
    async def test_no_interceptors(self) -> None:
        """Base handler is returned directly when no interceptors."""
        base = AsyncMock(
            return_value=CallToolResult(content=[TextContent(type="text", text="ok")])
        )
        handler = build_interceptor_chain(base, None)
        request = MCPToolCallRequest(name="test", args={}, server_name="s1")
        result = await handler(request)
        base.assert_awaited_once_with(request)
        assert isinstance(result, CallToolResult)

    @pytest.mark.asyncio
    async def test_single_interceptor(self) -> None:
        """Single interceptor wraps the base handler."""
        call_order: list[str] = []

        async def base(req: MCPToolCallRequest) -> MCPToolCallResult:
            call_order.append("base")
            return CallToolResult(content=[TextContent(type="text", text="ok")])

        async def interceptor(
            req: MCPToolCallRequest,
            handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
        ) -> MCPToolCallResult:
            call_order.append("interceptor_before")
            result = await handler(req)
            call_order.append("interceptor_after")
            return result

        handler = build_interceptor_chain(base, [interceptor])  # type: ignore[list-item]
        request = MCPToolCallRequest(name="test", args={}, server_name="s1")
        await handler(request)
        assert call_order == ["interceptor_before", "base", "interceptor_after"]

    @pytest.mark.asyncio
    async def test_multiple_interceptors_onion_order(self) -> None:
        """Multiple interceptors execute in onion order (first registered = outermost)."""
        call_order: list[str] = []

        async def base(req: MCPToolCallRequest) -> MCPToolCallResult:
            call_order.append("base")
            return CallToolResult(content=[TextContent(type="text", text="ok")])

        async def interceptor_a(
            req: MCPToolCallRequest,
            handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
        ) -> MCPToolCallResult:
            call_order.append("a_before")
            result = await handler(req)
            call_order.append("a_after")
            return result

        async def interceptor_b(
            req: MCPToolCallRequest,
            handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
        ) -> MCPToolCallResult:
            call_order.append("b_before")
            result = await handler(req)
            call_order.append("b_after")
            return result

        handler = build_interceptor_chain(base, [interceptor_a, interceptor_b])  # type: ignore[list-item]
        request = MCPToolCallRequest(name="test", args={}, server_name="s1")
        await handler(request)
        assert call_order == [
            "a_before",
            "b_before",
            "base",
            "b_after",
            "a_after",
        ]

    @pytest.mark.asyncio
    async def test_interceptor_can_modify_request(self) -> None:
        """Interceptor can modify the request before passing to next handler."""

        async def base(req: MCPToolCallRequest) -> MCPToolCallResult:
            return CallToolResult(
                content=[TextContent(type="text", text=req.args.get("key", "none"))]
            )

        async def add_arg_interceptor(
            req: MCPToolCallRequest,
            handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
        ) -> MCPToolCallResult:
            modified = req.override(args={**req.args, "key": "injected"})
            return await handler(modified)

        handler = build_interceptor_chain(base, [add_arg_interceptor])  # type: ignore[list-item]
        request = MCPToolCallRequest(name="test", args={}, server_name="s1")
        result = await handler(request)
        assert result.content[0].text == "injected"  # type: ignore[union-attr]


class TestConvertMcpContentToLcBlock:
    def test_text_content(self) -> None:
        content = TextContent(type="text", text="hello world")
        result = convert_mcp_content_to_lc_block(content)
        assert result["type"] == "text"
        assert result["text"] == "hello world"

    def test_image_content(self) -> None:
        content = ImageContent(type="image", data="base64data", mimeType="image/png")
        result = convert_mcp_content_to_lc_block(content)
        assert result["type"] == "image"

    def test_embedded_text_resource(self) -> None:
        content = EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri=AnyUrl("file://test.txt"),
                text="some text",
            ),
        )
        result = convert_mcp_content_to_lc_block(content)
        assert result["type"] == "text"
        assert result["text"] == "some text"


class TestConvertCallToolResult:
    def test_successful_result(self) -> None:
        result = CallToolResult(
            content=[
                TextContent(type="text", text="line 1"),
                TextContent(type="text", text="line 2"),
            ]
        )
        blocks = convert_call_tool_result(result)
        assert len(blocks) == 2

    def test_error_result_raises_tool_exception(self) -> None:
        result = CallToolResult(
            content=[TextContent(type="text", text="Something went wrong")],
            isError=True,
        )
        with pytest.raises(ToolException, match="Something went wrong"):
            convert_call_tool_result(result)

    def test_empty_content(self) -> None:
        result = CallToolResult(content=[])
        blocks = convert_call_tool_result(result)
        assert blocks == []
