"""Tests for CallToolTool — meta-tool for calling specific MCP tools by name."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import ToolException
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from oidcauthlib.auth.exceptions.authorization_needed_exception import (
    AuthorizationNeededException,
)
from pydantic import AnyUrl

from languagemodelcommon.tools.mcp.call_tool_tool import (
    CallToolInput,
    CallToolTool,
    _call_tool_result_to_text,
)
from languagemodelcommon.mcp.tool_catalog import ToolCatalog
from languagemodelcommon.configs.schemas.config_schema import AgentConfig
from mcp.types import Tool as MCPTool


def _agent_config() -> AgentConfig:
    return AgentConfig(name="server1", url="https://example.com/mcp")


def _make_call_tool_tool(
    catalog: ToolCatalog,
    mcp_tool_provider: Any = None,
    auth_interceptor: Any = None,
    resolver: Any = None,
) -> CallToolTool:
    """Create a CallToolTool bypassing Pydantic validation for mock dependencies."""
    return CallToolTool.model_construct(
        name="call_tool",
        description="Call a specific tool by name with the given arguments.",
        args_schema=CallToolInput,
        response_format="content",
        catalog=catalog,
        mcp_tool_provider=mcp_tool_provider or MagicMock(),
        auth_interceptor=auth_interceptor or MagicMock(),
        resolver=resolver,
    )


class TestCallToolResultToText:
    def test_text_content(self) -> None:
        result = CallToolResult(content=[TextContent(type="text", text="Hello world")])
        assert _call_tool_result_to_text(result) == "Hello world"

    def test_multiple_text_blocks(self) -> None:
        result = CallToolResult(
            content=[
                TextContent(type="text", text="Line 1"),
                TextContent(type="text", text="Line 2"),
            ]
        )
        assert _call_tool_result_to_text(result) == "Line 1\nLine 2"

    def test_image_content(self) -> None:
        result = CallToolResult(
            content=[ImageContent(type="image", data="base64", mimeType="image/png")]
        )
        assert _call_tool_result_to_text(result) == "[Image: image/png]"

    def test_embedded_text_resource(self) -> None:
        result = CallToolResult(
            content=[
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=AnyUrl("file://test.txt"),
                        text="resource text",
                    ),
                )
            ]
        )
        assert _call_tool_result_to_text(result) == "resource text"

    def test_error_result(self) -> None:
        result = CallToolResult(
            content=[TextContent(type="text", text="Something failed")],
            isError=True,
        )
        text = _call_tool_result_to_text(result)
        assert text.startswith("Tool call failed:")
        assert "Something failed" in text


class TestCallToolTool:
    @pytest.mark.asyncio
    async def test_tool_not_found(self) -> None:
        catalog = ToolCatalog()
        tool = _make_call_tool_tool(catalog=catalog)
        with pytest.raises(ToolException, match="not found"):
            await tool._arun(name="nonexistent", arguments={})

    @pytest.mark.asyncio
    async def test_successful_call(self) -> None:
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="my_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="tool output")]
            )
        )
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = _make_call_tool_tool(catalog=catalog, mcp_tool_provider=mock_provider)
        result = await tool._arun(name="my_tool", arguments={"key": "value"})
        content, artifact = result
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "tool output"
        assert artifact is None
        mock_provider.execute_mcp_tool.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_image_content_reaches_the_llm_as_a_real_image_block(self) -> None:
        """Regression test for the discovery path's former lossy behavior:
        ImageContent must reach the model as an actual image content block
        (via convert_call_tool_result), not collapsed to placeholder text
        like "[Image: image/png]" — see mcp-fhir-agent's ADR 0017 for a
        concrete downstream consequence of the old behavior."""
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="image_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(
                content=[
                    ImageContent(type="image", data="base64data", mimeType="image/png")
                ]
            )
        )
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = _make_call_tool_tool(catalog=catalog, mcp_tool_provider=mock_provider)
        content, _artifact = await tool._arun(name="image_tool", arguments={})

        assert len(content) == 1
        assert content[0]["type"] == "image"
        assert content[0]["base64"] == "base64data"
        assert content[0]["mime_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_embedded_binary_resource_reaches_the_llm_as_a_file_block(
        self,
    ) -> None:
        """Same regression as above, for a non-image EmbeddedResource blob --
        must become a file content block, not a "[Resource: uri]" placeholder
        that discards the bytes entirely."""
        from mcp.types import BlobResourceContents

        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="pdf_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(
                content=[
                    EmbeddedResource(
                        type="resource",
                        resource=BlobResourceContents(
                            uri=AnyUrl("file://doc.pdf"),
                            blob="base64pdf",
                            mimeType="application/pdf",
                        ),
                    )
                ]
            )
        )
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = _make_call_tool_tool(catalog=catalog, mcp_tool_provider=mock_provider)
        content, _artifact = await tool._arun(name="pdf_tool", arguments={})

        assert len(content) == 1
        assert content[0]["type"] == "file"
        assert content[0]["base64"] == "base64pdf"
        assert content[0]["mime_type"] == "application/pdf"

    @pytest.mark.asyncio
    async def test_fetch_mcp_app_embed_still_receives_a_plain_text_summary(
        self,
    ) -> None:
        """fetch_mcp_app_embed seeds an MCP Apps UI iframe's
        window.__MCP_TOOL_RESULT__ with tool_result_text -- it must keep
        getting the plain-text summary (_call_tool_result_to_text), not the
        new multimodal content_blocks list, regardless of content type."""
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="image_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(
                content=[
                    ImageContent(type="image", data="base64data", mimeType="image/png")
                ]
            )
        )
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = _make_call_tool_tool(catalog=catalog, mcp_tool_provider=mock_provider)
        await tool._arun(name="image_tool", arguments={})

        call_kwargs = mock_provider.fetch_mcp_app_embed.call_args.kwargs
        assert call_kwargs["tool_result_text"] == "[Image: image/png]"

    @pytest.mark.asyncio
    async def test_embedded_text_resource_reaches_the_llm_as_a_text_block(
        self,
    ) -> None:
        """Full _arun integration for the EmbeddedResource(TextResourceContents)
        case -- the lower-level conversion is already covered by
        TestConvertCallToolResult/content_conversion.py's own tests, but that
        doesn't prove _arun actually wires it through end to end."""
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="doc_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(
                content=[
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri=AnyUrl("file://notes.txt"),
                            text="resource text",
                        ),
                    )
                ]
            )
        )
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = _make_call_tool_tool(catalog=catalog, mcp_tool_provider=mock_provider)
        content, _artifact = await tool._arun(name="doc_tool", arguments={})

        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "resource text"

    @pytest.mark.asyncio
    async def test_isError_checked_before_content_conversion_that_could_raise(
        self,
    ) -> None:
        """convert_call_tool_result raises NotImplementedError on AudioContent
        (unsupported). An MCP server reporting isError=True with an
        AudioContent block must still surface as a clean ToolException, not
        crash on an unrelated conversion error -- proves the isError check
        runs before convert_call_tool_result is ever called, not after."""
        from mcp.types import AudioContent

        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="rejecting_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(
                content=[
                    AudioContent(type="audio", data="base64audio", mimeType="audio/wav")
                ],
                isError=True,
            )
        )
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = _make_call_tool_tool(catalog=catalog, mcp_tool_provider=mock_provider)
        with pytest.raises(ToolException) as exc_info:
            await tool._arun(name="rejecting_tool", arguments={})
        assert str(exc_info.value).startswith("Tool call failed:")
        mock_provider.fetch_mcp_app_embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_result_with_unconvertible_content_falls_back_to_text(
        self,
    ) -> None:
        """Regression test: convert_call_tool_result raises NotImplementedError
        on AudioContent. Before this fix, a *successful* MCP tool result
        containing AudioContent (e.g. a TTS/voice tool) would be turned into a
        ToolException by the outer `except Exception` handler -- a
        success-to-error regression, since the pre-existing
        _call_tool_result_to_text tolerated any content type via str(block).
        Must fall back to the tolerant text summary instead of failing the
        call."""
        from mcp.types import AudioContent

        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="tts_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(
                content=[
                    AudioContent(type="audio", data="base64audio", mimeType="audio/wav")
                ]
            )
        )
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = _make_call_tool_tool(catalog=catalog, mcp_tool_provider=mock_provider)
        content, _artifact = await tool._arun(name="tts_tool", arguments={})

        assert len(content) == 1
        assert content[0]["type"] == "text"
        mock_provider.fetch_mcp_app_embed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_call_failure_returns_error_string(self) -> None:
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="failing_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )

        tool = _make_call_tool_tool(catalog=catalog, mcp_tool_provider=mock_provider)
        with pytest.raises(ToolException) as exc_info:
            await tool._arun(name="failing_tool")
        assert "RuntimeError" in str(exc_info.value)
        assert "connection lost" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_inner_tool_error_result_raises_tool_exception(self) -> None:
        """The wrapped MCP tool itself can return isError=True without raising —
        CallToolTool must turn that into a ToolException so the resulting
        ToolMessage carries status="error" instead of looking like a success."""
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="rejecting_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="validation failed")],
                isError=True,
            )
        )
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = _make_call_tool_tool(catalog=catalog, mcp_tool_provider=mock_provider)
        with pytest.raises(ToolException) as exc_info:
            await tool._arun(name="rejecting_tool", arguments={})
        assert str(exc_info.value).startswith("Tool call failed:")
        assert "validation failed" in str(exc_info.value)
        # fetch_mcp_app_embed is a no-op on the error path
        mock_provider.fetch_mcp_app_embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_inner_tool_error_result_sets_tool_message_status_error(
        self,
    ) -> None:
        """End-to-end: going through the real BaseTool.arun (not just _arun
        directly) must produce a ToolMessage with status="error", since that's
        the actual signal the calling model/harness reacts to."""
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="rejecting_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="validation failed")],
                isError=True,
            )
        )
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = CallToolTool.model_construct(
            name="call_tool",
            description="Call a specific tool by name with the given arguments.",
            args_schema=CallToolInput,
            response_format="content_and_artifact",
            handle_tool_error=True,
            catalog=catalog,
            mcp_tool_provider=mock_provider,
            auth_interceptor=MagicMock(),
        )
        message = await tool.arun(
            tool_input={"name": "rejecting_tool", "arguments": {}},
            tool_call_id="call-1",
        )
        assert message.status == "error"
        assert "validation failed" in message.content

    def test_sync_run_raises(self) -> None:
        catalog = ToolCatalog()
        tool = _make_call_tool_tool(catalog=catalog)
        with pytest.raises(NotImplementedError):
            tool._run(name="test")

    @pytest.mark.asyncio
    async def test_default_arguments(self) -> None:
        """Arguments default to empty dict when None."""
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.add_tools(
            server_name="server1",
            category=None,
            tools=[MCPTool(name="my_tool", inputSchema={"type": "object"})],
            agent_config=config,
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(content=[TextContent(type="text", text="ok")])
        )
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = _make_call_tool_tool(catalog=catalog, mcp_tool_provider=mock_provider)
        await tool._arun(name="my_tool", arguments=None)
        call_kwargs = mock_provider.execute_mcp_tool.call_args.kwargs
        assert call_kwargs["arguments"] == {}

    @pytest.mark.asyncio
    async def test_lazily_resolves_unresolved_server_when_tool_missing(self) -> None:
        """Regression test: the catalog is rebuilt fresh per request, so a
        server can be registered-but-unresolved even though search_tools
        already surfaced this tool name in an earlier turn. call_tool must
        resolve on demand instead of failing immediately."""
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.register_server(
            server_name="server1", category="connections", agent_config=config
        )

        resolver = MagicMock()
        resolver.resolve_tools = AsyncMock(
            return_value=[
                MCPTool(name="list_connections", inputSchema={"type": "object"})
            ]
        )

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="connections: []")]
            )
        )
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = _make_call_tool_tool(
            catalog=catalog, mcp_tool_provider=mock_provider, resolver=resolver
        )
        content, _artifact = await tool._arun(name="list_connections", arguments={})

        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "connections: []"
        resolver.resolve_tools.assert_awaited_once()
        mock_provider.execute_mcp_tool.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_still_not_found_after_resolving_all_unresolved_servers(
        self,
    ) -> None:
        """Resolution runs, but no unresolved server actually has the
        requested tool — must still raise, not loop or silently succeed."""
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.register_server(
            server_name="server1", category="connections", agent_config=config
        )

        resolver = MagicMock()
        resolver.resolve_tools = AsyncMock(
            return_value=[MCPTool(name="other_tool", inputSchema={"type": "object"})]
        )

        tool = _make_call_tool_tool(catalog=catalog, resolver=resolver)
        with pytest.raises(ToolException, match="not found"):
            await tool._arun(name="list_connections", arguments={})

        resolver.resolve_tools.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resolution_auth_needed_propagates(self) -> None:
        """If resolving the server requires auth, that exception must
        propagate so the gateway can render a login prompt — it must not be
        swallowed into a generic 'not found' ToolException."""
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.register_server(
            server_name="server1", category="connections", agent_config=config
        )

        resolver = MagicMock()
        resolver.resolve_tools = AsyncMock(
            side_effect=AuthorizationNeededException(message="login required")
        )

        tool = _make_call_tool_tool(catalog=catalog, resolver=resolver)
        with pytest.raises(AuthorizationNeededException):
            await tool._arun(name="list_connections", arguments={})

    @pytest.mark.asyncio
    async def test_unrelated_auth_required_server_does_not_block_others(self) -> None:
        """An unresolved server that requires auth must not abort resolution
        of other unresolved servers -- the target tool may live on one that
        resolves without auth."""
        catalog = ToolCatalog()
        auth_config = AgentConfig(
            name="auth-server", url="https://auth.example.com/mcp"
        )
        catalog.register_server(
            server_name="auth-server", category="auth", agent_config=auth_config
        )
        ok_config = _agent_config()
        catalog.register_server(
            server_name="server1", category="connections", agent_config=ok_config
        )

        async def resolve_tools(*, agent_config: AgentConfig) -> list[MCPTool]:
            if agent_config.url == auth_config.url:
                raise AuthorizationNeededException(message="login required")
            return [MCPTool(name="list_connections", inputSchema={"type": "object"})]

        resolver = MagicMock()
        resolver.resolve_tools = AsyncMock(side_effect=resolve_tools)

        mock_provider = MagicMock()
        mock_provider.execute_mcp_tool = AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="connections: []")]
            )
        )
        mock_provider.fetch_mcp_app_embed = AsyncMock(return_value=None)

        tool = _make_call_tool_tool(
            catalog=catalog, mcp_tool_provider=mock_provider, resolver=resolver
        )
        content, _artifact = await tool._arun(name="list_connections", arguments={})

        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "connections: []"
        assert resolver.resolve_tools.await_count == 2

    @pytest.mark.asyncio
    async def test_raises_auth_exception_only_after_all_servers_attempted(
        self,
    ) -> None:
        """If the tool is still missing after every unresolved server has
        been attempted, and at least one required auth, surface that auth
        exception (not a generic 'not found') so the gateway can render a
        login prompt."""
        catalog = ToolCatalog()
        ok_config = _agent_config()
        catalog.register_server(
            server_name="server1", category="connections", agent_config=ok_config
        )
        auth_config = AgentConfig(
            name="auth-server", url="https://auth.example.com/mcp"
        )
        catalog.register_server(
            server_name="auth-server", category="auth", agent_config=auth_config
        )

        async def resolve_tools(*, agent_config: AgentConfig) -> list[MCPTool]:
            if agent_config.url == auth_config.url:
                raise AuthorizationNeededException(message="login required")
            return [MCPTool(name="other_tool", inputSchema={"type": "object"})]

        resolver = MagicMock()
        resolver.resolve_tools = AsyncMock(side_effect=resolve_tools)

        tool = _make_call_tool_tool(catalog=catalog, resolver=resolver)
        with pytest.raises(AuthorizationNeededException):
            await tool._arun(name="list_connections", arguments={})

        assert resolver.resolve_tools.await_count == 2

    @pytest.mark.asyncio
    async def test_no_resolver_skips_resolution_and_fails_fast(self) -> None:
        """Without a resolver (backward compatible default), an unresolved
        server must not be treated as a resolution opportunity — the
        original immediate 'not found' behavior is preserved."""
        catalog = ToolCatalog()
        config = _agent_config()
        catalog.register_server(
            server_name="server1", category="connections", agent_config=config
        )

        tool = _make_call_tool_tool(catalog=catalog)
        with pytest.raises(ToolException, match="not found"):
            await tool._arun(name="list_connections", arguments={})
