"""Tests for mcp_client — session management, interceptor chain, content conversion."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import (
    CallToolResult,
    ElicitRequest,
    ElicitRequestFormParams,
    EmbeddedResource,
    ImageContent,
    InputRequiredResult,
    TextContent,
    TextResourceContents,
)

from languagemodelcommon.mcp.mcp_client.content_conversion import (
    convert_call_tool_result,
    convert_mcp_content_to_lc_block,
)
from languagemodelcommon.mcp.mcp_client.session import MCPConnectionConfig
from languagemodelcommon.mcp.mcp_client.tool_invocation import (
    _execute_tool_call_with_heartbeat,
    build_interceptor_chain,
    call_mcp_tool_raw,
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
        handler = build_interceptor_chain(base_handler=base, tool_interceptors=None)
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

        handler = build_interceptor_chain(
            base_handler=base,
            tool_interceptors=[interceptor],  # type: ignore[list-item]
        )
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

        handler = build_interceptor_chain(
            base_handler=base,
            tool_interceptors=[interceptor_a, interceptor_b],  # type: ignore[list-item]
        )
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

        handler = build_interceptor_chain(
            base_handler=base,
            tool_interceptors=[add_arg_interceptor],  # type: ignore[list-item]
        )
        request = MCPToolCallRequest(name="test", args={}, server_name="s1")
        result = await handler(request)
        assert result.content[0].text == "injected"  # type: ignore[union-attr]


def test_mcp_tool_call_request_carries_input_responses_and_request_state() -> None:
    """MCPToolCallRequest accepts the SEP-2322 retry fields, defaulting to None."""
    from mcp.types import ElicitResult

    bare = MCPToolCallRequest(name="test", args={}, server_name="s1")
    assert bare.input_responses is None
    assert bare.request_state is None

    retry = MCPToolCallRequest(
        name="test",
        args={},
        server_name="s1",
        input_responses={
            "confirm": ElicitResult(action="accept", content={"confirm": True})
        },
        request_state="opaque-state-123",
    )
    assert retry.request_state == "opaque-state-123"
    assert retry.input_responses is not None
    assert retry.input_responses["confirm"].action == "accept"  # type: ignore[union-attr]


class TestExecuteToolCallWithHeartbeat:
    @pytest.mark.asyncio
    async def test_returns_input_required_result_when_allowed(self) -> None:
        """A guard-tool ask is returned to the caller, not raised, when
        allow_input_required=True."""
        input_required = InputRequiredResult(
            result_type="input_required",
            input_requests={
                "confirm": ElicitRequest(
                    method="elicitation/create",
                    params=ElicitRequestFormParams(
                        message="Proceed?",
                        requested_schema={
                            "type": "object",
                            "properties": {"confirm": {"type": "boolean"}},
                            "required": ["confirm"],
                        },
                    ),
                )
            },
        )
        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=input_required)

        result = await _execute_tool_call_with_heartbeat(
            session=session,
            name="save_fhir_resource",
            arguments={"resource": {"resourceType": "Patient"}},
            progress_callback=None,
            server_name="mcp-fhir-agent",
            heartbeat_interval_seconds=15.0,
            input_responses=None,
            request_state=None,
            allow_input_required=True,
        )

        assert result is input_required
        session.call_tool.assert_awaited_once_with(
            "save_fhir_resource",
            {"resource": {"resourceType": "Patient"}},
            progress_callback=None,
            input_responses=None,
            request_state=None,
            allow_input_required=True,
        )


@asynccontextmanager
async def _fake_session_cm() -> AsyncIterator[AsyncMock]:
    session = AsyncMock()
    session.initialize = AsyncMock()
    yield session


@asynccontextmanager
async def _fake_session_cm_with(session: AsyncMock) -> AsyncIterator[AsyncMock]:
    """Like ``_fake_session_cm`` but wraps a caller-provided, pre-configured
    session (e.g. one with a custom ``call_tool`` side effect)."""
    session.initialize = AsyncMock()
    session.get_server_capabilities = MagicMock(return_value=None)
    yield session


class TestCallMcpToolRaw:
    @pytest.mark.asyncio
    async def test_forwards_input_responses_and_request_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """call_mcp_tool_raw forwards retry fields (and an explicit
        allow_input_required=True opt-in) to the underlying session call."""
        from mcp.types import ElicitResult

        captured: dict[str, Any] = {}

        async def fake_execute_tool_call_with_heartbeat(
            **kwargs: Any,
        ) -> CallToolResult:
            captured.update(kwargs)
            return CallToolResult(content=[TextContent(type="text", text="saved")])

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.tool_invocation._execute_tool_call_with_heartbeat",
            fake_execute_tool_call_with_heartbeat,
        )
        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.session.create_mcp_session",
            lambda *args, **kwargs: _fake_session_cm(),
        )

        result = await call_mcp_tool_raw(
            config=MCPConnectionConfig(url="https://example.test/mcp"),
            tool_name="save_fhir_resource",
            arguments={"resource": {"resourceType": "Patient"}},
            server_name="mcp-fhir-agent",
            input_responses={
                "confirm": ElicitResult(action="accept", content={"confirm": True})
            },
            request_state="opaque-state-123",
            allow_input_required=True,
        )

        assert isinstance(result, CallToolResult)
        assert captured["request_state"] == "opaque-state-123"
        assert captured["input_responses"]["confirm"].action == "accept"
        assert captured["allow_input_required"] is True

    @pytest.mark.asyncio
    async def test_default_allow_input_required_preserves_pre_sep2322_raise_behavior(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression test: a caller that never passes allow_input_required
        (e.g. baileyai-skills-service's meta-tools, pinned to the old
        CallToolResult-only contract) must keep getting a RuntimeError raised
        on an InputRequiredResult, not the value silently returned -- this
        mirrors ClientSession.call_tool's own raise-vs-return contract.
        """

        async def fake_call_tool(
            name: str,
            arguments: dict[str, Any],
            progress_callback: Any = None,
            *,
            input_responses: Any = None,
            request_state: str | None = None,
            allow_input_required: bool = False,
        ) -> CallToolResult:
            if allow_input_required:
                raise AssertionError("expected allow_input_required=False by default")
            raise RuntimeError("Server requires input before call_tool can complete")

        session = AsyncMock()
        session.call_tool = fake_call_tool

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.session.create_mcp_session",
            lambda *args, **kwargs: _fake_session_cm_with(session),
        )

        with pytest.raises(
            RuntimeError, match="Server requires input before call_tool can complete"
        ):
            await call_mcp_tool_raw(
                config=MCPConnectionConfig(url="https://example.test/mcp"),
                tool_name="save_fhir_resource",
                arguments={"resource": {"resourceType": "Patient"}},
                server_name="mcp-fhir-agent",
            )

    @pytest.mark.asyncio
    async def test_allow_input_required_true_returns_input_required_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With allow_input_required=True explicitly requested, an
        InputRequiredResult is returned to the caller instead of raising."""
        input_required = InputRequiredResult(
            result_type="input_required",
            input_requests={
                "confirm": ElicitRequest(
                    method="elicitation/create",
                    params=ElicitRequestFormParams(
                        message="Proceed?",
                        requested_schema={
                            "type": "object",
                            "properties": {"confirm": {"type": "boolean"}},
                            "required": ["confirm"],
                        },
                    ),
                )
            },
            request_state="opaque-state-123",
        )

        async def fake_call_tool(
            name: str,
            arguments: dict[str, Any],
            progress_callback: Any = None,
            *,
            input_responses: Any = None,
            request_state: str | None = None,
            allow_input_required: bool = False,
        ) -> CallToolResult | InputRequiredResult:
            if not allow_input_required:
                raise AssertionError("expected allow_input_required=True")
            return input_required

        session = AsyncMock()
        session.call_tool = fake_call_tool

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.session.create_mcp_session",
            lambda *args, **kwargs: _fake_session_cm_with(session),
        )

        result = await call_mcp_tool_raw(
            config=MCPConnectionConfig(url="https://example.test/mcp"),
            tool_name="save_fhir_resource",
            arguments={"resource": {"resourceType": "Patient"}},
            server_name="mcp-fhir-agent",
            allow_input_required=True,
        )

        assert result is input_required

    @pytest.mark.asyncio
    async def test_session_pool_branch_forwards_retry_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The `session_pool is not None` branch of `_make_execute_tool`'s
        execute_tool() forwards input_responses/request_state/allow_input_required
        to the underlying session.call_tool, the same way the one-shot
        fallback branches (tested above) already do."""
        from mcp.types import ElicitResult

        from languagemodelcommon.mcp.mcp_client.session_pool import McpSessionPool

        captured: dict[str, Any] = {}

        async def fake_call_tool(
            name: str,
            arguments: dict[str, Any],
            progress_callback: Any = None,
            *,
            input_responses: Any = None,
            request_state: str | None = None,
            allow_input_required: bool = False,
        ) -> CallToolResult:
            captured["input_responses"] = input_responses
            captured["request_state"] = request_state
            captured["allow_input_required"] = allow_input_required
            return CallToolResult(content=[TextContent(type="text", text="saved")])

        mock_session = AsyncMock()
        mock_session.call_tool = fake_call_tool
        mock_session.get_server_capabilities = MagicMock(return_value=None)
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.session.create_mcp_session",
            MagicMock(return_value=mock_cm),
        )

        config: MCPConnectionConfig = {"url": "https://example.test/mcp"}

        async with McpSessionPool() as pool:
            result = await call_mcp_tool_raw(
                config=config,
                tool_name="save_fhir_resource",
                arguments={"resource": {"resourceType": "Patient"}},
                server_name="mcp-fhir-agent",
                session_pool=pool,
                input_responses={
                    "confirm": ElicitResult(action="accept", content={"confirm": True})
                },
                request_state="opaque-state-123",
                allow_input_required=True,
            )

        assert isinstance(result, CallToolResult)
        assert captured["request_state"] == "opaque-state-123"
        assert captured["input_responses"]["confirm"].action == "accept"
        assert captured["allow_input_required"] is True


class TestOneShotFallbackCmLifecycle:
    """BAI-889: the one-shot fallback (no session_pool) now gets its
    session from open_initialized_mcp_session and must exit that CM in a
    finally block itself, unlike the old `async with create_mcp_session(...)`.
    These pin that the CM is exited exactly once regardless of how the tool
    call finishes, and that a tool-call exception is re-raised as-is (not
    re-wrapped) after the CM is closed."""

    @staticmethod
    def _patch_open_initialized_mcp_session(
        monkeypatch: pytest.MonkeyPatch, *, session: AsyncMock
    ) -> AsyncMock:
        mock_cm = AsyncMock()
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        async def fake_open(*args: Any, **kwargs: Any) -> tuple[AsyncMock, AsyncMock]:
            return mock_cm, session

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.tool_invocation.open_initialized_mcp_session",
            fake_open,
        )
        return mock_cm

    @pytest.mark.asyncio
    async def test_exits_cm_once_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = AsyncMock()
        session.get_server_capabilities = MagicMock(return_value=None)
        mock_cm = self._patch_open_initialized_mcp_session(monkeypatch, session=session)

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.tool_invocation._execute_tool_call_with_heartbeat",
            AsyncMock(
                return_value=CallToolResult(
                    content=[TextContent(type="text", text="ok")]
                )
            ),
        )

        result = await call_mcp_tool_raw(
            config=MCPConnectionConfig(url="https://example.test/mcp"),
            tool_name="save_fhir_resource",
            arguments={},
            server_name="mcp-fhir-agent",
        )

        assert isinstance(result, CallToolResult)
        mock_cm.__aexit__.assert_awaited_once_with(None, None, None)

    @pytest.mark.asyncio
    async def test_exits_cm_and_reraises_unwrapped_on_tool_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = AsyncMock()
        session.get_server_capabilities = MagicMock(return_value=None)
        mock_cm = self._patch_open_initialized_mcp_session(monkeypatch, session=session)

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.tool_invocation._execute_tool_call_with_heartbeat",
            AsyncMock(side_effect=RuntimeError("Server requires input")),
        )

        with pytest.raises(RuntimeError, match="Server requires input"):
            await call_mcp_tool_raw(
                config=MCPConnectionConfig(url="https://example.test/mcp"),
                tool_name="save_fhir_resource",
                arguments={},
                server_name="mcp-fhir-agent",
            )

        mock_cm.__aexit__.assert_awaited_once_with(None, None, None)

    @pytest.mark.asyncio
    async def test_exits_cm_on_cancellation_during_tool_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CancelledError is a BaseException, not an Exception, so it skips
        the fallback's `except Exception as e:` -- the CM must still be
        exited via the outer `finally`, and cancellation must propagate."""
        session = AsyncMock()
        session.get_server_capabilities = MagicMock(return_value=None)
        mock_cm = self._patch_open_initialized_mcp_session(monkeypatch, session=session)

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.tool_invocation._execute_tool_call_with_heartbeat",
            AsyncMock(side_effect=asyncio.CancelledError()),
        )

        with pytest.raises(asyncio.CancelledError):
            await call_mcp_tool_raw(
                config=MCPConnectionConfig(url="https://example.test/mcp"),
                tool_name="save_fhir_resource",
                arguments={},
                server_name="mcp-fhir-agent",
            )

        mock_cm.__aexit__.assert_awaited_once_with(None, None, None)


class TestConvertMcpContentToLcBlock:
    def test_text_content(self) -> None:
        content = TextContent(type="text", text="hello world")
        result = convert_mcp_content_to_lc_block(content)
        assert result["type"] == "text"
        assert result["text"] == "hello world"

    def test_image_content(self) -> None:
        content = ImageContent(type="image", data="base64data", mime_type="image/png")
        result = convert_mcp_content_to_lc_block(content)
        assert result["type"] == "image"

    def test_embedded_text_resource(self) -> None:
        content = EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri="file://test.txt",
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

    def test_error_result_returns_error_text(self) -> None:
        result = CallToolResult(
            content=[TextContent(type="text", text="Something went wrong")],
            is_error=True,
        )
        blocks = convert_call_tool_result(result)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Error: Something went wrong"

    def test_empty_content(self) -> None:
        result = CallToolResult(content=[])
        blocks = convert_call_tool_result(result)
        assert blocks == []
