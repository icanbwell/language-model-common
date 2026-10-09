"""BAI-1118 / ADR 0003: configurable MCP protocol negotiation mode."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from languagemodelcommon.mcp.mcp_client.negotiation_mode import (
    McpProtocolNegotiationMode,
)
from languagemodelcommon.mcp.mcp_client.session import (
    open_initialized_mcp_session,
)
from languagemodelcommon.mcp.mcp_client.session_pool import McpSessionPool
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)

SESSION_MODULE = "languagemodelcommon.mcp.mcp_client.session"


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (None, McpProtocolNegotiationMode.LEGACY),
        ("", McpProtocolNegotiationMode.LEGACY),
        ("legacy", McpProtocolNegotiationMode.LEGACY),
        ("auto", McpProtocolNegotiationMode.AUTO),
        (" AUTO ", McpProtocolNegotiationMode.AUTO),
        ("2026-07-28", McpProtocolNegotiationMode.LEGACY),
        ("garbage", McpProtocolNegotiationMode.LEGACY),
    ],
)
def test_env_var_maps_to_mode(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str | None,
    expected: McpProtocolNegotiationMode,
) -> None:
    if raw_value is None:
        monkeypatch.delenv("MCP_PROTOCOL_NEGOTIATION_MODE", raising=False)
    else:
        monkeypatch.setenv("MCP_PROTOCOL_NEGOTIATION_MODE", raw_value)

    assert (
        LanguageModelCommonEnvironmentVariables().mcp_protocol_negotiation_mode
        == expected
    )


@pytest.mark.parametrize(
    "mode", [McpProtocolNegotiationMode.LEGACY, McpProtocolNegotiationMode.AUTO]
)
@pytest.mark.asyncio
async def test_open_session_retries_transient_failure_in_both_modes(
    mode: McpProtocolNegotiationMode,
) -> None:
    good_session = MagicMock()
    seen_modes: list[McpProtocolNegotiationMode | None] = []

    def _create(*_args: object, **kwargs: object) -> object:
        seen_modes.append(kwargs.get("negotiation_mode"))  # type: ignore[arg-type]
        first_attempt = len(seen_modes) == 1

        @asynccontextmanager
        async def _cm() -> AsyncIterator[MagicMock]:
            if first_attempt:
                raise ConnectionError("reset")
            yield good_session

        return _cm()

    with patch(f"{SESSION_MODULE}.create_mcp_session", new=_create):
        cm, session = await open_initialized_mcp_session(
            {"url": "http://x"}, base_delay_seconds=0.0, negotiation_mode=mode
        )
        await cm.__aexit__(None, None, None)

    assert session is good_session
    assert seen_modes == [mode, mode]


@pytest.mark.asyncio
async def test_open_session_does_not_retry_cancellation() -> None:
    import asyncio

    calls = 0

    def _create(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1

        @asynccontextmanager
        async def _cm() -> AsyncIterator[MagicMock]:
            raise asyncio.CancelledError()
            yield MagicMock()  # pragma: no cover

        return _cm()

    with patch(f"{SESSION_MODULE}.create_mcp_session", new=_create):
        with pytest.raises(asyncio.CancelledError):
            await open_initialized_mcp_session(
                {"url": "http://x"},
                base_delay_seconds=0.0,
                negotiation_mode=McpProtocolNegotiationMode.AUTO,
            )

    assert calls == 1


@pytest.mark.asyncio
async def test_open_session_defaults_to_legacy() -> None:
    seen: list[object] = []

    def _create(*_args: object, **kwargs: object) -> object:
        seen.append(kwargs.get("negotiation_mode"))

        @asynccontextmanager
        async def _cm() -> AsyncIterator[MagicMock]:
            yield MagicMock()

        return _cm()

    with patch(f"{SESSION_MODULE}.create_mcp_session", new=_create):
        cm, _ = await open_initialized_mcp_session({"url": "http://x"})
        await cm.__aexit__(None, None, None)

    assert seen == [McpProtocolNegotiationMode.LEGACY]


@pytest.mark.asyncio
async def test_pool_passes_its_mode_to_session_open() -> None:
    opened = AsyncMock(return_value=(MagicMock(__aexit__=AsyncMock()), MagicMock()))
    with patch(
        "languagemodelcommon.mcp.mcp_client.session_pool.open_initialized_mcp_session",
        new=opened,
    ):
        async with McpSessionPool(
            negotiation_mode=McpProtocolNegotiationMode.AUTO
        ) as pool:
            await pool.get_session({"url": "http://x"})

    assert (
        opened.await_args_list[0].kwargs["negotiation_mode"]
        is McpProtocolNegotiationMode.AUTO
    )


def _patch_one_shot_session(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch the one-shot session open and tool call; return the open mock."""
    from mcp.types import CallToolResult, TextContent

    session = AsyncMock()
    session.get_server_capabilities = MagicMock(return_value=None)
    mock_cm = AsyncMock()
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    opened = AsyncMock(return_value=(mock_cm, session))
    monkeypatch.setattr(
        "languagemodelcommon.mcp.mcp_client.tool_invocation.open_initialized_mcp_session",
        opened,
    )
    monkeypatch.setattr(
        "languagemodelcommon.mcp.mcp_client.tool_invocation._execute_tool_call_with_heartbeat",
        AsyncMock(
            return_value=CallToolResult(content=[TextContent(type="text", text="ok")])
        ),
    )
    return opened


@pytest.mark.asyncio
async def test_call_mcp_tool_raw_forwards_mode_on_one_shot_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from languagemodelcommon.mcp.mcp_client.tool_invocation import call_mcp_tool_raw

    opened = _patch_one_shot_session(monkeypatch)

    await call_mcp_tool_raw(
        config={"url": "https://example.test/mcp"},
        tool_name="some_tool",
        arguments={},
        server_name="some-server",
        negotiation_mode=McpProtocolNegotiationMode.AUTO,
    )

    assert (
        opened.await_args_list[0].kwargs["negotiation_mode"]
        is McpProtocolNegotiationMode.AUTO
    )


@pytest.mark.asyncio
async def test_call_mcp_tool_raw_defaults_to_legacy_on_one_shot_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from languagemodelcommon.mcp.mcp_client.tool_invocation import call_mcp_tool_raw

    opened = _patch_one_shot_session(monkeypatch)

    await call_mcp_tool_raw(
        config={"url": "https://example.test/mcp"},
        tool_name="some_tool",
        arguments={},
        server_name="some-server",
    )

    assert (
        opened.await_args_list[0].kwargs["negotiation_mode"]
        is McpProtocolNegotiationMode.LEGACY
    )


def test_conflicting_mode_with_pool_raises_instead_of_being_ignored() -> None:
    from languagemodelcommon.mcp.callbacks import _MCPCallbacks
    from languagemodelcommon.mcp.mcp_client.tool_invocation import _make_execute_tool

    pool = McpSessionPool(negotiation_mode=McpProtocolNegotiationMode.AUTO)

    with pytest.raises(ValueError, match="conflicts with the session pool's mode"):
        _make_execute_tool(
            config={"url": "https://example.test/mcp"},
            mcp_callbacks=_MCPCallbacks(),
            session_pool=pool,
            negotiation_mode=McpProtocolNegotiationMode.LEGACY,
        )


@pytest.mark.parametrize(
    "call_mode", [None, McpProtocolNegotiationMode.AUTO], ids=["omitted", "matching"]
)
def test_omitted_or_matching_mode_with_pool_is_accepted(
    call_mode: McpProtocolNegotiationMode | None,
) -> None:
    from languagemodelcommon.mcp.callbacks import _MCPCallbacks
    from languagemodelcommon.mcp.mcp_client.tool_invocation import _make_execute_tool

    pool = McpSessionPool(negotiation_mode=McpProtocolNegotiationMode.AUTO)

    handler = _make_execute_tool(
        config={"url": "https://example.test/mcp"},
        mcp_callbacks=_MCPCallbacks(),
        session_pool=pool,
        negotiation_mode=call_mode,
    )

    assert callable(handler)


def test_langchain_adapter_rejects_conflicting_mode_with_pool() -> None:
    from mcp.types import Tool

    from languagemodelcommon.mcp.mcp_client.langchain_adapter import (
        mcp_tool_to_langchain_tool,
    )

    pool = McpSessionPool(negotiation_mode=McpProtocolNegotiationMode.LEGACY)

    with pytest.raises(ValueError, match="conflicts with the session pool's mode"):
        mcp_tool_to_langchain_tool(
            Tool(name="t", input_schema={"type": "object"}),
            connection={"url": "https://example.test/mcp"},
            session_pool=pool,
            negotiation_mode=McpProtocolNegotiationMode.AUTO,
        )
