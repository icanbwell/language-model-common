"""BAI-1118 / ADR 0003: configurable MCP protocol negotiation mode."""

import importlib
from collections.abc import Iterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from languagemodelcommon.mcp.mcp_client.negotiation_mode import (
    McpProtocolNegotiationMode,
)
from languagemodelcommon.mcp.mcp_client.session import (
    negotiate_session,
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


@pytest.mark.asyncio
async def test_legacy_mode_only_initializes() -> None:
    session = MagicMock()
    session.initialize = AsyncMock()
    with patch("mcp.client._probe.negotiate_auto", new=AsyncMock()) as auto:
        await negotiate_session(session, mode=McpProtocolNegotiationMode.LEGACY)

    session.initialize.assert_awaited_once()
    auto.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_mode_is_legacy() -> None:
    session = MagicMock()
    session.initialize = AsyncMock()
    with patch("mcp.client._probe.negotiate_auto", new=AsyncMock()) as auto:
        await negotiate_session(session)

    session.initialize.assert_awaited_once()
    auto.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_mode_delegates_to_sdk_probe() -> None:
    session = MagicMock()
    session.initialize = AsyncMock()
    with patch("mcp.client._probe.negotiate_auto", new=AsyncMock()) as auto:
        await negotiate_session(session, mode=McpProtocolNegotiationMode.AUTO)

    auto.assert_awaited_once_with(session)
    session.initialize.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_mode_adopts_discover_result_without_initialize() -> None:
    from mcp_types import DiscoverResult
    from mcp_types.version import MODERN_PROTOCOL_VERSIONS

    modern = MODERN_PROTOCOL_VERSIONS[-1]
    session = MagicMock()
    session.initialize = AsyncMock()
    session.adopt = MagicMock()
    session.send_discover = AsyncMock(return_value={})

    with patch.object(
        DiscoverResult,
        "model_validate",
        return_value=MagicMock(supported_versions=[modern]),
    ):
        await negotiate_session(session, mode=McpProtocolNegotiationMode.AUTO)

    session.adopt.assert_called_once()
    session.initialize.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_mode_falls_back_to_initialize_on_legacy_rejection() -> None:
    from mcp.shared.exceptions import MCPError

    session = MagicMock()
    session.initialize = AsyncMock()
    session.send_discover = AsyncMock(
        side_effect=MCPError(code=-32601, message="Method not found")
    )

    await negotiate_session(session, mode=McpProtocolNegotiationMode.AUTO)

    session.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_mode_propagates_transport_errors() -> None:
    session = MagicMock()
    session.initialize = AsyncMock()
    session.send_discover = AsyncMock(side_effect=ConnectionError("down"))

    with pytest.raises(ConnectionError):
        await negotiate_session(session, mode=McpProtocolNegotiationMode.AUTO)

    session.initialize.assert_not_awaited()


def test_sdk_probe_symbol_is_importable() -> None:
    """Contract check: ADR 0003 imports a private SDK symbol. Fail loudly if
    an ``mcp`` upgrade moves it."""
    probe = importlib.import_module("mcp.client._probe")

    assert callable(probe.negotiate_auto)


def test_session_module_has_no_import_time_reference_to_sdk_probe() -> None:
    """The private probe is imported lazily inside ``negotiate_session``, so an
    ``mcp`` upgrade that moves it can only break ``auto`` mode, never importing
    this library or the default legacy path."""
    from languagemodelcommon.mcp.mcp_client import session as session_module

    assert not hasattr(session_module, "negotiate_auto")


@pytest.fixture
def fake_session_cm() -> Iterator[MagicMock]:
    session = MagicMock()

    @asynccontextmanager
    async def _create(*_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        yield session

    with patch(f"{SESSION_MODULE}.create_mcp_session", new=_create):
        yield session


@pytest.mark.parametrize(
    "mode", [McpProtocolNegotiationMode.LEGACY, McpProtocolNegotiationMode.AUTO]
)
@pytest.mark.asyncio
async def test_open_session_retries_transient_failure_in_both_modes(
    fake_session_cm: MagicMock, mode: McpProtocolNegotiationMode
) -> None:
    calls = AsyncMock(side_effect=[ConnectionError("reset"), None])
    with patch(f"{SESSION_MODULE}.negotiate_session", new=calls):
        cm, session = await open_initialized_mcp_session(
            {"url": "http://x"},
            base_delay_seconds=0.0,
            negotiation_mode=mode,
        )
        await cm.__aexit__(None, None, None)

    assert session is fake_session_cm
    assert calls.await_count == 2
    assert all(c.kwargs["mode"] is mode for c in calls.await_args_list)


@pytest.mark.asyncio
async def test_open_session_does_not_retry_cancellation(
    fake_session_cm: MagicMock,
) -> None:
    import asyncio

    calls = AsyncMock(side_effect=asyncio.CancelledError())
    with patch(f"{SESSION_MODULE}.negotiate_session", new=calls):
        with pytest.raises(asyncio.CancelledError):
            await open_initialized_mcp_session(
                {"url": "http://x"},
                base_delay_seconds=0.0,
                negotiation_mode=McpProtocolNegotiationMode.AUTO,
            )

    assert calls.await_count == 1


@pytest.mark.asyncio
async def test_open_session_defaults_to_legacy(fake_session_cm: MagicMock) -> None:
    calls = AsyncMock()
    with patch(f"{SESSION_MODULE}.negotiate_session", new=calls):
        cm, _ = await open_initialized_mcp_session({"url": "http://x"})
        await cm.__aexit__(None, None, None)

    assert calls.await_args_list[0].kwargs["mode"] is McpProtocolNegotiationMode.LEGACY


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
