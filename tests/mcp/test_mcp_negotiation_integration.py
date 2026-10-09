"""BAI-1118 / ADR 0003: protocol negotiation against a real MCP server.

Starts the SDK's own ``MCPServer`` over streamable HTTP on a loopback port (uvicorn
is already locked as a dependency of ``mcp``) and connects through this library's
``open_initialized_mcp_session``. No mocks of the SDK, so these tests fail if the
``Client``-based handshake stops matching what the library relied on before.
"""

import json
import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS, MODERN_PROTOCOL_VERSIONS
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from languagemodelcommon.mcp.mcp_client.negotiation_mode import (
    McpProtocolNegotiationMode,
)
from languagemodelcommon.mcp.mcp_client.session import (
    MCPConnectionConfig,
    open_initialized_mcp_session,
)
from languagemodelcommon.mcp.mcp_client.session_pool import McpSessionPool
from languagemodelcommon.mcp.mcp_client.tool_invocation import call_mcp_tool_raw
from languagemodelcommon.mcp.mcp_client.tool_list_cache import list_all_tools


class _RejectServerDiscover:
    """ASGI wrapper that answers ``server/discover`` with JSON-RPC
    "method not found", the way a server that predates the modern protocol does,
    and passes every other request through untouched."""

    def __init__(self, *, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self._app(scope, receive, send)
            return
        chunks: list[bytes] = []
        more_body = True
        while more_body:
            message = await receive()
            chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)
        body = b"".join(chunks)

        parsed = json.loads(body) if body else None
        if isinstance(parsed, dict) and parsed.get("method") == "server/discover":
            payload = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": parsed.get("id"),
                    "error": {"code": -32601, "message": "Method not found"},
                }
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": payload})
            return

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self._app(scope, replay, send)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _serve(*, app: ASGIApp) -> Iterator[str]:
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("test MCP server did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _build_server() -> MCPServer:
    server = MCPServer("negotiation-test")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return server


@pytest.fixture(scope="module", params=[True, False], ids=["stateless", "stateful"])
def modern_server_url(request: pytest.FixtureRequest) -> Iterator[str]:
    app = _build_server().streamable_http_app(stateless_http=request.param)
    yield from _serve(app=app)


@pytest.fixture(scope="module")
def legacy_only_server_url() -> Iterator[str]:
    app = _build_server().streamable_http_app(stateless_http=True)
    yield from _serve(app=_RejectServerDiscover(app=app))


def _config(*, url: str) -> MCPConnectionConfig:
    return {"url": url, "transport": "streamable_http"}


async def _protocol_version_and_sum(
    *, url: str, mode: McpProtocolNegotiationMode
) -> tuple[str | None, str, list[str]]:
    cm, session = await open_initialized_mcp_session(
        _config(url=url), negotiation_mode=mode
    )
    try:
        tools = await list_all_tools(session)
        result = await session.call_tool("add", {"a": 2, "b": 3})
        text = result.content[0].text  # type: ignore[union-attr]
        return session.protocol_version, text, [tool.name for tool in tools]
    finally:
        await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_legacy_mode_uses_handshake_era_version(modern_server_url: str) -> None:
    version, text, tools = await _protocol_version_and_sum(
        url=modern_server_url, mode=McpProtocolNegotiationMode.LEGACY
    )

    assert version in HANDSHAKE_PROTOCOL_VERSIONS
    assert text == "5"
    assert tools == ["add"]


@pytest.mark.asyncio
async def test_auto_mode_adopts_modern_version_when_server_supports_it(
    modern_server_url: str,
) -> None:
    version, text, tools = await _protocol_version_and_sum(
        url=modern_server_url, mode=McpProtocolNegotiationMode.AUTO
    )

    assert version in MODERN_PROTOCOL_VERSIONS
    assert text == "5"
    assert tools == ["add"]


@pytest.mark.asyncio
async def test_auto_mode_falls_back_to_legacy_when_discover_is_rejected(
    legacy_only_server_url: str,
) -> None:
    version, text, tools = await _protocol_version_and_sum(
        url=legacy_only_server_url, mode=McpProtocolNegotiationMode.AUTO
    )

    assert version in HANDSHAKE_PROTOCOL_VERSIONS
    assert text == "5"
    assert tools == ["add"]


@pytest.mark.parametrize(
    "mode", [McpProtocolNegotiationMode.LEGACY, McpProtocolNegotiationMode.AUTO]
)
@pytest.mark.asyncio
async def test_call_mcp_tool_raw_one_shot_and_pooled(
    modern_server_url: str, mode: McpProtocolNegotiationMode
) -> None:
    one_shot = await call_mcp_tool_raw(
        config=_config(url=modern_server_url),
        tool_name="add",
        arguments={"a": 4, "b": 5},
        server_name="negotiation-test",
        negotiation_mode=mode,
    )
    async with McpSessionPool(negotiation_mode=mode) as pool:
        first = await call_mcp_tool_raw(
            config=_config(url=modern_server_url),
            tool_name="add",
            arguments={"a": 6, "b": 7},
            server_name="negotiation-test",
            session_pool=pool,
        )
        second = await call_mcp_tool_raw(
            config=_config(url=modern_server_url),
            tool_name="add",
            arguments={"a": 1, "b": 1},
            server_name="negotiation-test",
            session_pool=pool,
        )

    assert one_shot.content[0].text == "9"  # type: ignore[union-attr]
    assert first.content[0].text == "13"  # type: ignore[union-attr]
    assert second.content[0].text == "2"  # type: ignore[union-attr]
