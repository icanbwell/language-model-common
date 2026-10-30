"""BAI-1388 / ADR 0008: validate tool results from the shared tool-list cache.

The session under test is the SDK's real ``ClientSession``: its own
``call_tool`` / ``validate_tool_result`` / ``list_tools`` logic runs unmodified.
Only the transport (``send_request``) is faked, by a scripted server that counts
``tools/list`` and ``tools/call`` requests. The cache is the real ``ToolListCache``
over the real ``McpToolListStore`` (in-memory backing store), so the schema round
trip through the store's serialization is exercised too.
"""

import importlib.metadata
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from key_value.aio.stores.memory import MemoryStore
from mcp import ClientSession
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ContentBlock,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
)
from mcp.types import Tool as MCPTool

from languagemodelcommon.mcp.mcp_client.cached_output_validation import (
    CachedOutputSchemaValidator,
    ToolListCacheSchemaSource,
    install_cached_output_validation,
)
from languagemodelcommon.mcp.mcp_client.mcp_tool_list_store import McpToolListStore
from languagemodelcommon.mcp.mcp_client.negotiation_mode import (
    McpProtocolNegotiationMode,
)
from languagemodelcommon.mcp.mcp_client.session import MCPConnectionConfig
from languagemodelcommon.mcp.mcp_client.tool_invocation import call_mcp_tool_raw
from languagemodelcommon.mcp.mcp_client.tool_list_cache import ToolListCache
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)

MCP_VERSION = tuple(int(p) for p in importlib.metadata.version("mcp").split(".")[:2])
# validate_tool_result as a public, per-instance-overridable method that lists
# tools lazily exists from mcp 2.2 (the pin in pyproject.toml).
pytestmark = pytest.mark.skipif(
    MCP_VERSION < (2, 2),
    reason=(
        "BAI-1388 relies on ClientSession.validate_tool_result (mcp>=2.2.0); "
        f"installed mcp is {importlib.metadata.version('mcp')}: these tests are SKIPPED, not passing"
    ),
)

URL = "https://catalog.example.test/mcp"
OBJ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"count": {"type": "integer"}},
    "required": ["count"],
}
CHANGED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"total": {"type": "integer"}},
    "required": ["total"],
}
# The "$ref" points at a definition that does not exist in the document.
UNRESOLVABLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"count": {"$ref": "#/$defs/missing"}},
}
INVALID_SCHEMA: dict[str, Any] = {"type": "not-a-json-schema-type"}


def _tool(*, name: str = "get_data", output_schema: dict[str, Any] | None) -> MCPTool:
    return MCPTool(
        name=name,
        description="synthetic",
        input_schema={"type": "object"},
        output_schema=output_schema,
    )


def _result(*, structured: dict[str, Any] | None) -> CallToolResult:
    """``structured=None`` means the server sent no structuredContent at all."""
    content: list[ContentBlock] = [TextContent(type="text", text="ok")]
    if structured is None:
        return CallToolResult(content=content)
    return CallToolResult(content=content, structured_content=structured)


@dataclass
class FakeToolServer:
    """Scripted downstream MCP server: what it lists and what it returns."""

    tools: list[MCPTool]
    result: CallToolResult
    list_calls: int = 0
    call_calls: int = 0
    fail_list: bool = False


class _TransportFakedSession(ClientSession):
    """Real ClientSession; only the wire (send_request) is scripted."""

    def __init__(self, *, server: FakeToolServer) -> None:
        # Streams are never used: send_request is overridden below.
        write_send, read_recv = anyio.create_memory_object_stream[Any](1)
        super().__init__(read_recv, write_send)
        self.server = server

    async def send_request(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(request, ListToolsRequest):
            self.server.list_calls += 1
            if self.server.fail_list:
                raise ConnectionError("downstream list failed")
            return ListToolsResult(tools=list(self.server.tools))
        if isinstance(request, CallToolRequest):
            self.server.call_calls += 1
            return self.server.result
        raise AssertionError(f"unexpected request {request!r}")


class _Exit:
    async def __aexit__(self, *args: object) -> None:
        return None


class _FakePool:
    """Stands in for McpSessionPool: hands back one long-lived session."""

    negotiation_mode = McpProtocolNegotiationMode.LEGACY

    def __init__(self, *, session: ClientSession) -> None:
        self._session = session
        self.evict = AsyncMock()

    async def get_session(self, config: Any, *, mcp_callbacks: Any) -> ClientSession:
        return self._session


@dataclass
class Harness:
    server: FakeToolServer
    cache: ToolListCache
    sessions: list[_TransportFakedSession] = field(default_factory=list)

    async def call(
        self,
        *,
        flag: bool,
        pooled: bool,
        use_cache: bool = True,
        tool_name: str = "get_data",
    ) -> CallToolResult:
        def new_session() -> _TransportFakedSession:
            session = _TransportFakedSession(server=self.server)
            self.sessions.append(session)
            return session

        config = MCPConnectionConfig(url=URL, transport="streamable_http")
        kwargs: dict[str, Any] = {
            "config": config,
            "tool_name": tool_name,
            "arguments": {},
            "server_name": "catalog",
            "tool_list_cache": self.cache if use_cache else None,
            "validate_output_from_tool_list_cache": flag,
        }
        if pooled:
            kwargs["session_pool"] = _FakePool(session=new_session())
            result = await call_mcp_tool_raw(**kwargs)
        else:
            with patch(
                "languagemodelcommon.mcp.mcp_client.tool_invocation.open_initialized_mcp_session",
                new=AsyncMock(side_effect=lambda *a, **k: (_Exit(), new_session())),
            ):
                result = await call_mcp_tool_raw(**kwargs)
        assert isinstance(result, CallToolResult)
        return result


def _cache() -> ToolListCache:
    return ToolListCache(
        store=McpToolListStore(store=MemoryStore(), collection="tool-list")
    )


async def _harness(
    *,
    served: list[MCPTool],
    cached: list[MCPTool] | None,
    structured: dict[str, Any] | None,
) -> Harness:
    cache = _cache()
    if cached is not None:
        await cache.put_async(key=ToolListCache.make_key(URL), tools=cached)
    return Harness(
        server=FakeToolServer(tools=served, result=_result(structured=structured)),
        cache=cache,
    )


@pytest.fixture(params=[False, True], ids=["one-shot", "pooled"])
def pooled(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


class TestFlagOffIsTodaysBehavior:
    async def test_flag_off_pays_one_list_tools_per_call_even_with_a_cache_hit(
        self, pooled: bool
    ) -> None:
        harness = await _harness(
            served=[_tool(output_schema=OBJ_SCHEMA)],
            cached=[_tool(output_schema=OBJ_SCHEMA)],
            structured={"count": 1},
        )

        await harness.call(flag=False, pooled=pooled)

        assert harness.server.list_calls == 1
        assert harness.server.call_calls == 1

    def test_flag_defaults_off_in_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MCP_VALIDATE_TOOL_OUTPUT_FROM_CACHE", raising=False)
        env = LanguageModelCommonEnvironmentVariables()
        assert env.mcp_validate_tool_output_from_cache is False

    @pytest.mark.parametrize("raw", ["true", "1", "True"])
    def test_flag_reads_environment(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv("MCP_VALIDATE_TOOL_OUTPUT_FROM_CACHE", raw)
        env = LanguageModelCommonEnvironmentVariables()
        assert env.mcp_validate_tool_output_from_cache is True


@dataclass(frozen=True)
class Scenario:
    id: str
    cached: list[MCPTool] | None
    served: list[MCPTool]
    structured: dict[str, Any] | None
    expected_error: str | None  # substring of the RuntimeError, None = passes
    flag_on_list_calls: int
    use_cache: bool = True


SCENARIOS = [
    Scenario(
        id="cached-no-schema-skips-lookup-and-validation",
        cached=[_tool(output_schema=None)],
        served=[_tool(output_schema=None)],
        structured=None,
        expected_error=None,
        flag_on_list_calls=0,
    ),
    Scenario(
        id="cached-schema-conforming",
        cached=[_tool(output_schema=OBJ_SCHEMA)],
        served=[_tool(output_schema=OBJ_SCHEMA)],
        structured={"count": 3},
        expected_error=None,
        flag_on_list_calls=0,
    ),
    Scenario(
        id="cached-schema-non-conforming-fails-after-one-refetch",
        cached=[_tool(output_schema=OBJ_SCHEMA)],
        served=[_tool(output_schema=OBJ_SCHEMA)],
        structured={"count": "three"},
        expected_error="Invalid structured content returned by tool get_data",
        flag_on_list_calls=1,
    ),
    Scenario(
        id="cached-schema-missing-structured-content-fails-after-one-refetch",
        cached=[_tool(output_schema=OBJ_SCHEMA)],
        served=[_tool(output_schema=OBJ_SCHEMA)],
        structured=None,
        expected_error="has an output schema but did not return structured content",
        flag_on_list_calls=1,
    ),
    Scenario(
        id="stale-cached-schema-refetch-then-passes",
        cached=[_tool(output_schema=OBJ_SCHEMA)],
        served=[_tool(output_schema=CHANGED_SCHEMA)],
        structured={"total": 9},
        expected_error=None,
        flag_on_list_calls=1,
    ),
    Scenario(
        id="stale-cached-schema-refetch-then-still-fails",
        cached=[_tool(output_schema=CHANGED_SCHEMA)],
        served=[_tool(output_schema=OBJ_SCHEMA)],
        structured={"count": "x"},
        expected_error="Invalid structured content returned by tool get_data",
        flag_on_list_calls=1,
    ),
    Scenario(
        id="stale-cached-schema-server-dropped-schema-refetch-passes",
        cached=[_tool(output_schema=OBJ_SCHEMA)],
        served=[_tool(output_schema=None)],
        structured=None,
        expected_error=None,
        flag_on_list_calls=1,
    ),
    Scenario(
        id="stale-cached-no-schema-server-now-has-schema-refetch-passes",
        cached=[_tool(output_schema=None)],
        served=[_tool(output_schema=OBJ_SCHEMA)],
        structured={"count": 1},
        expected_error=None,
        flag_on_list_calls=1,
    ),
    Scenario(
        id="stale-cached-no-schema-server-now-has-schema-refetch-fails",
        cached=[_tool(output_schema=None)],
        served=[_tool(output_schema=OBJ_SCHEMA)],
        structured={"count": "x"},
        expected_error="Invalid structured content returned by tool get_data",
        flag_on_list_calls=1,
    ),
    Scenario(
        id="cache-miss-falls-back-to-sdk-and-validates",
        cached=None,
        served=[_tool(output_schema=OBJ_SCHEMA)],
        structured={"count": "x"},
        expected_error="Invalid structured content returned by tool get_data",
        flag_on_list_calls=1,
    ),
    Scenario(
        id="cache-miss-falls-back-to-sdk-and-passes",
        cached=None,
        served=[_tool(output_schema=OBJ_SCHEMA)],
        structured={"count": 1},
        expected_error=None,
        flag_on_list_calls=1,
    ),
    Scenario(
        id="tool-missing-from-cached-list-falls-back-to-sdk",
        cached=[_tool(name="other_tool", output_schema=None)],
        served=[_tool(output_schema=OBJ_SCHEMA)],
        structured={"count": "x"},
        expected_error="Invalid structured content returned by tool get_data",
        flag_on_list_calls=1,
    ),
    Scenario(
        id="ad-hoc-server-never-cached-falls-back-to-sdk",
        cached=[
            _tool(output_schema=None)
        ],  # present, but cache is not given to the call
        served=[_tool(output_schema=OBJ_SCHEMA)],
        structured={"count": "x"},
        expected_error="Invalid structured content returned by tool get_data",
        flag_on_list_calls=1,
        use_cache=False,
    ),
    Scenario(
        id="cached-unresolvable-ref-refetches-then-fails",
        cached=[_tool(output_schema=UNRESOLVABLE_SCHEMA)],
        served=[_tool(output_schema=UNRESOLVABLE_SCHEMA)],
        structured={"count": 1},
        expected_error="Invalid schema for tool get_data",
        flag_on_list_calls=1,
    ),
    Scenario(
        id="cached-invalid-schema-refetches-then-fails",
        cached=[_tool(output_schema=INVALID_SCHEMA)],
        served=[_tool(output_schema=INVALID_SCHEMA)],
        structured={"count": 1},
        expected_error="Invalid schema for tool get_data",
        flag_on_list_calls=1,
    ),
]


class TestFlagOnMatchesTheSdk:
    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
    async def test_outcome_and_list_tools_count(
        self, scenario: Scenario, pooled: bool
    ) -> None:
        failed: dict[bool, bool] = {}
        list_calls: dict[bool, int] = {}
        for flag in (False, True):
            harness = await _harness(
                served=scenario.served,
                cached=scenario.cached,
                structured=scenario.structured,
            )
            try:
                await harness.call(
                    flag=flag, pooled=pooled, use_cache=scenario.use_cache
                )
                failed[flag] = False
            except RuntimeError:
                failed[flag] = True
            list_calls[flag] = harness.server.list_calls

        expect_failure = scenario.expected_error is not None
        assert failed == {False: expect_failure, True: expect_failure}
        assert list_calls[False] == 1  # today: one lookup per fresh session
        # Pooled sessions are left to the SDK's own per-session schema map.
        assert list_calls[True] == (1 if pooled else scenario.flag_on_list_calls)

    @pytest.mark.parametrize(
        "scenario",
        [s for s in SCENARIOS if s.expected_error is not None],
        ids=lambda s: s.id,
    )
    async def test_flag_on_raises_the_same_error_text_as_the_sdk(
        self, scenario: Scenario, pooled: bool
    ) -> None:
        messages: dict[bool, str] = {}
        for flag in (False, True):
            harness = await _harness(
                served=scenario.served,
                cached=scenario.cached,
                structured=scenario.structured,
            )
            with pytest.raises(RuntimeError) as excinfo:
                await harness.call(
                    flag=flag, pooled=pooled, use_cache=scenario.use_cache
                )
            messages[flag] = str(excinfo.value)

        assert messages[True] == messages[False]
        assert scenario.expected_error is not None
        assert scenario.expected_error in messages[True]


class TestCacheMaintenance:
    async def test_stale_schema_refetch_updates_the_shared_cache(self) -> None:
        harness = await _harness(
            served=[_tool(output_schema=CHANGED_SCHEMA)],
            cached=[_tool(output_schema=OBJ_SCHEMA)],
            structured={"total": 1},
        )

        await harness.call(flag=True, pooled=False)
        assert harness.server.list_calls == 1
        await harness.call(flag=True, pooled=False)

        assert (
            harness.server.list_calls == 1
        )  # second call validated from the refreshed cache
        refreshed = await harness.cache.get_async(key=ToolListCache.make_key(URL))
        assert refreshed is not None
        assert refreshed[0].output_schema == CHANGED_SCHEMA

    async def test_pooled_session_lists_tools_once_across_many_calls_as_the_sdk_does(
        self,
    ) -> None:
        harness = await _harness(
            served=[_tool(output_schema=OBJ_SCHEMA)],
            cached=[_tool(output_schema=OBJ_SCHEMA)],
            structured={"count": 1},
        )
        session = _TransportFakedSession(server=harness.server)
        pool = _FakePool(session=session)
        for _ in range(3):
            await call_mcp_tool_raw(
                config=MCPConnectionConfig(url=URL, transport="streamable_http"),
                tool_name="get_data",
                arguments={},
                server_name="catalog",
                session_pool=pool,  # type: ignore[arg-type]
                tool_list_cache=harness.cache,
                validate_output_from_tool_list_cache=True,
            )

        assert harness.server.call_calls == 3
        assert harness.server.list_calls == 1

    async def test_cache_entry_is_looked_up_by_the_called_server_url_only(
        self,
    ) -> None:
        # Another server's entry (different URL) must never answer for this call.
        cache = _cache()
        await cache.put_async(
            key=ToolListCache.make_key("https://other.example.test/mcp"),
            tools=[_tool(output_schema=None)],
        )
        server = FakeToolServer(
            tools=[_tool(output_schema=OBJ_SCHEMA)],
            result=_result(structured={"count": "x"}),
        )
        harness = Harness(server=server, cache=cache)

        with pytest.raises(RuntimeError, match="Invalid structured content"):
            await harness.call(flag=True, pooled=False)

        assert server.list_calls == 1  # fell back to the SDK path


class TestCacheStoreErrors:
    @pytest.mark.parametrize("pooled_session", [False, True])
    async def test_store_read_error_falls_back_to_sdk_validation(
        self, pooled_session: bool
    ) -> None:
        harness = await _harness(
            served=[_tool(output_schema=OBJ_SCHEMA)],
            cached=[_tool(output_schema=None)],
            structured={"count": "x"},
        )
        with (
            patch.object(
                harness.cache, "get_async", AsyncMock(side_effect=OSError("mongo down"))
            ),
            pytest.raises(RuntimeError, match="Invalid structured content"),
        ):
            await harness.call(flag=True, pooled=pooled_session)

        assert harness.server.list_calls == 1

    async def test_store_write_error_after_refetch_still_validates_fresh_schema(
        self,
    ) -> None:
        harness = await _harness(
            served=[_tool(output_schema=CHANGED_SCHEMA)],
            cached=[_tool(output_schema=OBJ_SCHEMA)],
            structured={"total": 1},
        )
        with patch.object(
            harness.cache, "put_async", AsyncMock(side_effect=OSError("mongo down"))
        ):
            await harness.call(flag=True, pooled=False)  # passes on the fresh schema

        assert harness.server.list_calls == 1

    async def test_refetch_failure_falls_back_to_sdk_and_never_skips_validation(
        self,
    ) -> None:
        harness = await _harness(
            served=[_tool(output_schema=OBJ_SCHEMA)],
            cached=[_tool(output_schema=OBJ_SCHEMA)],
            structured={"count": "x"},
        )
        harness.server.fail_list = True

        with pytest.raises(ConnectionError):
            await harness.call(flag=True, pooled=False)

        # one refetch attempt, then the SDK's own lookup (which also fails): the
        # result is never accepted unvalidated.
        assert harness.server.list_calls == 2


class TestInstall:
    async def test_install_overrides_only_the_given_instance(self) -> None:
        server = FakeToolServer(tools=[], result=_result(structured=None))
        first = _TransportFakedSession(server=server)
        second = _TransportFakedSession(server=server)
        source = ToolListCacheSchemaSource(cache=_cache(), cache_key=URL)

        installed = install_cached_output_validation(session=first, source=source)

        assert installed is True
        assert isinstance(first.validate_tool_result, CachedOutputSchemaValidator)
        assert not isinstance(second.validate_tool_result, CachedOutputSchemaValidator)

    async def test_reinstall_on_a_pooled_session_does_not_chain_overrides(
        self,
    ) -> None:
        server = FakeToolServer(
            tools=[_tool(output_schema=OBJ_SCHEMA)], result=_result(structured=None)
        )
        session = _TransportFakedSession(server=server)
        source = ToolListCacheSchemaSource(cache=_cache(), cache_key=URL)

        install_cached_output_validation(session=session, source=source)
        install_cached_output_validation(session=session, source=source)

        # Cache miss -> fallback must be the SDK's own method (one list_tools),
        # not the first override (which would recurse through itself).
        with pytest.raises(RuntimeError, match="did not return structured content"):
            await session.validate_tool_result("get_data", _result(structured=None))
        assert server.list_calls == 1

    def test_sdk_exposes_the_public_method_this_relies_on(self) -> None:
        # Fails loudly if a future SDK renames/removes validate_tool_result, so the
        # feature cannot silently degrade into "flag on, no effect".
        assert callable(getattr(ClientSession, "validate_tool_result", None))

    async def test_install_is_skipped_when_the_sdk_method_is_missing(self) -> None:
        class _NoValidate:
            pass

        installed = install_cached_output_validation(
            session=_NoValidate(),  # type: ignore[arg-type]
            source=ToolListCacheSchemaSource(cache=_cache(), cache_key=URL),
        )
        assert installed is False


@pytest.mark.skipif(
    MCP_VERSION < (2, 3),
    reason=(
        "explicit structuredContent=null is validated as a value from mcp 2.3; "
        f"installed mcp is {importlib.metadata.version('mcp')}: SKIPPED, not passing"
    ),
)
class TestExplicitNullStructuredContent:
    @pytest.mark.parametrize(
        ("schema", "expect_failure"),
        [
            ({"type": ["object", "null"]}, False),
            (OBJ_SCHEMA, True),
        ],
        ids=["nullable-schema-accepts-null", "object-schema-rejects-null"],
    )
    async def test_matches_the_sdk(
        self, schema: dict[str, Any], expect_failure: bool, pooled: bool
    ) -> None:
        failed: dict[bool, bool] = {}
        for flag in (False, True):
            harness = await _harness(
                served=[_tool(output_schema=schema)],
                cached=[_tool(output_schema=schema)],
                structured={"unused": 1},
            )
            harness.server.result = CallToolResult(
                content=[TextContent(type="text", text="ok")], structured_content=None
            )
            try:
                await harness.call(flag=flag, pooled=pooled)
                failed[flag] = False
            except RuntimeError:
                failed[flag] = True

        assert failed == {False: expect_failure, True: expect_failure}
