from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from key_value.aio.stores.memory import MemoryStore
from langchain_core.messages import ToolMessage
from langchain_core.runnables.schema import StandardStreamEvent

from languagemodelcommon.converters.stream_buffer import StreamBufferManager
from languagemodelcommon.converters.stream_debug_output_manager import (
    StreamDebugOutputManager,
)
from languagemodelcommon.converters.tool_event_handlers import ToolEventHandler
from languagemodelcommon.file_managers.file_writer import FileWriter
from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.learned_title_store import (
    LearnedTitleStore,
    LearnedTitleStoreProtocol,
)
from languagemodelcommon.utilities.request_information import RequestInformation
from languagemodelcommon.utilities.tool_display_name_mapper import ToolDisplayNameMapper
from tests.converters.test_tool_event_handlers import _FakeChatRequestWrapper


def _pod(
    *, store: LearnedTitleStoreProtocol | None
) -> tuple[ToolEventHandler, ToolDisplayNameMapper]:
    """One pod: its own singleton mapper (and in-process cache) and handler."""
    singleton = ToolDisplayNameMapper()
    handler = ToolEventHandler(
        debug_file_writer=AsyncMock(spec=FileWriter),
        environment_variables=LanguageModelCommonEnvironmentVariables(),
        tool_display_name_mapper=singleton,
        learned_title_store=store,
        stream_buffer_manager=StreamBufferManager(
            flush_interval_seconds=10.0, enabled=False
        ),
        stream_debug_output_manager=StreamDebugOutputManager(),
    )
    return handler, singleton


def _shared_store() -> LearnedTitleStore:
    return LearnedTitleStore(
        store=MemoryStore(),
        collection="learned_tool_titles",
        ttl_seconds=3600,
        timeout_seconds=1,
    )


async def _search_tools_end(
    *, handler: ToolEventHandler, request_information: RequestInformation
) -> None:
    event = cast(
        StandardStreamEvent,
        {
            "event": "on_tool_end",
            "name": "search_tools",
            "data": {
                "input": {"query": "vitals"},
                "output": ToolMessage(
                    content=[
                        {
                            "type": "text",
                            "text": '[{"name": "get_vitals", "title": "🩺 Checking vitals: {vitals|all}"}]',
                        }
                    ],
                    tool_call_id="tc1",
                    name="search_tools",
                ),
            },
        },
    )
    async for _ in handler.handle_tool_end(
        event=event,
        chat_request_wrapper=cast(ChatRequestWrapper, _FakeChatRequestWrapper()),
        request_information=request_information,
        tool_start_times={},
    ):
        pass


async def _call_tool_start_line(
    *, handler: ToolEventHandler, request_information: RequestInformation
) -> str | None:
    wrapper = _FakeChatRequestWrapper()
    event = cast(
        StandardStreamEvent,
        {
            "event": "on_tool_start",
            "name": "call_tool",
            "data": {"input": {"name": "get_vitals", "arguments": {"vitals": ["bmi"]}}},
        },
    )
    async for _ in handler.handle_tool_start(
        event=event,
        chat_request_wrapper=cast(ChatRequestWrapper, wrapper),
        request_information=request_information,
        tool_start_times={},
    ):
        pass
    return wrapper.last_tool_start_display_name


class TestLearnedTitleStoreWiring:
    """BAI-1306: a title learned on one pod reaches the first call_tool start
    line on a pod that never saw a search_tools/call_tool result."""

    @pytest.mark.asyncio
    async def test_cold_pod_renders_title_learned_by_another_pod(self) -> None:
        store = _shared_store()
        pod_a, singleton_a = _pod(store=store)
        await _search_tools_end(
            handler=pod_a,
            request_information=RequestInformation(
                request_id="a",
                tool_display_name_mapper=singleton_a.with_tools(tools=[]),
            ),
        )

        pod_b, singleton_b = _pod(store=store)
        line = await _call_tool_start_line(
            handler=pod_b,
            request_information=RequestInformation(
                request_id="b",
                tool_display_name_mapper=singleton_b.with_tools(tools=[]),
            ),
        )

        assert line == "🩺 Checking vitals: bmi"

    @pytest.mark.asyncio
    async def test_without_a_store_cold_pod_uses_humanized_name(self) -> None:
        pod_b, singleton_b = _pod(store=None)

        line = await _call_tool_start_line(
            handler=pod_b,
            request_information=RequestInformation(
                request_id="b",
                tool_display_name_mapper=singleton_b.with_tools(tools=[]),
            ),
        )

        assert line == "Get Vitals"

    @pytest.mark.asyncio
    async def test_store_miss_falls_back_to_humanized_name(self) -> None:
        missing = AsyncMock(spec=LearnedTitleStoreProtocol)
        missing.get_title.return_value = None
        pod, singleton = _pod(store=missing)

        line = await _call_tool_start_line(
            handler=pod,
            request_information=RequestInformation(
                request_id="b", tool_display_name_mapper=singleton.with_tools(tools=[])
            ),
        )

        assert line == "Get Vitals"
        missing.get_title.assert_awaited_once_with(tool_name="get_vitals")

    @pytest.mark.asyncio
    async def test_store_is_not_read_once_the_in_process_cache_has_the_title(
        self,
    ) -> None:
        counting: Any = AsyncMock(spec=LearnedTitleStoreProtocol)
        counting.get_title.return_value = "🩺 Checking vitals: {vitals|all}"
        pod, singleton = _pod(store=counting)
        info = RequestInformation(
            request_id="b", tool_display_name_mapper=singleton.with_tools(tools=[])
        )

        await _call_tool_start_line(handler=pod, request_information=info)
        await _call_tool_start_line(handler=pod, request_information=info)

        assert counting.get_title.await_count == 1

    @pytest.mark.asyncio
    async def test_static_config_wins_over_stored_title(self) -> None:
        store = _shared_store()
        await store.put_titles(titles={"get_vitals": "🩺 From store"})
        handler, _ = _pod(store=store)
        configured = ToolDisplayNameMapper(
            name_to_display_name={"get_vitals": "📋 From config"}
        )

        line = await _call_tool_start_line(
            handler=handler,
            request_information=RequestInformation(
                request_id="b", tool_display_name_mapper=configured.with_tools(tools=[])
            ),
        )

        assert line == "📋 From config"

    @pytest.mark.asyncio
    async def test_call_tool_result_title_is_not_persisted(self) -> None:
        store = _shared_store()
        handler, singleton = _pod(store=store)
        info = RequestInformation(
            request_id="a", tool_display_name_mapper=singleton.with_tools(tools=[])
        )
        event = cast(
            StandardStreamEvent,
            {
                "event": "on_tool_end",
                "name": "call_tool",
                "data": {
                    "input": {"name": "get_vitals", "arguments": {}},
                    "output": ToolMessage(
                        content="ok",
                        tool_call_id="tc1",
                        name="call_tool",
                        artifact={
                            "structured_content": {"tool_title": "🩺 Per-call text"}
                        },
                    ),
                },
            },
        )

        async for _ in handler.handle_tool_end(
            event=event,
            chat_request_wrapper=cast(ChatRequestWrapper, _FakeChatRequestWrapper()),
            request_information=info,
            tool_start_times={},
        ):
            pass

        assert await store.get_title(tool_name="get_vitals") is None
        assert info.tool_display_name_mapper is not None
        assert (
            info.tool_display_name_mapper.has_title_for_call_tool_target(
                tool_name="get_vitals"
            )
            is True
        )
