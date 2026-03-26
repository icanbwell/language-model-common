from typing import Any, AsyncGenerator

import pytest

from languagemodelcommon.converters.langgraph_to_openai_converter import (
    LangGraphToOpenAIConverter,
)
from languagemodelcommon.exceptions.bailey_exception import BaileyException
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.request_information import RequestInformation


class _ReadTimeoutNamedException(Exception):
    pass


class _GenericException(Exception):
    pass


class GraphRecursionError(Exception):
    pass


class _FakeCompiledStateGraph:
    def __init__(
        self,
        *,
        events: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.events = events or []
        self.error = error
        self.last_stream_config: dict[str, Any] | None = None
        self.last_ainvoke_config: dict[str, Any] | None = None

    async def astream_events(
        self,
        *,
        input: Any,
        version: str,
        config: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        self.last_stream_config = config
        if self.error is not None:
            raise self.error
        for event in self.events:
            yield event

    async def ainvoke(self, *, input: Any, config: dict[str, Any]) -> dict[str, Any]:
        self.last_ainvoke_config = config
        return {"messages": []}


def _build_converter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recursion_limit: str = "88",
) -> LangGraphToOpenAIConverter:
    monkeypatch.setenv("LANGGRAPH_RECURSION_LIMIT", recursion_limit)
    converter = object.__new__(LangGraphToOpenAIConverter)
    converter.environment_variables = LanguageModelCommonEnvironmentVariables()
    return converter


def _request_information() -> RequestInformation:
    return RequestInformation(
        request_id="req-1",
        conversation_thread_id="thread-1",
        user_id="user-1",
    )


def test_is_timeout_exception_returns_true_for_builtin_timeout() -> None:
    assert LangGraphToOpenAIConverter._is_timeout_exception(TimeoutError("timeout"))


def test_is_timeout_exception_returns_true_for_named_timeout_class() -> None:
    assert LangGraphToOpenAIConverter._is_timeout_exception(
        _ReadTimeoutNamedException("timed out")
    )


def test_is_timeout_exception_returns_true_for_wrapped_timeout() -> None:
    wrapped_exception = _GenericException("wrapper")
    wrapped_exception.__cause__ = _ReadTimeoutNamedException("inner timeout")

    assert LangGraphToOpenAIConverter._is_timeout_exception(wrapped_exception)


def test_is_timeout_exception_returns_false_for_non_timeout_error() -> None:
    assert not LangGraphToOpenAIConverter._is_timeout_exception(
        _GenericException("not timeout")
    )


@pytest.mark.asyncio
async def test_stream_graph_adds_default_recursion_limit_to_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _build_converter(monkeypatch, recursion_limit="77")
    fake_graph = _FakeCompiledStateGraph(events=[{"event": "on_chain_start"}])

    events = [
        event
        async for event in converter._stream_graph_with_messages_async(
            messages=[],
            compiled_state_graph=fake_graph,  # type: ignore[arg-type]
            request_information=_request_information(),
            config=None,
            state={"messages": []},  # type: ignore[arg-type]
        )
    ]

    assert len(events) == 1
    assert fake_graph.last_stream_config is not None
    assert fake_graph.last_stream_config["recursion_limit"] == 77
    assert fake_graph.last_stream_config["configurable"]["thread_id"] == "thread-1"
    assert fake_graph.last_stream_config["configurable"]["user_id"] == "user-1"


@pytest.mark.asyncio
async def test_run_graph_respects_explicit_recursion_limit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _build_converter(monkeypatch, recursion_limit="66")
    fake_graph = _FakeCompiledStateGraph()

    await converter._run_graph_with_messages_async(
        messages=[],
        compiled_state_graph=fake_graph,  # type: ignore[arg-type]
        request_information=_request_information(),
        config={"recursion_limit": 123, "configurable": {"thread_id": "thread-2"}},
        state={"messages": []},  # type: ignore[arg-type]
    )

    assert fake_graph.last_ainvoke_config is not None
    assert fake_graph.last_ainvoke_config["recursion_limit"] == 123
    assert fake_graph.last_ainvoke_config["configurable"]["thread_id"] == "thread-2"
    assert fake_graph.last_ainvoke_config["configurable"]["user_id"] == "user-1"


@pytest.mark.asyncio
async def test_stream_graph_maps_graph_recursion_error_to_bailey_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _build_converter(monkeypatch)
    fake_graph = _FakeCompiledStateGraph(
        error=GraphRecursionError(
            "Recursion limit of 25 reached without stop condition"
        )
    )

    with pytest.raises(BaileyException, match="recursion limit"):
        _ = [
            event
            async for event in converter._stream_graph_with_messages_async(
                messages=[],
                compiled_state_graph=fake_graph,  # type: ignore[arg-type]
                request_information=_request_information(),
                config=None,
                state={"messages": []},  # type: ignore[arg-type]
            )
        ]
