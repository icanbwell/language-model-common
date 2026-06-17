import asyncio
from typing import Any, AsyncGenerator

import pytest
from botocore.exceptions import TokenRetrievalError
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from languagemodelcommon.converters.langgraph_to_openai_converter import (
    LangGraphToOpenAIConverter,
)
from languagemodelcommon.exceptions.bailey_exception import BaileyException
from languagemodelcommon.history.context_compactor import ContextCompactor
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.request_information import RequestInformation


async def _no_sleep(*_args: Any, **_kwargs: Any) -> None:
    """Replacement for asyncio.sleep that returns immediately in tests."""
    return None


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
        fail_count: int = 0,
    ) -> None:
        self.events = events or []
        self.error = error
        self.fail_count = fail_count
        self._call_count = 0
        self.last_stream_config: dict[str, Any] | None = None
        self.last_ainvoke_config: dict[str, Any] | None = None

    async def astream_events(
        self,
        *,
        input: Any,
        version: str,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        self.last_stream_config = config
        self._call_count += 1
        if self.error is not None and self._call_count <= self.fail_count:
            raise self.error
        if self.error is not None and self.fail_count == 0:
            raise self.error
        for event in self.events:
            yield event

    async def ainvoke(
        self, *, input: Any, config: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
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
    converter._context_compactor = ContextCompactor()
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


@pytest.mark.parametrize(
    "error_message",
    [
        "could not resolve credentials from session",
        "Unable to locate credentials",
        "Expired credentials",
        "invalid identity token received",
    ],
)
@pytest.mark.asyncio
async def test_stream_graph_raises_token_retrieval_error_for_credential_failures(
    monkeypatch: pytest.MonkeyPatch,
    error_message: str,
) -> None:
    converter = _build_converter(monkeypatch)
    fake_graph = _FakeCompiledStateGraph(
        error=RuntimeError(error_message),
    )

    with pytest.raises(TokenRetrievalError):
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


@pytest.mark.asyncio
async def test_stream_graph_raises_token_retrieval_error_for_wrapped_credential_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _build_converter(monkeypatch)
    root_cause = RuntimeError("could not resolve credentials from session")
    wrapper = ValueError("something went wrong")
    wrapper.__cause__ = root_cause
    fake_graph = _FakeCompiledStateGraph(error=wrapper)

    with pytest.raises(TokenRetrievalError):
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


def test_is_credential_resolution_error_returns_false_for_unrelated_errors() -> None:
    assert not LangGraphToOpenAIConverter._is_credential_resolution_error(
        RuntimeError("something else entirely")
    )


@pytest.mark.parametrize(
    "error_message",
    [
        "Error code: 400 - {'message': 'Input is too long for requested model.'}",
        "context_length_exceeded: max tokens is 200000",
        "This model's maximum context length is 128000 tokens",
        "prompt is too long: 250000 tokens > 200000 maximum",
    ],
)
@pytest.mark.asyncio
async def test_stream_graph_raises_bailey_exception_for_input_too_long(
    monkeypatch: pytest.MonkeyPatch,
    error_message: str,
) -> None:
    converter = _build_converter(monkeypatch)
    fake_graph = _FakeCompiledStateGraph(
        error=RuntimeError(error_message),
    )

    with pytest.raises(BaileyException, match="context window"):
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


@pytest.mark.asyncio
async def test_stream_graph_raises_bailey_exception_for_wrapped_input_too_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _build_converter(monkeypatch)
    root_cause = RuntimeError("Input is too long for requested model.")
    wrapper = ValueError("streaming failed")
    wrapper.__cause__ = root_cause
    fake_graph = _FakeCompiledStateGraph(error=wrapper)

    with pytest.raises(BaileyException, match="context window"):
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


def test_is_input_too_long_error_returns_false_for_unrelated_errors() -> None:
    assert not LangGraphToOpenAIConverter._is_input_too_long_error(
        RuntimeError("something else entirely")
    )


@pytest.mark.asyncio
async def test_stream_graph_compacts_and_retries_on_input_too_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When input-too-long fires and messages can be compacted, retry succeeds."""
    converter = _build_converter(monkeypatch)

    # Build messages with a large tool result that compaction will shrink
    messages: list[AnyMessage] = [
        HumanMessage(content="What is the patient status?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_patient", "args": {}, "id": "tc1"}],
        ),
        ToolMessage(content="x" * 5000, tool_call_id="tc1", name="get_patient"),
        AIMessage(content="The patient is stable."),
        HumanMessage(content="Tell me more details"),
    ]

    # Fail on first call, succeed on second (after compaction)
    fake_graph = _FakeCompiledStateGraph(
        events=[{"event": "on_chain_start"}],
        error=RuntimeError("Input is too long for requested model."),
        fail_count=1,
    )

    events = [
        event
        async for event in converter._stream_graph_with_messages_async(
            messages=messages,
            compiled_state_graph=fake_graph,  # type: ignore[arg-type]
            request_information=_request_information(),
            config=None,
            state=None,
        )
    ]

    # The retry succeeded and yielded events
    assert len(events) == 1
    assert fake_graph._call_count == 2


@pytest.mark.asyncio
async def test_stream_graph_skips_compaction_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When CONTEXT_COMPACTION_ENABLED=false, raises immediately without retry."""
    monkeypatch.setenv("CONTEXT_COMPACTION_ENABLED", "false")
    converter = _build_converter(monkeypatch)

    messages: list[AnyMessage] = [
        HumanMessage(content="What is the patient status?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_patient", "args": {}, "id": "tc1"}],
        ),
        ToolMessage(content="x" * 5000, tool_call_id="tc1", name="get_patient"),
        AIMessage(content="The patient is stable."),
        HumanMessage(content="Tell me more details"),
    ]

    fake_graph = _FakeCompiledStateGraph(
        events=[{"event": "on_chain_start"}],
        error=RuntimeError("Input is too long for requested model."),
        fail_count=1,
    )

    with pytest.raises(BaileyException, match="context window"):
        _ = [
            event
            async for event in converter._stream_graph_with_messages_async(
                messages=messages,
                compiled_state_graph=fake_graph,  # type: ignore[arg-type]
                request_information=_request_information(),
                config=None,
                state=None,
            )
        ]

    # Only called once — no retry
    assert fake_graph._call_count == 1


class _AnthropicRateLimitError(Exception):
    """Mimics anthropic.RateLimitError: carries a 429 status code."""

    def __init__(
        self,
        message: str = (
            "Error code: 429 - {'message': 'Too many tokens, "
            "please wait before trying again.'}"
        ),
    ) -> None:
        super().__init__(message)
        self.status_code = 429


class RateLimitError(Exception):
    """Class name alone (matches anthropic/openai) should signal rate limiting."""


def test_is_rate_limit_error_true_for_429_status() -> None:
    assert LangGraphToOpenAIConverter._is_rate_limit_error(_AnthropicRateLimitError())


def test_is_rate_limit_error_true_for_too_many_tokens_message() -> None:
    assert LangGraphToOpenAIConverter._is_rate_limit_error(
        RuntimeError("Too many tokens, please wait before trying again.")
    )


def test_is_rate_limit_error_true_for_named_class() -> None:
    assert LangGraphToOpenAIConverter._is_rate_limit_error(
        RateLimitError("rate limited")
    )


def test_is_rate_limit_error_true_for_wrapped_cause() -> None:
    wrapped = _GenericException("wrapper")
    wrapped.__cause__ = _AnthropicRateLimitError()
    assert LangGraphToOpenAIConverter._is_rate_limit_error(wrapped)


def test_is_rate_limit_error_false_for_unrelated_error() -> None:
    assert not LangGraphToOpenAIConverter._is_rate_limit_error(
        RuntimeError("something else entirely")
    )


@pytest.mark.asyncio
async def test_stream_retries_on_rate_limit_before_first_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 raised before any event is yielded is retried with backoff."""
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    converter = _build_converter(monkeypatch)

    # First two attempts hit the rate limit, third succeeds.
    fake_graph = _FakeCompiledStateGraph(
        events=[{"event": "on_chain_start"}],
        error=_AnthropicRateLimitError(),
        fail_count=2,
    )

    events = [
        event
        async for event in converter._stream_graph_with_messages_async(
            messages=[HumanMessage(content="hi")],
            compiled_state_graph=fake_graph,  # type: ignore[arg-type]
            request_information=_request_information(),
            config=None,
            state=None,
        )
    ]

    assert len(events) == 1
    assert fake_graph._call_count == 3


@pytest.mark.asyncio
async def test_stream_raises_after_exhausting_rate_limit_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent 429s raise once the retry budget is exhausted."""
    monkeypatch.setenv("RATE_LIMIT_MAX_RETRIES", "2")
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    converter = _build_converter(monkeypatch)

    fake_graph = _FakeCompiledStateGraph(
        events=[{"event": "on_chain_start"}],
        error=_AnthropicRateLimitError(),
        fail_count=5,
    )

    with pytest.raises(BaileyException, match="rate limit"):
        _ = [
            event
            async for event in converter._stream_graph_with_messages_async(
                messages=[HumanMessage(content="hi")],
                compiled_state_graph=fake_graph,  # type: ignore[arg-type]
                request_information=_request_information(),
                config=None,
                state=None,
            )
        ]

    # initial attempt + 2 retries
    assert fake_graph._call_count == 3


@pytest.mark.asyncio
async def test_stream_does_not_retry_rate_limit_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RATE_LIMIT_RETRY_ENABLED=false surfaces the 429 without retrying."""
    monkeypatch.setenv("RATE_LIMIT_RETRY_ENABLED", "false")
    converter = _build_converter(monkeypatch)

    fake_graph = _FakeCompiledStateGraph(
        events=[{"event": "on_chain_start"}],
        error=_AnthropicRateLimitError(),
        fail_count=1,
    )

    with pytest.raises(BaileyException, match="rate limit"):
        _ = [
            event
            async for event in converter._stream_graph_with_messages_async(
                messages=[HumanMessage(content="hi")],
                compiled_state_graph=fake_graph,  # type: ignore[arg-type]
                request_information=_request_information(),
                config=None,
                state=None,
            )
        ]

    assert fake_graph._call_count == 1
