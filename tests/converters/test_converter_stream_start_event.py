"""The LangGraph converter opens Responses streams with `response.created` (BAI-1256)."""

import json
from typing import Any, AsyncGenerator, cast
from unittest.mock import MagicMock

import pytest

from languagemodelcommon.converters.langgraph_to_openai_converter import (
    LangGraphToOpenAIConverter,
)
from languagemodelcommon.schema.openai.responses import ResponsesRequest
from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)
from languagemodelcommon.structures.openai.request.responses_api_request_wrapper import (
    ResponsesApiRequestWrapper,
)
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.request_information import RequestInformation


def _build_converter(monkeypatch: pytest.MonkeyPatch) -> LangGraphToOpenAIConverter:
    monkeypatch.setenv("LANGGRAPH_RECURSION_LIMIT", "88")
    converter = object.__new__(LangGraphToOpenAIConverter)
    converter.environment_variables = LanguageModelCommonEnvironmentVariables()
    return converter


def _responses_wrapper(*, stream: bool) -> ResponsesApiRequestWrapper:
    env = MagicMock()
    env.debug_prefixes = ("DEBUG:", "/debug ")
    request = ResponsesRequest.model_validate(
        {"model": "gpt-4", "input": "hello", "stream": stream}
    )
    return ResponsesApiRequestWrapper(
        chat_request=request,
        enable_debug_logging=False,
        environment_variables=env,
    )


def _request_information(*, thread_id: str | None) -> RequestInformation:
    return RequestInformation(
        request_id="req-1", conversation_thread_id=thread_id, user_id="user-1"
    )


async def _no_events(**_kwargs: Any) -> AsyncGenerator[Any, None]:
    return
    yield  # pragma: no cover - makes this an async generator


async def _collect(
    *, converter: LangGraphToOpenAIConverter, wrapper: ResponsesApiRequestWrapper
) -> list[str]:
    return [
        chunk
        async for chunk in converter._stream_resp_async_generator(
            chat_request_wrapper=cast(ChatRequestWrapper, wrapper),
            compiled_state_graph=None,  # type: ignore[arg-type]
            messages=[],
            request_information=_request_information(thread_id="u1:chat-ab12"),
            config=None,
            state=None,
        )
    ]


@pytest.mark.asyncio
async def test_streaming_responses_start_with_response_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _build_converter(monkeypatch)
    monkeypatch.setattr(converter, "astream_events", _no_events)
    wrapper = _responses_wrapper(stream=True)
    wrapper.conversation_id = "u1:chat-ab12"

    chunks = await _collect(converter=converter, wrapper=wrapper)

    first = json.loads(chunks[0][len("data: ") :])
    assert first["type"] == "response.created"
    assert first["response"]["conversation"] == {"id": "u1:chat-ab12"}
    assert '"response.completed"' in chunks[-1]


@pytest.mark.asyncio
async def test_streaming_without_a_start_event_emits_only_the_final_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _build_converter(monkeypatch)
    monkeypatch.setattr(converter, "astream_events", _no_events)

    class _NoStartEventWrapper:
        enable_debug_logging = False

        def create_stream_start_sse_event(
            self, *, request_id: str, source: str
        ) -> str | None:
            return None

        def create_final_sse_message(
            self, *, request_id: str, usage_metadata: Any, source: str
        ) -> str:
            return "final"

    chunks = [
        chunk
        async for chunk in converter._stream_resp_async_generator(
            chat_request_wrapper=cast(ChatRequestWrapper, _NoStartEventWrapper()),
            compiled_state_graph=None,  # type: ignore[arg-type]
            messages=[],
            request_information=_request_information(thread_id="t"),
            config=None,
            state=None,
        )
    ]

    assert chunks == ["final"]


@pytest.mark.asyncio
async def test_failing_stream_start_event_goes_through_stream_error_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _build_converter(monkeypatch)
    monkeypatch.setattr(converter, "astream_events", _no_events)

    class _FailingStartEventWrapper:
        enable_debug_logging = False

        def create_stream_start_sse_event(
            self, *, request_id: str, source: str
        ) -> str | None:
            raise RuntimeError("boom")

        def create_sse_message(
            self,
            *,
            request_id: str,
            content: str | None,
            usage_metadata: Any,
            source: str,
        ) -> str:
            return f"{source}:{content}"

        def create_final_sse_message(
            self, *, request_id: str, usage_metadata: Any, source: str
        ) -> str:
            return "final"

    chunks = [
        chunk
        async for chunk in converter._stream_resp_async_generator(
            chat_request_wrapper=cast(ChatRequestWrapper, _FailingStartEventWrapper()),
            compiled_state_graph=None,  # type: ignore[arg-type]
            messages=[],
            request_information=_request_information(thread_id="t"),
            config=None,
            state=None,
        )
    ]

    assert chunks[0].startswith("error:")
    assert "boom" not in chunks[0]
    assert chunks[-1] == "final"


@pytest.mark.asyncio
async def test_stream_error_still_opens_with_response_created_and_ends_with_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = _build_converter(monkeypatch)

    async def _explode(**_kwargs: Any) -> AsyncGenerator[Any, None]:
        raise RuntimeError("boom")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(converter, "astream_events", _explode)
    wrapper = _responses_wrapper(stream=True)
    wrapper.conversation_id = "u1:chat-ab12"

    chunks = await _collect(converter=converter, wrapper=wrapper)

    assert json.loads(chunks[0][len("data: ") :])["type"] == "response.created"
    assert '"response.completed"' in chunks[-1]
    assert len(chunks) > 2
