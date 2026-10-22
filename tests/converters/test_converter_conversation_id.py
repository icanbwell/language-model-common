"""The LangGraph converter adopts the thread id as the conversation id and
opens Responses streams with `response.created` (BAI-1256)."""

import json
from typing import Any, AsyncGenerator, List, cast
from unittest.mock import MagicMock

import pytest
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, AnyMessage

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


async def _ainvoke_ok(**_kwargs: Any) -> List[AnyMessage]:
    return [AIMessage(content="ok")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("thread_id", "caller_set_id", "expected"),
    [
        ("u1:chat-ab12", None, {"id": "u1:chat-ab12"}),
        ("u1:chat-ab12", "caller-set", {"id": "caller-set"}),
        (None, None, None),
        ("", None, None),
    ],
)
async def test_non_streaming_response_adopts_the_thread_id(
    monkeypatch: pytest.MonkeyPatch,
    thread_id: str | None,
    caller_set_id: str | None,
    expected: dict[str, str] | None,
) -> None:
    converter = _build_converter(monkeypatch)
    monkeypatch.setattr(converter, "ainvoke", _ainvoke_ok)
    wrapper = _responses_wrapper(stream=False)
    wrapper.conversation_id = caller_set_id

    response = await converter.call_agent_with_input(
        chat_request_wrapper=cast(ChatRequestWrapper, wrapper),
        compiled_state_graph=None,  # type: ignore[arg-type]
        system_messages=[],
        request_information=_request_information(thread_id=thread_id),
        config=None,
        state=None,
    )

    body = json.loads(bytes(cast(JSONResponse, response).body))
    assert body["conversation"] == expected
