"""The LangGraph converter adopts the thread id as the conversation id only when
`ADOPT_THREAD_ID_AS_CONVERSATION_ID` is on (BAI-1264)."""

import json
from typing import Any, AsyncGenerator, List, cast
from unittest.mock import MagicMock

import pytest
from fastapi.responses import JSONResponse, StreamingResponse
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

FLAG = "ADOPT_THREAD_ID_AS_CONVERSATION_ID"


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


async def _ainvoke_ok(**_kwargs: Any) -> List[AnyMessage]:
    return [AIMessage(content="ok")]


async def _no_events(**_kwargs: Any) -> AsyncGenerator[Any, None]:
    return
    yield  # pragma: no cover - makes this an async generator


async def _non_streaming_conversation(
    *,
    converter: LangGraphToOpenAIConverter,
    thread_id: str | None,
    caller_set_id: str | None = None,
) -> Any:
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
    return json.loads(bytes(cast(JSONResponse, response).body))["conversation"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("thread_id", "caller_set_id", "expected"),
    [
        ("u1:chat-ab12", None, {"id": "u1:chat-ab12"}),
        ("u1:chat-ab12", "caller-set", {"id": "caller-set"}),
        (None, None, None),
        ("", None, None),
        ("x" * 257, None, None),
    ],
)
async def test_non_streaming_response_adopts_the_thread_id_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    thread_id: str | None,
    caller_set_id: str | None,
    expected: dict[str, str] | None,
) -> None:
    monkeypatch.setenv(FLAG, "true")
    converter = _build_converter(monkeypatch)
    monkeypatch.setattr(converter, "ainvoke", _ainvoke_ok)

    conversation = await _non_streaming_conversation(
        converter=converter, thread_id=thread_id, caller_set_id=caller_set_id
    )

    assert conversation == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("flag_value", [None, "false", "0", ""])
async def test_thread_id_is_not_adopted_unless_enabled(
    monkeypatch: pytest.MonkeyPatch, flag_value: str | None
) -> None:
    if flag_value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag_value)
    converter = _build_converter(monkeypatch)
    monkeypatch.setattr(converter, "ainvoke", _ainvoke_ok)

    conversation = await _non_streaming_conversation(
        converter=converter, thread_id="u1:chat-ab12"
    )

    assert conversation is None


@pytest.mark.asyncio
async def test_a_caller_set_id_is_kept_when_adoption_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    converter = _build_converter(monkeypatch)
    monkeypatch.setattr(converter, "ainvoke", _ainvoke_ok)

    conversation = await _non_streaming_conversation(
        converter=converter, thread_id="u1:chat-ab12", caller_set_id="caller-set"
    )

    assert conversation == {"id": "caller-set"}


@pytest.mark.asyncio
async def test_streaming_response_adopts_the_thread_id_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FLAG, "true")
    converter = _build_converter(monkeypatch)
    monkeypatch.setattr(converter, "astream_events", _no_events)
    wrapper = _responses_wrapper(stream=True)

    response = await converter.call_agent_with_input(
        chat_request_wrapper=cast(ChatRequestWrapper, wrapper),
        compiled_state_graph=None,  # type: ignore[arg-type]
        system_messages=[],
        request_information=_request_information(thread_id="u1:chat-ab12"),
        config=None,
        state=None,
    )

    chunks = [
        chunk if isinstance(chunk, str) else bytes(chunk).decode()
        async for chunk in cast(StreamingResponse, response).body_iterator
    ]
    completed = next(
        json.loads(line[len("data: ") :])
        for chunk in chunks
        for line in chunk.split("\n")
        if line.startswith("data: ") and '"response.completed"' in line
    )
    assert completed["response"]["conversation"] == {"id": "u1:chat-ab12"}


async def _stream_via_entry_point(
    *, converter: LangGraphToOpenAIConverter, thread_id: str | None
) -> Any:
    """Drives get_streaming_response_async directly, as baileyai's ExecuteGraphStep does."""
    wrapper = _responses_wrapper(stream=True)
    generator = await converter.get_streaming_response_async(
        chat_request_wrapper=cast(ChatRequestWrapper, wrapper),
        compiled_state_graph=None,  # type: ignore[arg-type]
        system_messages=[],
        request_information=_request_information(thread_id=thread_id),
        config=None,
        state=None,
    )
    completed: dict[str, Any] | None = None
    async for chunk in generator:
        for line in chunk.split("\n"):
            if line.startswith("data: ") and '"response.completed"' in line:
                completed = json.loads(line[len("data: ") :])
    assert completed is not None
    return completed["response"]["conversation"]


@pytest.mark.asyncio
async def test_the_streaming_entry_point_adopts_the_thread_id_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FLAG, "true")
    converter = _build_converter(monkeypatch)
    monkeypatch.setattr(converter, "astream_events", _no_events)

    conversation = await _stream_via_entry_point(
        converter=converter, thread_id="u1:chat-ab12"
    )

    assert conversation == {"id": "u1:chat-ab12"}


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id", [None, "", "x" * 257, "sub:chat\x01x"])
async def test_the_streaming_entry_point_skips_unusable_thread_ids(
    monkeypatch: pytest.MonkeyPatch, thread_id: str | None
) -> None:
    monkeypatch.setenv(FLAG, "true")
    converter = _build_converter(monkeypatch)
    monkeypatch.setattr(converter, "astream_events", _no_events)

    conversation = await _stream_via_entry_point(
        converter=converter, thread_id=thread_id
    )

    assert conversation is None


@pytest.mark.asyncio
async def test_the_streaming_entry_point_does_not_adopt_unless_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    converter = _build_converter(monkeypatch)
    monkeypatch.setattr(converter, "astream_events", _no_events)

    conversation = await _stream_via_entry_point(
        converter=converter, thread_id="u1:chat-ab12"
    )

    assert conversation is None
