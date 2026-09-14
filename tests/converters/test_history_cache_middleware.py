"""Tests for HistoryCacheMiddleware (BAI-706 ADR Phase 2).

Regression coverage for the within-turn history cache breakpoint: it must
only fire from the second model call in a turn onward (a ToolMessage in the
message list is the signal a round-trip already happened), must never mutate
the request it receives, and must not depend on message-list length or
position -- the actual re-anchoring after trimming happens downstream in
langchain_anthropic, based on whatever the live message list looks like on
each call.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_aws import ChatAnthropicBedrock
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from languagemodelcommon.converters.history_cache_middleware import (
    HistoryCacheMiddleware,
)


def _anthropic_bedrock_model() -> BaseChatModel:
    return MagicMock(spec=ChatAnthropicBedrock)


def _stub_response() -> ModelResponse[Any]:
    return ModelResponse(result=[AIMessage(content="response")])


def _round_one_messages() -> list[Any]:
    return [HumanMessage(content="What are my recent labs?")]


def _round_two_messages() -> list[Any]:
    return [
        HumanMessage(content="What are my recent labs?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_labs", "args": {}, "id": "call_1"}],
        ),
        ToolMessage(content="[]", tool_call_id="call_1"),
    ]


def _make_request(*, model: BaseChatModel, messages: list[Any]) -> ModelRequest[Any]:
    return ModelRequest(model=model, messages=messages)


class TestHistoryCacheMiddlewareAwrapModelCall:
    """HistoryCacheMiddleware.awrap_model_call"""

    @pytest.mark.asyncio
    async def test_first_round_of_turn_is_not_cached(self) -> None:
        middleware = HistoryCacheMiddleware()
        request = _make_request(
            model=_anthropic_bedrock_model(), messages=_round_one_messages()
        )
        captured: list[ModelRequest[Any]] = []

        async def handler(req: ModelRequest[Any]) -> ModelResponse[Any]:
            captured.append(req)
            return _stub_response()

        await middleware.awrap_model_call(request, handler)

        assert "cache_control" not in captured[0].model_settings

    @pytest.mark.asyncio
    async def test_second_round_of_turn_is_cached(self) -> None:
        middleware = HistoryCacheMiddleware()
        request = _make_request(
            model=_anthropic_bedrock_model(), messages=_round_two_messages()
        )
        captured: list[ModelRequest[Any]] = []

        async def handler(req: ModelRequest[Any]) -> ModelResponse[Any]:
            captured.append(req)
            return _stub_response()

        await middleware.awrap_model_call(request, handler)

        assert captured[0].model_settings["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_non_anthropic_bedrock_model_is_never_cached(self) -> None:
        middleware = HistoryCacheMiddleware()
        request = _make_request(
            model=MagicMock(spec=BaseChatModel), messages=_round_two_messages()
        )
        captured: list[ModelRequest[Any]] = []

        async def handler(req: ModelRequest[Any]) -> ModelResponse[Any]:
            captured.append(req)
            return _stub_response()

        await middleware.awrap_model_call(request, handler)

        assert "cache_control" not in captured[0].model_settings

    @pytest.mark.asyncio
    async def test_does_not_mutate_original_request(self) -> None:
        """request.override() must return a new request, not mutate in place --
        the original request object is owned by the agent loop, not this
        middleware."""
        middleware = HistoryCacheMiddleware()
        request = _make_request(
            model=_anthropic_bedrock_model(), messages=_round_two_messages()
        )
        original_model_settings = request.model_settings

        async def handler(req: ModelRequest[Any]) -> ModelResponse[Any]:
            return _stub_response()

        await middleware.awrap_model_call(request, handler)

        assert request.model_settings is original_model_settings
        assert "cache_control" not in request.model_settings

    @pytest.mark.asyncio
    async def test_does_not_depend_on_message_list_length_or_position(self) -> None:
        """Re-anchoring after a trim happens downstream (langchain_anthropic
        recomputes the breakpoint from whatever the live message list is on
        each call) -- this middleware's own decision must not hinge on a
        fixed message count or index, only on whether a round-trip already
        happened in this turn."""
        middleware = HistoryCacheMiddleware()
        model = _anthropic_bedrock_model()

        long_history = [
            HumanMessage(content="What are my recent labs?"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_labs", "args": {}, "id": "call_1"}],
            ),
            ToolMessage(content="[]", tool_call_id="call_1"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_notes", "args": {}, "id": "call_2"}],
            ),
            ToolMessage(content="[]", tool_call_id="call_2"),
        ]
        # Simulates history trimming mid-turn: fewer messages, but still past
        # round one (a ToolMessage survived the trim).
        trimmed_history = [
            ToolMessage(content="[]", tool_call_id="call_2"),
        ]

        for messages in (long_history, trimmed_history):
            request = _make_request(model=model, messages=messages)
            captured: list[ModelRequest[Any]] = []

            async def handler(req: ModelRequest[Any]) -> ModelResponse[Any]:
                captured.append(req)
                return _stub_response()

            await middleware.awrap_model_call(request, handler)

            assert captured[0].model_settings["cache_control"] == {"type": "ephemeral"}


class TestHistoryCacheMiddlewareWrapModelCall:
    """HistoryCacheMiddleware.wrap_model_call (sync counterpart)"""

    def test_second_round_of_turn_is_cached(self) -> None:
        middleware = HistoryCacheMiddleware()
        request = _make_request(
            model=_anthropic_bedrock_model(), messages=_round_two_messages()
        )
        captured: list[ModelRequest[Any]] = []

        def handler(req: ModelRequest[Any]) -> ModelResponse[Any]:
            captured.append(req)
            return _stub_response()

        middleware.wrap_model_call(request, handler)

        assert captured[0].model_settings["cache_control"] == {"type": "ephemeral"}

    def test_first_round_of_turn_is_not_cached(self) -> None:
        middleware = HistoryCacheMiddleware()
        request = _make_request(
            model=_anthropic_bedrock_model(), messages=_round_one_messages()
        )
        captured: list[ModelRequest[Any]] = []

        def handler(req: ModelRequest[Any]) -> ModelResponse[Any]:
            captured.append(req)
            return _stub_response()

        middleware.wrap_model_call(request, handler)

        assert "cache_control" not in captured[0].model_settings
