from __future__ import annotations

import pytest
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from languagemodelcommon.utilities.message_preprocessing.composite_message_preprocessor import (
    CompositeMessagePreprocessor,
)


class _AppendingPreprocessor:
    def __init__(self, *, suffix: str) -> None:
        self._suffix = suffix

    async def preprocess(
        self, *, messages: list[AnyMessage], headers: dict[str, str] | None = None
    ) -> list[AnyMessage]:
        return [*messages, SystemMessage(content=self._suffix)]


class TestCompositeMessagePreprocessor:
    @pytest.mark.asyncio
    async def test_empty_preprocessors_returns_messages_unchanged(self) -> None:
        composite = CompositeMessagePreprocessor(preprocessors=[])
        messages: list[AnyMessage] = [HumanMessage(content="hello")]

        result = await composite.preprocess(messages=messages)

        assert result == messages

    @pytest.mark.asyncio
    async def test_single_preprocessor_is_applied(self) -> None:
        composite = CompositeMessagePreprocessor(
            preprocessors=[_AppendingPreprocessor(suffix="added")]
        )
        messages: list[AnyMessage] = [HumanMessage(content="hello")]

        result = await composite.preprocess(messages=messages)

        assert len(result) == 2
        assert result[0].content == "hello"
        assert result[1].content == "added"

    @pytest.mark.asyncio
    async def test_multiple_preprocessors_compose_in_order(self) -> None:
        composite = CompositeMessagePreprocessor(
            preprocessors=[
                _AppendingPreprocessor(suffix="first"),
                _AppendingPreprocessor(suffix="second"),
            ]
        )
        messages: list[AnyMessage] = [HumanMessage(content="hello")]

        result = await composite.preprocess(messages=messages)

        assert len(result) == 3
        assert result[0].content == "hello"
        assert result[1].content == "first"
        assert result[2].content == "second"

    @pytest.mark.asyncio
    async def test_does_not_mutate_original_list(self) -> None:
        composite = CompositeMessagePreprocessor(
            preprocessors=[_AppendingPreprocessor(suffix="new")]
        )
        messages: list[AnyMessage] = [HumanMessage(content="hello")]
        original_len = len(messages)

        await composite.preprocess(messages=messages)

        assert len(messages) == original_len
