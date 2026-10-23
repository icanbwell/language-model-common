"""Tests for ConversationHistoryManager: when history is left alone, summarized, or trimmed.

tiktoken downloads its encoding on first use, so the encoding is replaced with a
whitespace tokenizer (one word == one token) and the summarizer LLM with a stub.
"""

from typing import Any, Sequence, cast

import pytest
import tiktoken
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from languagemodelcommon.history.conversation_history_manager import (
    ConversationHistoryManager,
)
from languagemodelcommon.state.messages_state import MyMessagesState


class _WordEncoding:
    """One token per whitespace-separated word."""

    def encode(self, text: str) -> list[str]:
        return text.split()


class _StubSummarizerLlm:
    def __init__(self, *, summary: str = "the summary", fail: bool = False) -> None:
        self.summary = summary
        self.fail = fail
        self.calls: list[list[BaseMessage]] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("summarizer unavailable")
        return AIMessage(content=self.summary)


@pytest.fixture(autouse=True)
def _word_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tiktoken, "encoding_for_model", lambda model: _WordEncoding(), raising=True
    )


def _state(messages: Sequence[BaseMessage]) -> MyMessagesState:
    return cast(MyMessagesState, {"messages": list(messages)})


def _conversation(count: int) -> list[BaseMessage]:
    """Alternating human/ai messages m0..m{count-1}, one token each."""
    return [
        HumanMessage(content=f"m{i}") if i % 2 == 0 else AIMessage(content=f"m{i}")
        for i in range(count)
    ]


def _contents(state: MyMessagesState) -> list[str]:
    return [str(m.content) for m in state["messages"]]


class TestInit:
    def test_unknown_model_falls_back_to_cl100k_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requested: list[str] = []

        def _unknown_model(model: str) -> Any:
            raise KeyError(model)

        def _get_encoding(name: str) -> _WordEncoding:
            requested.append(name)
            return _WordEncoding()

        monkeypatch.setattr(tiktoken, "encoding_for_model", _unknown_model)
        monkeypatch.setattr(tiktoken, "get_encoding", _get_encoding)

        manager = ConversationHistoryManager(model="not-a-real-model")

        assert requested == ["cl100k_base"]
        assert manager.count_tokens(messages=[HumanMessage(content="a b c")]) == 3


class TestCountTokens:
    @pytest.mark.parametrize(
        ("contents", "expected"),
        [([], 0), (["one"], 1), (["one two", "three four five"], 5)],
    )
    def test_sums_tokens_across_messages(
        self, contents: list[str], expected: int
    ) -> None:
        manager = ConversationHistoryManager()

        count = manager.count_tokens(
            messages=[HumanMessage(content=text) for text in contents]
        )

        assert count == expected


class TestWithinLimits:
    @pytest.mark.asyncio
    async def test_state_within_both_limits_is_returned_untouched(self) -> None:
        manager = ConversationHistoryManager(max_messages=5, max_tokens=100)
        llm = _StubSummarizerLlm()
        state = _state(_conversation(5))

        result = await manager.manage_history(state=state, llm=cast(Any, llm))

        assert result is state
        assert _contents(result) == ["m0", "m1", "m2", "m3", "m4"]
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_within_limits_history_is_not_reordered_or_rebuilt(self) -> None:
        manager = ConversationHistoryManager(max_messages=10, max_tokens=100)
        messages: list[BaseMessage] = [
            HumanMessage(content="m0"),
            SystemMessage(content="late system note"),
            AIMessage(content="m1"),
        ]
        state = _state(messages)
        original_list = state["messages"]

        result = await manager.manage_history(
            state=state, llm=cast(Any, _StubSummarizerLlm())
        )

        assert result["messages"] is original_list
        assert _contents(result) == ["m0", "late system note", "m1"]

    @pytest.mark.asyncio
    async def test_state_without_messages_is_returned_untouched(self) -> None:
        manager = ConversationHistoryManager()
        llm = _StubSummarizerLlm()
        state = cast(MyMessagesState, {})

        result = await manager.manage_history(state=state, llm=cast(Any, llm))

        assert result is state
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_history_exactly_at_both_limits_is_left_alone(self) -> None:
        """Exactly max_messages messages and exactly max_tokens tokens is still within
        limits: the late system note must not be moved ahead of the conversation."""
        manager = ConversationHistoryManager(max_messages=3, max_tokens=3)
        messages: list[BaseMessage] = [
            HumanMessage(content="m0"),
            SystemMessage(content="late"),
            AIMessage(content="m1"),
        ]
        llm = _StubSummarizerLlm()
        state = _state(messages)

        result = await manager.manage_history(state=state, llm=cast(Any, llm))

        assert manager.count_tokens(messages=messages) == 3
        assert _contents(result) == ["m0", "late", "m1"]
        assert llm.calls == []


class TestSummarization:
    @pytest.mark.asyncio
    async def test_old_messages_are_replaced_by_a_summary_and_recent_are_kept(
        self,
    ) -> None:
        manager = ConversationHistoryManager(
            max_messages=6, max_tokens=1000, summary_threshold=6, keep_recent=3
        )
        llm = _StubSummarizerLlm(summary="the summary")
        state = _state(_conversation(10))

        result = await manager.manage_history(state=state, llm=cast(Any, llm))

        summary, *recent = result["messages"]
        assert isinstance(summary, SystemMessage)
        assert str(summary.content) == (
            "[Previous conversation summary (7 messages)]\nthe summary"
        )
        assert [str(m.content) for m in recent] == ["m7", "m8", "m9"]

    @pytest.mark.asyncio
    async def test_summarizer_is_given_only_the_old_messages(self) -> None:
        manager = ConversationHistoryManager(
            max_messages=6, max_tokens=1000, summary_threshold=6, keep_recent=3
        )
        llm = _StubSummarizerLlm()

        await manager.manage_history(
            state=_state(_conversation(10)), llm=cast(Any, llm)
        )

        assert len(llm.calls) == 1
        prompt = str(llm.calls[0][-1].content)
        assert "HumanMessage: m0" in prompt
        assert "HumanMessage: m6" in prompt
        assert "m7" not in prompt

    @pytest.mark.asyncio
    async def test_system_messages_are_kept_and_moved_ahead_of_the_summary(
        self,
    ) -> None:
        manager = ConversationHistoryManager(
            max_messages=6, max_tokens=1000, summary_threshold=6, keep_recent=3
        )
        messages = _conversation(10)
        messages.insert(4, SystemMessage(content="be brief"))
        llm = _StubSummarizerLlm()

        result = await manager.manage_history(
            state=_state(messages), llm=cast(Any, llm)
        )

        first, second, *_ = result["messages"]
        assert str(first.content) == "be brief"
        assert str(second.content).startswith("[Previous conversation summary")
        assert "be brief" not in str(llm.calls[0][-1].content)

    @pytest.mark.asyncio
    async def test_summarizer_failure_falls_back_to_the_recent_messages(self) -> None:
        manager = ConversationHistoryManager(
            max_messages=6, max_tokens=1000, summary_threshold=6, keep_recent=3
        )
        llm = _StubSummarizerLlm(fail=True)

        result = await manager.manage_history(
            state=_state(_conversation(10)), llm=cast(Any, llm)
        )

        assert _contents(result) == ["m7", "m8", "m9"]

    @pytest.mark.asyncio
    async def test_nothing_is_summarized_when_no_more_than_keep_recent_messages(
        self,
    ) -> None:
        manager = ConversationHistoryManager(
            max_messages=2, max_tokens=1000, summary_threshold=1, keep_recent=5
        )
        llm = _StubSummarizerLlm()

        result = await manager.manage_history(
            state=_state(_conversation(4)), llm=cast(Any, llm)
        )

        assert _contents(result) == ["m0", "m1", "m2", "m3"]
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_too_many_messages_under_the_summary_threshold_are_kept_whole(
        self,
    ) -> None:
        manager = ConversationHistoryManager(
            max_messages=3, max_tokens=1000, summary_threshold=15, keep_recent=10
        )
        llm = _StubSummarizerLlm()

        result = await manager.manage_history(
            state=_state(_conversation(5)), llm=cast(Any, llm)
        )

        assert _contents(result) == ["m0", "m1", "m2", "m3", "m4"]
        assert llm.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("conversation_length", "summarizer_calls"),
        [(6, 0), (7, 1)],
        ids=["exactly-summary-threshold", "one-over-summary-threshold"],
    )
    async def test_summarization_starts_above_the_summary_threshold(
        self, conversation_length: int, summarizer_calls: int
    ) -> None:
        manager = ConversationHistoryManager(
            max_messages=3, max_tokens=1000, summary_threshold=6, keep_recent=3
        )
        llm = _StubSummarizerLlm()

        result = await manager.manage_history(
            state=_state(_conversation(conversation_length)), llm=cast(Any, llm)
        )

        assert len(llm.calls) == summarizer_calls
        if summarizer_calls == 0:
            assert len(result["messages"]) == conversation_length

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("conversation_length", "summarizer_calls"),
        [(4, 0), (5, 1)],
        ids=["exactly-keep-recent", "one-over-keep-recent"],
    )
    async def test_only_messages_beyond_keep_recent_are_ever_summarized(
        self, conversation_length: int, summarizer_calls: int
    ) -> None:
        manager = ConversationHistoryManager(
            max_messages=2, max_tokens=1000, summary_threshold=1, keep_recent=4
        )
        llm = _StubSummarizerLlm()

        result = await manager.manage_history(
            state=_state(_conversation(conversation_length)), llm=cast(Any, llm)
        )

        assert len(llm.calls) == summarizer_calls
        if summarizer_calls == 0:
            assert _contents(result) == [f"m{i}" for i in range(conversation_length)]


class TestTokenTrimming:
    @staticmethod
    def _words(count: int, label: str) -> str:
        return " ".join(f"{label}{i}" for i in range(count))

    @pytest.mark.asyncio
    async def test_keeps_the_most_recent_messages_that_fit_in_order(self) -> None:
        manager = ConversationHistoryManager(
            max_messages=10, max_tokens=10, summary_threshold=100, keep_recent=10
        )
        messages = [
            HumanMessage(content=self._words(4, "a")),
            AIMessage(content=self._words(4, "b")),
            HumanMessage(content=self._words(4, "c")),
            AIMessage(content=self._words(4, "d")),
            HumanMessage(content=self._words(4, "e")),
            AIMessage(content=self._words(4, "f")),
        ]
        llm = _StubSummarizerLlm()

        result = await manager.manage_history(
            state=_state(messages), llm=cast(Any, llm)
        )

        assert _contents(result) == [self._words(4, "e"), self._words(4, "f")]

    @pytest.mark.asyncio
    async def test_system_messages_always_survive_trimming(self) -> None:
        manager = ConversationHistoryManager(
            max_messages=10, max_tokens=8, summary_threshold=100, keep_recent=10
        )
        messages: list[BaseMessage] = [
            SystemMessage(content="be brief"),
            HumanMessage(content=self._words(5, "a")),
            AIMessage(content=self._words(5, "b")),
        ]

        result = await manager.manage_history(
            state=_state(messages), llm=cast(Any, _StubSummarizerLlm())
        )

        assert _contents(result) == ["be brief", self._words(5, "b")]

    @pytest.mark.asyncio
    async def test_trimming_stops_at_the_first_message_that_does_not_fit(
        self,
    ) -> None:
        """Documents current behavior: an oversized message ends the walk back, so
        smaller, older messages are dropped too."""
        manager = ConversationHistoryManager(
            max_messages=10, max_tokens=10, summary_threshold=100, keep_recent=10
        )
        messages: list[BaseMessage] = [
            HumanMessage(content="tiny"),
            AIMessage(content=self._words(20, "big")),
            HumanMessage(content="latest"),
        ]

        result = await manager.manage_history(
            state=_state(messages), llm=cast(Any, _StubSummarizerLlm())
        )

        assert _contents(result) == ["latest"]

    @pytest.mark.asyncio
    async def test_a_newest_message_larger_than_the_budget_leaves_no_conversation(
        self,
    ) -> None:
        """Documents current behavior: when the newest message alone exceeds
        max_tokens, nothing but system messages remains."""
        manager = ConversationHistoryManager(
            max_messages=10, max_tokens=10, summary_threshold=100, keep_recent=10
        )
        messages: list[BaseMessage] = [
            SystemMessage(content="be brief"),
            HumanMessage(content="earlier question"),
            AIMessage(content="earlier answer"),
            HumanMessage(content=self._words(30, "huge")),
        ]

        result = await manager.manage_history(
            state=_state(messages), llm=cast(Any, _StubSummarizerLlm())
        )

        assert _contents(result) == ["be brief"]

    @pytest.mark.asyncio
    async def test_an_oversized_summary_is_never_trimmed_and_displaces_every_recent_message(
        self,
    ) -> None:
        """Documents current behavior: the summary is a SystemMessage, trimming always
        keeps system messages, so a summary larger than max_tokens is kept whole and
        the recent messages are all dropped, leaving the history over budget."""
        manager = ConversationHistoryManager(
            max_messages=6, max_tokens=10, summary_threshold=6, keep_recent=3
        )
        llm = _StubSummarizerLlm(summary=self._words(40, "s"))

        result = await manager.manage_history(
            state=_state(_conversation(10)), llm=cast(Any, llm)
        )

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], SystemMessage)
        assert manager.count_tokens(messages=result["messages"]) > 10

    @pytest.mark.asyncio
    async def test_a_history_exactly_at_the_token_budget_is_kept_whole(self) -> None:
        manager = ConversationHistoryManager(
            max_messages=2, max_tokens=8, summary_threshold=100, keep_recent=10
        )
        messages: list[BaseMessage] = [
            HumanMessage(content=self._words(3, "a")),
            AIMessage(content=self._words(3, "b")),
            HumanMessage(content=self._words(2, "c")),
        ]

        result = await manager.manage_history(
            state=_state(messages), llm=cast(Any, _StubSummarizerLlm())
        )

        assert manager.count_tokens(messages=messages) == 8
        assert _contents(result) == [
            self._words(3, "a"),
            self._words(3, "b"),
            self._words(2, "c"),
        ]

    @pytest.mark.asyncio
    async def test_messages_that_fit_the_budget_exactly_are_kept(self) -> None:
        manager = ConversationHistoryManager(
            max_messages=10, max_tokens=8, summary_threshold=100, keep_recent=10
        )
        messages: list[BaseMessage] = [
            HumanMessage(content=self._words(2, "old")),
            AIMessage(content=self._words(4, "mid")),
            HumanMessage(content=self._words(4, "new")),
        ]

        result = await manager.manage_history(
            state=_state(messages), llm=cast(Any, _StubSummarizerLlm())
        )

        assert _contents(result) == [self._words(4, "mid"), self._words(4, "new")]
