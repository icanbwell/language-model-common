import pytest
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from languagemodelcommon.history.context_compactor import ContextCompactor


@pytest.fixture
def compactor() -> ContextCompactor:
    return ContextCompactor()


class TestTruncateOldToolResults:
    def test_preserves_current_turn_tool_messages(
        self, *, compactor: ContextCompactor
    ) -> None:
        large_content = "x" * 5000
        messages: list[AnyMessage] = [
            HumanMessage(content="query"),
            AIMessage(
                content="",
                tool_calls=[{"name": "tool1", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content=large_content, tool_call_id="tc1", name="tool1"),
            AIMessage(content="result"),
            HumanMessage(content="follow up"),  # current turn starts here
        ]

        result = compactor.compact(messages=messages)

        # Current turn's HumanMessage preserved intact
        assert any(
            isinstance(m, HumanMessage) and m.content == "follow up" for m in result
        )

    def test_truncates_large_tool_results_in_older_turns(
        self, *, compactor: ContextCompactor
    ) -> None:
        large_content = "x" * 5000
        messages: list[AnyMessage] = [
            HumanMessage(content="first question"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content=large_content, tool_call_id="tc1", name="search"),
            AIMessage(content="Here is the answer."),
            HumanMessage(content="second question"),
        ]

        result = compactor.compact(messages=messages)

        # Should be smaller than original
        original_size = sum(len(str(m.content)) for m in messages)
        compacted_size = sum(len(str(m.content)) for m in result)
        assert compacted_size < original_size

    def test_leaves_small_tool_results_unchanged(
        self, *, compactor: ContextCompactor
    ) -> None:
        messages: list[AnyMessage] = [
            HumanMessage(content="query"),
            AIMessage(
                content="",
                tool_calls=[{"name": "tool1", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="small result", tool_call_id="tc1", name="tool1"),
            AIMessage(content="done"),
            HumanMessage(content="next"),
        ]

        result = compactor.compact(messages=messages)

        # With small content, pass 1 won't hit 50% reduction target,
        # so subsequent passes may apply, but the tool message text should remain short
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        for tm in tool_msgs:
            assert len(str(tm.content)) <= 200


class TestDropOldToolPairs:
    def test_drops_old_tool_pairs_when_pass1_insufficient(
        self, *, compactor: ContextCompactor
    ) -> None:
        # Create messages where tool results are small but there are many
        # large human/AI messages — pass 1 truncation won't help much,
        # forcing pass 2 to drop tool pairs
        messages: list[AnyMessage] = []
        for i in range(10):
            messages.append(HumanMessage(content=f"question {i} " + "q" * 300))
            messages.append(
                AIMessage(
                    content="a" * 300,
                    tool_calls=[{"name": f"tool_{i}", "args": {}, "id": f"tc{i}"}],
                )
            )
            messages.append(
                ToolMessage(
                    content="result " + "y" * 100,
                    tool_call_id=f"tc{i}",
                    name=f"tool_{i}",
                )
            )
            messages.append(AIMessage(content=f"answer {i} " + "z" * 300))
        messages.append(HumanMessage(content="final question"))

        result = compactor.compact(messages=messages)

        # Compaction should have reduced total content significantly
        original_size = sum(len(str(m.content)) for m in messages)
        compacted_size = sum(len(str(m.content)) for m in result)
        assert compacted_size < original_size
        # Last human message should be preserved
        assert any(
            isinstance(m, HumanMessage) and m.content == "final question"
            for m in result
        )


class TestSummarizeOldExchanges:
    def test_summarizes_when_many_old_messages(
        self, *, compactor: ContextCompactor
    ) -> None:
        # Lots of older exchanges with moderately large content
        messages: list[AnyMessage] = []
        for i in range(20):
            messages.append(HumanMessage(content=f"question {i} " + "w" * 500))
            messages.append(AIMessage(content=f"answer {i} " + "z" * 500))
        messages.append(HumanMessage(content="current question"))

        result = compactor.compact(messages=messages)

        # Should be significantly smaller
        assert len(result) < len(messages)
        # Last human message preserved
        assert any(
            isinstance(m, HumanMessage) and m.content == "current question"
            for m in result
        )


class TestEdgeCases:
    def test_empty_messages_returns_empty(self, *, compactor: ContextCompactor) -> None:
        assert compactor.compact(messages=[]) == []

    def test_single_human_message_unchanged(
        self, *, compactor: ContextCompactor
    ) -> None:
        messages: list[AnyMessage] = [HumanMessage(content="hello")]
        result = compactor.compact(messages=messages)
        assert len(result) == 1
        assert result[0].content == "hello"

    def test_preserves_system_messages(self, *, compactor: ContextCompactor) -> None:
        messages: list[AnyMessage] = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="first"),
            AIMessage(
                content="",
                tool_calls=[{"name": "t", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="x" * 5000, tool_call_id="tc1", name="t"),
            AIMessage(content="response"),
            HumanMessage(content="second"),
        ]

        result = compactor.compact(messages=messages)

        system_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert any("helpful assistant" in str(m.content) for m in system_msgs)

    def test_includes_compaction_notice(self, *, compactor: ContextCompactor) -> None:
        messages: list[AnyMessage] = [
            HumanMessage(content="first question"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="x" * 5000, tool_call_id="tc1", name="search"),
            AIMessage(content="answer"),
            HumanMessage(content="second question"),
        ]

        result = compactor.compact(messages=messages)

        system_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert any("CONTEXT COMPACTED" in str(m.content) for m in system_msgs)
        notice = next(m for m in system_msgs if "CONTEXT COMPACTED" in str(m.content))
        assert "truncate_tool_results" in str(notice.content)
