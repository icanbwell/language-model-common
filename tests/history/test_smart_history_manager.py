"""Tests for SmartHistoryManager: server checkpoint first, client history as fallback.

The checkpointer and the summarizer LLM are external boundaries and are faked;
ConversationHistoryManager is the real class. tiktoken downloads its encoding on
first use, so it is replaced with a whitespace tokenizer.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence, cast

import pytest
import tiktoken
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from languagemodelcommon.history.conversation_history_manager import (
    ConversationHistoryManager,
)
from languagemodelcommon.history.smart_history_manager import SmartHistoryManager
from languagemodelcommon.state.messages_state import MyMessagesState


class _WordEncoding:
    def encode(self, text: str) -> list[str]:
        return text.split()


@dataclass
class _Checkpoint:
    channel_values: dict[str, Any]
    id: str = "ckpt-1"


@dataclass
class _FakeCheckpointer:
    checkpoint: _Checkpoint | None = None
    error: Exception | None = None
    configs: list[RunnableConfig] = field(default_factory=list)

    async def aget(self, config: RunnableConfig) -> _Checkpoint | None:
        self.configs.append(config)
        if self.error is not None:
            raise self.error
        return self.checkpoint


class _SummarizerWasCalled(BaseException):
    """BaseException so ConversationHistoryManager's `except Exception` cannot swallow it."""


class _UnusedLlm:
    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        raise _SummarizerWasCalled("history within limits must not call the LLM")


@pytest.fixture(autouse=True)
def _word_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tiktoken, "encoding_for_model", lambda model: _WordEncoding(), raising=True
    )


def _manager(
    *,
    checkpointer: _FakeCheckpointer | None,
    max_messages: int = 100,
    max_tokens: int = 10_000,
) -> SmartHistoryManager:
    return SmartHistoryManager(
        checkpointer=cast(Any, checkpointer),
        history_manager=ConversationHistoryManager(
            max_messages=max_messages,
            max_tokens=max_tokens,
            summary_threshold=1_000,
            keep_recent=10,
        ),
        llm=cast(Any, _UnusedLlm()),
    )


def _state(messages: Sequence[BaseMessage]) -> MyMessagesState:
    return cast(MyMessagesState, {"messages": list(messages)})


def _config(thread_id: str | None = "thread-1") -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}} if thread_id else {}


def _contents(state: MyMessagesState) -> list[str]:
    return [str(m.content) for m in state["messages"]]


def _server_history() -> list[BaseMessage]:
    return [
        HumanMessage(content="server q1"),
        AIMessage(content="server a1"),
        HumanMessage(content="server q2"),
        AIMessage(content="server a2"),
    ]


class TestClientHistoryFallback:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scenario",
        [
            "no-thread-id",
            "empty-thread-id",
            "no-checkpointer",
            "no-checkpoint-for-thread",
            "checkpoint-lookup-raises",
            "checkpoint-without-messages",
        ],
    )
    async def test_uses_the_client_history(self, scenario: str) -> None:
        checkpointer: _FakeCheckpointer | None = _FakeCheckpointer(
            checkpoint=_Checkpoint(channel_values={"messages": _server_history()})
        )
        thread_id: str | None = "thread-1"
        if scenario == "no-thread-id":
            thread_id = None
        elif scenario == "empty-thread-id":
            thread_id = ""
        elif scenario == "no-checkpointer":
            checkpointer = None
        elif scenario == "no-checkpoint-for-thread":
            checkpointer = _FakeCheckpointer(checkpoint=None)
        elif scenario == "checkpoint-lookup-raises":
            checkpointer = _FakeCheckpointer(error=RuntimeError("mongo down"))
        elif scenario == "checkpoint-without-messages":
            checkpointer = _FakeCheckpointer(checkpoint=_Checkpoint(channel_values={}))
        manager = _manager(checkpointer=checkpointer)
        client_messages = [HumanMessage(content="c1"), HumanMessage(content="c2")]

        result = await manager.select_history(
            state=_state(client_messages), config=_config(thread_id)
        )

        assert _contents(result) == ["c1", "c2"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("thread_id", [None, ""])
    async def test_checkpointer_is_not_consulted_without_a_thread_id(
        self, thread_id: str | None
    ) -> None:
        checkpointer = _FakeCheckpointer(
            checkpoint=_Checkpoint(channel_values={"messages": _server_history()})
        )
        manager = _manager(checkpointer=checkpointer)

        await manager.select_history(
            state=_state([HumanMessage(content="c1")]), config=_config(thread_id)
        )

        assert checkpointer.configs == []


class TestServerHistoryIsTheSourceOfTruth:
    @pytest.mark.asyncio
    async def test_looks_the_checkpoint_up_with_the_request_config(self) -> None:
        checkpointer = _FakeCheckpointer(
            checkpoint=_Checkpoint(channel_values={"messages": _server_history()})
        )
        config = _config("thread-42")

        await _manager(checkpointer=checkpointer).select_history(
            state=_state([HumanMessage(content="new")]), config=config
        )

        assert checkpointer.configs == [config]

    @pytest.mark.asyncio
    async def test_a_client_sending_only_the_newest_message_gets_the_stored_history(
        self,
    ) -> None:
        checkpointer = _FakeCheckpointer(
            checkpoint=_Checkpoint(channel_values={"messages": _server_history()})
        )

        result = await _manager(checkpointer=checkpointer).select_history(
            state=_state([HumanMessage(content="newest question")]),
            config=_config(),
        )

        assert _contents(result) == [
            "server q1",
            "server a1",
            "server q2",
            "server a2",
            "newest question",
        ]

    @pytest.mark.asyncio
    async def test_the_client_history_is_ignored_in_favour_of_the_server_history(
        self,
    ) -> None:
        checkpointer = _FakeCheckpointer(
            checkpoint=_Checkpoint(channel_values={"messages": _server_history()})
        )
        client_messages = [
            HumanMessage(content="client q1"),
            AIMessage(content="client a1"),
            HumanMessage(content="newest question"),
        ]

        result = await _manager(checkpointer=checkpointer).select_history(
            state=_state(client_messages), config=_config()
        )

        assert _contents(result)[:-1] == [
            "server q1",
            "server a1",
            "server q2",
            "server a2",
        ]
        assert _contents(result)[-1] == "newest question"

    @pytest.mark.asyncio
    async def test_a_request_with_no_messages_gets_the_stored_history(self) -> None:
        checkpointer = _FakeCheckpointer(
            checkpoint=_Checkpoint(channel_values={"messages": _server_history()})
        )

        result = await _manager(checkpointer=checkpointer).select_history(
            state=_state([]), config=_config()
        )

        assert _contents(result) == [
            "server q1",
            "server a1",
            "server q2",
            "server a2",
        ]

    @pytest.mark.asyncio
    async def test_a_message_already_in_the_stored_history_is_not_appended_again(
        self,
    ) -> None:
        checkpointer = _FakeCheckpointer(
            checkpoint=_Checkpoint(channel_values={"messages": _server_history()})
        )

        result = await _manager(checkpointer=checkpointer).select_history(
            state=_state([AIMessage(content="server a2")]), config=_config()
        )

        assert _contents(result) == [
            "server q1",
            "server a1",
            "server q2",
            "server a2",
        ]

    @pytest.mark.asyncio
    async def test_a_repeated_user_message_is_dropped_as_a_duplicate(self) -> None:
        """Documents current behavior: a user who really sends the same text again
        ("yes", "thanks") within the last five messages gets it silently dropped."""
        history: list[BaseMessage] = [
            HumanMessage(content="ready?"),
            AIMessage(content="Shall I continue?"),
            HumanMessage(content="yes"),
            AIMessage(content="Done. Shall I continue?"),
        ]
        checkpointer = _FakeCheckpointer(
            checkpoint=_Checkpoint(channel_values={"messages": history})
        )

        result = await _manager(checkpointer=checkpointer).select_history(
            state=_state([HumanMessage(content="yes")]), config=_config()
        )

        assert _contents(result) == [
            "ready?",
            "Shall I continue?",
            "yes",
            "Done. Shall I continue?",
        ]

    @pytest.mark.asyncio
    async def test_the_selected_history_is_trimmed_to_the_token_budget(self) -> None:
        history: list[BaseMessage] = [
            HumanMessage(content="one two three four"),
            AIMessage(content="five six seven eight"),
            HumanMessage(content="nine ten eleven twelve"),
        ]
        checkpointer = _FakeCheckpointer(
            checkpoint=_Checkpoint(channel_values={"messages": history})
        )
        manager = _manager(checkpointer=checkpointer, max_messages=2, max_tokens=8)

        result = await manager.select_history(
            state=_state([HumanMessage(content="newest question")]),
            config=_config(),
        )

        assert _contents(result) == ["nine ten eleven twelve", "newest question"]

    @pytest.mark.asyncio
    async def test_system_messages_in_the_stored_history_survive_trimming(
        self,
    ) -> None:
        history: list[BaseMessage] = [
            SystemMessage(content="be brief"),
            HumanMessage(content="one two three four"),
            AIMessage(content="five six seven eight"),
        ]
        checkpointer = _FakeCheckpointer(
            checkpoint=_Checkpoint(channel_values={"messages": history})
        )
        manager = _manager(checkpointer=checkpointer, max_messages=2, max_tokens=8)

        result = await manager.select_history(
            state=_state([HumanMessage(content="newest question")]),
            config=_config(),
        )

        assert _contents(result)[0] == "be brief"
        assert _contents(result)[-1] == "newest question"


class TestRealCheckpointerShape:
    @pytest.mark.xfail(
        strict=True,
        raises=AttributeError,
        reason=(
            "SmartHistoryManager reads `checkpoint.channel_values` as an attribute, but "
            "langgraph's BaseCheckpointSaver.aget returns a Checkpoint dict. Remove this "
            "marker once the production code reads checkpoint['channel_values']."
        ),
    )
    @pytest.mark.asyncio
    async def test_a_real_langgraph_checkpoint_supplies_the_stored_history(
        self,
    ) -> None:
        saver = InMemorySaver()
        graph = StateGraph(MessagesState)
        graph.add_node("reply", lambda state: {"messages": [("ai", "stored answer")]})
        graph.add_edge(START, "reply")
        compiled = graph.compile(checkpointer=saver)
        config = _config("real-thread")
        await compiled.ainvoke(
            {"messages": [HumanMessage(content="stored question")]}, config
        )
        manager = SmartHistoryManager(
            checkpointer=saver,
            history_manager=ConversationHistoryManager(),
            llm=cast(Any, _UnusedLlm()),
        )

        result = await manager.select_history(
            state=_state([HumanMessage(content="newest question")]), config=config
        )

        assert _contents(result) == [
            "stored question",
            "stored answer",
            "newest question",
        ]


class TestIsDuplicate:
    @pytest.mark.parametrize(
        ("new_message", "existing", "expected"),
        [
            (HumanMessage(content="hi"), [], False),
            (HumanMessage(content="hi"), [HumanMessage(content="hi")], True),
            (HumanMessage(content="hi"), [AIMessage(content="hi")], False),
            (HumanMessage(content="hi"), [HumanMessage(content="bye")], False),
            (
                HumanMessage(content="hi"),
                [HumanMessage(content="hi")]
                + [AIMessage(content=f"filler {i}") for i in range(5)],
                False,
            ),
            (
                HumanMessage(content="hi"),
                [HumanMessage(content="hi")]
                + [AIMessage(content=f"filler {i}") for i in range(4)],
                True,
            ),
        ],
        ids=[
            "empty-history",
            "same-type-and-content",
            "same-content-different-type",
            "different-content",
            "older-than-the-last-five",
            "within-the-last-five",
        ],
    )
    def test_only_the_last_five_messages_are_checked(
        self,
        new_message: BaseMessage,
        existing: list[BaseMessage],
        expected: bool,
    ) -> None:
        manager = _manager(checkpointer=None)

        assert (
            manager._is_duplicate(new_message=new_message, existing_messages=existing)
            is expected
        )


class TestAsGraphNode:
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=(
            "select_history returns the trimmed messages as a state update, but "
            "MyMessagesState.messages uses the add_messages reducer, which never drops "
            "messages, so the next node still receives the untrimmed history. Remove this "
            "marker once trimming actually reduces what the next node sees."
        ),
    )
    @pytest.mark.asyncio
    async def test_the_next_node_receives_the_trimmed_history(self) -> None:
        seen_by_next_node: list[list[str]] = []

        def _capture(state: MyMessagesState) -> dict[str, Any]:
            seen_by_next_node.append([str(m.content) for m in state["messages"]])
            return {}

        manager = _manager(checkpointer=None, max_messages=2, max_tokens=8)
        graph = StateGraph(MyMessagesState)
        graph.add_node("select_history", RunnableLambda(manager.select_history))
        graph.add_node("capture", _capture)
        graph.add_edge(START, "select_history")
        graph.add_edge("select_history", "capture")
        graph.add_edge("capture", END)
        compiled = graph.compile()
        history: list[BaseMessage] = [
            HumanMessage(content="one two three four"),
            AIMessage(content="five six seven eight"),
            HumanMessage(content="nine ten eleven twelve"),
        ]

        await compiled.ainvoke(_state(history))

        assert seen_by_next_node == [["five six seven eight", "nine ten eleven twelve"]]
