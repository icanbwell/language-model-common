from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from languagemodelcommon.state.messages_state import MyMessagesState
from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)
from languagemodelcommon.utilities.request_information import RequestInformation


@dataclass
class PipelineContext:
    """Shared mutable state that flows through all pipeline steps."""

    chat_request_wrapper: ChatRequestWrapper
    request: Any = None

    request_information: RequestInformation | None = None

    messages: list[AnyMessage] = field(default_factory=list)
    prior_messages: list[AnyMessage] = field(default_factory=list)
    previous_context: dict[str, Any] | None = None

    state: MyMessagesState | None = None
    config: RunnableConfig | None = None
    graph: CompiledStateGraph[MyMessagesState] | None = None

    content_stream: AsyncGenerator[str, None] | None = None

    accumulated_content: str = ""
    response_messages: list[AnyMessage] = field(default_factory=list)

    conversation_id: str = ""
    user_id: str = ""
