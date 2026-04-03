import logging
from typing import Sequence, cast, Any

import tiktoken
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from languagemodelcommon.state.messages_state import MyMessagesState
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.LLM)


# Add this helper class
class ConversationHistoryManager:
    def __init__(
        self,
        max_messages: int = 20,
        max_tokens: int = 4000,
        summary_threshold: int = 15,
        keep_recent: int = 10,
        model: str = "gpt-4",
    ):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold
        self.keep_recent = keep_recent
        self.encoding = tiktoken.encoding_for_model(model)

    def count_tokens(self, messages: Sequence[BaseMessage]) -> int:
        """Count tokens in messages"""
        return sum(len(self.encoding.encode(str(msg.content))) for msg in messages)

    async def manage_history(
        self, state: MyMessagesState, llm: BaseChatModel
    ) -> MyMessagesState:
        """Manage conversation history with summarization"""
        messages = state["messages"]

        # If under threshold, return as-is
        if len(messages) <= self.max_messages:
            token_count = self.count_tokens(messages)
            if token_count <= self.max_tokens:
                return state

        logger.info(
            f"Managing history: {len(messages)} messages, "
            f"{self.count_tokens(messages)} tokens"
        )

        # Separate system messages from conversation
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        conversation_messages = [
            m for m in messages if not isinstance(m, SystemMessage)
        ]

        # If we have too many conversation messages, summarize old ones
        if len(conversation_messages) > self.summary_threshold:
            managed_messages = await self._summarize_with_recent(
                conversation_messages, llm
            )
        else:
            managed_messages = list(conversation_messages)

        # Combine system messages with managed conversation
        result_messages = system_messages + managed_messages

        # Final token-based trimming if still needed
        if self.count_tokens(result_messages) > self.max_tokens:
            result_messages = self._trim_by_tokens(result_messages)

        logger.info(
            f"History managed: {len(result_messages)} messages, "
            f"{self.count_tokens(result_messages)} tokens"
        )

        state["messages"] = cast(list[Any], result_messages)
        return state

    async def _summarize_with_recent(
        self, messages: Sequence[BaseMessage], llm: BaseChatModel
    ) -> list[BaseMessage]:
        """Summarize old messages, keep recent ones"""
        if len(messages) <= self.keep_recent:
            return list(messages)

        old_messages = messages[: -self.keep_recent]
        recent_messages = messages[-self.keep_recent :]

        # Create summary of old messages
        summary_content = "\n".join(
            [f"{m.__class__.__name__}: {m.content}" for m in old_messages]
        )

        summary_prompt = f"""Summarize this conversation history concisely, preserving:
- Key facts and decisions
- Important context
- User preferences or requirements
- Any unresolved issues

Conversation:
{summary_content}

Provide a brief summary (2-3 paragraphs max):"""

        try:
            summary_response = await llm.ainvoke(
                [
                    SystemMessage(content="You are a conversation summarizer."),
                    HumanMessage(content=summary_prompt),
                ]
            )
            summary_text = summary_response.content
        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            # Fallback: just truncate
            return list(messages[-self.keep_recent :])

        # Create new message list with summary + recent
        result: list[BaseMessage] = [
            SystemMessage(
                content=f"[Previous conversation summary ({len(old_messages)} messages)]\n{summary_text}"
            )
        ]
        result.extend(recent_messages)

        return result

    def _trim_by_tokens(self, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        """Trim messages to fit token limit, keeping system messages"""
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        other_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        # Always keep system messages
        result: list[BaseMessage] = list(system_messages)
        current_tokens = self.count_tokens(result)

        # Add other messages from most recent until we hit limit
        for msg in reversed(other_messages):
            msg_tokens = len(self.encoding.encode(str(msg.content)))
            if current_tokens + msg_tokens <= self.max_tokens:
                result.insert(len(system_messages), msg)
                current_tokens += msg_tokens
            else:
                break

        return result
