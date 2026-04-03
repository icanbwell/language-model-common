"""
Conversation history management for context window optimization.
"""

import logging
from typing import Sequence

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
import tiktoken

from languagemodelcommon.state.messages_state import MyMessagesState

logger = logging.getLogger(__name__)


class ConversationHistoryManager:
    """
    Manages conversation history to fit within model context windows.

    Provides strategies for trimming and summarizing long conversation histories
    while preserving important context and recent messages.

    Attributes:
        max_messages: Maximum number of messages before triggering management
        max_tokens: Maximum token count allowed in conversation history
        summary_threshold: Number of messages that triggers summarization
        keep_recent: Number of recent messages to always preserve
        model: Model name for token counting
    """

    def __init__(
        self,
        max_messages: int = 20,
        max_tokens: int = 4000,
        summary_threshold: int = 15,
        keep_recent: int = 10,
        model: str = "gpt-4",
    ):
        """
        Initialize the history manager.

        Args:
            max_messages: Maximum messages before management kicks in
            max_tokens: Maximum tokens allowed in history
            summary_threshold: When to start summarizing old messages
            keep_recent: How many recent messages to always keep
            model: Model name for token encoding
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold
        self.keep_recent = keep_recent

        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Model {model} not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages: Sequence[BaseMessage]) -> int:
        """
        Count total tokens in a list of messages.

        Args:
            messages: List of messages to count

        Returns:
            Total token count
        """
        return sum(len(self.encoding.encode(str(msg.content))) for msg in messages)

    async def manage_history(
        self, state: MyMessagesState, llm: BaseChatModel
    ) -> MyMessagesState:
        """
        Manage conversation history with summarization and trimming.

        Strategy:
        1. If under thresholds, return as-is
        2. If over summary_threshold, summarize old messages
        3. If still over max_tokens, trim by tokens

        Args:
            state: Current conversation state
            llm: Language model for generating summaries

        Returns:
            Updated state with managed history
        """
        messages = state.get("messages", [])

        # Check if management is needed
        if len(messages) <= self.max_messages:
            token_count = self.count_tokens(messages)
            if token_count <= self.max_tokens:
                logger.debug(
                    f"History within limits: {len(messages)} messages, "
                    f"{token_count} tokens"
                )
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

        # Apply summarization if needed
        managed_messages: list[BaseMessage]
        if len(conversation_messages) > self.summary_threshold:
            managed_messages = await self._summarize_with_recent(
                conversation_messages, llm
            )
        else:
            managed_messages = list(conversation_messages)

        # Combine system messages with managed conversation
        result_messages: list[BaseMessage] = list(system_messages) + managed_messages

        # Final token-based trimming if still needed
        if self.count_tokens(result_messages) > self.max_tokens:
            result_messages = self._trim_by_tokens(result_messages)

        logger.info(
            f"History managed: {len(result_messages)} messages, "
            f"{self.count_tokens(result_messages)} tokens"
        )

        state["messages"] = result_messages  # type: ignore[typeddict-item]
        return state

    async def _summarize_with_recent(
        self, messages: Sequence[BaseMessage], llm: BaseChatModel
    ) -> list[BaseMessage]:
        """
        Summarize old messages while keeping recent ones intact.

        Args:
            messages: Conversation messages to process
            llm: Language model for summarization

        Returns:
            List with summary message + recent messages
        """
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
            summary_text = str(summary_response.content)

            logger.info(
                f"Summarized {len(old_messages)} messages into summary "
                f"({len(self.encoding.encode(summary_text))} tokens)"
            )
        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            # Fallback: just keep recent messages
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
        """
        Trim messages to fit token limit, preserving system messages.

        Args:
            messages: Messages to trim

        Returns:
            Trimmed message list within token limit
        """
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
                logger.debug(
                    f"Skipping message ({msg_tokens} tokens) - would exceed limit"
                )
                break

        logger.info(f"Trimmed to {len(result)} messages, {current_tokens} tokens")

        return result
