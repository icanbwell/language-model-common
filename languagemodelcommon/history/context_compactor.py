import logging
from typing import List

import tiktoken
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)


class ContextCompactor:
    """
    Compacts conversation messages when the model's context window is exceeded.

    Applies a multi-pass reduction strategy:
    1. Strip tool result content from older turns (keep tool call names only)
    2. Drop older tool call/result pairs entirely
    3. Summarize older human/assistant exchanges into a system message

    The most recent human message and the current turn's tool interactions
    are always preserved intact.
    """

    def __init__(self, *, model: str = "gpt-4") -> None:
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def compact(self, *, messages: List[AnyMessage]) -> List[AnyMessage]:
        """
        Compact messages to reduce total token count.

        Applies progressively more aggressive reduction until the message
        list is meaningfully smaller than the input.

        Args:
            messages: The full list of messages that exceeded the context window.

        Returns:
            A compacted list of messages preserving the most recent turn.
        """
        if not messages:
            return messages

        original_tokens = self._count_tokens(messages)
        logger.info(
            "[CONTEXT_COMPACTOR] Starting compaction: %d messages, ~%d tokens",
            len(messages),
            original_tokens,
        )

        system_messages, conversation_messages = self._split_system_messages(
            messages=messages
        )

        # Pass 1: Truncate tool result content from older turns
        compacted = self._truncate_old_tool_results(messages=conversation_messages)

        tokens_after_pass1 = self._count_tokens(system_messages + compacted)
        logger.info(
            "[CONTEXT_COMPACTOR] After pass 1 (truncate tool results): %d messages, ~%d tokens",
            len(system_messages) + len(compacted),
            tokens_after_pass1,
        )

        if tokens_after_pass1 < original_tokens * 0.5:
            return system_messages + compacted

        # Pass 2: Drop older tool call/result pairs entirely
        compacted = self._drop_old_tool_pairs(messages=compacted)

        tokens_after_pass2 = self._count_tokens(system_messages + compacted)
        logger.info(
            "[CONTEXT_COMPACTOR] After pass 2 (drop tool pairs): %d messages, ~%d tokens",
            len(system_messages) + len(compacted),
            tokens_after_pass2,
        )

        if tokens_after_pass2 < original_tokens * 0.5:
            return system_messages + compacted

        # Pass 3: Summarize older exchanges, keep only recent turn
        compacted = self._summarize_old_exchanges(messages=compacted)

        tokens_after_pass3 = self._count_tokens(system_messages + compacted)
        logger.info(
            "[CONTEXT_COMPACTOR] After pass 3 (summarize old): %d messages, ~%d tokens",
            len(system_messages) + len(compacted),
            tokens_after_pass3,
        )

        return system_messages + compacted

    def _split_system_messages(
        self, *, messages: List[AnyMessage]
    ) -> tuple[List[AnyMessage], List[AnyMessage]]:
        system: List[AnyMessage] = []
        conversation: List[AnyMessage] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system.append(msg)
            else:
                conversation.append(msg)
        return system, conversation

    def _truncate_old_tool_results(
        self, *, messages: List[AnyMessage]
    ) -> List[AnyMessage]:
        """Replace tool result content with a short summary in older turns.

        Preserves the last human message and any tool messages that follow it
        (current turn). Older ToolMessages get their content replaced with
        a truncated version.
        """
        last_human_idx = self._find_last_human_index(messages=messages)
        result: List[AnyMessage] = []

        for i, msg in enumerate(messages):
            if i >= last_human_idx:
                result.append(msg)
            elif isinstance(msg, ToolMessage):
                content_str = str(msg.content) if msg.content else ""
                if len(content_str) > 200:
                    truncated_content = content_str[:100] + "\n...[truncated]..."
                    result.append(
                        ToolMessage(
                            content=truncated_content,
                            tool_call_id=msg.tool_call_id,
                            name=getattr(msg, "name", None) or "",
                        )
                    )
                else:
                    result.append(msg)
            elif isinstance(msg, AIMessage) and msg.tool_calls:
                # Keep the AI message but truncate any large content blocks
                content = msg.content
                if isinstance(content, str) and len(content) > 500:
                    content = content[:200] + "\n...[truncated]..."
                result.append(
                    AIMessage(
                        content=content,
                        tool_calls=msg.tool_calls,
                    )
                )
            else:
                result.append(msg)

        return result

    def _drop_old_tool_pairs(self, *, messages: List[AnyMessage]) -> List[AnyMessage]:
        """Remove tool call/result pairs from older turns entirely.

        Keeps the AI message text but removes tool_calls, and drops
        corresponding ToolMessages. Current turn is preserved.
        """
        last_human_idx = self._find_last_human_index(messages=messages)
        result: List[AnyMessage] = []

        for i, msg in enumerate(messages):
            if i >= last_human_idx:
                result.append(msg)
            elif isinstance(msg, ToolMessage):
                continue
            elif isinstance(msg, AIMessage) and msg.tool_calls:
                text_content = self._extract_text_content(msg=msg)
                if text_content:
                    result.append(AIMessage(content=text_content))
            else:
                result.append(msg)

        return result

    def _summarize_old_exchanges(
        self, *, messages: List[AnyMessage]
    ) -> List[AnyMessage]:
        """Collapse older exchanges into a single summary system message.

        Keeps only the most recent turn (last human message + everything after).
        Older messages are collapsed into a brief textual summary.
        """
        last_human_idx = self._find_last_human_index(messages=messages)

        if last_human_idx <= 0:
            return messages

        old_messages = messages[:last_human_idx]
        recent_messages = messages[last_human_idx:]

        summary_parts: List[str] = []
        for msg in old_messages:
            role = msg.__class__.__name__.replace("Message", "").lower()
            content = self._extract_text_content(msg=msg)
            if content:
                # Take first 150 chars of each old message
                abbreviated = content[:150].strip()
                if len(content) > 150:
                    abbreviated += "..."
                summary_parts.append(f"[{role}]: {abbreviated}")

        if summary_parts:
            summary_text = (
                "[Previous conversation summary — "
                f"{len(old_messages)} messages compacted]\n"
                + "\n".join(summary_parts[-10:])  # keep last 10 at most
            )
            summary_msg = SystemMessage(content=summary_text)
            return [summary_msg] + recent_messages

        return recent_messages

    def _find_last_human_index(self, *, messages: List[AnyMessage]) -> int:
        """Find the index of the last HumanMessage."""
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                return i
        return 0

    def _extract_text_content(self, *, msg: BaseMessage) -> str:
        """Extract text content from a message, handling list content."""
        if isinstance(msg.content, str):
            return msg.content
        if isinstance(msg.content, list):
            text_parts = []
            for block in msg.content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return " ".join(text_parts)
        return str(msg.content) if msg.content else ""

    def _count_tokens(self, messages: List[AnyMessage]) -> int:
        total = 0
        for msg in messages:
            content = self._extract_text_content(msg=msg)
            total += len(self.encoding.encode(content))
        return total
