"""ChatBedrockConverse subclass that sanitizes tool names before sending to Bedrock.

AWS Bedrock ConverseStream validates tool names against [a-zA-Z0-9_-]+.
When an LLM hallucinates a tool_use block with invalid characters in the name
(e.g., dots, colons, slashes), that response gets checkpointed by LangGraph.
On the next turn, the invalid name is replayed to Bedrock and causes a
ValidationException.

This subclass intercepts messages before serialization and replaces invalid
characters in tool_use names with underscores.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Iterator, List

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import BaseMessage, AIMessage

logger = logging.getLogger(__name__)

_INVALID_TOOL_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_tool_name(name: str) -> str:
    return _INVALID_TOOL_NAME_CHARS.sub("_", name)


def _sanitize_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Sanitize tool_use names in AI messages to satisfy Bedrock constraints."""
    sanitized = False
    result: List[BaseMessage] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            result.append(msg)
            continue

        needs_fix = False
        if msg.tool_calls:
            for tc in msg.tool_calls:
                name = (
                    tc.get("name", "")
                    if isinstance(tc, dict)
                    else getattr(tc, "name", "")
                )
                if name and _INVALID_TOOL_NAME_CHARS.search(name):
                    needs_fix = True
                    break

        if not needs_fix and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "")
                    if name and _INVALID_TOOL_NAME_CHARS.search(name):
                        needs_fix = True
                        break

        if not needs_fix:
            result.append(msg)
            continue

        sanitized = True
        new_tool_calls = None
        if msg.tool_calls:
            new_tool_calls = []
            for tc in msg.tool_calls:
                if isinstance(tc, dict):
                    tc_copy = dict(tc)
                    if tc_copy.get("name"):
                        tc_copy["name"] = _sanitize_tool_name(str(tc_copy["name"]))
                    new_tool_calls.append(tc_copy)
                else:
                    new_tool_calls.append(tc)

        new_content = msg.content
        if isinstance(msg.content, list):
            new_content = []
            for block in msg.content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name")
                ):
                    block_copy = dict(block)
                    block_copy["name"] = _sanitize_tool_name(block_copy["name"])
                    new_content.append(block_copy)
                else:
                    new_content.append(block)

        patched = msg.model_copy(
            update={
                "content": new_content,
                **(
                    {"tool_calls": new_tool_calls} if new_tool_calls is not None else {}
                ),
            }
        )
        result.append(patched)

    if sanitized:
        logger.warning(
            "Sanitized invalid tool_use names in messages before sending to Bedrock"
        )

    return result


class SanitizingChatBedrockConverse(ChatBedrockConverse):
    """ChatBedrockConverse that sanitizes tool names to comply with Bedrock's regex."""

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        return super()._stream(
            _sanitize_messages(messages), stop, run_manager, **kwargs
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        return super()._generate(
            _sanitize_messages(messages), stop, run_manager, **kwargs
        )
