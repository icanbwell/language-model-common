"""
Chat Message Helper Utilities for LangChain/OpenAI Message Conversion.

This module provides utilities for converting between LangChain message types
and OpenAI API message formats (Chat Completions and Responses API).

Request flow context:
    LangGraph agent → LangChain messages → **chat_message_helpers** → OpenAI format → SSE/JSON

Actively used functions:
- `convert_message_content_to_string` – Used by streaming_manager, bailey_agent_services
- `langchain_to_chat_message` – Used by chat_completion_api_request_wrapper
- `remove_tool_calls` – Reserved for Anthropic streaming support

Deprecated functions (see individual docstrings for details):
- `convert_message_content_to_list` – Superseded by streaming_manager
- `langchain_to_response_message` – Superseded by direct ResponseOutputMessage construction
"""

import warnings
from dataclasses import dataclass
from typing import Dict, Any, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)
from openai.types.chat import ChatCompletionMessage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from languagemodelcommon.utilities.text_humanizer import Humanizer


@dataclass(frozen=True)
class ContentChunks:
    text_chunks: list[str]
    non_text_blocks: list[dict[str, Any]]


def iter_message_content_text_chunks(
    content: str | list[str | Dict[str, Any]],
    include_non_text_placeholders: bool = True,
) -> ContentChunks:
    """
    Normalize message content into text chunks for streaming and capture non-text blocks.

    Non-text blocks (e.g., images) are returned separately so callers can emit
    safe debug summaries without leaking payloads into OpenAI-compatible output.
    """
    if isinstance(content, str):
        return ContentChunks(text_chunks=[content], non_text_blocks=[])
    text_chunks: list[str] = []
    non_text_blocks: list[dict[str, Any]] = []
    for content_item in content:
        if isinstance(content_item, str):
            text_chunks.append(content_item)
        elif isinstance(content_item, dict):
            content_item_type: Optional[str] = content_item.get("type")
            # text_chunks.append(f"({content_item_type or 'unknown'})")  # Placeholder for all content types
            if content_item_type in ("text", "input_text", "output_text"):
                text_item = content_item.get("text")
                if isinstance(text_item, str) and text_item:
                    text_chunks.append(text_item)
            elif content_item_type == "refusal":
                refusal_text = content_item.get("refusal")
                if isinstance(refusal_text, str) and refusal_text:
                    text_chunks.append(refusal_text)
                else:
                    text_chunks.append("[refusal]")
                non_text_blocks.append(content_item)
            elif content_item_type == "image_url":
                image_url = content_item.get("image_url")
                url_value = (
                    image_url.get("url") if isinstance(image_url, dict) else image_url
                )
                if include_non_text_placeholders:
                    text_chunks.append(
                        f"[image_url:{url_value}]"
                        if isinstance(url_value, str) and url_value
                        else "[image_url]"
                    )
                non_text_blocks.append(content_item)
            elif content_item_type in ("input_image", "output_image"):
                image_url = content_item.get("image_url")
                url_value = (
                    image_url.get("url") if isinstance(image_url, dict) else image_url
                )
                if include_non_text_placeholders:
                    text_chunks.append(
                        f"[{content_item_type}:{url_value}]"
                        if isinstance(url_value, str) and url_value
                        else f"[{content_item_type}]"
                    )
                non_text_blocks.append(content_item)
            elif content_item_type in ("input_audio", "output_audio"):
                if include_non_text_placeholders:
                    text_chunks.append(f"[{content_item_type}]")
                non_text_blocks.append(content_item)
            elif content_item_type == "tool_use":
                tool_name = content_item.get("name")
                tool_id = content_item.get("id")
                if include_non_text_placeholders:
                    # text_chunks.append(f"[tool_use:{json.dumps(content_item)}]")
                    if isinstance(tool_name, str) and tool_name:
                        # suffix = (
                        #     f"#{tool_id}"
                        #     if isinstance(tool_id, str) and tool_id
                        #     else ""
                        # )
                        text_chunks.append(
                            f" Using {Humanizer.humanize_tool_name(tool_name)} {tool_id} Skill. "
                        )
                    else:
                        text_chunks.append("[tool_use]")
                non_text_blocks.append(content_item)
            elif content_item_type == "tool_result":
                tool_name = content_item.get("name")
                tool_id = content_item.get("tool_use_id")
                label = (
                    tool_name
                    if isinstance(tool_name, str) and tool_name
                    else "tool_result"
                )
                suffix = f"#{tool_id}" if isinstance(tool_id, str) and tool_id else ""
                if include_non_text_placeholders:
                    text_chunks.append(f"[{label}{suffix}]")
                non_text_blocks.append(content_item)
            elif content_item_type in ("reasoning_content", "reasoning"):
                # Extended thinking / reasoning blocks from Anthropic models.
                # Extract the reasoning text and store in non_text_blocks
                # so callers (e.g. streaming_manager) can render it in
                # debug mode via <details> without leaking it into the
                # main response text.
                reasoning_text: str | None = None
                if content_item_type == "reasoning_content":
                    rc = content_item.get("reasoning_content", {})
                    if isinstance(rc, dict):
                        reasoning_text = rc.get("text")
                elif content_item_type == "reasoning":
                    reasoning_text = content_item.get("reasoning")
                non_text_blocks.append(content_item)
                if include_non_text_placeholders and reasoning_text:
                    text_chunks.append("[reasoning]")
            else:
                non_text_blocks.append(content_item)
                if include_non_text_placeholders:
                    text_chunks.append(f"[{content_item_type or 'unknown'}]")
        else:
            raise TypeError(
                "iter_message_content_text_chunks: Unsupported content item type: "
                f"{type(content_item)}: {content_item}"
            )
    return ContentChunks(text_chunks=text_chunks, non_text_blocks=non_text_blocks)


def convert_message_content_to_string(content: str | list[str | Dict[str, Any]]) -> str:
    """
    Convert message content (string or list of content blocks) to a single string.

    Handles both simple string content and structured content blocks (e.g., from
    multi-modal messages or tool responses).

    Args:
        content: Either a string or a list of strings/dicts with 'type' and 'text' keys.

    Returns:
        Concatenated text content as a single string.

    Raises:
        TypeError: If content contains unsupported item types.
    """
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
        elif isinstance(content_item, dict):
            content_item_type: Optional[str] = content_item.get("type")
            if content_item_type == "text":
                text.append(content_item.get("text") or "")
        else:
            raise TypeError(
                f"convert_message_content_to_string: Unsupported content item type: {type(content_item)}: {content_item}"
            )
    return "".join(text)


def convert_message_content_to_list(
    content: str | list[str | Dict[str, Any]],
) -> list[str | Dict[str, Any]]:
    """
    Convert message content to a list format.

    .. deprecated::
        This function is unused and superseded by
        `LangGraphStreamingManager.get_structured_content_from_tool_message()`
        in `converters/streaming_manager.py`, which provides more specialized
        handling for MCP tool response formats.

        Scheduled for removal in a future release.

    Args:
        content: Either a string or a list of strings/dicts.

    Returns:
        Content wrapped in a list if it was a string, or the original list.

    Raises:
        TypeError: If content is neither a string nor a list.
    """
    warnings.warn(
        "convert_message_content_to_list is deprecated and will be removed. "
        "Use LangGraphStreamingManager.get_structured_content_from_tool_message() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(content, str):
        return [content]
    elif isinstance(content, list):
        return content
    else:
        raise TypeError(
            f"convert_message_content_to_list: Unsupported content type: {type(content)}: {content}"
        )


def langchain_to_chat_message(message: BaseMessage) -> Optional[ChatCompletionMessage]:
    """
    Convert a LangChain message to an OpenAI ChatCompletionMessage.

    Only AIMessage and ToolMessage (with artifact) are converted; other message
    types raise ValueError since they shouldn't appear in assistant responses.

    Args:
        message: A LangChain BaseMessage instance.

    Returns:
        ChatCompletionMessage for AIMessage/ToolMessage, None for ToolMessage without artifact.

    Raises:
        ValueError: For SystemMessage, HumanMessage, LangchainChatMessage, or unknown types.
    """
    match message:
        case SystemMessage():
            raise ValueError(
                "System messages should not be converted to ChatCompletionMessage"
            )
        case HumanMessage():
            raise ValueError(
                "Human messages should not be converted to ChatCompletionMessage"
            )
        case AIMessage():
            ai_message = ChatCompletionMessage(
                role="assistant",
                content=convert_message_content_to_string(message.content),
            )
            return ai_message
        case ToolMessage():
            artifact: str = message.artifact
            if artifact:
                ai_message = ChatCompletionMessage(
                    role="assistant",
                    content=f"\n[{artifact}]\n",
                )
                return ai_message
        case LangchainChatMessage():
            raise ValueError(
                "Chat messages should not be converted to ChatCompletionMessage"
            )
        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")
    return None


def langchain_to_response_message(
    message: BaseMessage,
) -> Optional[ResponseOutputMessage]:
    """
    Convert a LangChain message to an OpenAI ResponseOutputMessage.

    .. deprecated::
        This function is unused and superseded by direct construction of
        `ResponseOutputMessage` objects in `BaileyAgentService.process_responses_request()`
        and `stream_responses_request()` methods.

        The Responses API handling was implemented to build response objects inline
        rather than converting from LangChain messages through this helper.

        Scheduled for removal in a future release.

    Args:
        message: A LangChain BaseMessage instance.

    Returns:
        ResponseOutputMessage for AIMessage/ToolMessage, None for ToolMessage without artifact.

    Raises:
        ValueError: For SystemMessage, HumanMessage, LangchainChatMessage, or unknown types.
    """
    warnings.warn(
        "langchain_to_response_message is deprecated and will be removed. "
        "Build ResponseOutputMessage objects directly in BaileyAgentService instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    match message:
        case SystemMessage():
            raise ValueError(
                "System messages should not be converted to ChatCompletionMessage"
            )
        case HumanMessage():
            raise ValueError(
                "Human messages should not be converted to ChatCompletionMessage"
            )
        case AIMessage():
            ai_message: ResponseOutputMessage = ResponseOutputMessage(
                id=message.id or "0",
                role="assistant",
                content=[
                    ResponseOutputText(
                        type="output_text",
                        text=convert_message_content_to_string(message.content),
                        annotations=[],
                    )
                ],
                status="completed",
                type="message",
            )
            return ai_message
        case ToolMessage():
            artifact: str = message.artifact
            if artifact:
                ai_message = ResponseOutputMessage(
                    id=message.id or "0",
                    role="assistant",
                    content=[
                        ResponseOutputText(
                            type="output_text", text=f"\n[{artifact}]\n", annotations=[]
                        )
                    ],
                    status="completed",
                    type="message",
                )
                return ai_message
        case LangchainChatMessage():
            raise ValueError(
                "Chat messages should not be converted to ChatCompletionMessage"
            )
        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")
    return None


def remove_tool_calls(
    content: str | list[str | dict[str, Any]],
) -> str | list[str | dict[str, Any]]:
    """
    Remove Anthropic tool_use content blocks from message content.

    When streaming responses from Anthropic models, tool calls appear as content
    blocks with type 'tool_use'. This function filters them out so only text
    content is returned to the client.

    Note: Currently reserved for future Anthropic streaming support. The streaming
    manager handles tool calls via on_tool_start/on_tool_end events rather than
    content block filtering.

    Args:
        content: Either a string (returned as-is) or a list of content blocks.

    Returns:
        Original string, or list with tool_use blocks removed.
    """
    if isinstance(content, str):
        return content
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item.get("type") != "tool_use"
    ]
