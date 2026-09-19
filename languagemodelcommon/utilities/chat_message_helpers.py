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

from languagemodelcommon.utilities.text_humanizer import Humanizer


@dataclass(frozen=True)
class ContentChunks:
    text_chunks: list[str]
    non_text_blocks: list[dict[str, Any]]


def iter_message_content_text_chunks(
    *,
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
            elif content_item_type == "image":
                # LangChain's own ImageContentBlock (langchain_core.messages.content),
                # e.g. from convert_call_tool_result. Never a text chunk on its own;
                # extract_image_output_parts is how callers recover the image itself.
                if include_non_text_placeholders:
                    text_chunks.append("[image]")
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
                    if isinstance(tool_name, str) and tool_name:
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


def extract_image_output_parts(
    *,
    non_text_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build OpenAI-style ``output_image`` content parts from the non-text
    content blocks captured by `iter_message_content_text_chunks`.

    Recognizes both LangChain's own `ImageContentBlock` (`type: "image"`,
    with `url`/`base64`/`mime_type` -- e.g. from `convert_call_tool_result`)
    and the OpenAI/Anthropic-style `image_url`/`input_image`/`output_image`
    dict shapes. Blocks with no resolvable URL (e.g. a bare `file_id`
    reference with no inline data) are skipped rather than emitting a part
    with nothing to render.
    """
    image_parts: list[dict[str, Any]] = []
    for block in non_text_blocks:
        block_type = block.get("type")
        if block_type == "image":
            mime_type = block.get("mime_type")
            url = block.get("url")
            base64_data = block.get("base64")
            if not url and base64_data:
                url = f"data:{mime_type or 'application/octet-stream'};base64,{base64_data}"
            if not url:
                continue
            image_parts.append(
                {"type": "output_image", "image_url": url, "mime_type": mime_type}
            )
        elif block_type in ("image_url", "input_image", "output_image"):
            image_url = block.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not isinstance(url, str) or not url:
                continue
            image_parts.append(
                {"type": "output_image", "image_url": url, "mime_type": None}
            )
    return image_parts


def build_openai_message_content(
    *,
    content: str | list[str | Dict[str, Any]],
) -> str | list[dict[str, Any]]:
    """
    Convert LangChain message content into an OpenAI-compatible content value.

    Returns a plain `str` when the content has no image parts -- byte-for-byte
    what `convert_message_content_to_string` would have produced, so every
    existing text-only consumer is unaffected. Only when an image content
    block (see `extract_image_output_parts`) is present does this return a
    list of content parts (`text` + `output_image`) instead of silently
    dropping the image.
    """
    if isinstance(content, str):
        return content
    chunks = iter_message_content_text_chunks(
        content=content, include_non_text_placeholders=False
    )
    text = "".join(chunks.text_chunks)
    image_parts = extract_image_output_parts(non_text_blocks=chunks.non_text_blocks)
    if not image_parts:
        return text
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    parts.extend(image_parts)
    return parts


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
            content_value = build_openai_message_content(content=message.content)
            if isinstance(content_value, list):
                # openai's ChatCompletionMessage.content is declared Optional[str];
                # a content-part list is an additive extension of that contract
                # (see BAI-806 design doc), so bypass field validation via
                # model_construct rather than raising on the list we intend to
                # carry. Serialization (model_dump/model_dump_json) still emits
                # exactly what was assigned.
                return ChatCompletionMessage.model_construct(
                    role="assistant",
                    content=content_value,  # type: ignore[arg-type]
                )
            ai_message = ChatCompletionMessage(
                role="assistant",
                content=content_value,
            )
            return ai_message
        case ToolMessage():
            artifact: str = message.artifact
            # Non-text content (e.g. an image from an MCP tool result) is
            # only present when message.content is a content-block list, not
            # a plain string.
            content_text: str = ""
            image_parts: list[dict[str, Any]] = []
            if not isinstance(message.content, str):
                content_chunks = iter_message_content_text_chunks(
                    content=message.content,
                    include_non_text_placeholders=False,
                )
                content_text = "".join(content_chunks.text_chunks)
                image_parts = extract_image_output_parts(
                    non_text_blocks=content_chunks.non_text_blocks
                )
            if artifact:
                if image_parts:
                    parts: list[dict[str, Any]] = [
                        {"type": "text", "text": f"\n[{artifact}]\n"}
                    ]
                    parts.extend(image_parts)
                    return ChatCompletionMessage.model_construct(
                        role="assistant",
                        content=parts,  # type: ignore[arg-type]
                    )
                ai_message = ChatCompletionMessage(
                    role="assistant",
                    content=f"\n[{artifact}]\n",
                )
                return ai_message
            if image_parts:
                # BAI-806 fix: preserve any accompanying text (e.g. the MCP
                # response_format="content_and_artifact" shape where the
                # model's summary text and the image are both in
                # message.content, and artifact is always None) instead of
                # dropping it -- mirrors build_openai_message_content's
                # text-then-image ordering used for the AIMessage case.
                parts = []
                if content_text:
                    parts.append({"type": "text", "text": content_text})
                parts.extend(image_parts)
                return ChatCompletionMessage.model_construct(
                    role="assistant",
                    content=parts,  # type: ignore[arg-type]
                )
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
