"""Tests for chat message helper utilities."""

import pytest
from typing import Any, Dict, List, Union

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from languagemodelcommon.utilities.chat_message_helpers import (
    build_openai_message_content,
    convert_message_content_to_string,
    extract_image_output_parts,
    iter_message_content_text_chunks,
    remove_tool_calls,
    langchain_to_chat_message,
)
from openai.types.chat import ChatCompletionMessage


class TestConvertMessageContentToString:
    """Tests for convert_message_content_to_string function."""

    def test_string_content_returned_as_is(self) -> None:
        """Test that string content is returned unchanged."""
        content = "Hello, world!"
        result = convert_message_content_to_string(content)
        assert result == content

    def test_list_with_multiple_text_dicts(self) -> None:
        """Test list with multiple text dictionaries."""
        content: List[Union[str, Dict[str, Any]]] = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " world"},
        ]
        result = convert_message_content_to_string(content)
        assert result == "Hello world"

    def test_list_with_mixed_string_and_dict(self) -> None:
        """Test list with mixed string and dictionary content."""
        content: List[Union[str, Dict[str, Any]]] = [
            "Hello",
            {"type": "text", "text": " world"},
        ]
        result = convert_message_content_to_string(content)
        assert result == "Hello world"

    def test_list_with_non_text_dict_ignored(self) -> None:
        """Test that non-text dictionaries are ignored."""
        content: List[Union[str, Dict[str, Any]]] = [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
            {"type": "text", "text": " world"},
        ]
        result = convert_message_content_to_string(content)
        assert result == "Hello world"

    def test_list_with_text_dict_missing_text_field(self) -> None:
        """Test text dict without 'text' field is skipped."""
        content: List[Union[str, Dict[str, Any]]] = [
            {"type": "text", "text": "Hello"},
            {"type": "text"},  # Missing 'text' field
            {"type": "text", "text": " world"},
        ]
        result = convert_message_content_to_string(content)
        assert result == "Hello world"

    def test_unsupported_content_type_raises_error(self) -> None:
        """Test that unsupported content types raise TypeError."""
        with pytest.raises(TypeError):
            invalid_content: list[Any] = [
                "text",
                {"type": "text", "text": "more text"},
                123,
            ]
            convert_message_content_to_string(invalid_content)


class TestRemoveToolCalls:
    """Tests for remove_tool_calls function."""

    def test_string_content_returned_as_is(self) -> None:
        """Test that string content is returned unchanged."""
        content = "Hello, world!"
        result = remove_tool_calls(content)
        assert result == content

    def test_list_without_tool_use_unchanged(self) -> None:
        """Test list without tool_use items is unchanged."""
        content: List[Union[str, Dict[str, Any]]] = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " world"},
        ]
        result = remove_tool_calls(content)
        assert result == content

    def test_list_with_single_tool_use_removed(self) -> None:
        """Test that single tool_use item is removed."""
        content: List[Union[str, Dict[str, Any]]] = [
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "id": "123", "name": "get_weather"},
            {"type": "text", "text": " world"},
        ]
        result = remove_tool_calls(content)
        expected: List[Union[str, Dict[str, Any]]] = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " world"},
        ]
        assert result == expected

    def test_list_with_mixed_types_preserves_non_tool_use(self) -> None:
        """Test that non-tool_use types are preserved."""
        content: List[Union[str, Dict[str, Any]]] = [
            "Hello",
            {"type": "tool_use", "id": "123", "name": "get_weather"},
            {"type": "text", "text": " world"},
            {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
        ]
        result = remove_tool_calls(content)
        expected: List[Union[str, Dict[str, Any]]] = [
            "Hello",
            {"type": "text", "text": " world"},
            {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
        ]
        assert result == expected

    def test_dict_without_type_field_preserved(self) -> None:
        """Test that dictionaries without 'type' field are preserved."""
        content: List[Union[str, Dict[str, Any]]] = [
            {"type": "text", "text": "Hello"},
            {"name": "some_data", "value": 123},  # No 'type' field
            {"type": "tool_use", "id": "123", "name": "get_weather"},
        ]
        result = remove_tool_calls(content)
        expected: List[Union[str, Dict[str, Any]]] = [
            {"type": "text", "text": "Hello"},
            {"name": "some_data", "value": 123},
        ]
        assert result == expected


class TestLangchainToChatMessage:
    """Tests for langchain_to_chat_message function."""

    def test_ai_message_converts_to_chat_completion_message(self) -> None:
        message = AIMessage(content="hello")
        result = langchain_to_chat_message(message)
        assert isinstance(result, ChatCompletionMessage)
        assert result.role == "assistant"
        assert result.content == "hello"

    def test_tool_message_with_artifact_wraps_output(self) -> None:
        message = ToolMessage(
            content="ignored", tool_call_id="tool_1", artifact="trace"
        )
        result = langchain_to_chat_message(message)
        assert isinstance(result, ChatCompletionMessage)
        assert result.content == "\n[trace]\n"

    def test_tool_message_without_artifact_returns_none(self) -> None:
        message = ToolMessage(content="ignored", tool_call_id="tool_1", artifact="")
        assert langchain_to_chat_message(message) is None

    def test_ai_message_with_image_content_returns_content_part_list(self) -> None:
        message = AIMessage(
            content=[
                {"type": "text", "text": "here you go"},
                {
                    "type": "image",
                    "url": "https://example.com/chart.png",
                    "mime_type": "image/png",
                },
            ]
        )
        result = langchain_to_chat_message(message)
        assert isinstance(result, ChatCompletionMessage)
        assert isinstance(result.content, list)
        assert result.content[0] == {"type": "text", "text": "here you go"}
        assert result.content[1] == {
            "type": "output_image",
            "image_url": "https://example.com/chart.png",
            "mime_type": "image/png",
        }

    def test_tool_message_with_artifact_and_image_content_includes_both(
        self,
    ) -> None:
        message = ToolMessage(
            content=[
                {
                    "type": "image",
                    "url": "https://example.com/chart.png",
                    "mime_type": "image/png",
                }
            ],
            tool_call_id="tool_1",
            artifact="trace",
        )
        result = langchain_to_chat_message(message)
        assert isinstance(result, ChatCompletionMessage)
        assert isinstance(result.content, list)
        assert result.content[0] == {"type": "text", "text": "\n[trace]\n"}
        assert result.content[1]["type"] == "output_image"

    def test_tool_message_with_image_content_and_no_artifact_returns_image(
        self,
    ) -> None:
        """Regression for BAI-806: the MCP tool-calling path
        (response_format='content_and_artifact') always returns
        artifact=None, so message.content -- not artifact -- is where the
        image actually lives; this must not be silently dropped."""
        message = ToolMessage(
            content=[
                {
                    "type": "image",
                    "base64": "Zm9v",
                    "mime_type": "image/png",
                }
            ],
            tool_call_id="tool_1",
            artifact=None,
        )
        result = langchain_to_chat_message(message)
        assert isinstance(result, ChatCompletionMessage)
        assert isinstance(result.content, list)
        assert result.content[0] == {
            "type": "output_image",
            "image_url": "data:image/png;base64,Zm9v",
            "mime_type": "image/png",
        }

    def test_system_message_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            langchain_to_chat_message(SystemMessage(content="sys"))

    def test_unsupported_message_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            langchain_to_chat_message(HumanMessage(content="user"))


class TestIterMessageContentTextChunksImage:
    """Tests for the LangChain `ImageContentBlock` (`type: "image"`) branch."""

    def test_image_block_yields_no_text_chunk_by_default(self) -> None:
        result = iter_message_content_text_chunks(
            content=[
                {"type": "text", "text": "hello"},
                {"type": "image", "url": "https://example.com/a.png"},
            ],
            include_non_text_placeholders=False,
        )
        assert result.text_chunks == ["hello"]
        assert result.non_text_blocks == [
            {"type": "image", "url": "https://example.com/a.png"}
        ]

    def test_image_block_yields_placeholder_when_enabled(self) -> None:
        result = iter_message_content_text_chunks(
            content=[{"type": "image", "url": "https://example.com/a.png"}],
            include_non_text_placeholders=True,
        )
        assert result.text_chunks == ["[image]"]


class TestExtractImageOutputParts:
    """Tests for extract_image_output_parts."""

    def test_langchain_image_block_with_url(self) -> None:
        parts = extract_image_output_parts(
            [
                {
                    "type": "image",
                    "url": "https://example.com/a.png",
                    "mime_type": "image/png",
                }
            ]
        )
        assert parts == [
            {
                "type": "output_image",
                "image_url": "https://example.com/a.png",
                "mime_type": "image/png",
            }
        ]

    def test_langchain_image_block_with_base64_builds_data_uri(self) -> None:
        parts = extract_image_output_parts(
            [{"type": "image", "base64": "Zm9v", "mime_type": "image/jpeg"}]
        )
        assert parts == [
            {
                "type": "output_image",
                "image_url": "data:image/jpeg;base64,Zm9v",
                "mime_type": "image/jpeg",
            }
        ]

    def test_langchain_image_block_with_neither_url_nor_base64_skipped(self) -> None:
        parts = extract_image_output_parts([{"type": "image", "file_id": "file-123"}])
        assert parts == []

    @pytest.mark.parametrize("block_type", ["image_url", "input_image", "output_image"])
    def test_openai_style_dict_image_url(self, block_type: str) -> None:
        parts = extract_image_output_parts(
            [{"type": block_type, "image_url": {"url": "https://example.com/b.png"}}]
        )
        assert parts == [
            {
                "type": "output_image",
                "image_url": "https://example.com/b.png",
                "mime_type": None,
            }
        ]

    def test_openai_style_bare_string_image_url(self) -> None:
        parts = extract_image_output_parts(
            [{"type": "input_image", "image_url": "https://example.com/c.png"}]
        )
        assert parts == [
            {
                "type": "output_image",
                "image_url": "https://example.com/c.png",
                "mime_type": None,
            }
        ]

    def test_non_image_blocks_ignored(self) -> None:
        parts = extract_image_output_parts([{"type": "reasoning", "reasoning": "x"}])
        assert parts == []


class TestBuildOpenAiMessageContent:
    """Tests for build_openai_message_content."""

    def test_string_content_returned_unchanged(self) -> None:
        assert build_openai_message_content("hello") == "hello"

    def test_text_only_list_returns_string(self) -> None:
        result = build_openai_message_content(
            [{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}]
        )
        assert result == "hello world"

    def test_text_and_image_returns_content_parts(self) -> None:
        result = build_openai_message_content(
            [
                {"type": "text", "text": "here"},
                {"type": "image", "url": "https://example.com/a.png"},
            ]
        )
        assert result == [
            {"type": "text", "text": "here"},
            {
                "type": "output_image",
                "image_url": "https://example.com/a.png",
                "mime_type": None,
            },
        ]

    def test_image_only_returns_content_parts_with_no_text_part(self) -> None:
        result = build_openai_message_content(
            [{"type": "image", "url": "https://example.com/a.png"}]
        )
        assert result == [
            {
                "type": "output_image",
                "image_url": "https://example.com/a.png",
                "mime_type": None,
            }
        ]
