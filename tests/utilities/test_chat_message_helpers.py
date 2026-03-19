"""Tests for chat message helper utilities."""

import pytest
from typing import Any, Dict, List, Union

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import ChatMessage as LangchainChatMessage
from languagemodelcommon.utilities.chat_message_helpers import (
    convert_message_content_to_string,
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

    def test_empty_string_content(self) -> None:
        """Test empty string content."""
        content = ""
        result = convert_message_content_to_string(content)
        assert result == ""

    def test_list_with_single_text_dict(self) -> None:
        """Test list with single text dictionary."""
        content: List[Union[str, Dict[str, Any]]] = [{"type": "text", "text": "Hello"}]
        result = convert_message_content_to_string(content)
        assert result == "Hello"

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

    def test_list_with_only_non_text_dicts(self) -> None:
        """Test list with only non-text dictionaries returns empty string."""
        content: List[Union[str, Dict[str, Any]]] = [
            {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
        ]
        result = convert_message_content_to_string(content)
        assert result == ""

    def test_empty_list_content(self) -> None:
        """Test empty list returns empty string."""
        content: List[Union[str, Dict[str, Any]]] = []
        result = convert_message_content_to_string(content)
        assert result == ""

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
        # Use content variable to avoid unused variable warning
        with pytest.raises(TypeError):
            invalid_content: list[Any] = [
                "text",
                {"type": "text", "text": "more text"},
                123,
            ]
            # Directly pass the invalid content
            convert_message_content_to_string(invalid_content)


class TestRemoveToolCalls:
    """Tests for remove_tool_calls function."""

    def test_string_content_returned_as_is(self) -> None:
        """Test that string content is returned unchanged."""
        content = "Hello, world!"
        result = remove_tool_calls(content)
        assert result == content

    def test_empty_string_content(self) -> None:
        """Test empty string content."""
        content = ""
        result = remove_tool_calls(content)
        assert result == ""

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

    def test_list_with_multiple_tool_use_removed(self) -> None:
        """Test that multiple tool_use items are removed."""
        content: List[Union[str, Dict[str, Any]]] = [
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "id": "123", "name": "get_weather"},
            {"type": "text", "text": " world"},
            {"type": "tool_use", "id": "456", "name": "search"},
        ]
        result = remove_tool_calls(content)
        expected: List[Union[str, Dict[str, Any]]] = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " world"},
        ]
        assert result == expected

    def test_list_with_only_tool_use(self) -> None:
        """Test list with only tool_use items returns empty list."""
        content: List[Union[str, Dict[str, Any]]] = [
            {"type": "tool_use", "id": "123", "name": "get_weather"},
            {"type": "tool_use", "id": "456", "name": "search"},
        ]
        result = remove_tool_calls(content)
        assert result == []

    def test_empty_list_content(self) -> None:
        """Test empty list returns empty list."""
        content: List[Union[str, Dict[str, Any]]] = []
        result = remove_tool_calls(content)
        assert result == []

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

    def test_system_message_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            langchain_to_chat_message(SystemMessage(content="sys"))

    def test_human_message_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            langchain_to_chat_message(HumanMessage(content="user"))

    def test_chat_message_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            langchain_to_chat_message(
                LangchainChatMessage(role="assistant", content="hi")
            )
