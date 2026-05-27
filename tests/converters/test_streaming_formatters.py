from langchain_core.messages import ToolMessage

from languagemodelcommon.converters.streaming_formatters import (
    make_tool_key,
    safe_json,
    convert_message_content_into_string,
    get_structured_content_from_tool_message,
    format_message_content,
    format_tool_input_labels,
    extract_reasoning_text,
)


class TestMakeToolKey:
    def test_generates_key_from_name_and_input(self) -> None:
        key = make_tool_key("search", {"query": "hello"})
        assert key.startswith("search:")

    def test_none_name_defaults_to_unknown(self) -> None:
        key = make_tool_key(None, {"query": "hello"})
        assert key.startswith("unknown:")

    def test_same_inputs_produce_same_key(self) -> None:
        key1 = make_tool_key("tool", {"a": 1, "b": 2})
        key2 = make_tool_key("tool", {"b": 2, "a": 1})
        assert key1 == key2

    def test_different_inputs_produce_different_keys(self) -> None:
        key1 = make_tool_key("tool", {"a": 1})
        key2 = make_tool_key("tool", {"a": 2})
        assert key1 != key2


class TestSafeJson:
    def test_parses_valid_json(self) -> None:
        assert safe_json('{"key": "value"}') == {"key": "value"}

    def test_returns_none_for_invalid_json(self) -> None:
        assert safe_json("not json") is None

    def test_parses_json_array(self) -> None:
        assert safe_json("[1, 2, 3]") == [1, 2, 3]


class TestConvertMessageContentIntoString:
    def test_string_content_returned_directly(self) -> None:
        msg = ToolMessage(content="hello world", tool_call_id="tc1")
        result = convert_message_content_into_string(tool_message=msg)
        assert "hello world" in result

    def test_json_result_field_extracted(self) -> None:
        msg = ToolMessage(
            content=[{"type": "text", "text": '{"result": "extracted"}'}],
            tool_call_id="tc1",
        )
        result = convert_message_content_into_string(tool_message=msg)
        assert result == "extracted"

    def test_list_content_joined(self) -> None:
        msg = ToolMessage(content=["part1", "part2"], tool_call_id="tc1")
        result = convert_message_content_into_string(tool_message=msg)
        assert "part1" in result
        assert "part2" in result


class TestGetStructuredContentFromToolMessage:
    def test_single_element_list_returned(self) -> None:
        msg = ToolMessage(content=[{"key": "value"}], tool_call_id="tc1")
        result = get_structured_content_from_tool_message(tool_message=msg)
        assert result == {"key": "value"}

    def test_string_content_returns_none(self) -> None:
        msg = ToolMessage(content="just text", tool_call_id="tc1")
        result = get_structured_content_from_tool_message(tool_message=msg)
        assert result is None


class TestFormatMessageContent:
    def test_string_returned_as_is(self) -> None:
        assert format_message_content("hello") == "hello"

    def test_list_of_strings_joined(self) -> None:
        result = format_message_content(["line1", "line2"])
        assert result == "line1\nline2"

    def test_list_of_dicts_extracts_text(self) -> None:
        result = format_message_content([{"text": "hello"}, {"text": "world"}])
        assert result == "hello\nworld"


class TestFormatToolInputLabels:
    def test_formats_keys_as_labels(self) -> None:
        result = format_tool_input_labels(
            tool_input={"user_name": "test", "query": "x"}
        )
        assert "User Name" in result
        assert "Query" in result

    def test_hides_auth_token_state_runtime(self) -> None:
        result = format_tool_input_labels(
            tool_input={"auth_token": "x", "state": "y", "runtime": "z", "query": "q"}
        )
        assert "Auth Token" not in result
        assert "State" not in result
        assert "Runtime" not in result
        assert "Query" in result

    def test_empty_input_returns_none(self) -> None:
        assert format_tool_input_labels(tool_input=None) == "none"
        assert format_tool_input_labels(tool_input={}) == "none"


class TestExtractReasoningText:
    def test_extracts_reasoning_content_type(self) -> None:
        block = {
            "type": "reasoning_content",
            "reasoning_content": {"text": "thinking..."},
        }
        assert extract_reasoning_text(block) == "thinking..."

    def test_extracts_reasoning_type(self) -> None:
        block = {"type": "reasoning", "reasoning": "I think..."}
        assert extract_reasoning_text(block) == "I think..."

    def test_returns_none_for_unknown_type(self) -> None:
        block = {"type": "text", "text": "hello"}
        assert extract_reasoning_text(block) is None
