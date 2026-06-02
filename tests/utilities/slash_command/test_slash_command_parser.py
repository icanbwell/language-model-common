from languagemodelcommon.utilities.slash_command.slash_command_parser import (
    parse_slash_command,
)


class TestParseSlashCommand:
    def test_simple_command_with_message(self) -> None:
        result = parse_slash_command(content="/debug hello world")
        assert result is not None
        assert result.command_name == "debug"
        assert result.remaining_message == "hello world"

    def test_command_with_hyphenated_name(self) -> None:
        result = parse_slash_command(
            content="/uspstf-depression-screening what should I check?"
        )
        assert result is not None
        assert result.command_name == "uspstf-depression-screening"
        assert result.remaining_message == "what should I check?"

    def test_command_only_no_message(self) -> None:
        result = parse_slash_command(content="/list-skills")
        assert result is not None
        assert result.command_name == "list-skills"
        assert result.remaining_message == ""

    def test_no_slash_prefix_returns_none(self) -> None:
        result = parse_slash_command(content="hello world")
        assert result is None

    def test_slash_in_middle_of_message_returns_none(self) -> None:
        result = parse_slash_command(content="use /debug mode")
        assert result is None

    def test_url_in_message_returns_none(self) -> None:
        result = parse_slash_command(content="check https://example.com/path")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = parse_slash_command(content="")
        assert result is None

    def test_just_slash_returns_none(self) -> None:
        result = parse_slash_command(content="/")
        assert result is None

    def test_slash_with_spaces_before_command_returns_none(self) -> None:
        result = parse_slash_command(content="  /debug hello")
        assert result is None

    def test_multiple_spaces_between_command_and_message(self) -> None:
        result = parse_slash_command(content="/debug   hello")
        assert result is not None
        assert result.command_name == "debug"
        assert result.remaining_message == "hello"

    def test_colon_style_prefix_returns_none(self) -> None:
        result = parse_slash_command(content="DEBUG: hello")
        assert result is None

    def test_command_with_underscores(self) -> None:
        result = parse_slash_command(content="/vaccine_eligibility check patient")
        assert result is not None
        assert result.command_name == "vaccine_eligibility"
        assert result.remaining_message == "check patient"

    def test_command_with_numeric_suffix(self) -> None:
        result = parse_slash_command(content="/uspstf-6 check depression")
        assert result is not None
        assert result.command_name == "uspstf-6"
        assert result.remaining_message == "check depression"
