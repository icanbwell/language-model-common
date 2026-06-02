import pytest

from languagemodelcommon.utilities.text_humanizer import Humanizer


class TestHumanizer:
    @pytest.mark.parametrize(
        "input_key,expected",
        [
            ("get_weather", "Get Weather"),
            ("search-documents", "Search Documents"),
            ("get_user_id", "Get User ID"),
            ("fetch_fhir_resource", "Fetch FHIR Resource"),
            ("list_mcp_servers", "List MCP Servers"),
            ("validate_jwt_token", "Validate JWT Token"),
            ("check_oidc_config", "Check OIDC Config"),
            ("get_url_by_ids", "Get URL By IDS"),
            ("get_uri_path", "Get URI Path"),
            ("", ""),
            ("single", "Single"),
            ("already_UPPER", "Already Upper"),
            ("___multiple___underscores___", "Multiple Underscores"),
            ("mixed-and_separators", "Mixed And Separators"),
        ],
    )
    def test_humanize_tool_name(self, *, input_key: str, expected: str) -> None:
        result = Humanizer.humanize_tool_name(key=input_key)
        assert result == expected
