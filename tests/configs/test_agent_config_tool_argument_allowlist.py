"""Tests for AgentConfig.tool_argument_allowlist (BAI-801)."""

from languagemodelcommon.configs.schemas.config_schema import AgentConfig


class TestAgentConfigToolArgumentAllowlist:
    def test_defaults_to_none(self) -> None:
        config = AgentConfig(name="rxutility", url="https://example.com/mcp")
        assert config.tool_argument_allowlist is None

    def test_accepts_map_of_tool_name_to_allowed_args(self) -> None:
        config = AgentConfig(
            name="rxutility",
            url="https://example.com/mcp",
            tool_argument_allowlist={"get_prescription": ["patient_id", "ndc_code"]},
        )
        assert config.tool_argument_allowlist == {
            "get_prescription": ["patient_id", "ndc_code"]
        }

    def test_empty_map_is_distinct_from_none(self) -> None:
        config = AgentConfig(
            name="rxutility", url="https://example.com/mcp", tool_argument_allowlist={}
        )
        assert config.tool_argument_allowlist == {}
        assert config.tool_argument_allowlist is not None
