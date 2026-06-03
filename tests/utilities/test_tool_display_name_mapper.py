from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

from langchain_core.tools import BaseTool

from languagemodelcommon.utilities.tool_display_name_mapper import (
    ToolDisplayNameMapper,
)


def test_empty_string_suppresses_display(tmp_path: Path) -> None:
    config_path = tmp_path / "names.json"
    config_path.write_text('{"hidden_tool": ""}', encoding="utf-8")

    mapper = ToolDisplayNameMapper.from_config_path(config_path=str(config_path))

    assert mapper.get_display_name(tool_name="hidden_tool") == ""
    assert (
        mapper.get_message_for_tool(tool_name="hidden_tool", tool_input={}) == ""
    )


def test_mapper_prefers_display_name(tmp_path: Path) -> None:
    config_path = tmp_path / "tool_display_names.json"
    config_path.write_text('{"fhir_server": "FHIR Server"}', encoding="utf-8")

    mapper = ToolDisplayNameMapper.from_config_path(config_path=str(config_path))

    assert mapper.get_display_name(tool_name="fhir_server") == "🛠️ FHIR Server"


def _make_tool_stub(name: str, metadata: Dict[str, Any] | None = None) -> BaseTool:
    stub = MagicMock(spec=BaseTool)
    stub.name = name
    stub.metadata = metadata
    return stub


import pytest


class TestParameterSubstitution:
    @pytest.mark.parametrize(
        "template,params,expected",
        [
            (
                "🧠 Using skill {skill_name}",
                {"skill_name": "weather"},
                "🧠 Using skill weather",
            ),
            (
                "📋 Checking {resource} for {patient_id}",
                {"resource": "vitals", "patient_id": "P123"},
                "📋 Checking vitals for P123",
            ),
            (
                "🔍 Searching {query}",
                {"query": "lab results", "unused_param": "ignored"},
                "🔍 Searching lab results",
            ),
            (
                "🛠️ Static display name",
                {"anything": "value"},
                "🛠️ Static display name",
            ),
            (
                "🧠 Has param {param}",
                {"param": "hello"},
                "🧠 Has param hello",
            ),
        ],
    )
    def test_substitutes_params_in_display_name(
        self, template: str, params: Dict[str, Any], expected: str
    ) -> None:
        mapper = ToolDisplayNameMapper(
            name_to_display_name={"my_tool": template}
        )
        result = mapper.get_display_name(tool_name="my_tool", tool_input=params)
        assert result == expected

    def test_no_substitution_without_tool_input(self) -> None:
        mapper = ToolDisplayNameMapper(
            name_to_display_name={"my_tool": "🧠 Using skill {skill_name}"}
        )
        result = mapper.get_display_name(tool_name="my_tool")
        assert result == "🧠 Using skill {skill_name}"

    def test_get_name_for_tool_passes_inputs(self) -> None:
        mapper = ToolDisplayNameMapper(
            name_to_display_name={"load_skill": "🧠 Using skill {skill_name}"}
        )
        result = mapper.get_name_for_tool(
            tool_name="load_skill", tool_input={"skill_name": "scheduling"}
        )
        assert result == "🧠 Using skill scheduling"

    def test_missing_param_resolves_to_none(self) -> None:
        mapper = ToolDisplayNameMapper(
            name_to_display_name={"my_tool": "🧠 Skill {name} with {missing}"}
        )
        result = mapper.get_display_name(
            tool_name="my_tool", tool_input={"name": "test"}
        )
        assert result == "🧠 Skill test with None"

    def test_call_tool_substitutes_from_arguments(self) -> None:
        mapper = ToolDisplayNameMapper(
            name_to_display_name={
                "provider_search": "🏥 Searching for {specialty} providers"
            }
        )
        result = mapper.get_name_for_tool(
            tool_name="call_tool",
            tool_input={
                "name": "provider_search",
                "arguments": {"specialty": "cardiology", "zip": "90210"},
            },
        )
        assert result == "🏥 Searching for cardiology providers"


class TestRegisterFromTools:
    def test_registers_mcp_title(self) -> None:
        mapper = ToolDisplayNameMapper()
        tool = _make_tool_stub("get_weather", {"mcp_title": "Weather Info"})

        mapper.register_from_tools([tool])

        assert mapper.get_display_name(tool_name="get_weather") == "🛠️ Weather Info"

    def test_static_config_takes_precedence(self, tmp_path: Path) -> None:
        config_path = tmp_path / "names.json"
        config_path.write_text('{"get_weather": "Custom Weather"}', encoding="utf-8")
        mapper = ToolDisplayNameMapper.from_config_path(config_path=str(config_path))

        tool = _make_tool_stub("get_weather", {"mcp_title": "MCP Weather Title"})
        mapper.register_from_tools([tool])

        assert mapper.get_display_name(tool_name="get_weather") == "🛠️ Custom Weather"

    def test_skips_tools_without_metadata(self) -> None:
        mapper = ToolDisplayNameMapper()
        tool = _make_tool_stub("get_weather", None)

        mapper.register_from_tools([tool])

        # Falls back to humanized name
        assert "Get Weather" in mapper.get_display_name(tool_name="get_weather")

    def test_skips_tools_without_mcp_title(self) -> None:
        mapper = ToolDisplayNameMapper()
        tool = _make_tool_stub("get_weather", {"mcp_description": "desc"})

        mapper.register_from_tools([tool])

        assert "Get Weather" in mapper.get_display_name(tool_name="get_weather")

    def test_registers_multiple_tools(self) -> None:
        mapper = ToolDisplayNameMapper()
        tools = [
            _make_tool_stub("tool_a", {"mcp_title": "Tool Alpha"}),
            _make_tool_stub("tool_b", {"mcp_title": "Tool Beta"}),
            _make_tool_stub("tool_c", None),
        ]

        mapper.register_from_tools(tools)

        assert mapper.get_display_name(tool_name="tool_a") == "🛠️ Tool Alpha"
        assert mapper.get_display_name(tool_name="tool_b") == "🛠️ Tool Beta"
        assert "Tool C" in mapper.get_display_name(tool_name="tool_c")
