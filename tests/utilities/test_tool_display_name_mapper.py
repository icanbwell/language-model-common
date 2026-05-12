from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

from langchain_core.tools import BaseTool

from languagemodelcommon.utilities.tool_display_name_mapper import (
    ToolDisplayNameMapper,
)


def test_mapper_prefers_display_name(tmp_path: Path) -> None:
    config_path = tmp_path / "tool_display_names.json"
    config_path.write_text('{"fhir_server": "FHIR Server"}', encoding="utf-8")

    mapper = ToolDisplayNameMapper.from_config_path(config_path=str(config_path))

    assert mapper.get_display_name(tool_name="fhir_server") == "FHIR Server"


def test_mapper_builds_skill_start_text() -> None:
    mapper = ToolDisplayNameMapper()

    text = mapper.get_message_for_tool(
        tool_name="load_skill", tool_input={"skill_name": "medication_review"}
    )

    assert "Using skill" in text
    assert "Medication Review" in text


def _make_tool_stub(name: str, metadata: Dict[str, Any] | None = None) -> BaseTool:
    stub = MagicMock(spec=BaseTool)
    stub.name = name
    stub.metadata = metadata
    return stub


class TestRegisterFromTools:
    def test_registers_mcp_title(self) -> None:
        mapper = ToolDisplayNameMapper()
        tool = _make_tool_stub("get_weather", {"mcp_title": "Weather Info"})

        mapper.register_from_tools([tool])

        assert mapper.get_display_name(tool_name="get_weather") == "Weather Info"

    def test_static_config_takes_precedence(self, tmp_path: Path) -> None:
        config_path = tmp_path / "names.json"
        config_path.write_text('{"get_weather": "Custom Weather"}', encoding="utf-8")
        mapper = ToolDisplayNameMapper.from_config_path(config_path=str(config_path))

        tool = _make_tool_stub("get_weather", {"mcp_title": "MCP Weather Title"})
        mapper.register_from_tools([tool])

        assert mapper.get_display_name(tool_name="get_weather") == "Custom Weather"

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

        assert mapper.get_display_name(tool_name="tool_a") == "Tool Alpha"
        assert mapper.get_display_name(tool_name="tool_b") == "Tool Beta"
        assert "Tool C" in mapper.get_display_name(tool_name="tool_c")
