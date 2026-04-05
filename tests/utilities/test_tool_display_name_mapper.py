from pathlib import Path

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
