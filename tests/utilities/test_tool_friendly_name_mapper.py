from pathlib import Path

from languagemodelcommon.utilities.tool_friendly_name_mapper import (
    ToolFriendlyNameMapper,
)


def test_mapper_prefers_friendly_name(tmp_path: Path) -> None:
    config_path = tmp_path / "tool_friendly_names.json"
    config_path.write_text('{"fhir_server": "FHIR Server"}', encoding="utf-8")

    mapper = ToolFriendlyNameMapper.from_config_path(config_path=str(config_path))

    assert mapper.get_display_name(tool_name="fhir_server") == "FHIR Server"


def test_mapper_builds_skill_start_text() -> None:
    mapper = ToolFriendlyNameMapper()

    text = mapper.get_message_for_tool(
        tool_name="load_skill", tool_input={"skill_name": "medication_review"}
    )

    assert "Using skill" in text
    assert "Medication Review" in text
