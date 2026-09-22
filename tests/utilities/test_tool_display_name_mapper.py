import pytest

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
    assert mapper.get_message_for_tool(tool_name="hidden_tool", tool_input={}) == ""


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
        mapper = ToolDisplayNameMapper(name_to_display_name={"my_tool": template})
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

    def test_missing_param_leaves_placeholder(self) -> None:
        mapper = ToolDisplayNameMapper(
            name_to_display_name={"my_tool": "🧠 Skill {name} with {missing}"}
        )
        result = mapper.get_display_name(
            tool_name="my_tool", tool_input={"name": "test"}
        )
        assert result == "🧠 Skill test with {missing}"

    def test_missing_param_with_default(self) -> None:
        mapper = ToolDisplayNameMapper(
            name_to_display_name={"my_tool": "🧠 Skill {name} with {missing|fallback}"}
        )
        result = mapper.get_display_name(
            tool_name="my_tool", tool_input={"name": "test"}
        )
        assert result == "🧠 Skill test with fallback"

    def test_list_multiple_elements_wrapped_in_parens(self) -> None:
        mapper = ToolDisplayNameMapper(
            name_to_display_name={"my_tool": "🩺 Checking your {vitals|vitals}"}
        )
        result = mapper.get_display_name(
            tool_name="my_tool",
            tool_input={"vitals": ["blood_pressure", "heart_rate"]},
        )
        assert result == "🩺 Checking your (blood_pressure, heart_rate)"

    def test_list_single_element_no_parens(self) -> None:
        mapper = ToolDisplayNameMapper(
            name_to_display_name={"my_tool": "🩺 Checking your {vitals|vitals}"}
        )
        result = mapper.get_display_name(
            tool_name="my_tool",
            tool_input={"vitals": ["blood_pressure"]},
        )
        assert result == "🩺 Checking your blood_pressure"

    def test_empty_list_uses_default(self) -> None:
        mapper = ToolDisplayNameMapper(
            name_to_display_name={"my_tool": "🩺 Checking your {vitals|vitals}"}
        )
        result = mapper.get_display_name(tool_name="my_tool", tool_input={"vitals": []})
        assert result == "🩺 Checking your vitals"

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


class TestWithTools:
    """with_tools() must merge live tool titles without mutating self.

    Unlike register_from_tools(), this is the safe entry point for a
    process-wide singleton mapper: callers holding one shared instance
    across concurrent requests must not have one request's ad-hoc/
    request-scoped tool titles leak into another request or accumulate
    unboundedly in the singleton's dict over the process lifetime.
    """

    def test_returns_new_instance_with_merged_title(self) -> None:
        mapper = ToolDisplayNameMapper()
        tool = _make_tool_stub("get_weather", {"mcp_title": "Weather Info"})

        merged = mapper.with_tools(tools=[tool])

        assert merged is not mapper
        assert merged.get_display_name(tool_name="get_weather") == "🛠️ Weather Info"

    def test_does_not_mutate_singleton(self) -> None:
        singleton = ToolDisplayNameMapper()
        tool = _make_tool_stub("get_weather", {"mcp_title": "Weather Info"})

        singleton.with_tools(tools=[tool])

        # The singleton itself must be untouched -- a second, unrelated
        # request sharing this instance must not see the first request's
        # ad-hoc tool title.
        assert "Get Weather" in singleton.get_display_name(tool_name="get_weather")

    def test_preserves_static_config_precedence(self, tmp_path: Path) -> None:
        config_path = tmp_path / "names.json"
        config_path.write_text('{"get_weather": "Custom Weather"}', encoding="utf-8")
        static_mapper = ToolDisplayNameMapper.from_config_path(
            config_path=str(config_path)
        )
        tool = _make_tool_stub("get_weather", {"mcp_title": "MCP Weather Title"})

        merged = static_mapper.with_tools(tools=[tool])

        assert merged.get_display_name(tool_name="get_weather") == "🛠️ Custom Weather"

    def test_concurrent_requests_do_not_see_each_others_titles(self) -> None:
        singleton = ToolDisplayNameMapper()
        request_a_tool = _make_tool_stub(
            "custom_search", {"mcp_title": "Client A Search"}
        )
        request_b_tool = _make_tool_stub(
            "custom_search", {"mcp_title": "Client B Search"}
        )

        mapper_a = singleton.with_tools(tools=[request_a_tool])
        mapper_b = singleton.with_tools(tools=[request_b_tool])

        assert (
            mapper_a.get_display_name(tool_name="custom_search") == "🛠️ Client A Search"
        )
        assert (
            mapper_b.get_display_name(tool_name="custom_search") == "🛠️ Client B Search"
        )


class TestRegisterTitle:
    """register_title() lets callers learn a tool's title at runtime (BAI-903),
    for tools dispatched via call_tool that are never bound as tool objects.
    """

    def test_registers_runtime_title(self) -> None:
        mapper = ToolDisplayNameMapper()

        mapper.register_title(tool_name="start_onboarding", title="🚀 Start Onboarding")

        assert (
            mapper.get_display_name(tool_name="start_onboarding")
            == "🚀 Start Onboarding"
        )

    def test_static_config_takes_precedence_over_runtime_title(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "names.json"
        config_path.write_text(
            '{"start_onboarding": "Custom Onboarding"}', encoding="utf-8"
        )
        mapper = ToolDisplayNameMapper.from_config_path(config_path=str(config_path))

        mapper.register_title(tool_name="start_onboarding", title="🚀 Start Onboarding")

        assert (
            mapper.get_display_name(tool_name="start_onboarding")
            == "🛠️ Custom Onboarding"
        )

    def test_first_registered_runtime_title_wins(self) -> None:
        mapper = ToolDisplayNameMapper()

        mapper.register_title(tool_name="start_onboarding", title="🚀 Start Onboarding")
        mapper.register_title(tool_name="start_onboarding", title="🔁 Different Title")

        assert (
            mapper.get_display_name(tool_name="start_onboarding")
            == "🚀 Start Onboarding"
        )

    def test_ignores_blank_title(self) -> None:
        mapper = ToolDisplayNameMapper()

        mapper.register_title(tool_name="start_onboarding", title="   ")

        assert "Start Onboarding" in mapper.get_display_name(
            tool_name="start_onboarding"
        )


class TestGetMessageForToolFormat:
    """Format invariants for streamed tool-progress messages.

    A single `\\n` is a Markdown soft break and collapses to a space in
    chat UIs, so the wrapper must use `\\n\\n` on both sides to render
    the tool-progress message on its own paragraph.
    """

    def test_emoji_message_uses_paragraph_break_wrap(self) -> None:
        mapper = ToolDisplayNameMapper(
            name_to_display_name={"load_skill": "🧠 Loading skill: {skill_name}"}
        )
        message = mapper.get_message_for_tool(
            tool_name="load_skill", tool_input={"skill_name": "scheduling"}
        )
        assert message == "\n\n🧠 Loading skill: scheduling.\n\n"

    def test_non_emoji_message_uses_paragraph_break_wrap(self) -> None:
        mapper = ToolDisplayNameMapper(
            name_to_display_name={"plain_tool": "Plain message"}
        )
        message = mapper.get_message_for_tool(tool_name="plain_tool", tool_input={})
        assert message == "\n\n🛠️ Plain message.\n\n"

    def test_empty_template_returns_empty_string(self) -> None:
        mapper = ToolDisplayNameMapper(name_to_display_name={"silent_tool": ""})
        assert mapper.get_message_for_tool(tool_name="silent_tool", tool_input={}) == ""

    def test_missing_tool_name_returns_empty_string(self) -> None:
        mapper = ToolDisplayNameMapper()
        assert mapper.get_message_for_tool(tool_name=None, tool_input={}) == ""
