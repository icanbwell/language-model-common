import pytest

from languagemodelcommon.utilities.tool_display_name_mapper import (
    ToolDisplayNameMapper,
    is_storable_title,
)


def _call_tool_line(
    *, mapper: ToolDisplayNameMapper, name: str, arguments: dict[str, object]
) -> str:
    return mapper.get_name_for_tool(
        tool_name="call_tool", tool_input={"name": name, "arguments": arguments}
    )


class TestTitlesAreRequestScoped:
    """No title is shared between mapper instances: not through the singleton,
    not through a process-wide cache (BAI-1732, ADR 0011)."""

    def test_two_mappers_never_share_learned_titles(self) -> None:
        singleton = ToolDisplayNameMapper()
        request_a = singleton.with_tools(tools=[])
        request_a.register_title(tool_name="get_vitals", title="🩺 Vitals")

        request_b = singleton.with_tools(tools=[])

        assert (
            _call_tool_line(mapper=request_b, name="get_vitals", arguments={})
            == "Get Vitals"
        )

    def test_independently_constructed_mappers_never_share_titles(self) -> None:
        ToolDisplayNameMapper().register_title(
            tool_name="get_vitals", title="🩺 Vitals"
        )

        assert (
            _call_tool_line(
                mapper=ToolDisplayNameMapper(), name="get_vitals", arguments={}
            )
            == "Get Vitals"
        )

    def test_title_learned_in_a_request_renders_with_placeholders(self) -> None:
        request = ToolDisplayNameMapper().with_tools(tools=[])
        request.register_title(
            tool_name="get_vitals", title="🩺 Checking your vitals: {vitals|all}"
        )

        assert (
            _call_tool_line(
                mapper=request,
                name="get_vitals",
                arguments={"vitals": ["bmi", "weight"]},
            )
            == "🩺 Checking your vitals: (bmi, weight)"
        )

    def test_singleton_config_dict_is_not_mutated(self) -> None:
        singleton = ToolDisplayNameMapper()
        request = singleton.with_tools(tools=[])
        request.register_title(tool_name="get_vitals", title="🩺 Vitals")
        request.note_store_lookup_miss(tool_name="other_tool")

        assert singleton._name_to_display_name == {}
        assert not singleton.has_missed_store_lookup(tool_name="other_tool")

    def test_static_config_still_wins_over_learned_title(self) -> None:
        singleton = ToolDisplayNameMapper(
            name_to_display_name={"get_vitals": "📋 From config"}
        )
        request = singleton.with_tools(tools=[])
        request.register_title(tool_name="get_vitals", title="🩺 From title")

        assert (
            _call_tool_line(mapper=request, name="get_vitals", arguments={})
            == "📋 From config"
        )

    def test_learned_title_does_not_relabel_directly_bound_tool(self) -> None:
        singleton = ToolDisplayNameMapper()
        singleton.with_tools(tools=[]).register_title(
            tool_name="search", title="🔎 Catalog search"
        )

        assert (
            singleton.with_tools(tools=[]).get_display_name(tool_name="search")
            == "🛠️ Search"
        )

    def test_store_lookup_miss_is_remembered_per_mapper_only(self) -> None:
        first = ToolDisplayNameMapper()
        first.note_store_lookup_miss(tool_name="untitled")

        assert first.has_missed_store_lookup(tool_name="untitled")
        assert not first.with_tools(tools=[]).has_missed_store_lookup(
            tool_name="untitled"
        )


class TestIsStorableTitle:
    @pytest.mark.parametrize(
        ("tool_name", "title", "expected"),
        [
            ("get_vitals", "🩺 Vitals", True),
            ("get_vitals", "x" * 256, True),
            ("get_vitals", "x" * 257, False),
            ("n" * 128, "T", True),
            ("n" * 129, "T", False),
        ],
    )
    def test_length_caps(self, tool_name: str, title: str, expected: bool) -> None:
        assert is_storable_title(tool_name=tool_name, title=title) is expected
