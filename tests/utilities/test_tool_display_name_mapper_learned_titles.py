from languagemodelcommon.utilities.tool_display_name_mapper import (
    LearnedTitleCache,
    ToolDisplayNameMapper,
)


def _call_tool_line(
    *, mapper: ToolDisplayNameMapper, name: str, arguments: dict[str, object]
) -> str:
    return mapper.get_name_for_tool(
        tool_name="call_tool", tool_input={"name": name, "arguments": arguments}
    )


class TestLearnedTitlesSharedAcrossRequests:
    """A title learned by one request's mapper must reach the *first*
    ``call_tool`` start line of a later request (no ``search_tools`` first),
    without the singleton's own config dict ever being mutated.
    """

    def test_later_request_renders_title_learned_by_earlier_request(self) -> None:
        singleton = ToolDisplayNameMapper()
        request_a = singleton.with_tools(tools=[])
        request_a.register_title(
            tool_name="get_vitals", title="🩺 Checking your vitals: {vitals|all}"
        )

        request_b = singleton.with_tools(tools=[])

        assert (
            _call_tool_line(
                mapper=request_b,
                name="get_vitals",
                arguments={"vitals": ["bmi", "weight"]},
            )
            == "🩺 Checking your vitals: (bmi, weight)"
        )

    def test_singleton_config_dict_is_not_mutated(self) -> None:
        singleton = ToolDisplayNameMapper()
        singleton.with_tools(tools=[]).register_title(
            tool_name="get_vitals", title="🩺 Vitals"
        )

        assert singleton._name_to_display_name == {}

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

    def test_unrelated_mapper_does_not_see_learned_titles(self) -> None:
        ToolDisplayNameMapper().with_tools(tools=[]).register_title(
            tool_name="get_vitals", title="🩺 Vitals"
        )

        assert (
            _call_tool_line(
                mapper=ToolDisplayNameMapper(), name="get_vitals", arguments={}
            )
            == "Get Vitals"
        )

    def test_cache_is_bounded_and_evicts_least_recently_used(self) -> None:
        cache = LearnedTitleCache(max_entries=2)
        cache.put(tool_name="a", title="A")
        cache.put(tool_name="b", title="B")
        assert cache.get(tool_name="a") == "A"
        cache.put(tool_name="c", title="C")

        assert cache.get(tool_name="b") is None
        assert cache.get(tool_name="a") == "A"
        assert cache.get(tool_name="c") == "C"

    def test_overlong_title_is_not_cached(self) -> None:
        cache = LearnedTitleCache()
        cache.put(tool_name="a", title="x" * 1000)

        assert cache.get(tool_name="a") is None

    def test_overlong_tool_name_is_not_cached(self) -> None:
        cache = LearnedTitleCache()
        cache.put(tool_name="n" * 1000, title="T")

        assert cache.get(tool_name="n" * 1000) is None

    def test_learned_title_does_not_relabel_directly_bound_tool(self) -> None:
        singleton = ToolDisplayNameMapper()
        singleton.with_tools(tools=[]).register_title(
            tool_name="search", title="🔎 Catalog search"
        )

        assert (
            singleton.with_tools(tools=[]).get_display_name(tool_name="search")
            == "🛠️ Search"
        )
