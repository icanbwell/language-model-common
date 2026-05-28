from datetime import datetime, timezone

from languagemodelcommon.converters.stream_debug_output_manager import (
    StreamDebugOutputManager,
)


class TestStreamDebugOutputManager:
    def test_append_and_pop_text(self) -> None:
        manager = StreamDebugOutputManager()
        manager.append_fragment(text="hello ")
        manager.append_fragment(text="world")
        result = manager.pop_text()
        assert result == "hello world"

    def test_pop_text_empty_returns_none(self) -> None:
        manager = StreamDebugOutputManager()
        assert manager.pop_text() is None

    def test_clear_removes_output(self) -> None:
        manager = StreamDebugOutputManager()
        manager.append_fragment(text="text")
        manager.clear()
        assert manager.pop_text() is None

    def test_append_empty_string_is_noop(self) -> None:
        manager = StreamDebugOutputManager()
        manager.append_fragment(text="")
        assert manager.pop_text() is None

    def test_start_streamed_output_sets_metadata(self) -> None:
        manager = StreamDebugOutputManager()
        start = datetime(2026, 5, 28, 6, 30, 0, tzinfo=timezone.utc)
        manager.start_streamed_output(
            event_name="agent_node",
            start_time=start,
        )
        manager.append_fragment(text="response")
        output = manager.pop_streamed_output()
        assert output is not None
        assert output.event_name == "agent_node"
        assert output.start_time == start
        assert "".join(output.text_fragments) == "response"

    def test_pop_streamed_output_clears_state(self) -> None:
        manager = StreamDebugOutputManager()
        manager.append_fragment(text="data")
        output = manager.pop_streamed_output()
        assert output is not None
        assert manager.pop_streamed_output() is None
