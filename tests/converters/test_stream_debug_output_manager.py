import time
from datetime import datetime, timezone

import pytest

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


class TestLlmCallTiming:
    """Per-run timing used by the llm_call stream items (BAI-1440)."""

    def test_pop_open_llm_call_runtimes_returns_and_forgets_open_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = iter([0.0, 10.0, 25.0, 25.0])
        monkeypatch.setattr(time, "perf_counter", lambda: next(clock))
        manager = StreamDebugOutputManager()
        manager.record_llm_call_start(run_id="run-a")
        manager.record_llm_call_start(run_id="run-b")

        open_runtimes = manager.pop_open_llm_call_runtimes()

        assert open_runtimes == {"run-a": 25.0, "run-b": 15.0}
        assert manager.pop_open_llm_call_runtimes() == {}

    def test_injected_clock_times_llm_calls(self) -> None:
        clock = iter([100.0, 107.5])
        manager = StreamDebugOutputManager(clock=lambda: next(clock))
        manager.record_llm_call_start(run_id="run-a")

        assert manager.pop_llm_call_runtime_seconds(run_id="run-a") == 7.5

    def test_clear_does_not_drop_in_flight_llm_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = iter([0.0, 4.0])
        monkeypatch.setattr(time, "perf_counter", lambda: next(clock))
        manager = StreamDebugOutputManager()
        manager.record_llm_call_start(run_id="run-a")

        manager.clear()

        assert manager.pop_llm_call_runtime_seconds(run_id="run-a") == 4.0

    def test_unknown_run_id_has_no_runtime(self) -> None:
        assert (
            StreamDebugOutputManager().pop_llm_call_runtime_seconds(run_id="missing")
            is None
        )
