from dataclasses import dataclass, field
import time
from datetime import datetime, timezone


def elapsed_seconds_since(*, started: float) -> float:
    """Seconds since a ``time.perf_counter()`` reading, never negative."""
    return max(0.0, time.perf_counter() - started)


@dataclass
class StreamedOutput:
    event_name: str
    start_time: datetime
    text_fragments: list[str] = field(default_factory=list)


class StreamDebugOutputManager:
    def __init__(self) -> None:
        self._streamed_output: StreamedOutput | None = None
        # Start times of in-flight model invocations, keyed by run id, so
        # overlapping invocations time independently. Monotonic clock: a wall
        # clock step must not produce a negative duration. Deliberately not
        # reset by clear(), which fires on every nested chain end.
        self._llm_call_started: dict[str, float] = {}

    def record_llm_call_start(self, *, run_id: str) -> None:
        """Remember when the model invocation identified by ``run_id`` began."""
        self._llm_call_started[run_id] = time.perf_counter()

    def pop_llm_call_runtime_seconds(self, *, run_id: str) -> float | None:
        """Return the seconds since ``run_id`` started and forget it.

        Consumes the entry; returns ``None`` when no start was recorded.
        """
        started = self._llm_call_started.pop(run_id, None)
        if started is None:
            return None
        return elapsed_seconds_since(started=started)

    def pop_open_llm_call_runtimes(self) -> dict[str, float]:
        """Return the seconds each still-open invocation has run, then forget them.

        Used when an attempt fails or is retried and the open invocations will
        never see an ``on_chat_model_end``.
        """
        open_runtimes = {
            run_id: elapsed_seconds_since(started=started)
            for run_id, started in self._llm_call_started.items()
        }
        self._llm_call_started.clear()
        return open_runtimes

    def start_streamed_output(self, *, event_name: str, start_time: datetime) -> None:
        self._streamed_output = StreamedOutput(
            event_name=event_name, start_time=start_time
        )

    def append_fragment(self, *, text: str) -> None:
        if not text:
            return
        if self._streamed_output is None:
            self._streamed_output = StreamedOutput(
                event_name="unknown",
                start_time=datetime.now(tz=timezone.utc),
            )
        self._streamed_output.text_fragments.append(text)

    def pop_streamed_output(self) -> StreamedOutput | None:
        output = self._streamed_output
        self._streamed_output = None
        return output

    def pop_text(self) -> str | None:
        if not self._streamed_output or not self._streamed_output.text_fragments:
            return None
        combined = "".join(self._streamed_output.text_fragments)
        return combined if combined else None

    def clear(self) -> None:
        self._streamed_output = None
