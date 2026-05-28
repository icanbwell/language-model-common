from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class StreamedOutput:
    event_name: str
    start_time: datetime
    text_fragments: list[str] = field(default_factory=list)


class StreamDebugOutputManager:
    def __init__(self) -> None:
        self._streamed_output: StreamedOutput | None = None

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
