import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class StreamBuffer:
    chunks: list[str] = field(default_factory=list)
    last_flush_ts: float = 0.0


@dataclass
class StreamedOutput:
    event_name: str
    start_time: datetime
    text_fragments: list[str] = field(default_factory=list)


class StreamBufferManager:
    def __init__(
        self,
        *,
        flush_interval_seconds: float,
        enabled: bool,
    ) -> None:
        self._flush_interval_seconds = flush_interval_seconds
        self._enabled = enabled
        self._stream_buffers: dict[str, StreamBuffer] = {}
        self._streamed_outputs: dict[str, StreamedOutput] = {}

    async def buffer_content(
        self,
        *,
        request_id: str,
        content_text: str,
        force_flush: bool = False,
    ) -> str | None:
        if not self._enabled:
            existing_buffer = self._stream_buffers.pop(request_id, None)
            existing_text = (
                "".join(existing_buffer.chunks)
                if existing_buffer is not None and existing_buffer.chunks
                else ""
            )
            immediate_text = f"{existing_text}{content_text}"
            return immediate_text or None

        buffer = self._stream_buffers.setdefault(
            request_id,
            StreamBuffer(chunks=[], last_flush_ts=time.monotonic()),
        )
        if content_text:
            buffer.chunks.append(content_text)
        if not buffer.chunks and force_flush:
            self._stream_buffers.pop(request_id, None)
            return None
        if not buffer.chunks:
            return None
        now = time.monotonic()
        should_flush = (
            force_flush
            or ("\n" in content_text if content_text else False)
            or (now - buffer.last_flush_ts) >= self._flush_interval_seconds
        )
        if not should_flush:
            return None
        combined = "".join(buffer.chunks)
        buffer.chunks.clear()
        buffer.last_flush_ts = now
        if not combined:
            if force_flush:
                self._stream_buffers.pop(request_id, None)
            return None
        if force_flush:
            self._stream_buffers.pop(request_id, None)
        return combined

    def start_streamed_output(
        self, *, request_id: str, event_name: str, start_time: datetime
    ) -> None:
        self._streamed_outputs[request_id] = StreamedOutput(
            event_name=event_name, start_time=start_time
        )

    def append_streamed_text_fragment(self, *, request_id: str, text: str) -> None:
        if not text:
            return
        output = self._streamed_outputs.get(request_id)
        if output is None:
            output = StreamedOutput(
                event_name="unknown",
                start_time=datetime.now(tz=timezone.utc),
            )
            self._streamed_outputs[request_id] = output
        output.text_fragments.append(text)

    def pop_streamed_output(self, request_id: str) -> StreamedOutput | None:
        return self._streamed_outputs.pop(request_id, None)

    def pop_streamed_text(self, request_id: str) -> str | None:
        output = self._streamed_outputs.get(request_id)
        if not output or not output.text_fragments:
            return None
        combined = "".join(output.text_fragments)
        return combined if combined else None

    def clear_request_streamed_text(self, request_id: str) -> None:
        self._streamed_outputs.pop(request_id, None)
