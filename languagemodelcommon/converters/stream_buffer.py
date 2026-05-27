import time
from dataclasses import dataclass, field


@dataclass
class StreamBuffer:
    chunks: list[str] = field(default_factory=list)
    last_flush_ts: float = 0.0


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
        self._streamed_text_fragments: dict[str, list[str]] = {}

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

    def append_streamed_text_fragment(self, *, request_id: str, text: str) -> None:
        if not text:
            return
        fragments = self._streamed_text_fragments.setdefault(request_id, [])
        fragments.append(text)

    def pop_streamed_text(self, request_id: str) -> str | None:
        fragments = self._streamed_text_fragments.pop(request_id, None)
        if not fragments:
            return None
        combined = "".join(fragments)
        return combined if combined else None

    def clear_request_streamed_text(self, request_id: str) -> None:
        self._streamed_text_fragments.pop(request_id, None)
