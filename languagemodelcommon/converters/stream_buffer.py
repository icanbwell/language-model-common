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
        self._buffer = StreamBuffer(chunks=[], last_flush_ts=time.monotonic())
        self._last_emitted_char: str | None = None
        self._pending_invocation_boundary = False

    def mark_new_invocation_boundary(self) -> None:
        """Flag that the next content flows from a fresh chat-model invocation.

        LangGraph re-invokes the chat model independently after each tool
        call, so the resumed completion has no guaranteed leading whitespace
        against whatever this request already streamed. The flag is consumed
        by the next non-empty ``content_text`` passed to ``buffer_content``.
        """
        self._pending_invocation_boundary = True

    def _apply_invocation_boundary(self, *, content_text: str) -> str:
        if not self._pending_invocation_boundary:
            return content_text
        self._pending_invocation_boundary = False
        needs_separator = (
            self._last_emitted_char is not None
            and not self._last_emitted_char.isspace()
            and not content_text[0].isspace()
        )
        return f" {content_text}" if needs_separator else content_text

    async def buffer_content(
        self,
        *,
        content_text: str,
        force_flush: bool = False,
    ) -> str | None:
        if content_text:
            content_text = self._apply_invocation_boundary(content_text=content_text)
            self._last_emitted_char = content_text[-1]

        if not self._enabled:
            existing_text = "".join(self._buffer.chunks) if self._buffer.chunks else ""
            self._buffer.chunks.clear()
            immediate_text = f"{existing_text}{content_text}"
            return immediate_text or None

        if content_text:
            self._buffer.chunks.append(content_text)
        if not self._buffer.chunks and force_flush:
            return None
        if not self._buffer.chunks:
            return None
        now = time.monotonic()
        should_flush = (
            force_flush
            or ("\n" in content_text if content_text else False)
            or (now - self._buffer.last_flush_ts) >= self._flush_interval_seconds
        )
        if not should_flush:
            return None
        combined = "".join(self._buffer.chunks)
        self._buffer.chunks.clear()
        self._buffer.last_flush_ts = now
        if not combined:
            return None
        return combined
