# Streaming Manager Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `LangGraphStreamingManager` (1,058 lines) into 4 focused modules for maintainability.

**Architecture:** Extract tool event handling, stream buffering, and formatting utilities into their own modules. The orchestrator retains event routing and chat model handlers. New classes are injected via the IoC container.

**Tech Stack:** Python 3.12, pytest, langchain-core, simple_container

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `languagemodelcommon/converters/streaming_formatters.py` | Create | Stateless formatting/parsing utility functions |
| `languagemodelcommon/converters/stream_buffer.py` | Create | `StreamBuffer` dataclass + `StreamBufferManager` class |
| `languagemodelcommon/converters/tool_event_handlers.py` | Create | `ToolEventHandler` class with tool start/end/error handling |
| `languagemodelcommon/converters/streaming_manager.py` | Modify | Remove extracted code, inject new dependencies |
| `languagemodelcommon/container/container_factory.py` | Modify | Register `StreamBufferManager` and `ToolEventHandler` |
| `tests/converters/test_streaming_formatters.py` | Create | Tests for pure formatting functions |
| `tests/converters/test_stream_buffer.py` | Create | Buffer tests (migrated from test_streaming_manager.py) |
| `tests/converters/test_tool_event_handlers.py` | Create | Tool handler tests |
| `tests/converters/test_streaming_manager.py` | Modify | Remove migrated buffer tests, update remaining tests |

---

### Task 1: Create `streaming_formatters.py`

**Files:**
- Create: `languagemodelcommon/converters/streaming_formatters.py`
- Test: `tests/converters/test_streaming_formatters.py`

- [ ] **Step 1: Write failing tests for formatting functions**

Create `tests/converters/test_streaming_formatters.py`:

```python
import pytest
from langchain_core.messages import ToolMessage

from languagemodelcommon.converters.streaming_formatters import (
    make_tool_key,
    safe_json,
    convert_message_content_into_string,
    get_structured_content_from_tool_message,
    format_message_content,
    format_tool_input_labels,
    extract_reasoning_text,
)


class TestMakeToolKey:
    def test_generates_key_from_name_and_input(self) -> None:
        key = make_tool_key("search", {"query": "hello"})
        assert key.startswith("search:")

    def test_none_name_defaults_to_unknown(self) -> None:
        key = make_tool_key(None, {"query": "hello"})
        assert key.startswith("unknown:")

    def test_same_inputs_produce_same_key(self) -> None:
        key1 = make_tool_key("tool", {"a": 1, "b": 2})
        key2 = make_tool_key("tool", {"b": 2, "a": 1})
        assert key1 == key2

    def test_different_inputs_produce_different_keys(self) -> None:
        key1 = make_tool_key("tool", {"a": 1})
        key2 = make_tool_key("tool", {"a": 2})
        assert key1 != key2


class TestSafeJson:
    def test_parses_valid_json(self) -> None:
        assert safe_json('{"key": "value"}') == {"key": "value"}

    def test_returns_none_for_invalid_json(self) -> None:
        assert safe_json("not json") is None

    def test_parses_json_array(self) -> None:
        assert safe_json("[1, 2, 3]") == [1, 2, 3]


class TestConvertMessageContentIntoString:
    def test_string_content_returned_directly(self) -> None:
        msg = ToolMessage(content="hello world", tool_call_id="tc1")
        result = convert_message_content_into_string(tool_message=msg)
        assert "hello world" in result

    def test_json_result_field_extracted(self) -> None:
        msg = ToolMessage(
            content=[{"type": "text", "text": '{"result": "extracted"}'}],
            tool_call_id="tc1",
        )
        result = convert_message_content_into_string(tool_message=msg)
        assert result == "extracted"

    def test_list_content_joined(self) -> None:
        msg = ToolMessage(content=["part1", "part2"], tool_call_id="tc1")
        result = convert_message_content_into_string(tool_message=msg)
        assert "part1" in result
        assert "part2" in result


class TestGetStructuredContentFromToolMessage:
    def test_dict_content_returned(self) -> None:
        msg = ToolMessage(content={"key": "value"}, tool_call_id="tc1")
        result = get_structured_content_from_tool_message(tool_message=msg)
        assert result == {"key": "value"}

    def test_single_element_list_returned(self) -> None:
        msg = ToolMessage(content=[{"key": "value"}], tool_call_id="tc1")
        result = get_structured_content_from_tool_message(tool_message=msg)
        assert result == {"key": "value"}

    def test_string_content_returns_none(self) -> None:
        msg = ToolMessage(content="just text", tool_call_id="tc1")
        result = get_structured_content_from_tool_message(tool_message=msg)
        assert result is None


class TestFormatMessageContent:
    def test_string_returned_as_is(self) -> None:
        assert format_message_content("hello") == "hello"

    def test_list_of_strings_joined(self) -> None:
        result = format_message_content(["line1", "line2"])
        assert result == "line1\nline2"

    def test_list_of_dicts_extracts_text(self) -> None:
        result = format_message_content([{"text": "hello"}, {"text": "world"}])
        assert result == "hello\nworld"


class TestFormatToolInputLabels:
    def test_formats_keys_as_labels(self) -> None:
        result = format_tool_input_labels(tool_input={"user_name": "test", "query": "x"})
        assert "User Name" in result
        assert "Query" in result

    def test_hides_auth_token_state_runtime(self) -> None:
        result = format_tool_input_labels(
            tool_input={"auth_token": "x", "state": "y", "runtime": "z", "query": "q"}
        )
        assert "Auth Token" not in result
        assert "State" not in result
        assert "Runtime" not in result
        assert "Query" in result

    def test_empty_input_returns_none(self) -> None:
        assert format_tool_input_labels(tool_input=None) == "none"
        assert format_tool_input_labels(tool_input={}) == "none"


class TestExtractReasoningText:
    def test_extracts_reasoning_content_type(self) -> None:
        block = {"type": "reasoning_content", "reasoning_content": {"text": "thinking..."}}
        assert extract_reasoning_text(block) == "thinking..."

    def test_extracts_reasoning_type(self) -> None:
        block = {"type": "reasoning", "reasoning": "I think..."}
        assert extract_reasoning_text(block) == "I think..."

    def test_returns_none_for_unknown_type(self) -> None:
        block = {"type": "text", "text": "hello"}
        assert extract_reasoning_text(block) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/imranqureshi/git/language-model-common && uv run pytest tests/converters/test_streaming_formatters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'languagemodelcommon.converters.streaming_formatters'`

- [ ] **Step 3: Create `streaming_formatters.py` with all functions**

Create `languagemodelcommon/converters/streaming_formatters.py`:

```python
import json
from typing import Any, Dict, Optional

from langchain_core.messages import ToolMessage

from languagemodelcommon.utilities.text_humanizer import Humanizer


def make_tool_key(
    tool_name: Optional[str], tool_input: Optional[Dict[str, Any]]
) -> str:
    if tool_name is None:
        tool_name = "unknown"
    try:
        tool_input_str = json.dumps(tool_input, sort_keys=True, default=str)
    except Exception:
        tool_input_str = str(tool_input)
    return f"{tool_name}:{hash(tool_input_str)}"


def safe_json(string: str) -> Any:
    try:
        return json.loads(string)
    except json.JSONDecodeError:
        return None


def convert_message_content_into_string(*, tool_message: ToolMessage) -> str:
    if isinstance(tool_message.content, str):
        return _format_text_resource_contents(text=tool_message.content)

    if (
        isinstance(tool_message.content, list)
        and len(tool_message.content) == 1
        and isinstance(tool_message.content[0], dict)
        and "text" in tool_message.content[0]
    ):
        text = tool_message.content[0]["text"]
        json_object: dict[str, Any] | None = safe_json(text)
        if json_object is not None and isinstance(json_object, dict):
            if "result" in json_object:
                return str(json_object.get("result"))

    return " ".join([str(c) for c in tool_message.content])


def get_structured_content_from_tool_message(
    *, tool_message: ToolMessage
) -> dict[str, Any] | None:
    if isinstance(tool_message.content, dict):
        return tool_message.content
    elif (
        isinstance(tool_message.content, list)
        and len(tool_message.content) == 1
        and isinstance(tool_message.content[0], dict)
    ):
        return tool_message.content[0]
    return None


def format_message_content(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def format_tool_input_labels(*, tool_input: Dict[str, Any] | None) -> str:
    if not tool_input:
        return "none"
    hidden_keys = {"auth_token", "state", "runtime"}
    labels: list[str] = []
    for key in tool_input.keys():
        if key in hidden_keys:
            continue
        labels.append(Humanizer.humanize_tool_name(key))
    return ", ".join(labels) if labels else "none"


def extract_reasoning_text(block: dict[str, Any]) -> str | None:
    block_type = block.get("type")
    if block_type == "reasoning_content":
        rc = block.get("reasoning_content", {})
        if isinstance(rc, dict):
            return rc.get("text")
    elif block_type == "reasoning":
        return block.get("reasoning")
    return None


def _format_text_resource_contents(text: str) -> str:
    result = ""
    json_object: Any = safe_json(text)
    if json_object is not None and isinstance(json_object, dict):
        if "result" in json_object:
            result += str(json_object.get("result")) + "\n"
        if "error" in json_object:
            result += "Error: " + str(json_object.get("error")) + "\n"
        if "meta" in json_object:
            meta = json_object.get("meta", {})
            if isinstance(meta, dict) and len(meta) > 0:
                result += "Metadata:\n"
                for key, value in meta.items():
                    result += f"- {key}: {value}\n"
        if "urls" in json_object:
            urls = json_object.get("urls", [])
            if isinstance(urls, list) and len(urls) > 0:
                result += "Related URLs:\n"
                for url in urls:
                    result += f"- {url}\n"
        if "result" not in json_object and "error" not in json_object:
            result += text + "\n"
    else:
        result += text + "\n"
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/imranqureshi/git/language-model-common && uv run pytest tests/converters/test_streaming_formatters.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/imranqureshi/git/language-model-common
git add languagemodelcommon/converters/streaming_formatters.py tests/converters/test_streaming_formatters.py
git commit -m "Extract streaming_formatters module from streaming_manager"
```

---

### Task 2: Create `stream_buffer.py`

**Files:**
- Create: `languagemodelcommon/converters/stream_buffer.py`
- Create: `tests/converters/test_stream_buffer.py`

- [ ] **Step 1: Write failing tests for StreamBufferManager**

Create `tests/converters/test_stream_buffer.py`:

```python
import pytest

from languagemodelcommon.converters.stream_buffer import StreamBufferManager


class _FakeClock:
    def __init__(self) -> None:
        self._current = 0.0

    def advance(self, delta: float) -> None:
        self._current += delta

    def monotonic(self) -> float:
        return self._current


@pytest.mark.asyncio
async def test_buffer_flushes_on_newline() -> None:
    manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=True,
    )

    assert (
        await manager.buffer_content(request_id="req", content_text="Hello")
        is None
    )

    flushed = await manager.buffer_content(
        request_id="req", content_text=" world\n"
    )

    assert flushed == "Hello world\n"


@pytest.mark.asyncio
async def test_buffer_flushes_after_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_clock = _FakeClock()
    monkeypatch.setattr(
        "languagemodelcommon.converters.stream_buffer.time.monotonic",
        fake_clock.monotonic,
    )

    manager = StreamBufferManager(
        flush_interval_seconds=0.05,
        enabled=True,
    )

    assert (
        await manager.buffer_content(request_id="req", content_text="a")
        is None
    )

    fake_clock.advance(0.051)

    flushed = await manager.buffer_content(
        request_id="req", content_text="b"
    )

    assert flushed == "ab"


@pytest.mark.asyncio
async def test_buffer_disabled_returns_content_immediately() -> None:
    manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=False,
    )

    first = await manager.buffer_content(request_id="req", content_text="Hello")
    second = await manager.buffer_content(request_id="req", content_text=" world")

    assert first == "Hello"
    assert second == " world"


@pytest.mark.asyncio
async def test_force_flush_returns_buffered_content() -> None:
    manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=True,
    )

    await manager.buffer_content(request_id="req", content_text="buffered")
    flushed = await manager.buffer_content(
        request_id="req", content_text="", force_flush=True
    )

    assert flushed == "buffered"


@pytest.mark.asyncio
async def test_force_flush_empty_buffer_returns_none() -> None:
    manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=True,
    )

    result = await manager.buffer_content(
        request_id="req", content_text="", force_flush=True
    )

    assert result is None


class TestStreamedTextFragments:
    def test_append_and_pop(self) -> None:
        manager = StreamBufferManager(flush_interval_seconds=1.0, enabled=True)
        manager.append_streamed_text_fragment("req", "hello ")
        manager.append_streamed_text_fragment("req", "world")
        result = manager.pop_streamed_text("req")
        assert result == "hello world"

    def test_pop_empty_returns_none(self) -> None:
        manager = StreamBufferManager(flush_interval_seconds=1.0, enabled=True)
        assert manager.pop_streamed_text("req") is None

    def test_clear_removes_fragments(self) -> None:
        manager = StreamBufferManager(flush_interval_seconds=1.0, enabled=True)
        manager.append_streamed_text_fragment("req", "text")
        manager.clear_request_streamed_text("req")
        assert manager.pop_streamed_text("req") is None

    def test_append_empty_string_is_noop(self) -> None:
        manager = StreamBufferManager(flush_interval_seconds=1.0, enabled=True)
        manager.append_streamed_text_fragment("req", "")
        assert manager.pop_streamed_text("req") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/imranqureshi/git/language-model-common && uv run pytest tests/converters/test_stream_buffer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'languagemodelcommon.converters.stream_buffer'`

- [ ] **Step 3: Create `stream_buffer.py`**

Create `languagemodelcommon/converters/stream_buffer.py`:

```python
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

    def append_streamed_text_fragment(self, request_id: str, text: str) -> None:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/imranqureshi/git/language-model-common && uv run pytest tests/converters/test_stream_buffer.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/imranqureshi/git/language-model-common
git add languagemodelcommon/converters/stream_buffer.py tests/converters/test_stream_buffer.py
git commit -m "Extract stream_buffer module from streaming_manager"
```

---

### Task 3: Create `tool_event_handlers.py`

**Files:**
- Create: `languagemodelcommon/converters/tool_event_handlers.py`
- Create: `tests/converters/test_tool_event_handlers.py`

- [ ] **Step 1: Write failing tests for ToolEventHandler**

Create `tests/converters/test_tool_event_handlers.py`:

```python
from typing import Any, Optional, cast
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.runnables.schema import StandardStreamEvent

from languagemodelcommon.converters.stream_buffer import StreamBufferManager
from languagemodelcommon.converters.tool_event_handlers import ToolEventHandler
from languagemodelcommon.file_managers.file_writer import FileWriter, DebugFileWriteResult
from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.request_information import RequestInformation
from languagemodelcommon.utilities.tool_display_name_mapper import ToolDisplayNameMapper


class _FakeChatRequestWrapper:
    def __init__(self, *, enable_debug_logging: bool = False) -> None:
        self.enable_debug_logging = enable_debug_logging

    def create_sse_message(
        self, *, request_id: str, content: str | None, usage_metadata: Any, source: str
    ) -> str:
        return content or ""

    def create_debug_sse_message(
        self, *, request_id: str, content: str | None, usage_metadata: Any, source: str
    ) -> str | None:
        return content

    def create_tool_start_sse_event(
        self, *, request_id: str, tool_name: str, tool_input: Any
    ) -> str | None:
        return None

    def create_tool_end_sse_event(
        self, *, request_id: str, tool_name: str, tool_input: Any, runtime_seconds: Any
    ) -> str | None:
        return None

    def create_mcp_app_sse_event(self, **kwargs: Any) -> str | None:
        return None


@pytest.fixture
def tool_event_handler(monkeypatch: pytest.MonkeyPatch) -> ToolEventHandler:
    monkeypatch.setenv("WRITE_TOOL_OUTPUT_TO_FILE", "false")
    environment_variables = LanguageModelCommonEnvironmentVariables()
    mock_file_writer = AsyncMock(spec=FileWriter)
    mock_file_writer.write_to_file_async = AsyncMock(return_value=None)
    stream_buffer_manager = StreamBufferManager(
        flush_interval_seconds=10.0,
        enabled=False,
    )
    return ToolEventHandler(
        debug_file_writer=mock_file_writer,
        environment_variables=environment_variables,
        tool_display_name_mapper=ToolDisplayNameMapper(),
        stream_buffer_manager=stream_buffer_manager,
    )


@pytest.mark.asyncio
async def test_tool_start_records_start_time(
    tool_event_handler: ToolEventHandler,
) -> None:
    event = cast(
        StandardStreamEvent,
        {
            "event": "on_tool_start",
            "name": "search_tool",
            "data": {"input": {"query": "test"}},
        },
    )
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=False),
    )
    request_information = RequestInformation(request_id="req-1")
    tool_start_times: dict[str, float] = {}

    chunks = [
        chunk
        async for chunk in tool_event_handler.handle_tool_start(
            event=event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
            tool_start_times=tool_start_times,
        )
        if chunk
    ]

    assert len(tool_start_times) == 1


@pytest.mark.asyncio
async def test_tool_end_yields_content(
    tool_event_handler: ToolEventHandler,
) -> None:
    event = cast(
        StandardStreamEvent,
        {
            "event": "on_tool_end",
            "name": "search_tool",
            "data": {
                "input": {"query": "test"},
                "output": ToolMessage(
                    content="search results here",
                    tool_call_id="tc1",
                    name="search_tool",
                ),
            },
        },
    )
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=False),
    )
    request_information = RequestInformation(request_id="req-1")
    tool_start_times: dict[str, float] = {}

    chunks = [
        chunk
        async for chunk in tool_event_handler.handle_tool_end(
            event=event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
            tool_start_times=tool_start_times,
        )
        if chunk
    ]

    # Should not error; may yield 0 chunks if no file writing or debug
    assert isinstance(chunks, list)


@pytest.mark.asyncio
async def test_tool_error_yields_error_message(
    tool_event_handler: ToolEventHandler,
) -> None:
    event = cast(
        StandardStreamEvent,
        {
            "event": "on_tool_error",
            "name": "failing_tool",
            "data": {
                "input": {"param": "value"},
                "error": "Something went wrong",
            },
        },
    )
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=False),
    )
    request_information = RequestInformation(request_id="req-1")
    tool_start_times: dict[str, float] = {}

    chunks = [
        chunk
        async for chunk in tool_event_handler.handle_tool_error(
            event=event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
            tool_start_times=tool_start_times,
        )
        if chunk
    ]

    assert len(chunks) >= 1
    assert "Something went wrong" in chunks[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/imranqureshi/git/language-model-common && uv run pytest tests/converters/test_tool_event_handlers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'languagemodelcommon.converters.tool_event_handlers'`

- [ ] **Step 3: Create `tool_event_handlers.py`**

Create `languagemodelcommon/converters/tool_event_handlers.py`:

```python
import copy
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

from langchain_core.messages import ToolMessage
from langchain_core.runnables.schema import CustomStreamEvent, StandardStreamEvent
from oidcauthlib.auth.exceptions.authorization_needed_exception import (
    AuthorizationNeededException,
)

from languagemodelcommon.converters.stream_buffer import StreamBufferManager
from languagemodelcommon.converters.streaming_formatters import (
    convert_message_content_into_string,
    make_tool_key,
)
from languagemodelcommon.file_managers.file_writer import (
    DebugFileWriteResult,
    FileWriter,
)
from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS
from languagemodelcommon.utilities.request_information import RequestInformation
from languagemodelcommon.utilities.tool_display_name_mapper import ToolDisplayNameMapper

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.LLM)


class ToolEventHandler:
    def __init__(
        self,
        *,
        debug_file_writer: FileWriter,
        environment_variables: LanguageModelCommonEnvironmentVariables,
        tool_display_name_mapper: ToolDisplayNameMapper,
        stream_buffer_manager: StreamBufferManager,
    ) -> None:
        if debug_file_writer is None:
            raise ValueError("debug_file_writer must not be None")
        if environment_variables is None:
            raise ValueError("environment_variables must not be None")
        if tool_display_name_mapper is None:
            raise ValueError("tool_display_name_mapper must not be None")
        if stream_buffer_manager is None:
            raise ValueError("stream_buffer_manager must not be None")

        self._debug_file_writer = debug_file_writer
        self._environment_variables = environment_variables
        self._tool_display_name_mapper = tool_display_name_mapper
        self._stream_buffer_manager = stream_buffer_manager

    async def handle_tool_start(
        self,
        *,
        event: StandardStreamEvent | CustomStreamEvent,
        chat_request_wrapper: ChatRequestWrapper,
        request_information: RequestInformation,
        tool_start_times: dict[str, float],
    ) -> AsyncGenerator[str, None]:
        tool_name: Optional[str] = event["name"] if "name" in event else None
        logger.debug("on_tool_start: %s: %s", tool_name, event)
        data = event["data"] if "data" in event else {}
        tool_input: Optional[Dict[str, Any]] = data.get("input")
        tool_input_display: Optional[Dict[str, Any]] = (
            tool_input.copy() if tool_input is not None else None
        )
        if tool_input_display and "auth_token" in tool_input_display:
            tool_input_display["auth_token"] = "***"
        if tool_input_display and "state" in tool_input_display:
            tool_input_display["state"] = "***"
        if tool_input_display and "runtime" in tool_input_display:
            tool_input_display.pop("runtime")
        tool_key: str = make_tool_key(tool_name, tool_input)
        tool_start_times[tool_key] = time.time()
        if tool_name:
            logger.debug("on_tool_start: %s %s", tool_name, tool_input_display)
            tool_start_event = chat_request_wrapper.create_tool_start_sse_event(
                request_id=request_information.request_id,
                tool_name=tool_name,
                tool_input=tool_input_display,
            )
            if tool_start_event:
                yield tool_start_event
            content_text: str = self._tool_display_name_mapper.get_message_for_tool(
                tool_name=tool_name, tool_input=tool_input
            )
            buffered_chunk = await self._stream_buffer_manager.buffer_content(
                request_id=str(request_information.request_id),
                content_text=content_text,
            )
            if buffered_chunk:
                yield chat_request_wrapper.create_sse_message(
                    request_id=request_information.request_id,
                    content=buffered_chunk,
                    usage_metadata=None,
                    source="on_tool_start",
                )
            if chat_request_wrapper.enable_debug_logging:
                self._stream_buffer_manager.append_streamed_text_fragment(
                    str(request_information.request_id),
                    f"\n--- Tool Call: {tool_name} ---\n{json.dumps(tool_input_display, indent=2, default=str)}\n",
                )
            debug_content_text: str = (
                f"\n\n<details>\n<summary>Agent: {tool_name}</summary>\n\n"
                f"```json\n{json.dumps(tool_input_display, indent=2, default=str)}\n```\n\n"
                f"</details>\n\n"
            )
            debug_message = chat_request_wrapper.create_debug_sse_message(
                request_id=request_information.request_id,
                content=debug_content_text,
                usage_metadata=None,
                source="on_tool_start",
            )
            if debug_message:
                yield debug_message

    async def handle_tool_end(
        self,
        *,
        event: StandardStreamEvent | CustomStreamEvent,
        chat_request_wrapper: ChatRequestWrapper,
        request_information: RequestInformation,
        tool_start_times: dict[str, float],
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        event_name: Optional[str] = event["name"] if "name" in event else None
        data = event["data"] if "data" in event else {}
        logger.debug(
            "on_tool_end: name=%s request_id=%s data=%s",
            event_name,
            request_information.request_id,
            data,
        )

        runtime_str: str = ""
        tool_message: Optional[ToolMessage] = data.get("output")
        if tool_message:
            tool_name: str = tool_message.name or event_name or "unknown"
            tool_input: Optional[Dict[str, Any]] = data.get("input")

            tool_key: str = make_tool_key(tool_name, tool_input)
            start_time: Optional[float] = tool_start_times.pop(tool_key, None)
            runtime_seconds: Optional[float] = None
            if start_time is not None:
                elapsed: float = time.time() - start_time
                runtime_seconds = elapsed
                runtime_str = f"{elapsed:.2f}s"
                logger.debug("Tool %s completed in %.2f seconds.", tool_name, elapsed)
            else:
                logger.warning(
                    "Tool %s end event received without matching start event.",
                    tool_name,
                )

            tool_end_event = chat_request_wrapper.create_tool_end_sse_event(
                request_id=request_information.request_id,
                tool_name=tool_name,
                tool_input=tool_input,
                runtime_seconds=runtime_seconds,
            )
            if tool_end_event:
                yield tool_end_event

            tool_message_content: str = (
                convert_message_content_into_string(tool_message=tool_message)
                if tool_message
                else ""
            )

            artifact: Optional[Any] = tool_message.artifact

            logger.debug(
                "Tool %s has artifact of type %s: %s",
                tool_name,
                type(artifact),
                artifact,
            )

            if isinstance(artifact, dict) and "mcp_app_embed" in artifact:
                mcp_app_embed = artifact["mcp_app_embed"]
                embed_html = getattr(mcp_app_embed, "html", None)
                embed_title = getattr(mcp_app_embed, "title", None)
                ui_meta = getattr(mcp_app_embed, "ui_meta", None)
                if embed_html:
                    mcp_app_event = chat_request_wrapper.create_mcp_app_sse_event(
                        html=embed_html,
                        title=embed_title,
                        csp=getattr(ui_meta, "csp", None) if ui_meta else None,
                        permissions=getattr(ui_meta, "permissions", None)
                        if ui_meta
                        else None,
                        prefers_border=getattr(ui_meta, "prefers_border", None)
                        if ui_meta
                        else None,
                        display_mode=getattr(ui_meta, "display_mode", None)
                        if ui_meta
                        else None,
                    )
                    if mcp_app_event:
                        yield mcp_app_event

            if self._environment_variables.write_tool_output_to_file and (
                chat_request_wrapper.enable_debug_logging or artifact is not None
            ):
                if self._environment_variables.log_input_and_output:
                    logger.debug(
                        f"Returning artifact: {artifact if artifact else tool_message_content}"
                    )
                tool_message_or_artifact_content = (
                    str(artifact) if artifact else tool_message_content
                )
                if chat_request_wrapper.enable_debug_logging:
                    self._stream_buffer_manager.append_streamed_text_fragment(
                        str(request_information.request_id),
                        f"\n--- Tool Output: {tool_name} ({runtime_str}) ---\n{tool_message_or_artifact_content}\n",
                    )

                tool_display_name: str = (
                    self._tool_display_name_mapper.get_name_for_tool(
                        tool_name=tool_name,
                        tool_input=tool_input,
                    )
                )
                write_result: (
                    DebugFileWriteResult | None
                ) = await self._debug_file_writer.write_to_file_async(
                    content=tool_message_or_artifact_content,
                    user_id=user_id,
                    file_name=tool_name,
                )
                if (
                    write_result is not None
                    and write_result.file_path
                    and write_result.file_url
                ):
                    content_text: str = f"\n\n[Click to download {tool_display_name} Output]({write_result.file_url})\n\n"
                    yield chat_request_wrapper.create_sse_message(
                        request_id=request_information.request_id,
                        content=content_text,
                        usage_metadata=None,
                        source="on_tool_end",
                    )

            if chat_request_wrapper.enable_debug_logging:
                structured_data: dict[str, Any] | None = (
                    artifact if isinstance(artifact, dict) else None
                )
                structured_data_without_result: dict[str, Any] | None = (
                    copy.deepcopy(structured_data)
                    if structured_data is not None
                    else None
                )
                if structured_data_without_result:
                    structured_data_without_result.pop("result", None)
                    structured_content = structured_data_without_result.get(
                        "structured_content"
                    )
                    if isinstance(structured_content, dict):
                        structured_content.pop("result", None)

                    structured_json = json.dumps(
                        structured_data_without_result, indent=2
                    )
                    structured_content_text: str = (
                        f"\n\n<details>\n<summary>{tool_name} output</summary>\n\n"
                        f"```json\n{structured_json}\n```\n\n"
                        f"</details>\n\n"
                    )
                    debug_message = chat_request_wrapper.create_debug_sse_message(
                        request_id=request_information.request_id,
                        content=structured_content_text,
                        usage_metadata=None,
                        source="on_tool_end",
                    )
                    if debug_message:
                        yield debug_message
        else:
            logger.debug("on_tool_end: no tool message output")
            content_text = (
                f"\n\n<details>\n<summary>Tool completed with no output</summary>\n\n"
                f"Runtime: {runtime_str}\n\n"
                f"</details>\n\n"
            )
            debug_message = chat_request_wrapper.create_debug_sse_message(
                request_id=request_information.request_id,
                content=content_text,
                usage_metadata=None,
                source="on_tool_end",
            )
            if debug_message:
                yield debug_message

    async def handle_tool_error(
        self,
        *,
        event: StandardStreamEvent | CustomStreamEvent,
        chat_request_wrapper: ChatRequestWrapper,
        request_information: RequestInformation,
        tool_start_times: dict[str, float],
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        tool_name: Optional[str] = event["name"] if "name" in event else None
        data = event["data"] if "data" in event else {}
        error_message: Any = data.get("error") or str(event)
        tool_input: Optional[Dict[str, Any]] = data.get("input")
        runtime_str: str = ""
        tool_key: str = make_tool_key(tool_name, tool_input)
        start_time: Optional[float] = tool_start_times.pop(tool_key, None)
        if start_time is not None:
            elapsed: float = time.time() - start_time
            runtime_str = f"{elapsed:.2f}s"
        logger.error(
            "Tool error in %s: (%s) %s [runtime: %s]",
            tool_name,
            type(error_message),
            error_message,
            runtime_str,
        )
        if isinstance(error_message, AuthorizationNeededException):
            return

        content_text: str = f"\n\n> Tool {tool_name} encountered an error: {error_message} [runtime: {runtime_str}]\n"

        yield chat_request_wrapper.create_sse_message(
            request_id=request_information.request_id,
            content=content_text,
            usage_metadata=None,
            source="on_tool_error",
        )

        if self._environment_variables.write_tool_output_to_file:
            error_content: str = (
                f"Tool: {tool_name}\nError: {error_message}\nRuntime: {runtime_str}"
            )
            self._stream_buffer_manager.append_streamed_text_fragment(
                str(request_information.request_id),
                f"\n--- Tool Error: {tool_name} ({runtime_str}) ---\n{error_message}\n",
            )
            tool_display_name: str = self._tool_display_name_mapper.get_name_for_tool(
                tool_name=tool_name or "unknown",
                tool_input=tool_input,
            )
            write_result: (
                DebugFileWriteResult | None
            ) = await self._debug_file_writer.write_to_file_async(
                content=error_content,
                user_id=user_id,
                file_name=tool_name or "unknown",
            )
            if (
                write_result is not None
                and write_result.file_path
                and write_result.file_url
            ):
                download_text: str = f"\n\n[Click to download {tool_display_name} Error Output]({write_result.file_url})\n\n"
                yield chat_request_wrapper.create_sse_message(
                    request_id=request_information.request_id,
                    content=download_text,
                    usage_metadata=None,
                    source="on_tool_error",
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/imranqureshi/git/language-model-common && uv run pytest tests/converters/test_tool_event_handlers.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/imranqureshi/git/language-model-common
git add languagemodelcommon/converters/tool_event_handlers.py tests/converters/test_tool_event_handlers.py
git commit -m "Extract tool_event_handlers module from streaming_manager"
```

---

### Task 4: Refactor `streaming_manager.py` to delegate to extracted modules

**Files:**
- Modify: `languagemodelcommon/converters/streaming_manager.py`
- Modify: `tests/converters/test_streaming_manager.py`

- [ ] **Step 1: Rewrite `streaming_manager.py` to use new modules**

Replace the contents of `languagemodelcommon/converters/streaming_manager.py` with:

```python
"""
Streaming Manager for LangGraph-to-OpenAI SSE Translation.

This module bridges LangGraph's internal event stream with OpenAI-compatible
Server-Sent Events (SSE). When a user sends a chat request to BaileyAI with
streaming enabled, the flow is:

    OpenWebUI → /bailey/v1/chat/completions → LangGraphToOpenAIConverter
        → LangGraph agent (astream_events) → **LangGraphStreamingManager** → SSE chunks → OpenWebUI

The `LangGraphStreamingManager` receives raw LangChain/LangGraph events
(e.g., `on_chat_model_stream`, `on_tool_start`, `on_tool_end`) and yields
formatted SSE strings that OpenWebUI can render in real-time.

See Also:
    - LangChain astream_events reference:
      https://python.langchain.com/docs/how_to/streaming/#using-stream-events
    - `LangGraphToOpenAIConverter` (converters/langgraph_to_openai_converter.py)
      which orchestrates streaming and calls this manager.
"""

from datetime import datetime, timezone
import json
import logging
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Optional,
    cast,
)

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables.schema import (
    CustomStreamEvent,
    EventData,
    StandardStreamEvent,
)

from languagemodelcommon.converters.stream_buffer import StreamBufferManager
from languagemodelcommon.converters.streaming_formatters import (
    extract_reasoning_text,
    format_message_content,
)
from languagemodelcommon.converters.tool_event_handlers import ToolEventHandler
from languagemodelcommon.file_managers.file_writer import (
    DebugFileWriteResult,
    FileWriter,
)
from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)
from languagemodelcommon.utilities.chat_message_helpers import (
    iter_message_content_text_chunks,
)
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS
from languagemodelcommon.utilities.request_information import RequestInformation
from languagemodelcommon.utilities.token_reducer.token_reducer import TokenReducer

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.LLM)


class LangGraphStreamingManager:
    """
    Dispatches LangGraph streaming events into OpenAI-compatible SSE chunks.

    This class listens for events emitted by LangGraph's `astream_events` and
    translates them into SSE messages that OpenWebUI can display. Key events:

    - ``on_chat_model_stream`` – Token-by-token LLM output (main response text).
    - ``on_tool_start`` / ``on_tool_end`` – MCP tool invocation lifecycle.
    - ``on_tool_error`` – Errors during tool execution.
    - ``on_chain_end`` – Final usage metadata for the request.

    Instantiated via DI container (`ContainerFactory`) and injected into
    `LangGraphToOpenAIConverter`.
    """

    def __init__(
        self,
        *,
        token_reducer: TokenReducer,
        debug_file_writer: FileWriter,
        environment_variables: LanguageModelCommonEnvironmentVariables,
        tool_event_handler: ToolEventHandler,
        stream_buffer_manager: StreamBufferManager,
    ) -> None:
        if token_reducer is None:
            raise ValueError("token_reducer must not be None")
        if not isinstance(token_reducer, TokenReducer):
            raise TypeError("token_reducer must be an instance of TokenReducer")
        self.token_reducer = token_reducer

        if debug_file_writer is None:
            raise ValueError("debug_file_writer must not be None")
        if not isinstance(debug_file_writer, FileWriter):
            raise TypeError("debug_file_writer must be an instance of FileWriter")
        self.debug_file_writer = debug_file_writer

        if environment_variables is None:
            raise ValueError("environment_variables must not be None")
        self.environment_variables = environment_variables

        if tool_event_handler is None:
            raise ValueError("tool_event_handler must not be None")
        if not isinstance(tool_event_handler, ToolEventHandler):
            raise TypeError(
                "tool_event_handler must be an instance of ToolEventHandler"
            )
        self._tool_event_handler = tool_event_handler

        if stream_buffer_manager is None:
            raise ValueError("stream_buffer_manager must not be None")
        if not isinstance(stream_buffer_manager, StreamBufferManager):
            raise TypeError(
                "stream_buffer_manager must be an instance of StreamBufferManager"
            )
        self._stream_buffer_manager = stream_buffer_manager

    async def handle_langchain_event(
        self,
        *,
        event: StandardStreamEvent | CustomStreamEvent,
        chat_request_wrapper: ChatRequestWrapper,
        request_information: RequestInformation,
        tool_start_times: dict[str, float],
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Route a single LangGraph event to the appropriate handler and yield SSE chunks."""
        try:
            event_type: str = event["event"]
            match event_type:
                case "on_chat_model_start":
                    async for chunk in self._handle_on_chat_model_start(
                        event=event,
                        chat_request_wrapper=chat_request_wrapper,
                        request_information=request_information,
                    ):
                        if chunk:
                            yield chunk
                case "on_chat_model_end":
                    async for chunk in self._handle_on_chat_model_end(
                        event=event,
                        chat_request_wrapper=chat_request_wrapper,
                        request_information=request_information,
                    ):
                        if chunk:
                            yield chunk
                case "on_chain_start":
                    pass
                case "on_chain_stream":
                    pass
                case "on_chat_model_stream":
                    async for chunk in self._handle_on_chat_model_stream(
                        event=event,
                        chat_request_wrapper=chat_request_wrapper,
                        request_information=request_information,
                    ):
                        if chunk:
                            yield chunk
                case "on_chain_end":
                    async for chunk in self._handle_on_chain_end(
                        event=event,
                        chat_request_wrapper=chat_request_wrapper,
                        request_information=request_information,
                    ):
                        if chunk:
                            yield chunk
                case "on_tool_start":
                    async for chunk in self._tool_event_handler.handle_tool_start(
                        event=event,
                        chat_request_wrapper=chat_request_wrapper,
                        request_information=request_information,
                        tool_start_times=tool_start_times,
                    ):
                        if chunk:
                            yield chunk
                case "on_tool_end":
                    async for chunk in self._tool_event_handler.handle_tool_end(
                        event=event,
                        chat_request_wrapper=chat_request_wrapper,
                        request_information=request_information,
                        tool_start_times=tool_start_times,
                        user_id=user_id,
                    ):
                        if chunk:
                            yield chunk
                case "on_tool_error":
                    async for chunk in self._tool_event_handler.handle_tool_error(
                        event=event,
                        chat_request_wrapper=chat_request_wrapper,
                        request_information=request_information,
                        tool_start_times=tool_start_times,
                        user_id=user_id,
                    ):
                        if chunk:
                            yield chunk
                case "on_custom_event":
                    async for chunk in self._handle_on_custom_event(
                        event=event,
                        chat_request_wrapper=chat_request_wrapper,
                        request_information=request_information,
                    ):
                        if chunk:
                            yield chunk
                case _:
                    logger.debug("Skipped event type: %s", event_type)
        except Exception:
            logger.exception("Error handling langchain event")

    async def _handle_on_chat_model_stream(
        self,
        *,
        event: StandardStreamEvent | CustomStreamEvent,
        chat_request_wrapper: ChatRequestWrapper,
        request_information: RequestInformation,
    ) -> AsyncGenerator[str, None]:
        data = event["data"] if "data" in event else {}
        chunk: AIMessageChunk | None = data.get("chunk")
        if chunk is not None:
            content: str | list[str | dict[str, Any]] = chunk.content
            content_chunks = iter_message_content_text_chunks(
                content,
                include_non_text_placeholders=False,
            )
            for content_text in content_chunks.text_chunks:
                if not isinstance(content_text, str):
                    raise TypeError(
                        f"content_text must be str, got {type(content_text)}"
                    )
                if self.environment_variables.log_input_and_output and content_text:
                    logger.debug("Returning content: %s", content_text)
                if content_text:
                    self._stream_buffer_manager.append_streamed_text_fragment(
                        str(request_information.request_id),
                        content_text,
                    )
                    buffered_chunk = await self._stream_buffer_manager.buffer_content(
                        request_id=str(request_information.request_id),
                        content_text=content_text,
                    )
                    if buffered_chunk:
                        yield chat_request_wrapper.create_sse_message(
                            request_id=request_information.request_id,
                            content=buffered_chunk,
                            usage_metadata=chunk.usage_metadata,
                            source="on_chat_model_stream",
                        )
            if chat_request_wrapper.enable_debug_logging:
                async for debug_chunk in self._handle_non_text_content_debug(
                    chat_request_wrapper=chat_request_wrapper,
                    request_information=request_information,
                    non_text_blocks=content_chunks.non_text_blocks,
                ):
                    if debug_chunk:
                        yield debug_chunk

    async def _handle_on_chain_end(
        self,
        *,
        event: StandardStreamEvent | CustomStreamEvent,
        chat_request_wrapper: ChatRequestWrapper,
        request_information: RequestInformation,
    ) -> AsyncGenerator[str, None]:
        data = event["data"] if "data" in event else {}
        output: Dict[str, Any] | str | None = data.get("output")
        buffered_chunk = await self._stream_buffer_manager.buffer_content(
            request_id=str(request_information.request_id),
            content_text="",
            force_flush=True,
        )
        if buffered_chunk:
            yield chat_request_wrapper.create_sse_message(
                request_id=request_information.request_id,
                content=buffered_chunk,
                usage_metadata=None,
                source="on_chat_model_stream",
            )
        if output and isinstance(output, dict) and "usage_metadata" in output:
            yield chat_request_wrapper.create_final_sse_message(
                request_id=request_information.request_id,
                usage_metadata=output["usage_metadata"],
                source="on_chain_end",
            )
        self._stream_buffer_manager.clear_request_streamed_text(
            str(request_information.request_id),
        )

    async def _handle_on_chat_model_start(
        self,
        *,
        event: StandardStreamEvent | CustomStreamEvent,
        chat_request_wrapper: ChatRequestWrapper,
        request_information: RequestInformation,
    ) -> AsyncGenerator[str | None, None]:
        yield None

    async def _handle_on_chat_model_end(
        self,
        *,
        event: StandardStreamEvent | CustomStreamEvent,
        chat_request_wrapper: ChatRequestWrapper,
        request_information: RequestInformation,
    ) -> AsyncGenerator[str | None, None]:
        if not chat_request_wrapper.enable_debug_logging:
            return

        data: EventData = event["data"] if "data" in event else {}
        input_messages_list: list[list[BaseMessage]] = cast(
            list[list[BaseMessage]],
            cast(dict[str, Any], data.get("input", {})).get("messages", []),
        )
        input_messages: list[BaseMessage] = (
            input_messages_list[0] if input_messages_list else []
        )
        streamed_output = self._stream_buffer_manager.pop_streamed_text(
            str(request_information.request_id),
        )
        event_name: str = event.get("name", "unknown")
        event_time: str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        content_text = ""
        for message_number, input_message in enumerate(input_messages):
            name_suffix = f" ({input_message.name})" if input_message.name else ""
            content_text += f"--- Message {message_number + 1} by {input_message.type}{name_suffix} | event: {event_name} | {event_time} ---\n"
            content_text += f"{format_message_content(input_message.content)}\n"
            if isinstance(input_message, AIMessage) and input_message.tool_calls:
                for tool_call in input_message.tool_calls:
                    content_text += f"  Tool Call: {tool_call.get('name', 'unknown')}({json.dumps(tool_call.get('args', {}), default=str)})\n"
        if streamed_output:
            content_text += "--- Streamed assistant output ---\n"
            content_text += f"{streamed_output}\n"

        write_result: (
            DebugFileWriteResult | None
        ) = await self.debug_file_writer.write_to_file_async(
            file_name="messages",
            content=content_text,
            user_id=request_information.user_id,
        )
        if write_result and write_result.file_url:
            message_content_text: str = f"\n\n[Click to download full messages log]({write_result.file_url})\n\n"
            yield chat_request_wrapper.create_debug_sse_message(
                request_id=request_information.request_id,
                content=message_content_text,
                usage_metadata=None,
                source="on_chat_model_end",
            )
        elif content_text:
            collapsed_text: str = (
                f"\n\n<details>\n<summary>Messages log</summary>\n\n"
                f"```\n{content_text}\n```\n\n"
                f"</details>\n\n"
            )
            yield chat_request_wrapper.create_debug_sse_message(
                request_id=request_information.request_id,
                content=collapsed_text,
                usage_metadata=None,
                source="on_chat_model_end",
            )

    async def _handle_on_custom_event(
        self,
        *,
        event: StandardStreamEvent | CustomStreamEvent,
        chat_request_wrapper: ChatRequestWrapper,
        request_information: RequestInformation,
    ) -> AsyncGenerator[str, None]:
        name = event.get("name")
        if name == "mcp_task_progress":
            data: Dict[str, Any] = dict(event.get("data", {}))
            chunk = chat_request_wrapper.create_task_progress_sse_event(
                request_id=request_information.request_id,
                task_id=data.get("task_id", ""),
                status=data.get("status", ""),
                message=data.get("message"),
            )
            if chunk:
                yield chunk
        else:
            logger.debug("Skipped custom event: %s", name)

    async def _handle_non_text_content_debug(
        self,
        *,
        chat_request_wrapper: ChatRequestWrapper,
        request_information: RequestInformation,
        non_text_blocks: list[dict[str, Any]],
    ) -> AsyncGenerator[str | None, None]:
        if not non_text_blocks:
            return
        for block in non_text_blocks:
            block_type = block.get("type", "unknown")
            if block_type in ("reasoning_content", "reasoning"):
                reasoning_text = extract_reasoning_text(block)
                if reasoning_text:
                    content_text = (
                        f"\n\n<details>\n<summary>Reasoning</summary>\n\n"
                        f"{reasoning_text}\n\n"
                        f"</details>\n\n"
                    )
                    message = chat_request_wrapper.create_debug_sse_message(
                        request_id=request_information.request_id,
                        content=content_text,
                        usage_metadata=None,
                        source="on_chat_model_stream",
                    )
                    if message:
                        yield message
```

- [ ] **Step 2: Update `tests/converters/test_streaming_manager.py`**

Replace the contents of `tests/converters/test_streaming_manager.py` with:

```python
from collections.abc import Callable
from typing import Any, Optional, cast
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables.schema import CustomStreamEvent, StandardStreamEvent

from languagemodelcommon.converters.stream_buffer import StreamBufferManager
from languagemodelcommon.converters.streaming_manager import LangGraphStreamingManager
from languagemodelcommon.converters.tool_event_handlers import ToolEventHandler
from languagemodelcommon.file_managers.file_writer import FileWriter
from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)
from languagemodelcommon.utilities.token_reducer.token_reducer import TokenReducer
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.request_information import RequestInformation
from languagemodelcommon.utilities.tool_display_name_mapper import ToolDisplayNameMapper


class _FakeChatRequestWrapper:
    def __init__(self, *, enable_debug_logging: bool) -> None:
        self.enable_debug_logging = enable_debug_logging

    def create_sse_message(
        self,
        *,
        request_id: str,
        content: str | None,
        usage_metadata: Optional[dict[str, Any]],
        source: str,
    ) -> str:
        return content or ""

    def create_debug_sse_message(
        self,
        *,
        request_id: str,
        content: str | None,
        usage_metadata: Optional[dict[str, Any]],
        source: str,
    ) -> str | None:
        return content

    def create_final_sse_message(
        self,
        *,
        request_id: str,
        usage_metadata: Optional[dict[str, Any]],
        source: str,
    ) -> str:
        return "final"


@pytest.fixture()
def streaming_manager_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], LangGraphStreamingManager]:
    def _factory() -> LangGraphStreamingManager:
        monkeypatch.setenv("BUFFER_FLUSH_INTERVAL_SECONDS", "10.0")
        monkeypatch.setenv("WRITE_TOOL_OUTPUT_TO_FILE", "false")
        environment_variables = LanguageModelCommonEnvironmentVariables()
        mock_file_writer = AsyncMock(spec=FileWriter)
        mock_file_writer.write_to_file_async = AsyncMock(return_value=None)
        stream_buffer_manager = StreamBufferManager(
            flush_interval_seconds=10.0,
            enabled=True,
        )
        tool_event_handler = ToolEventHandler(
            debug_file_writer=mock_file_writer,
            environment_variables=environment_variables,
            tool_display_name_mapper=ToolDisplayNameMapper(),
            stream_buffer_manager=stream_buffer_manager,
        )
        return LangGraphStreamingManager(
            token_reducer=TokenReducer(),
            environment_variables=environment_variables,
            debug_file_writer=mock_file_writer,
            tool_event_handler=tool_event_handler,
            stream_buffer_manager=stream_buffer_manager,
        )

    return _factory


@pytest.mark.asyncio
async def test_chat_model_end_includes_streamed_text_when_debug_enabled(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-1")
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=True),
    )

    stream_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content="Hello world")},
        },
    )
    streamed_chunks = [
        chunk
        async for chunk in manager._handle_on_chat_model_stream(
            event=stream_event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
        )
    ]
    assert streamed_chunks == []

    end_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_end",
            "data": {"input": {"messages": []}},
        },
    )
    debug_chunks = [
        chunk
        async for chunk in manager._handle_on_chat_model_end(
            event=end_event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
        )
        if chunk is not None
    ]

    assert len(debug_chunks) == 1
    assert "Streamed assistant output" in debug_chunks[0]
    assert "Hello world" in debug_chunks[0]


@pytest.mark.asyncio
async def test_chain_end_clears_streamed_text_when_chat_model_end_not_called(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-2")
    chat_request_wrapper = cast(
        ChatRequestWrapper,
        _FakeChatRequestWrapper(enable_debug_logging=False),
    )

    stream_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content="partial response")},
        },
    )
    _ = [
        chunk
        async for chunk in manager._handle_on_chat_model_stream(
            event=stream_event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
        )
    ]

    chain_end_event: StandardStreamEvent | CustomStreamEvent = cast(
        StandardStreamEvent,
        {
            "event": "on_chain_end",
            "data": {},
        },
    )
    _ = [
        chunk
        async for chunk in manager._handle_on_chain_end(
            event=chain_end_event,
            chat_request_wrapper=chat_request_wrapper,
            request_information=request_information,
        )
    ]

    # Verify fragments were cleared
    assert manager._stream_buffer_manager.pop_streamed_text("req-2") is None
```

- [ ] **Step 3: Run all tests to verify everything passes**

Run: `cd /Users/imranqureshi/git/language-model-common && uv run pytest tests/converters/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/imranqureshi/git/language-model-common
git add languagemodelcommon/converters/streaming_manager.py tests/converters/test_streaming_manager.py
git commit -m "Refactor streaming_manager to delegate to extracted modules"
```

---

### Task 5: Update container registrations

**Files:**
- Modify: `languagemodelcommon/container/container_factory.py`

- [ ] **Step 1: Update container_factory.py with new registrations**

Add imports at the top of `languagemodelcommon/container/container_factory.py`:

```python
from languagemodelcommon.converters.stream_buffer import StreamBufferManager
from languagemodelcommon.converters.tool_event_handlers import ToolEventHandler
```

Add registrations before the `LangGraphStreamingManager` registration (around line 146):

```python
container.singleton(
    StreamBufferManager,
    lambda c: StreamBufferManager(
        flush_interval_seconds=c.resolve(
            LanguageModelCommonEnvironmentVariables
        ).streaming_buffer_flush_interval_seconds,
        enabled=c.resolve(
            LanguageModelCommonEnvironmentVariables
        ).enable_streaming_buffering,
    ),
)

container.singleton(
    ToolEventHandler,
    lambda c: ToolEventHandler(
        debug_file_writer=c.resolve(FileWriter),
        environment_variables=c.resolve(LanguageModelCommonEnvironmentVariables),
        tool_display_name_mapper=c.resolve(ToolDisplayNameMapper),
        stream_buffer_manager=c.resolve(StreamBufferManager),
    ),
)
```

Update the existing `LangGraphStreamingManager` registration to:

```python
container.singleton(
    LangGraphStreamingManager,
    lambda c: LangGraphStreamingManager(
        environment_variables=c.resolve(
            LanguageModelCommonEnvironmentVariables
        ),
        debug_file_writer=c.resolve(FileWriter),
        token_reducer=c.resolve(TokenReducer),
        tool_event_handler=c.resolve(ToolEventHandler),
        stream_buffer_manager=c.resolve(StreamBufferManager),
    ),
)
```

Remove the `ToolDisplayNameMapper` import from the `LangGraphStreamingManager` registration (it's no longer a direct dependency of the manager).

- [ ] **Step 2: Run the full test suite**

Run: `cd /Users/imranqureshi/git/language-model-common && uv run pytest tests/ -v --timeout=60`
Expected: All tests PASS

- [ ] **Step 3: Run type checking**

Run: `cd /Users/imranqureshi/git/language-model-common && uv run mypy languagemodelcommon/converters/ languagemodelcommon/container/container_factory.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
cd /Users/imranqureshi/git/language-model-common
git add languagemodelcommon/container/container_factory.py
git commit -m "Update container registrations for streaming manager decomposition"
```

---

### Task 6: Update `langgraph_to_openai_converter.py` references

**Files:**
- Modify: `languagemodelcommon/converters/langgraph_to_openai_converter.py`

- [ ] **Step 1: Check if converter accesses internal state that moved**

The converter accesses `self.streaming_manager._pop_streamed_text()` (line 234) and `self.streaming_manager.debug_file_writer` (line 243). Both still exist on the refactored `LangGraphStreamingManager`:
- `debug_file_writer` remains a public attribute on the orchestrator
- `_pop_streamed_text` moved to `StreamBufferManager`, but the orchestrator exposes `_stream_buffer_manager`

Update `langgraph_to_openai_converter.py` line 234 from:
```python
streamed_output = self.streaming_manager._pop_streamed_text(
    request_id=str(request_information.request_id),
)
```
to:
```python
streamed_output = self.streaming_manager._stream_buffer_manager.pop_streamed_text(
    str(request_information.request_id),
)
```

- [ ] **Step 2: Run the full test suite**

Run: `cd /Users/imranqureshi/git/language-model-common && uv run pytest tests/ -v --timeout=60`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/imranqureshi/git/language-model-common
git add languagemodelcommon/converters/langgraph_to_openai_converter.py
git commit -m "Update converter to use StreamBufferManager directly"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/imranqureshi/git/language-model-common && uv run pytest tests/ -v --timeout=60`
Expected: All tests PASS

- [ ] **Step 2: Run linting**

Run: `cd /Users/imranqureshi/git/language-model-common && uv run ruff check languagemodelcommon/converters/ && uv run ruff format --check languagemodelcommon/converters/`
Expected: No errors

- [ ] **Step 3: Run type checking on all modified files**

Run: `cd /Users/imranqureshi/git/language-model-common && uv run mypy languagemodelcommon/converters/ languagemodelcommon/container/container_factory.py`
Expected: No errors

- [ ] **Step 4: Verify line counts**

Run: `wc -l languagemodelcommon/converters/streaming_manager.py languagemodelcommon/converters/tool_event_handlers.py languagemodelcommon/converters/stream_buffer.py languagemodelcommon/converters/streaming_formatters.py`
Expected: streaming_manager.py ~280 lines, tool_event_handlers.py ~280 lines, stream_buffer.py ~80 lines, streaming_formatters.py ~100 lines
