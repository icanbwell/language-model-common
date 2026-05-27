# Streaming Manager Decomposition

**Date:** 2026-05-27  
**Status:** Proposed  
**Driver:** Code health — file exceeds 800 lines with 8 embedded responsibilities

## Context

`languagemodelcommon/converters/streaming_manager.py` is 1,058 lines. `LangGraphStreamingManager` handles event routing, tool lifecycle management, stream buffering, debug file writing, content formatting, error handling, and custom event dispatch. This makes the file hard to navigate and increases the risk surface of any change.

## Decision

Conservative decomposition into 4 modules grouped by responsibility. The public API (`handle_langchain_event`) is unchanged — callers are unaffected.

## New Module Structure

```
languagemodelcommon/converters/
├── streaming_manager.py          (~250 lines) — orchestrator + chat model handlers
├── tool_event_handlers.py        (~350 lines) — tool start/end/error + MCP embed logic
├── stream_buffer.py              (~100 lines) — buffer state + flush logic + text fragments
├── streaming_formatters.py       (~150 lines) — stateless parsing/formatting utilities
├── langgraph_to_openai_converter.py  (unchanged)
├── streaming_tool_node.py            (unchanged)
└── __init__.py
```

## Module Responsibilities

### `streaming_manager.py` (Orchestrator)

Retains:
- `LangGraphStreamingManager.__init__` — constructor with DI (adds `ToolEventHandler` and `StreamBufferManager` as injected deps)
- `handle_langchain_event` — event dispatch (match-case router)
- `_handle_on_chat_model_stream` — token-by-token text extraction + buffering delegation
- `_handle_on_chain_end` — final SSE + force-flush
- `_handle_on_chat_model_start` — debug SSE for input messages (currently disabled)
- `_handle_on_chat_model_end` — debug SSE with input/output + file write
- `_handle_on_custom_event` — custom event dispatch
- `_handle_non_text_content_debug` — reasoning block debug output

Delegates to:
- `ToolEventHandler` for tool start/end/error events
- `StreamBufferManager` for buffering and text fragment tracking
- `streaming_formatters` for pure formatting functions

### `tool_event_handlers.py`

```python
class ToolEventHandler:
    def __init__(
        self,
        debug_file_writer: FileWriter,
        environment_variables: LanguageModelCommonEnvironmentVariables,
        tool_display_name_mapper: ToolDisplayNameMapper,
    ) -> None: ...

    async def handle_tool_start(
        self,
        event: StandardStreamEvent,
        chat_request_wrapper: ChatRequestWrapper,
        request_information: RequestInformation,
    ) -> AsyncGenerator[str, None]: ...

    async def handle_tool_end(
        self,
        event: StandardStreamEvent,
        chat_request_wrapper: ChatRequestWrapper,
        request_information: RequestInformation,
    ) -> AsyncGenerator[str, None]: ...

    async def handle_tool_error(
        self,
        event: StandardStreamEvent,
        chat_request_wrapper: ChatRequestWrapper,
        request_information: RequestInformation,
    ) -> AsyncGenerator[str, None]: ...
```

Internal state:
- `_tool_start_times: dict[str, float]` — moved from orchestrator

Private helpers (moved from orchestrator):
- `_format_text_resource_contents` — extracts JSON fields for display
- MCP app embed detection and metadata emission

### `stream_buffer.py`

```python
@dataclass
class StreamBuffer:
    chunks: list[str]
    last_flush_ts: float

class StreamBufferManager:
    def __init__(
        self,
        flush_interval_seconds: float,
        enabled: bool,
    ) -> None: ...

    async def buffer_content(
        self,
        request_id: str,
        content: str,
        force_flush: bool = False,
    ) -> AsyncGenerator[str, None]: ...

    def append_streamed_text_fragment(self, request_id: str, text: str) -> None: ...
    def pop_streamed_text(self, request_id: str) -> list[str]: ...
    def clear_request_streamed_text(self, request_id: str) -> None: ...
```

Internal state:
- `_stream_buffers: dict[str, StreamBuffer]` — per-request buffer
- `_streamed_text_fragments: dict[str, list[str]]` — per-request debug fragments

### `streaming_formatters.py`

Stateless module-level functions (no class):

```python
def make_tool_key(tool_name: str, tool_input: dict) -> str: ...
def safe_json(text: str) -> dict | list | None: ...
def convert_message_content_into_string(message: ToolMessage) -> str: ...
def get_structured_content_from_tool_message(message: ToolMessage) -> dict | None: ...
def format_message_content(content: str | list) -> str: ...
def format_tool_input_labels(tool_input: dict) -> str: ...
def extract_reasoning_text(message: AIMessage | AIMessageChunk) -> str | None: ...
```

## Container Registration

In `container_factory.py`:

```python
# New registrations
container.register_singleton(StreamBufferManager, lambda c: StreamBufferManager(
    flush_interval_seconds=c.resolve(LanguageModelCommonEnvironmentVariables).streaming_buffer_flush_interval_seconds,
    enabled=c.resolve(LanguageModelCommonEnvironmentVariables).enable_streaming_buffering,
))

container.register_singleton(ToolEventHandler, lambda c: ToolEventHandler(
    debug_file_writer=c.resolve(FileWriter),
    environment_variables=c.resolve(LanguageModelCommonEnvironmentVariables),
    tool_display_name_mapper=c.resolve(ToolDisplayNameMapper),
))

# Updated registration
container.register_singleton(LangGraphStreamingManager, lambda c: LangGraphStreamingManager(
    token_reducer=c.resolve(TokenReducer),
    debug_file_writer=c.resolve(FileWriter),
    environment_variables=c.resolve(LanguageModelCommonEnvironmentVariables),
    tool_display_name_mapper=c.resolve(ToolDisplayNameMapper),
    tool_event_handler=c.resolve(ToolEventHandler),
    stream_buffer_manager=c.resolve(StreamBufferManager),
))
```

## Test Migration

| Current Test | New Location | Change |
|---|---|---|
| `test_buffer_flushes_on_newline` | `tests/converters/test_stream_buffer.py` | Tests `StreamBufferManager` directly |
| `test_buffer_flushes_after_interval` | `tests/converters/test_stream_buffer.py` | Tests `StreamBufferManager` directly |
| `test_buffer_is_disabled_when_buffering_env_flag_is_false` | `tests/converters/test_stream_buffer.py` | Tests `StreamBufferManager` directly |
| `test_chat_model_end_includes_streamed_text_when_debug_enabled` | `tests/converters/test_streaming_manager.py` | Stays; injects mock `StreamBufferManager` |
| `test_chain_end_clears_streamed_text_when_chat_model_end_not_called` | `tests/converters/test_streaming_manager.py` | Stays; injects mock `StreamBufferManager` |

New test files:
- `tests/converters/test_stream_buffer.py` — buffer logic in isolation
- `tests/converters/test_tool_event_handlers.py` — tool handlers with mocked file writer
- `tests/converters/test_streaming_formatters.py` — pure function tests

## Constraints

- Public API unchanged: `handle_langchain_event` signature and return type are preserved
- IoC pattern: all new classes injected via container, no inline instantiation
- Existing callers (`langgraph_to_openai_converter.py`) require zero changes
- No behavioral changes — this is a pure structural refactor
