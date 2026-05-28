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
from languagemodelcommon.converters.stream_debug_output_manager import (
    StreamDebugOutputManager,
    StreamedOutput,
)
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
        stream_buffer_manager: StreamBufferManager | None = None,
        stream_debug_output_manager: StreamDebugOutputManager | None = None,
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

        self._static_stream_buffer_manager = stream_buffer_manager
        self._static_stream_debug_output_manager = stream_debug_output_manager

    @property
    def _stream_buffer_manager(self) -> StreamBufferManager:
        if self._static_stream_buffer_manager is not None:
            return self._static_stream_buffer_manager
        from languagemodelcommon.context.request_context import (
            get_stream_buffer_manager,
        )

        return get_stream_buffer_manager()

    @property
    def _stream_debug_output_manager(self) -> StreamDebugOutputManager:
        if self._static_stream_debug_output_manager is not None:
            return self._static_stream_debug_output_manager
        from languagemodelcommon.context.request_context import (
            get_stream_debug_output_manager,
        )

        return get_stream_debug_output_manager()

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
                content=content,
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
                    self._stream_debug_output_manager.append_fragment(
                        text=content_text,
                    )
                    buffered_chunk = await self._stream_buffer_manager.buffer_content(
                        content_text=content_text,
                    )
                    if buffered_chunk:
                        yield chat_request_wrapper.create_sse_message(
                            request_id=request_information.request_id,
                            content=buffered_chunk,
                            usage_metadata=chunk.usage_metadata if chunk else None,
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
        self._stream_debug_output_manager.clear()

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
        streamed_output_record: StreamedOutput | None = (
            self._stream_debug_output_manager.pop_streamed_output()
        )
        streamed_output: str | None = (
            "".join(streamed_output_record.text_fragments)
            if streamed_output_record and streamed_output_record.text_fragments
            else None
        )
        content_text = ""
        for message_number, input_message in enumerate(input_messages):
            name_suffix = f" ({input_message.name})" if input_message.name else ""
            content_text += f"--- Message {message_number + 1} by {input_message.type}{name_suffix} ---\n"
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
