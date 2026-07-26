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
from languagemodelcommon.converters.stream_context_mixin import StreamContextMixin
from languagemodelcommon.converters.stream_debug_output_manager import (
    StreamDebugOutputManager,
)
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

# Cap on the tool result/error text forwarded to the response.output_item.done
# SSE event, so a tool that echoes back large content (e.g. a full skill body)
# doesn't bloat every trace event; full output remains available via the
# existing debug-file-writer path for cases that need it.
TOOL_END_OUTPUT_TRACE_MAX_CHARS = 2000


def _truncate_for_trace(
    text: str, *, max_chars: int = TOOL_END_OUTPUT_TRACE_MAX_CHARS
) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[truncated {len(text) - max_chars} chars]"


class ToolEventHandler(StreamContextMixin):
    def __init__(
        self,
        *,
        debug_file_writer: FileWriter,
        environment_variables: LanguageModelCommonEnvironmentVariables,
        tool_display_name_mapper: ToolDisplayNameMapper,
        stream_buffer_manager: StreamBufferManager | None = None,
        stream_debug_output_manager: StreamDebugOutputManager | None = None,
    ) -> None:
        if debug_file_writer is None:
            raise ValueError("debug_file_writer must not be None")
        if environment_variables is None:
            raise ValueError("environment_variables must not be None")
        if tool_display_name_mapper is None:
            raise ValueError("tool_display_name_mapper must not be None")

        self._debug_file_writer = debug_file_writer
        self._environment_variables = environment_variables
        self._tool_display_name_mapper = tool_display_name_mapper
        self._static_stream_buffer_manager = stream_buffer_manager
        self._static_stream_debug_output_manager = stream_debug_output_manager

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
        tool_key: str = make_tool_key(tool_name=tool_name, tool_input=tool_input)
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
                self._stream_debug_output_manager.append_fragment(
                    text=f"\n--- Tool Call: {tool_name} ---\n{json.dumps(tool_input_display, indent=2, default=str)}\n",
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

            tool_key: str = make_tool_key(tool_name=tool_name, tool_input=tool_input)
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

            tool_message_content: str = (
                convert_message_content_into_string(tool_message=tool_message)
                if tool_message
                else ""
            )
            artifact: Optional[Any] = tool_message.artifact
            # tool_message.status only flips to "error" for an uncaught exception;
            # meta-tools like call_tool catch their own failures and report them via
            # the artifact instead (see CallToolTool), so check both signals.
            is_error: bool = tool_message.status == "error" or (
                isinstance(artifact, dict) and artifact.get("is_error") is True
            )

            tool_end_event = chat_request_wrapper.create_tool_end_sse_event(
                request_id=request_information.request_id,
                tool_name=tool_name,
                tool_input=tool_input,
                runtime_seconds=runtime_seconds,
                output=_truncate_for_trace(tool_message_content)
                if tool_message_content
                else None,
                is_error=is_error,
            )
            if tool_end_event:
                yield tool_end_event

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
                    self._stream_debug_output_manager.append_fragment(
                        text=f"\n--- Tool Output: {tool_name} ({runtime_str}) ---\n{tool_message_or_artifact_content}\n",
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
        tool_key: str = make_tool_key(tool_name=tool_name, tool_input=tool_input)
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
            self._stream_debug_output_manager.append_fragment(
                text=f"\n--- Tool Error: {tool_name} ({runtime_str}) ---\n{error_message}\n",
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
