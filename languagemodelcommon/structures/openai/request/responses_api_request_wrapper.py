import json
import logging
import time
from datetime import datetime, UTC
from typing import (
    AsyncIterable,
    Dict,
    Literal,
    Union,
    override,
    Optional,
    List,
    Any,
    cast,
)

from langchain_core.messages import AnyMessage
from langchain_core.messages.ai import UsageMetadata
from openai.types.responses import (
    ResponseInputParam,
    EasyInputMessageParam,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseCompletedEvent,
    Response,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseOutputRefusal,
    ResponseCreatedEvent,
    ResponseUsage,
)

from languagemodelcommon.configs.schemas.config_schema import AgentConfig
from languagemodelcommon.schema.openai.responses import ResponsesRequest
from languagemodelcommon.structures.openai.message.chat_message_wrapper import (
    ChatMessageWrapper,
)
from languagemodelcommon.structures.openai.message.responses_api_message_wrapper import (
    ResponsesApiMessageWrapper,
)
from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.LLM)


class ResponsesApiRequestWrapper(ChatRequestWrapper):
    def __init__(
        self, *, chat_request: ResponsesRequest, enable_debug_logging: bool
    ) -> None:
        """
        Wraps an OpenAI /responses API request and provides a unified interface so the code can use it

        """
        self.request: ResponsesRequest = chat_request

        self._messages: list[ChatMessageWrapper] = self.convert_from_responses_input(
            input_=self.request.input
        )
        self._enable_debug_logging: bool = enable_debug_logging
        self._apply_debug_prefix_toggle()

    def _apply_debug_prefix_toggle(self) -> None:
        debug_prefix = "DEBUG:"
        for index, message in enumerate(self._messages):
            if message.role != "user":
                continue
            content = message.content
            if not isinstance(content, str):
                continue
            if not content.startswith(debug_prefix):
                continue
            self._enable_debug_logging = True
            stripped_content = content[len(debug_prefix) :].lstrip()
            if isinstance(message, ResponsesApiMessageWrapper):
                self._set_message_input_text(message=message, content=stripped_content)
            self._update_request_input(index=index, content=stripped_content)
            break

    def _set_message_input_text(
        self, *, message: ResponsesApiMessageWrapper, content: str
    ) -> None:
        if isinstance(message.input_, dict):
            if message.input_.get("role") != "user":
                return
            input_item = cast(dict[str, Any], message.input_)
            if "content" in input_item:
                input_item["content"] = content
            elif "text" in input_item:
                input_item["text"] = content
        elif hasattr(message.input_, "content"):
            setattr(message.input_, "content", content)
        elif hasattr(message.input_, "text"):
            setattr(message.input_, "text", content)

    def _update_request_input(self, *, index: int, content: str) -> None:
        if isinstance(self.request.input, str):
            self.request.input = content
            return
        if not isinstance(self.request.input, list):
            return
        if index < 0 or index >= len(self.request.input):
            return
        input_item = self.request.input[index]
        if isinstance(input_item, dict):
            if input_item.get("role") != "user":
                return
            item_dict = cast(dict[str, Any], input_item)
            if "content" in item_dict:
                item_dict["content"] = content
            elif "text" in item_dict:
                item_dict["text"] = content
        elif hasattr(input_item, "content"):
            setattr(input_item, "content", content)
        elif hasattr(input_item, "text"):
            setattr(input_item, "text", content)

    @staticmethod
    def convert_from_responses_input(
        *, input_: Union[str, ResponseInputParam]
    ) -> list[ChatMessageWrapper]:
        if isinstance(input_, str):
            return [
                ResponsesApiMessageWrapper(
                    input_=EasyInputMessageParam(role="user", content=input_)
                )
            ]
        elif isinstance(input_, list):
            return [ResponsesApiMessageWrapper(input_=item) for item in input_]
        else:
            raise TypeError(
                f"input_ must be a str or list, got {type(input_).__name__}: {input_!r}"
            )

    @property
    @override
    def model(self) -> str:
        return self.request.model

    @property
    @override
    def messages(self) -> list[ChatMessageWrapper]:
        return self._messages

    @messages.setter
    def messages(self, value: list[ChatMessageWrapper]) -> None:
        self._messages = value

    @override
    def append_message(self, *, message: ChatMessageWrapper) -> None:
        self._messages.append(message)

    @override
    def create_system_message(self, *, content: str) -> ChatMessageWrapper:
        return ResponsesApiMessageWrapper.create_system_message(content=content)

    @override
    @property
    def stream(self) -> Literal[False, True] | None | bool:
        return self.request.stream

    @override
    @property
    def response_format(self) -> Literal["text", "json_object", "json_schema"] | None:
        return "json_object"  # in case of ResponsesRequest, we always use JSON object format

    @override
    @property
    def response_json_schema(self) -> str | None:
        return None  # Not applicable for ResponsesRequest

    @override
    def create_first_sse_message(self, *, request_id: str, source: str) -> str:
        # For the first SSE message, we can include any initial content if needed. Here we just return an empty message to indicate the start of the stream.
        parallel_tool_calls = self.effective_parallel_tool_calls()
        message: ResponseCreatedEvent = ResponseCreatedEvent(
            response=Response(
                id=request_id,
                model=self.model,
                status="in_progress",
                created_at=time.time(),
                object="response",
                output=[],
                parallel_tool_calls=(
                    parallel_tool_calls if parallel_tool_calls is not None else False
                ),
                tools=[],
                tool_choice="auto",
            ),
            type="response.created",
            sequence_number=0,
        )
        return f"data: {message.model_dump_json()}\n\n"

    @override
    def create_sse_message(
        self,
        *,
        request_id: str,
        content: str | None,
        usage_metadata: UsageMetadata | None,
        source: str,
    ) -> str:
        # Format a single SSE message chunk for streaming
        if content is None:
            return ""

        logger.debug(
            "Creating SSE message for request_id: %s from source: %s",
            request_id,
            source,
        )

        message: ResponseTextDeltaEvent = ResponseTextDeltaEvent(
            item_id=request_id,
            content_index=0,
            output_index=len(self._messages),
            delta=content,
            type="response.output_text.delta",
            sequence_number=len(self._messages),
            logprobs=[],
        )
        return f"data: {message.model_dump_json()}\n\n"

    def create_debug_sse_message(
        self,
        *,
        request_id: str,
        content: str | None,
        usage_metadata: UsageMetadata | None,
        source: str,
    ) -> str | None:

        return (
            self.create_sse_message(
                request_id=request_id,
                content=content,
                usage_metadata=usage_metadata,
                source=source,
            )
            if self._enable_debug_logging
            else None
        )

    @override
    def create_tool_start_sse_event(
        self,
        *,
        request_id: str,
        tool_name: str,
        tool_input: Dict[str, Any] | None,
    ) -> str | None:
        """Emit a ``response.output_item.added`` event with a ``function_call`` item."""
        event: Dict[str, Any] = {
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": len(self._messages),
            "item": {
                "type": "function_call",
                "id": f"fc_{request_id}_{tool_name}",
                "call_id": f"call_{request_id}_{tool_name}",
                "name": tool_name,
                "arguments": json.dumps(tool_input) if tool_input else "",
                "status": "in_progress",
            },
        }
        return f"data: {json.dumps(event)}\n\n"

    @override
    def create_tool_end_sse_event(
        self,
        *,
        request_id: str,
        tool_name: str,
        tool_input: Dict[str, Any] | None,
        runtime_seconds: float | None,
    ) -> str | None:
        """Emit a ``response.output_item.done`` event with a ``function_call`` item."""
        event: Dict[str, Any] = {
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": len(self._messages),
            "item": {
                "type": "function_call",
                "id": f"fc_{request_id}_{tool_name}",
                "call_id": f"call_{request_id}_{tool_name}",
                "name": tool_name,
                "arguments": json.dumps(tool_input) if tool_input else "",
                "status": "completed",
            },
        }
        return f"data: {json.dumps(event)}\n\n"

    @staticmethod
    def _convert_usage_to_response_usage(
        usages: list[UsageMetadata],
    ) -> ResponseUsage:
        """Convert LangChain UsageMetadata list to OpenAI ResponseUsage."""
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        for u in usages:
            input_tokens += u.get("input_tokens", 0)
            output_tokens += u.get("output_tokens", 0)
            total_tokens += u.get("total_tokens", 0)
        return ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    @override
    def create_final_sse_message(
        self, *, request_id: str, usage_metadata: UsageMetadata | None, source: str
    ) -> str:
        logger.debug(
            "Creating final SSE message for request_id: %s from source: %s",
            request_id,
            source,
        )
        # Build usage from accumulated metadata
        response_usage: ResponseUsage | None = None
        if usage_metadata:
            response_usage = self._convert_usage_to_response_usage([usage_metadata])

        parallel_tool_calls = self.effective_parallel_tool_calls()
        # Emit response.completed event with the full Response including usage
        completed_event: ResponseCompletedEvent = ResponseCompletedEvent(
            response=Response(
                id=request_id,
                model=self.model,
                status="completed",
                created_at=time.time(),
                object="response",
                output=[],
                parallel_tool_calls=(
                    parallel_tool_calls if parallel_tool_calls is not None else False
                ),
                tools=[],
                tool_choice="auto",
                usage=response_usage,
            ),
            type="response.completed",
            sequence_number=len(self._messages) + 1,
        )
        # Emit text done + completed event + [DONE]
        text_done: ResponseTextDoneEvent = ResponseTextDoneEvent(
            item_id=request_id,
            content_index=0,
            output_index=len(self._messages),
            type="response.output_text.done",
            sequence_number=len(self._messages),
            logprobs=[],
            text="",
        )
        return (
            f"data: {text_done.model_dump_json()}\n\n"
            f"data: {completed_event.model_dump_json()}\n\n"
            f"data: [DONE]\n\n"
        )

    @staticmethod
    def convert_message_content(
        input_content: str | list[str | dict[str, Any]],
    ) -> list[ResponseOutputText | ResponseOutputRefusal]:
        if isinstance(input_content, str):
            return [
                ResponseOutputText(
                    text=input_content, type="output_text", annotations=[]
                )
            ]
        elif isinstance(input_content, list):
            output_texts: list[ResponseOutputText | ResponseOutputRefusal] = []
            for item in input_content:
                if isinstance(item, str):
                    output_texts.append(
                        ResponseOutputText(
                            text=item, type="output_text", annotations=[]
                        )
                    )
                elif isinstance(item, dict):
                    output_texts.append(ResponseOutputText(**item))
            return output_texts
        else:
            return []

    @override
    def create_non_streaming_response(
        self,
        *,
        request_id: str,
        json_output_requested: Optional[bool],
        responses: List[AnyMessage],
    ) -> dict[str, Any]:
        # Build a non-streaming response dict
        output: list[ResponseOutputItem] = []
        for idx, msg in enumerate(responses):
            content: str | list[str | dict[str, Any]] = msg.content
            output.append(
                ResponseOutputMessage(
                    id=str(idx),
                    content=self.convert_message_content(input_content=content),
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        # Aggregate usage from all response messages
        usage_list: list[UsageMetadata] = [
            m.usage_metadata
            for m in responses
            if hasattr(m, "usage_metadata") and m.usage_metadata
        ]
        response_usage: ResponseUsage | None = (
            self._convert_usage_to_response_usage(usage_list) if usage_list else None
        )

        parallel_tool_calls = self.effective_parallel_tool_calls()
        response: Response = Response(
            id=request_id,
            created_at=datetime.now(UTC).timestamp(),
            output=output,
            model=self.model,
            object="response",
            parallel_tool_calls=(
                parallel_tool_calls if parallel_tool_calls is not None else False
            ),
            tools=[],
            tool_choice="auto",
            usage=response_usage,
        )
        return response.model_dump(mode="json")

    @override
    def to_dict(self) -> dict[str, Any]:
        return self.request.model_dump(mode="json")

    @staticmethod
    def extract_mcp_agent_configs(
        tools_in_request: list[dict[str, Any]],
    ) -> list[AgentConfig]:
        """
        Extract AgentConfig objects for MCP tools from the tools_in_request list.
        """
        return [
            AgentConfig(
                url=tool["server_url"],
                name=tool["server_label"],
                tools=",".join(
                    [
                        t["name"] if isinstance(t, dict) and "name" in t else str(t)
                        for t in tool["allowed_tools"]
                    ]
                )
                if isinstance(tool["allowed_tools"], (list, tuple))
                else "",
                headers=tool.get("headers"),
                auth="headers",
            )
            for tool in tools_in_request
            if tool["type"] == "mcp"
            and "server_url" in tool
            and "server_label" in tool
            and "allowed_tools" in tool
        ]

    @override
    def get_tools(self) -> list[AgentConfig]:
        """
        Return a list of tools passed in the request.
        """
        tools_in_request: list[dict[str, Any]] | None = self.request.tools
        if tools_in_request is None:
            return []
        return self.extract_mcp_agent_configs(tools_in_request)

    @override
    def stream_response(
        self,
        *,
        request_id: str,
        response_messages1: List[AnyMessage],
    ) -> AsyncIterable[str]:
        """Streams the response messages as Server-Sent Events (SSE) in the Responses API format."""
        from languagemodelcommon.utilities.chat_message_helpers import (
            convert_message_content_to_string,
        )

        async def response_stream() -> AsyncIterable[str]:
            yield self.create_first_sse_message(
                request_id=request_id,
                source="stream_response",
            )

            # Collect usage from all messages for the final event
            accumulated_usage: UsageMetadata | None = None
            for response_message in response_messages1:
                message_content: str = convert_message_content_to_string(
                    response_message.content
                )
                if message_content:
                    delta_message: str = self.create_sse_message(
                        request_id=request_id,
                        content=message_content + "\n",
                        usage_metadata=None,
                        source="stream_response",
                    )
                    if delta_message:
                        yield delta_message
                # Accumulate usage from each message
                if (
                    hasattr(response_message, "usage_metadata")
                    and response_message.usage_metadata
                ):
                    if accumulated_usage is None:
                        accumulated_usage = dict(response_message.usage_metadata)
                    else:
                        accumulated_usage["input_tokens"] += (
                            response_message.usage_metadata.get("input_tokens", 0)
                        )
                        accumulated_usage["output_tokens"] += (
                            response_message.usage_metadata.get("output_tokens", 0)
                        )
                        accumulated_usage["total_tokens"] += (
                            response_message.usage_metadata.get("total_tokens", 0)
                        )

            yield self.create_final_sse_message(
                request_id=request_id,
                usage_metadata=accumulated_usage,
                source="stream_response",
            )

        return response_stream()

    @override
    @property
    def instructions(self) -> Optional[str]:
        """ChatCompletion API does not have a separate instructions field, so we return None."""
        return self.request.instructions

    @override
    @property
    def previous_response_id(self) -> Optional[str]:
        """Responses API does have a previous_response_id."""
        return self.request.previous_response_id

    @override
    @property
    def store(self) -> Optional[bool]:
        """Responses API does have a store parameter."""
        return self.request.store

    @override
    @property
    def user_input(self) -> Optional[str]:
        """Extract the user input from the request."""
        if self.request.input is not None:
            if isinstance(self.request.input, str):
                return self.request.input
            elif isinstance(self.request.input, list):
                text_parts: list[str] = []
                for part in self.request.input:
                    if isinstance(part, dict):
                        text_value = part.get("text")
                        if isinstance(text_value, str):
                            text_parts.append(text_value)
                            continue
                        content_value = part.get("content")
                        if isinstance(content_value, str):
                            text_parts.append(content_value)
                            continue
                return " ".join(text_parts)
        return ""

    @override
    @property
    def metadata(self) -> Optional[dict[str, Any]]:
        """Responses API does have a metadata field."""
        return self.request.metadata

    @override
    @property
    def max_tokens(self) -> Optional[int]:
        """Responses API does have a max_output_tokens parameter, but it's not the same as max_tokens in ChatCompletion API, so we return None here to avoid confusion."""
        return None

    @override
    @property
    def max_output_tokens(self) -> Optional[int]:
        """Return the max_output_tokens parameter from the request, which is specific to the Responses API."""
        return self.request.max_output_tokens

    @override
    @property
    def temperature(self) -> Optional[float]:
        """Return the temperature parameter from the request, which is used in both ChatCompletion and Responses API."""
        return self.request.temperature

    @override
    @property
    def parallel_tool_calls(self) -> Optional[bool]:
        return self.request.parallel_tool_calls

    @override
    @property
    def enable_debug_logging(self) -> bool:
        """Return whether debug logging is enabled for this request."""
        return self._enable_debug_logging
