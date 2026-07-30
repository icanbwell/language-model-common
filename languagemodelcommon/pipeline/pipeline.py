from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Protocol

from starlette.responses import StreamingResponse

from languagemodelcommon.pipeline.context import PipelineContext
from languagemodelcommon.pipeline.step import PipelineStep

logger = logging.getLogger(__name__)


class OutputStep(Protocol):
    def format_response(self, *, context: PipelineContext) -> Any: ...
    def stream_response(
        self, *, context: PipelineContext
    ) -> AsyncGenerator[str, None]: ...
    def format_error(self, *, context: PipelineContext, error: Exception) -> Any: ...


class Pipeline:
    """Runs an ordered list of steps against a shared PipelineContext."""

    def __init__(
        self,
        *,
        pre_execution_steps: list[PipelineStep],
        execution_step: PipelineStep,
        post_execution_steps: list[PipelineStep],
        output_step: OutputStep,
    ) -> None:
        self._pre_execution_steps = pre_execution_steps
        self._execution_step = execution_step
        self._post_execution_steps = post_execution_steps
        self._output_step = output_step

    async def run_non_streaming(self, *, context: PipelineContext) -> Any:
        try:
            for step in self._pre_execution_steps:
                await step.run(context=context)
            await self._execution_step.run(context=context)
            await self._drain_stream(context=context)
            for step in self._post_execution_steps:
                await step.run(context=context)
            return self._output_step.format_response(context=context)
        except Exception as e:
            return self._output_step.format_error(context=context, error=e)

    @staticmethod
    async def _drain_stream(*, context: PipelineContext) -> None:
        """Consume content_stream fully, populating accumulated_content and response_messages."""
        if context.content_stream is None:
            return
        import json as _json

        from langchain_core.messages import AIMessage

        async for chunk in context.content_stream:
            for line in chunk.strip().splitlines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    continue
                try:
                    event = _json.loads(payload)
                    if "delta" in event and isinstance(event["delta"], str):
                        context.accumulated_content += event["delta"]
                    elif (
                        "choices" in event
                        and event["choices"]
                        and "delta" in event["choices"][0]
                    ):
                        delta_content = event["choices"][0]["delta"].get("content", "")
                        if delta_content:
                            context.accumulated_content += delta_content
                except (ValueError, KeyError, IndexError):
                    # Non-content SSE events (tool calls, metadata) don't match
                    # either delta format — safe to skip during stream draining.
                    logger.debug(
                        "Skipping non-content SSE event during drain: %s", payload[:100]
                    )
        if context.accumulated_content:
            context.response_messages = [AIMessage(content=context.accumulated_content)]

    async def run_streaming(
        self, *, context: PipelineContext
    ) -> AsyncGenerator[Any, None]:
        try:
            for step in self._pre_execution_steps:
                await step.run(context=context)
            await self._execution_step.run(context=context)
        except Exception as e:
            error_result = self._output_step.format_error(context=context, error=e)
            if isinstance(error_result, StreamingResponse):
                async for chunk in error_result.body_iterator:
                    yield chunk
            else:
                yield error_result
            return

        async for chunk in self._output_step.stream_response(context=context):
            yield chunk

        for step in self._post_execution_steps:
            await step.run(context=context)
