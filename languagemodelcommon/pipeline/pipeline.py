from __future__ import annotations

from typing import Any, AsyncGenerator, Protocol

from languagemodelcommon.pipeline.context import PipelineContext
from languagemodelcommon.pipeline.step import PipelineStep


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
        import re

        from langchain_core.messages import AIMessage

        async for chunk in context.content_stream:
            match = re.search(r'"delta":\s*\{\s*"content":\s*"([^"]*)"', chunk)
            if match:
                context.accumulated_content += match.group(1)
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
            yield self._output_step.format_error(context=context, error=e)
            return

        async for chunk in self._output_step.stream_response(context=context):
            yield chunk

        for step in self._post_execution_steps:
            await step.run(context=context)
