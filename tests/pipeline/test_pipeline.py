from typing import Any, AsyncGenerator

import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from languagemodelcommon.pipeline.context import PipelineContext
from languagemodelcommon.pipeline.pipeline import Pipeline


class FakeStep:
    def __init__(self) -> None:
        self.called = False

    async def run(self, *, context: PipelineContext) -> None:
        self.called = True


class FakeOutputStep:
    def __init__(self) -> None:
        self.format_called = False

    def format_response(self, *, context: PipelineContext) -> dict[str, Any]:
        self.format_called = True
        return {"content": context.accumulated_content}

    async def stream_response(
        self, *, context: PipelineContext
    ) -> AsyncGenerator[str, None]:
        yield "chunk1"
        yield "chunk2"

    def format_error(
        self, *, context: PipelineContext, error: Exception
    ) -> dict[str, str]:
        return {"error": str(error)}


class TestPipeline:
    @pytest.mark.asyncio
    async def test_run_non_streaming_executes_all_phases(self) -> None:
        step1 = FakeStep()
        step2 = FakeStep()
        exec_step = FakeStep()
        post_step = FakeStep()
        output = FakeOutputStep()

        pipeline = Pipeline(
            pre_execution_steps=[step1, step2],
            execution_step=exec_step,
            post_execution_steps=[post_step],
            output_step=output,
        )

        wrapper = MagicMock()
        wrapper.stream = False
        context = PipelineContext(chat_request_wrapper=wrapper)

        await pipeline.run_non_streaming(context=context)

        assert step1.called
        assert step2.called
        assert exec_step.called
        assert post_step.called
        assert output.format_called

    @pytest.mark.asyncio
    async def test_run_streaming_yields_chunks(self) -> None:
        exec_step = FakeStep()
        post_step = FakeStep()
        output = FakeOutputStep()

        pipeline = Pipeline(
            pre_execution_steps=[],
            execution_step=exec_step,
            post_execution_steps=[post_step],
            output_step=output,
        )

        wrapper = MagicMock()
        context = PipelineContext(chat_request_wrapper=wrapper)

        chunks = []
        async for chunk in pipeline.run_streaming(context=context):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2"]
        assert post_step.called

    @pytest.mark.asyncio
    async def test_run_non_streaming_catches_exception_and_formats_error(self) -> None:
        class FailingStep:
            async def run(self, *, context: PipelineContext) -> None:
                raise ValueError("something broke")

        output = FakeOutputStep()

        pipeline = Pipeline(
            pre_execution_steps=[FailingStep()],
            execution_step=FakeStep(),
            post_execution_steps=[],
            output_step=output,
        )

        wrapper = MagicMock()
        context = PipelineContext(chat_request_wrapper=wrapper)

        result = await pipeline.run_non_streaming(context=context)

        assert result == {"error": "something broke"}

    @pytest.mark.asyncio
    async def test_run_streaming_catches_pre_execution_error(self) -> None:
        class FailingStep:
            async def run(self, *, context: PipelineContext) -> None:
                raise ValueError("stream broke")

        output = FakeOutputStep()

        pipeline = Pipeline(
            pre_execution_steps=[FailingStep()],
            execution_step=FakeStep(),
            post_execution_steps=[],
            output_step=output,
        )

        wrapper = MagicMock()
        context = PipelineContext(chat_request_wrapper=wrapper)

        chunks: list[Any] = []
        async for chunk in pipeline.run_streaming(context=context):
            chunks.append(chunk)

        assert chunks == [{"error": "stream broke"}]

    @pytest.mark.asyncio
    async def test_drain_stream_parses_openai_delta_format(self) -> None:
        """Verify _drain_stream handles OpenAI choices[0].delta.content SSE format."""

        async def fake_stream() -> AsyncGenerator[str, None]:
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}\n'
            yield 'data: {"choices": [{"delta": {"content": " world"}}]}\n'
            yield "data: [DONE]\n"

        wrapper = MagicMock()
        context = PipelineContext(chat_request_wrapper=wrapper)
        context.content_stream = fake_stream()

        await Pipeline._drain_stream(context=context)

        assert context.accumulated_content == "Hello world"
        assert len(context.response_messages) == 1
        assert isinstance(context.response_messages[0], AIMessage)
        assert context.response_messages[0].content == "Hello world"

    @pytest.mark.asyncio
    async def test_drain_stream_parses_simple_delta_format(self) -> None:
        """Verify _drain_stream handles simple {"delta": "..."} SSE format."""

        async def fake_stream() -> AsyncGenerator[str, None]:
            yield 'data: {"delta": "Hi"}\ndata: {"delta": " there"}\n'

        wrapper = MagicMock()
        context = PipelineContext(chat_request_wrapper=wrapper)
        context.content_stream = fake_stream()

        await Pipeline._drain_stream(context=context)

        assert context.accumulated_content == "Hi there"

    @pytest.mark.asyncio
    async def test_drain_stream_noop_when_no_stream(self) -> None:
        """Verify _drain_stream does nothing when content_stream is None."""
        wrapper = MagicMock()
        context = PipelineContext(chat_request_wrapper=wrapper)
        assert context.content_stream is None

        await Pipeline._drain_stream(context=context)

        assert context.accumulated_content == ""
        assert context.response_messages == []
