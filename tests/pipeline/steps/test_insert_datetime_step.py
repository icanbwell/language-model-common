import pytest
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage, SystemMessage

from languagemodelcommon.pipeline.context import PipelineContext
from languagemodelcommon.pipeline.steps.insert_datetime_step import InsertDatetimeStep


class TestInsertDatetimeStep:
    @pytest.mark.asyncio
    async def test_inserts_datetime_as_first_message(self) -> None:
        step = InsertDatetimeStep()
        wrapper = MagicMock()
        context = PipelineContext(chat_request_wrapper=wrapper)
        context.messages = [HumanMessage(content="hello")]

        await step.run(context=context)

        assert len(context.messages) == 2
        assert isinstance(context.messages[0], SystemMessage)
        content = context.messages[0].content
        assert isinstance(content, str)
        assert "date" in content.lower()
        assert context.messages[1].content == "hello"

    @pytest.mark.asyncio
    async def test_works_with_empty_messages(self) -> None:
        step = InsertDatetimeStep()
        wrapper = MagicMock()
        context = PipelineContext(chat_request_wrapper=wrapper)
        context.messages = []

        await step.run(context=context)

        assert len(context.messages) == 1
        assert isinstance(context.messages[0], SystemMessage)
