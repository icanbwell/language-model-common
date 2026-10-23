from unittest.mock import MagicMock

import pytest

from languagemodelcommon.pipeline.context import PipelineContext
from languagemodelcommon.schema.openai.responses import HistoryStatus


def test_history_fields_default_to_unrecorded() -> None:
    context = PipelineContext(chat_request_wrapper=MagicMock())

    assert context.history_status is None
    assert context.history_loaded == 0


@pytest.mark.parametrize(
    ("status", "loaded"), [("loaded", 4), ("empty", 0), ("failed", 0)]
)
def test_history_fields_are_settable(status: HistoryStatus, loaded: int) -> None:
    context = PipelineContext(chat_request_wrapper=MagicMock())

    context.history_status = status
    context.history_loaded = loaded

    assert (context.history_status, context.history_loaded) == (status, loaded)


def test_history_status_is_typed_as_the_history_status_literal() -> None:
    """A producer typo should be a mypy error at the context, not only a runtime drop."""
    from typing import get_args, get_type_hints

    from languagemodelcommon.pipeline.context import PipelineContext
    from languagemodelcommon.schema.openai.responses import HistoryStatus

    members = get_args(get_type_hints(PipelineContext)["history_status"])
    literals = [member for member in members if member is not type(None)]

    assert type(None) in members
    assert len(literals) == 1
    assert set(get_args(literals[0])) == set(get_args(HistoryStatus))
