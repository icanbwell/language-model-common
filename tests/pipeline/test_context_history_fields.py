from unittest.mock import MagicMock

import pytest

from languagemodelcommon.pipeline.context import PipelineContext


def test_history_fields_default_to_unrecorded() -> None:
    context = PipelineContext(chat_request_wrapper=MagicMock())

    assert context.history_status is None
    assert context.history_loaded == 0


@pytest.mark.parametrize(
    ("status", "loaded"), [("loaded", 4), ("empty", 0), ("failed", 0)]
)
def test_history_fields_are_settable(status: str, loaded: int) -> None:
    context = PipelineContext(chat_request_wrapper=MagicMock())

    context.history_status = status
    context.history_loaded = loaded

    assert (context.history_status, context.history_loaded) == (status, loaded)
