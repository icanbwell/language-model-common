"""`ADOPT_THREAD_ID_AS_CONVERSATION_ID` is off unless explicitly enabled (BAI-1264)."""

import pytest

from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("false", False),
        ("0", False),
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
    ],
)
def test_adopt_thread_id_as_conversation_id(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool
) -> None:
    if value is None:
        monkeypatch.delenv("ADOPT_THREAD_ID_AS_CONVERSATION_ID", raising=False)
    else:
        monkeypatch.setenv("ADOPT_THREAD_ID_AS_CONVERSATION_ID", value)

    assert (
        LanguageModelCommonEnvironmentVariables().adopt_thread_id_as_conversation_id
        is expected
    )
