import pytest

from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        (None, 128),
        ("40", 40),
    ],
)
def test_tool_title_max_param_length(
    monkeypatch: pytest.MonkeyPatch, raw_value: str | None, expected: int
) -> None:
    if raw_value is None:
        monkeypatch.delenv("TOOL_TITLE_MAX_PARAM_LENGTH", raising=False)
    else:
        monkeypatch.setenv("TOOL_TITLE_MAX_PARAM_LENGTH", raw_value)

    env_vars = LanguageModelCommonEnvironmentVariables()

    assert env_vars.tool_title_max_param_length == expected
