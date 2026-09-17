import pytest

from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)


def test_rate_limit_max_backoff_seconds_defaults_to_60(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RATE_LIMIT_MAX_BACKOFF_SECONDS", raising=False)
    env_vars = LanguageModelCommonEnvironmentVariables()
    assert env_vars.rate_limit_max_backoff_seconds == 60.0


def test_rate_limit_max_backoff_seconds_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RATE_LIMIT_MAX_BACKOFF_SECONDS", "30")
    env_vars = LanguageModelCommonEnvironmentVariables()
    assert env_vars.rate_limit_max_backoff_seconds == 30.0
