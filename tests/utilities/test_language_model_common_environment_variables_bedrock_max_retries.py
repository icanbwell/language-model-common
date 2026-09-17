import pytest

from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)


def test_aws_bedrock_max_retries_defaults_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWS_BEDROCK_MAX_RETRIES", raising=False)
    monkeypatch.delenv("AWS_BEDROCK_MAX_ATTEMPTS", raising=False)
    env_vars = LanguageModelCommonEnvironmentVariables()
    assert env_vars.aws_bedrock_max_retries is None


def test_aws_bedrock_max_retries_reads_max_retries_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_BEDROCK_MAX_RETRIES", "5")
    monkeypatch.delenv("AWS_BEDROCK_MAX_ATTEMPTS", raising=False)
    env_vars = LanguageModelCommonEnvironmentVariables()
    assert env_vars.aws_bedrock_max_retries == 5


def test_aws_bedrock_max_retries_prefers_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_BEDROCK_MAX_ATTEMPTS", "7")
    monkeypatch.setenv("AWS_BEDROCK_MAX_RETRIES", "5")
    env_vars = LanguageModelCommonEnvironmentVariables()
    assert env_vars.aws_bedrock_max_retries == 7


def test_aws_bedrock_max_retries_returns_none_for_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_BEDROCK_MAX_RETRIES", "not-a-number")
    monkeypatch.delenv("AWS_BEDROCK_MAX_ATTEMPTS", raising=False)
    env_vars = LanguageModelCommonEnvironmentVariables()
    assert env_vars.aws_bedrock_max_retries is None
