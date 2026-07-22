import pytest

from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)


def test_mcp_tool_heartbeat_interval_seconds_defaults_to_15(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MCP_TOOL_HEARTBEAT_INTERVAL_SECONDS", raising=False)
    env_vars = LanguageModelCommonEnvironmentVariables()
    assert env_vars.mcp_tool_heartbeat_interval_seconds == 15.0


def test_mcp_tool_heartbeat_interval_seconds_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_TOOL_HEARTBEAT_INTERVAL_SECONDS", "5")
    env_vars = LanguageModelCommonEnvironmentVariables()
    assert env_vars.mcp_tool_heartbeat_interval_seconds == 5.0


def test_emit_tool_heartbeat_in_chat_completions_defaults_to_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMIT_TOOL_HEARTBEAT_IN_CHAT_COMPLETIONS", raising=False)
    env_vars = LanguageModelCommonEnvironmentVariables()
    assert env_vars.emit_tool_heartbeat_in_chat_completions is False


def test_emit_tool_heartbeat_in_chat_completions_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMIT_TOOL_HEARTBEAT_IN_CHAT_COMPLETIONS", "true")
    env_vars = LanguageModelCommonEnvironmentVariables()
    assert env_vars.emit_tool_heartbeat_in_chat_completions is True
