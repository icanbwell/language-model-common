from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from languagemodelcommon.configs.schemas.config_schema import (
    ChatModelConfig,
    ModelConfig,
    ModelParameterConfig,
)
from languagemodelcommon.models.model_factory import ModelFactory


def _make_config(
    *,
    model: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
    provider: str = "bedrock",
    parameters: list[ModelParameterConfig] | None = None,
) -> ChatModelConfig:
    return ChatModelConfig(
        id="test",
        name="test-model",
        model=ModelConfig(provider=provider, model=model),
        model_parameters=parameters,
    )


class TestPromptCachingEnabled:
    def test_prompt_caching_enabled_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prompt caching is ON by default — returns a Runnable (bound model)."""
        monkeypatch.delenv("PROMPT_CACHE_ENABLED", raising=False)
        monkeypatch.delenv("PROMPT_CACHE_TTL", raising=False)

        mock_client = MagicMock()
        factory = ModelFactory(
            aws_client_factory=MagicMock(create_bedrock_client=lambda: mock_client)
        )
        result = factory.get_model(_make_config())

        assert isinstance(result, Runnable)
        assert not isinstance(result, BaseChatModel)
        assert cast(Any, result).kwargs == {
            "cache_control": {"type": "ephemeral", "ttl": "5m"}
        }

    def test_prompt_caching_disabled_via_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When PROMPT_CACHE_ENABLED=false, returns a plain BaseChatModel."""
        monkeypatch.setenv("PROMPT_CACHE_ENABLED", "false")

        mock_client = MagicMock()
        factory = ModelFactory(
            aws_client_factory=MagicMock(create_bedrock_client=lambda: mock_client)
        )
        result = factory.get_model(_make_config())

        assert isinstance(result, BaseChatModel)

    def test_prompt_caching_ttl_1h(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PROMPT_CACHE_TTL=1h is respected."""
        monkeypatch.setenv("PROMPT_CACHE_TTL", "1h")
        monkeypatch.delenv("PROMPT_CACHE_ENABLED", raising=False)

        mock_client = MagicMock()
        factory = ModelFactory(
            aws_client_factory=MagicMock(create_bedrock_client=lambda: mock_client)
        )
        result = factory.get_model(_make_config())

        assert isinstance(result, Runnable)
        assert cast(Any, result).kwargs == {
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }

    def test_invalid_ttl_defaults_to_5m(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid PROMPT_CACHE_TTL values fall back to 5m."""
        monkeypatch.setenv("PROMPT_CACHE_TTL", "30m")
        monkeypatch.delenv("PROMPT_CACHE_ENABLED", raising=False)

        mock_client = MagicMock()
        factory = ModelFactory(
            aws_client_factory=MagicMock(create_bedrock_client=lambda: mock_client)
        )
        result = factory.get_model(_make_config())

        assert cast(Any, result).kwargs == {
            "cache_control": {"type": "ephemeral", "ttl": "5m"}
        }

    def test_caching_not_applied_to_openai_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prompt caching only applies to Bedrock models."""
        monkeypatch.delenv("PROMPT_CACHE_ENABLED", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        factory = ModelFactory()
        config = _make_config(provider="openai", model="gpt-4")
        result = factory.get_model(config)

        assert isinstance(result, BaseChatModel)

    def test_environment_variables_take_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment variables class is used when provided."""
        mock_env = MagicMock()
        mock_env.prompt_cache_enabled = True
        mock_env.prompt_cache_ttl = "1h"
        mock_env.aws_credentials_profile = None
        mock_env.aws_region = "us-east-1"
        mock_env.bedrock_use_anthropic_client = False

        mock_client = MagicMock()
        factory = ModelFactory(
            environment_variables=mock_env,
            aws_client_factory=MagicMock(create_bedrock_client=lambda: mock_client),
        )
        result = factory.get_model(_make_config())

        assert cast(Any, result).kwargs == {
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }
