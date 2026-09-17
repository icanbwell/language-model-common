"""Tests for ModelFactory's Bedrock Anthropic max_tokens resolution.

Regression coverage for BAI-343: ChatAnthropicBedrock silently capped
max_tokens at 4096 for Bedrock cross-region model IDs (e.g.
``us.anthropic.claude-sonnet-4-5-20250929-v1:0``) because its inherited
default-max-tokens lookup only recognizes bare Anthropic API model names,
not Bedrock-prefixed IDs.
"""

from unittest.mock import MagicMock, patch

import pytest

from languagemodelcommon.models.model_factory import ModelFactory
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)


class TestResolveAnthropicBedrockMaxTokens:
    """ModelFactory._resolve_anthropic_bedrock_max_tokens"""

    def test_resolves_known_bedrock_model_id(self) -> None:
        max_tokens = ModelFactory._resolve_anthropic_bedrock_max_tokens(
            model_name="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        assert max_tokens == 64000

    @pytest.mark.parametrize(
        "model_name",
        [
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
        ],
    )
    def test_resolves_across_region_prefixes(self, model_name: str) -> None:
        assert (
            ModelFactory._resolve_anthropic_bedrock_max_tokens(model_name=model_name)
            == 64000
        )

    def test_returns_none_for_unknown_model(self) -> None:
        assert (
            ModelFactory._resolve_anthropic_bedrock_max_tokens(
                model_name="us.anthropic.some-future-model-v1:0"
            )
            is None
        )


class TestCreateAnthropicBedrockModel:
    """ModelFactory._create_anthropic_bedrock_model passes an explicit max_tokens."""

    def test_sets_max_tokens_when_unset(self) -> None:
        factory = ModelFactory(aws_client_factory=MagicMock())
        model_parameters_dict: dict[str, object] = {
            "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        }

        with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat_cls:
            factory._create_anthropic_bedrock_model(
                model_name="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                aws_credentials_profile=None,
                aws_region_name="us-east-1",
                thinking_budget=None,
                model_parameters_dict=model_parameters_dict,
            )

        assert mock_chat_cls.call_args.kwargs["max_tokens"] == 64000

    def test_does_not_override_explicit_max_tokens(self) -> None:
        factory = ModelFactory(aws_client_factory=MagicMock())
        model_parameters_dict: dict[str, object] = {
            "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "max_tokens": 2048,
        }

        with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat_cls:
            factory._create_anthropic_bedrock_model(
                model_name="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                aws_credentials_profile=None,
                aws_region_name="us-east-1",
                thinking_budget=None,
                model_parameters_dict=model_parameters_dict,
            )

        assert mock_chat_cls.call_args.kwargs["max_tokens"] == 2048

    def test_leaves_max_tokens_unset_for_unknown_model(self) -> None:
        factory = ModelFactory(aws_client_factory=MagicMock())
        model_parameters_dict: dict[str, object] = {
            "model": "us.anthropic.some-future-model-v1:0",
        }

        with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat_cls:
            factory._create_anthropic_bedrock_model(
                model_name="us.anthropic.some-future-model-v1:0",
                aws_credentials_profile=None,
                aws_region_name="us-east-1",
                thinking_budget=None,
                model_parameters_dict=model_parameters_dict,
            )

        assert "max_tokens" not in mock_chat_cls.call_args.kwargs


class TestCreateAnthropicBedrockModelMaxRetries:
    """ModelFactory._create_anthropic_bedrock_model max_retries wiring (BAI-765).

    BEDROCK_USE_ANTHROPIC_CLIENT=true (set for client-sandbox and production)
    routes model creation through ChatAnthropicBedrock, which never consulted
    AWS_BEDROCK_MAX_RETRIES — that env var only reached the boto3-based
    Converse client's retry config, so ChatAnthropicBedrock silently fell back
    to langchain_aws's hardcoded default of 2 retries, causing Bedrock 429s to
    exhaust retries and surface to users.
    """

    def test_sets_max_retries_from_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AWS_BEDROCK_MAX_RETRIES=5 means 5 total attempts in boto3 semantics
        (matching AwsClientFactory.create_bedrock_client), so the Anthropic
        SDK's retries-after-first max_retries kwarg must be 4, not 5
        (BAI-765 review: parity between the two Bedrock client paths)."""
        monkeypatch.setenv("AWS_BEDROCK_MAX_RETRIES", "5")
        factory = ModelFactory(
            environment_variables=LanguageModelCommonEnvironmentVariables(),
            aws_client_factory=MagicMock(),
        )
        model_parameters_dict: dict[str, object] = {
            "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        }

        with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat_cls:
            factory._create_anthropic_bedrock_model(
                model_name="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                aws_credentials_profile=None,
                aws_region_name="us-east-1",
                thinking_budget=None,
                model_parameters_dict=model_parameters_dict,
            )

        assert mock_chat_cls.call_args.kwargs["max_retries"] == 4

    def test_prefers_max_attempts_over_max_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_BEDROCK_MAX_ATTEMPTS", "7")
        monkeypatch.setenv("AWS_BEDROCK_MAX_RETRIES", "5")
        factory = ModelFactory(
            environment_variables=LanguageModelCommonEnvironmentVariables(),
            aws_client_factory=MagicMock(),
        )
        model_parameters_dict: dict[str, object] = {
            "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        }

        with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat_cls:
            factory._create_anthropic_bedrock_model(
                model_name="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                aws_credentials_profile=None,
                aws_region_name="us-east-1",
                thinking_budget=None,
                model_parameters_dict=model_parameters_dict,
            )

        assert mock_chat_cls.call_args.kwargs["max_retries"] == 6

    def test_clamps_max_retries_to_zero_when_max_attempts_is_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AWS_BEDROCK_MAX_ATTEMPTS=1 means "no retries" in boto3 semantics
        (matching AwsClientFactory.create_bedrock_client's default); the
        Anthropic client must get max_retries=0, not -1 or 1."""
        monkeypatch.setenv("AWS_BEDROCK_MAX_ATTEMPTS", "1")
        factory = ModelFactory(
            environment_variables=LanguageModelCommonEnvironmentVariables(),
            aws_client_factory=MagicMock(),
        )
        model_parameters_dict: dict[str, object] = {
            "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        }

        with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat_cls:
            factory._create_anthropic_bedrock_model(
                model_name="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                aws_credentials_profile=None,
                aws_region_name="us-east-1",
                thinking_budget=None,
                model_parameters_dict=model_parameters_dict,
            )

        assert mock_chat_cls.call_args.kwargs["max_retries"] == 0

    def test_does_not_override_explicit_max_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_BEDROCK_MAX_RETRIES", "5")
        factory = ModelFactory(
            environment_variables=LanguageModelCommonEnvironmentVariables(),
            aws_client_factory=MagicMock(),
        )
        model_parameters_dict: dict[str, object] = {
            "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "max_retries": 1,
        }

        with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat_cls:
            factory._create_anthropic_bedrock_model(
                model_name="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                aws_credentials_profile=None,
                aws_region_name="us-east-1",
                thinking_budget=None,
                model_parameters_dict=model_parameters_dict,
            )

        assert mock_chat_cls.call_args.kwargs["max_retries"] == 1

    def test_leaves_max_retries_unset_when_env_var_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AWS_BEDROCK_MAX_RETRIES", raising=False)
        monkeypatch.delenv("AWS_BEDROCK_MAX_ATTEMPTS", raising=False)
        factory = ModelFactory(
            environment_variables=LanguageModelCommonEnvironmentVariables(),
            aws_client_factory=MagicMock(),
        )
        model_parameters_dict: dict[str, object] = {
            "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        }

        with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat_cls:
            factory._create_anthropic_bedrock_model(
                model_name="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                aws_credentials_profile=None,
                aws_region_name="us-east-1",
                thinking_budget=None,
                model_parameters_dict=model_parameters_dict,
            )

        # Falls back to ChatAnthropicBedrock's own default (2) instead of
        # this factory forcing a value.
        assert "max_retries" not in mock_chat_cls.call_args.kwargs

    def test_no_max_retries_set_without_environment_variables_instance(self) -> None:
        """factory constructed with environment_variables=None (e.g. some test
        setups) must not blow up resolving max_retries."""
        factory = ModelFactory(aws_client_factory=MagicMock())
        model_parameters_dict: dict[str, object] = {
            "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        }

        with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat_cls:
            factory._create_anthropic_bedrock_model(
                model_name="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                aws_credentials_profile=None,
                aws_region_name="us-east-1",
                thinking_budget=None,
                model_parameters_dict=model_parameters_dict,
            )

        assert "max_retries" not in mock_chat_cls.call_args.kwargs
