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


class TestResolveAnthropicBedrockMaxTokens:
    """ModelFactory._resolve_anthropic_bedrock_max_tokens"""

    def test_resolves_known_bedrock_model_id(self) -> None:
        max_tokens = ModelFactory._resolve_anthropic_bedrock_max_tokens(
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
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
        assert ModelFactory._resolve_anthropic_bedrock_max_tokens(model_name) == 64000

    def test_returns_none_for_unknown_model(self) -> None:
        assert (
            ModelFactory._resolve_anthropic_bedrock_max_tokens(
                "us.anthropic.some-future-model-v1:0"
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
