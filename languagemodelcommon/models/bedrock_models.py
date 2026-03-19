"""
AWS Bedrock model constants and validation utilities.

This module provides an allowlist of valid AWS Bedrock model identifiers
and validation functions to prevent SSRF attacks through model selection.
"""

from typing import FrozenSet


# Allowlist of valid AWS Bedrock model identifiers
# This prevents SSRF attacks by restricting model selection to known-good values
ALLOWED_BEDROCK_MODELS: FrozenSet[str] = frozenset(
    {
        # Anthropic Claude models
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-v2:1",
        "anthropic.claude-v2",
        "anthropic.claude-instant-v1",
        # Amazon Titan models
        "amazon.titan-text-premier-v1:0",
        "amazon.titan-text-express-v1",
        "amazon.titan-text-lite-v1",
        # Meta Llama models
        "meta.llama3-70b-instruct-v1:0",
        "meta.llama3-8b-instruct-v1:0",
        "meta.llama2-70b-chat-v1",
        "meta.llama2-13b-chat-v1",
        # Mistral models
        "mistral.mistral-large-2402-v1:0",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral.mistral-7b-instruct-v0:2",
        # Cohere models
        "cohere.command-r-plus-v1:0",
        "cohere.command-r-v1:0",
    }
)


def validate_bedrock_model(model_id: str, parameter_name: str) -> str:
    """
    Validate that a Bedrock model ID is in the allowlist.

    Args:
        model_id: The model identifier to validate
        parameter_name: Name of the parameter for error messages

    Returns:
        The validated model_id if valid

    Raises:
        ValueError: If model_id is not in the allowlist
    """
    if model_id not in ALLOWED_BEDROCK_MODELS:
        raise ValueError(
            f"Invalid {parameter_name}: '{model_id}' is not in the allowed models list. "
            f"Allowed models: {sorted(ALLOWED_BEDROCK_MODELS)}"
        )
    return model_id
