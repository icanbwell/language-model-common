"""
LLM utility functions for health safety evaluation.
Shared utilities for handling LLM responses across the module.
"""

from typing import Any


def extract_text_content(content: Any) -> str:
    """
    Extract text from LLM response content.

    Handles both string content and list-of-blocks format returned by some
    providers (e.g., AWS Bedrock returns [{'type': 'text', 'text': '...'}]).

    Args:
        content: The response content from an LLM invocation

    Returns:
        The extracted text as a string
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)

    # Fallback: convert to string
    return str(content)
