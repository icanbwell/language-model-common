"""
Prompts for health safety evaluation.

Contains evaluation prompts for scoring AI responses across multiple dimensions.
"""

from languagemodelcommon.langgraph.health_safety.prompts.evaluation_prompts import (
    MAX_SCORES,
    COMMUNICATION_PROMPT,
    ACCURACY_PROMPT,
    SCOPE_PROMPT,
    PRIVACY_PROMPT,
    UNCERTAINTY_PROMPT,
    EVALUATION_DIMENSION_PROMPTS,
    get_evaluation_dimension_prompts,
)

__all__ = [
    "MAX_SCORES",
    "COMMUNICATION_PROMPT",
    "ACCURACY_PROMPT",
    "SCOPE_PROMPT",
    "PRIVACY_PROMPT",
    "UNCERTAINTY_PROMPT",
    "EVALUATION_DIMENSION_PROMPTS",
    "get_evaluation_dimension_prompts",
]
