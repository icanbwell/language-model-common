"""
Configuration classes for health safety evaluation.

Contains configuration and result classes used by the health safety node.
"""

import os
from typing import Optional, List

from languagemodelcommon.langgraph.health_safety.models import (
    EvaluationScore,
    ViolationDetail,
)
from languagemodelcommon.models.bedrock_models import validate_bedrock_model


# Configuration constants
DEFAULT_TARGET_SCORE = 75.0
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_ENABLE_HEALTH_SAFETY = False  # Disabled by default


class HealthSafetyConfig:
    """Configuration for health safety evaluation."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        target_score: Optional[float] = None,
        max_iterations: Optional[int] = None,
        evaluation_model: Optional[str] = None,
        correction_model: Optional[str] = None,
    ):
        self.enabled = (
            enabled
            if enabled is not None
            else self._get_env_bool(
                "HEALTH_SAFETY_ENABLE_EVALUATOR", DEFAULT_ENABLE_HEALTH_SAFETY
            )
        )
        self.target_score = target_score or float(
            os.getenv("HEALTH_SAFETY_TARGET_SCORE", str(DEFAULT_TARGET_SCORE))
        )
        self.max_iterations = max_iterations or int(
            os.getenv("HEALTH_SAFETY_MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS))
        )

        # Get model identifiers with guaranteed str type using defaults
        default_evaluation_model = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        default_correction_model = "us.anthropic.claude-sonnet-4-20250514-v1:0"

        raw_evaluation_model: str = (
            evaluation_model
            if evaluation_model is not None
            else os.getenv("HEALTH_SAFETY_EVALUATION_MODEL", default_evaluation_model)
            or default_evaluation_model
        )
        raw_correction_model: str = (
            correction_model
            if correction_model is not None
            else os.getenv("HEALTH_SAFETY_CORRECTION_MODEL", default_correction_model)
            or default_correction_model
        )

        # Validate models against allowlist to prevent SSRF
        self.evaluation_model = validate_bedrock_model(
            raw_evaluation_model, "evaluation_model"
        )
        self.correction_model = validate_bedrock_model(
            raw_correction_model, "correction_model"
        )

    @staticmethod
    def _get_env_bool(key: str, default: bool) -> bool:
        """Get boolean from environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")


class IterationResult:
    """Result from a single evaluation iteration."""

    def __init__(
        self,
        iteration: int,
        response: str,
        evaluation: EvaluationScore,
        violations: List[ViolationDetail],
        duration: float,
    ):
        self.iteration = iteration
        self.response = response
        self.evaluation = evaluation
        self.violations = violations
        self.duration = duration
