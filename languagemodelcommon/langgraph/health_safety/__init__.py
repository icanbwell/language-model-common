"""
Health Safety evaluation for LangGraph workflows.

This package provides health/safety evaluation capabilities for AI responses,
including scoring across multiple dimensions, violation detection, and
iterative response refinement.
"""

# Configuration
from languagemodelcommon.langgraph.health_safety.config import (
    HealthSafetyConfig,
    IterationResult,
    DEFAULT_TARGET_SCORE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_ENABLE_HEALTH_SAFETY,
)

# Models
from languagemodelcommon.langgraph.health_safety.models import (
    MAX_SCORES,
    ViolationDetail,
    ViolationReport,
    CommunicationEval,
    AccuracyEval,
    ScopeEval,
    PrivacyEval,
    UncertaintyEval,
    EvaluationScore,
)

# Nodes
from languagemodelcommon.langgraph.health_safety.nodes import (
    HealthSafetyNode,
    create_health_safety_node,
)


# Evaluator
from languagemodelcommon.langgraph.health_safety.evaluator import Evaluator

# Utils
from languagemodelcommon.langgraph.health_safety.utils import (
    ViolationPattern,
    VIOLATION_PATTERNS,
    FastViolationDetector,
    HybridViolationDetector,
    ResponseCorrector,
    ExtractedFacts,
    ReportGenerator,
)

__all__ = [
    # Configuration
    "HealthSafetyConfig",
    "IterationResult",
    "DEFAULT_TARGET_SCORE",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_ENABLE_HEALTH_SAFETY",
    # Models
    "MAX_SCORES",
    "ViolationDetail",
    "ViolationReport",
    "CommunicationEval",
    "AccuracyEval",
    "ScopeEval",
    "PrivacyEval",
    "UncertaintyEval",
    "EvaluationScore",
    # Nodes
    "HealthSafetyNode",
    "create_health_safety_node",
    # Evaluator
    "Evaluator",
    # Utils
    "ViolationPattern",
    "VIOLATION_PATTERNS",
    "FastViolationDetector",
    "HybridViolationDetector",
    "ResponseCorrector",
    "ExtractedFacts",
    "ReportGenerator",
]
