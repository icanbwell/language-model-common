"""
Utility modules for health safety evaluation.

Contains support classes for violation detection, response correction,
fact extraction, and report generation.
"""

from languagemodelcommon.langgraph.health_safety.utils.violation_detector import (
    ViolationPattern,
    VIOLATION_PATTERNS,
    FastViolationDetector,
    HybridViolationDetector,
)
from languagemodelcommon.langgraph.health_safety.utils.corrector import (
    ResponseCorrector,
)
from languagemodelcommon.langgraph.health_safety.utils.fact_extractor import (
    ExtractedFacts,
    FACT_EXTRACTION_PROMPT,
    LLM_MODEL,
)
from languagemodelcommon.langgraph.health_safety.utils.report_generator import (
    ReportGenerator,
)
from languagemodelcommon.langgraph.health_safety.utils.llm_utils import (
    extract_text_content,
)

__all__ = [
    # Violation detection
    "ViolationPattern",
    "VIOLATION_PATTERNS",
    "FastViolationDetector",
    "HybridViolationDetector",
    # Correction
    "ResponseCorrector",
    # Fact extraction
    "ExtractedFacts",
    "FACT_EXTRACTION_PROMPT",
    "LLM_MODEL",
    # Report generation
    "ReportGenerator",
    # LLM utilities
    "extract_text_content",
]
