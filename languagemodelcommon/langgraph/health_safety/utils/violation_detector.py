"""
Violation detection using regex patterns.
Fast regex-based and hybrid violation detection for health AI responses.
"""

import re
from typing import List, Dict

from langchain_aws import ChatBedrockConverse
from loguru import logger

from languagemodelcommon.langgraph.health_safety.models import (
    ViolationDetail,
    ViolationReport,
)


# ---------------------------------------------------------------------------
# Violation Pattern Definitions
# ---------------------------------------------------------------------------
class ViolationPattern:
    """A regex-based violation pattern."""

    def __init__(
        self,
        pattern: str,
        violation_type: str,
        dimension: str,
        severity: str,
        explanation: str,
        replacement: str,
    ):
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.violation_type = violation_type
        self.dimension = dimension
        self.severity = severity
        self.explanation = explanation
        self.replacement = replacement


# Common violation patterns
VIOLATION_PATTERNS = [
    # Scope violations (CRITICAL)
    ViolationPattern(
        r"\b(I recommend|you should|I suggest|I advise|my recommendation is)\b",
        "direct_recommendation",
        "scope",
        "high",
        "Provides direct medical recommendation instead of deferring to provider",
        "Consider discussing with your healthcare provider",
    ),
    ViolationPattern(
        r"\b(you have|you may have|you\'re diagnosed with|you suffer from)\b",
        "diagnosis",
        "scope",
        "critical",
        "Provides diagnosis instead of presenting record information",
        "Your records show",
    ),
    ViolationPattern(
        r"\b(increase|decrease|stop|start|discontinue)\s+(your|taking|the)\s+(dose|dosage|medication)\b",
        "treatment_directive",
        "scope",
        "critical",
        "Gives specific treatment instructions",
        "Discuss any medication changes with your provider",
    ),
    # Accuracy violations
    ViolationPattern(
        r"\b(your doctor is likely|provider is probably|likely monitoring|probably tracking)\b",
        "provider_speculation",
        "accuracy",
        "medium",
        "Speculates about undocumented provider actions",
        "Your healthcare provider can advise",
    ),
    ViolationPattern(
        r"\b(this indicates|this suggests|this means)\b",
        "clinical_interpretation",
        "accuracy",
        "medium",
        "Provides clinical interpretation without qualification",
        "Your healthcare provider can interpret these results",
    ),
    # Communication violations
    ViolationPattern(
        r"\b(immediately|urgent|life-threatening|emergency|critical|right away)\b",
        "alarmist_language",
        "communication",
        "high",
        "Uses alarmist language that may cause unnecessary anxiety",
        "important to discuss with your doctor",
    ),
]


# ---------------------------------------------------------------------------
# Fast Regex-Based Violation Detector
# ---------------------------------------------------------------------------
class FastViolationDetector:
    """Fast regex-based violation detection."""

    def detect_violations(self, text: str) -> List[ViolationDetail]:
        """Find violations using regex patterns."""
        violations = []

        for pattern in VIOLATION_PATTERNS:
            for match in pattern.pattern.finditer(text):
                start = match.start()
                end = match.end()
                violated_text = match.group()

                # Get context
                context_start = max(0, start - 20)
                context_end = min(len(text), end + 20)
                context_before = text[context_start:start]
                context_after = text[end:context_end]

                violation = ViolationDetail(
                    violated_text=violated_text,
                    violation_type=pattern.violation_type,
                    dimension=pattern.dimension,
                    severity=pattern.severity,
                    explanation=pattern.explanation,
                    suggested_replacement=pattern.replacement,
                    context_before=context_before,
                    context_after=context_after,
                )

                violations.append(violation)

        return violations


# ---------------------------------------------------------------------------
# Hybrid Violation Detector
# ---------------------------------------------------------------------------
class HybridViolationDetector:
    """Combines fast regex detection."""

    def __init__(self, llm: ChatBedrockConverse):
        self.fast_detector = FastViolationDetector()
        self.llm = llm

    async def detect_violations(self, text: str) -> ViolationReport:
        """Detect violations using regex."""
        # Fast regex scan
        fast_violations = self.fast_detector.detect_violations(text)
        logger.debug(
            f"⚡ [FAST DETECTOR] Found {len(fast_violations)} pattern-based violations"
        )

        all_violations = fast_violations

        # Build severity breakdown
        severity_counts: Dict[str, int] = {}
        for v in all_violations:
            severity_counts[v.severity] = severity_counts.get(v.severity, 0) + 1

        return ViolationReport(
            violations=all_violations,
            total_violations=len(all_violations),
            severity_breakdown=severity_counts,
            has_critical_violations=any(
                v.severity == "critical" for v in all_violations
            ),
        )
