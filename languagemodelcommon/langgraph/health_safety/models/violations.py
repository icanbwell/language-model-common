"""
Violation detection models for health safety evaluation.

Defines models for tracking and reporting violations in AI responses.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class ViolationDetail(BaseModel):
    """A specific text violation with context."""

    violated_text: str = Field(description="The exact problematic text")
    violation_type: str = Field(description="Type of violation")
    dimension: str = Field(description="Which dimension violated")
    severity: str = Field(description="low, medium, high, critical")
    explanation: str = Field(description="Why this violates guidelines")
    suggested_replacement: Optional[str] = Field(None, description="How to fix it")
    context_before: str = Field(default="", description="Text before violation")
    context_after: str = Field(default="", description="Text after violation")


class ViolationReport(BaseModel):
    """Complete violation analysis."""

    violations: List[ViolationDetail] = Field(default_factory=list)
    total_violations: int = 0
    severity_breakdown: Dict[str, int] = Field(default_factory=dict)
    has_critical_violations: bool = False
