"""
Fact extraction for health AI responses.
Extracts structured medical facts from AI response text.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field

LLM_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"


class ExtractedFacts(BaseModel):
    """Structured facts extracted from an AI response."""

    lab_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Lab values with test name, value, date, reference range",
    )
    medications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Medications with name, dose, frequency, start date",
    )
    diagnoses: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Diagnoses/conditions with name and date if available",
    )
    procedures: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Procedures/surgeries with name, date, location",
    )
    vital_signs: List[Dict[str, Any]] = Field(
        default_factory=list, description="Vital signs with type, value, date"
    )
    providers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Provider names with specialty and visit dates",
    )
    temporal_facts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Timeline facts like 'last test was on X date'",
    )
    missing_information: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Information explicitly stated as NOT in records or unavailable",
    )


FACT_EXTRACTION_PROMPT = """
You are a fact extraction specialist. Your job is to extract ALL objective, verifiable facts from an AI assistant's response that appear to come from patient health records.

## What to Extract
Extract ONLY concrete facts - things that came from records:
- Specific numeric values (lab results, vital signs, dosages)
- Specific dates (test dates, appointment dates, prescription dates)
- Specific names (medications, providers, facilities, conditions)
- Timeline information (ranges, counts, frequencies)

## What NOT to Extract
Do NOT extract:
- Clinical interpretations ("this indicates", "excellent control")
- Recommendations ("you should", "I recommend")
- Speculative statements ("likely", "probably")
- General educational information
- Questions asked by the AI

## Output Format
Return ONLY a valid JSON object. Do NOT wrap in markdown code blocks.

## Example
Input: "Your A1c was 6.0% on January 10, 2025"
Output: {"lab_results": [{"test": "A1c", "value": "6.0%", "date": "January 10, 2025"}]}
""".strip()
