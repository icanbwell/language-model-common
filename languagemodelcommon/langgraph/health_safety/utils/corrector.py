"""
Response corrector for health AI responses.
Generates corrected responses based on evaluation results.
"""

import os
import time
from typing import Optional, Any

from boto3 import Session
from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from loguru import logger

from languagemodelcommon.langgraph.health_safety.models import EvaluationScore
from languagemodelcommon.langgraph.health_safety.utils.llm_utils import (
    extract_text_content,
)
from languagemodelcommon.utilities.security.prompt_sanitizer import PromptSanitizer


class ResponseCorrector:
    """Generates corrected responses based on evaluation results."""

    def __init__(self, model: str):
        self.model = model
        self.llm: Optional[ChatBedrockConverse] = None
        self._init_corrector()

    def _init_corrector(self) -> None:
        """Initialize Bedrock LLM for correction."""
        retries = {
            "max_attempts": int(os.getenv("AWS_BEDROCK_MAX_RETRIES", "3")),
            "mode": os.getenv("AWS_BEDROCK_RETRY_MODE", "adaptive"),
        }
        config = BotoConfig(retries=retries, connect_timeout=10, read_timeout=120)  # type: ignore[arg-type]

        aws_region = os.environ.get("AWS_REGION", "us-east-1")
        aws_profile = os.environ.get("AWS_CREDENTIALS_PROFILE")

        session = Session(profile_name=aws_profile, region_name=aws_region)
        bedrock_client = session.client(
            service_name="bedrock-runtime",
            config=config,
            region_name=aws_region,
        )

        self.llm = ChatBedrockConverse(
            client=bedrock_client,
            model=self.model,
            provider="anthropic",
            credentials_profile_name=aws_profile,
            region_name=aws_region,
            temperature=0.0,
            max_tokens=3000,
        )
        logger.debug(f"✅ [CORRECTOR] Initialized with {self.model}")

    async def generate_corrected_response(
        self,
        original_response: str,
        facts: Any,  # ExtractedFacts
        evaluation: EvaluationScore,
        user_query: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> str:
        """
        Generate corrected version of response.

        Args:
            original_response: The response to correct
            facts: Extracted medical facts to preserve
            evaluation: Evaluation results showing issues
            user_query: Original user query (optional)
            iteration: Current iteration number (optional, for context)
        """
        try:
            prompt = self._build_correction_prompt(
                original_response, facts, evaluation, user_query, iteration
            )

            iteration_str = f" (iteration {iteration}/4)" if iteration else ""
            logger.debug(
                f"🔧 [CORRECTOR] Generating corrected response{iteration_str}..."
            )
            start_time = time.time()

            if not self.llm:
                raise ValueError("LLM not initialized")

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            duration = time.time() - start_time

            corrected = extract_text_content(response.content).strip()

            logger.debug(
                f"✅ [CORRECTOR] Generated in {duration:.2f}s | "
                f"Length: {len(corrected)} chars"
            )

            return corrected

        except Exception as e:
            logger.exception(f"❌ [CORRECTOR] Failed: {e}")
            return f"⚠️ Could not generate correction: {type(e).__name__}"

    def _build_correction_prompt(
        self,
        original_response: str,
        facts: Any,
        evaluation: EvaluationScore,
        user_query: Optional[str],
        iteration: Optional[int],
    ) -> str:
        """Build prompt for correction with iteration context."""
        facts_text = self._format_facts_for_prompt(facts)
        violations_text = self._format_violations_for_prompt(evaluation)

        # Add iteration context if provided
        iteration_context = ""
        if iteration:
            iteration_context = f"""
## Iteration Context
This is attempt {iteration} of up to 4 refinement iterations.
Target score: 80/100 (current: {evaluation.total_score:.1f}/100)
Focus on addressing the specific issues identified below to improve the score.
"""

        # Identify weakest dimensions
        weak_dimensions = []
        if evaluation.patient_communication_score < 16:
            weak_dimensions.append(
                f"Communication ({evaluation.patient_communication_score:.0f}/20)"
            )
        if evaluation.information_accuracy_score < 16:
            weak_dimensions.append(
                f"Accuracy ({evaluation.information_accuracy_score:.0f}/20)"
            )
        if evaluation.scope_boundaries_score < 16:
            weak_dimensions.append(
                f"Scope ({evaluation.scope_boundaries_score:.0f}/20)"
            )
        if evaluation.privacy_score < 16:
            weak_dimensions.append(f"Privacy ({evaluation.privacy_score:.0f}/20)")
        if evaluation.uncertainty_score < 16:
            weak_dimensions.append(
                f"Uncertainty ({evaluation.uncertainty_score:.0f}/20)"
            )

        focus_areas = ""
        if weak_dimensions:
            focus_areas = f"""
## Priority Focus Areas
Pay special attention to these dimensions that scored below 16/20:
{chr(10).join(f"  • {dim}" for dim in weak_dimensions)}
"""

        prompt = f"""You are a medical AI response editor. Rewrite a response to be fully compliant with medical safety guidelines.
{iteration_context}
## Original User Query
{PromptSanitizer.wrap_user_content(user_query, max_length=1000, label="ORIGINAL USER QUERY") if user_query else "Not provided"}

## Current Response (WITH ISSUES)
{PromptSanitizer.wrap_user_content(original_response, max_length=2500, label="AI RESPONSE TO CORRECT")}

## Extracted Facts (MUST PRESERVE ALL)
{facts_text}

## Identified Violations
{violations_text}

## Evaluation Scores
- Communication: {evaluation.patient_communication_score:.0f}/20
- Accuracy: {evaluation.information_accuracy_score:.0f}/20
- Scope & Boundaries: {evaluation.scope_boundaries_score:.0f}/20
- Privacy: {evaluation.privacy_score:.0f}/20
- Uncertainty: {evaluation.uncertainty_score:.0f}/20

**Overall: {evaluation.total_score:.0f}/100** - {evaluation.compliance_level}
{focus_areas}
## Your Task
Rewrite to fix ALL violations and improve weak dimensions:

### MUST DO:
1. **Preserve ALL facts** - Every medication, date, value, name must appear exactly
2. **Keep structure** - Same sections, same flow, same markdown formatting
3. **Fix violations** - Apply suggested replacements
4. **Maintain tone** - Still helpful, clear, and compassionate
5. **Improve weak dimensions** - Focus on areas scoring below 16/20

### SPECIFIC FIXES:
- Replace "I recommend" → "Consider discussing with your healthcare provider"
- Replace "You should" → "Your doctor can advise whether"
- Replace "You have [condition]" → "Your records show [condition]"
- Replace "This indicates" → "Your healthcare provider can interpret"
- Add qualifiers: "typically," "generally" to general knowledge
- Add redirects: "Discuss with your provider" where appropriate
- Acknowledge uncertainty: "This information may be incomplete. Check with your provider."

### RULES:
- DO preserve all dates, values, names from facts
- DO keep markdown formatting (##, **, -, etc.)
- DO maintain helpful, informative tone
- DON'T add new medical advice
- DON'T remove factual information
- DON'T add extra content not in original

## Output
Return ONLY the corrected response. No explanations or meta-commentary.
Start immediately with the corrected text."""

        return prompt

    def _format_facts_for_prompt(self, facts: Any) -> str:
        """Format facts for correction prompt with sanitization to prevent injection."""
        parts = []

        if facts.medications:
            parts.append("Medications/Vaccinations:")
            for med in facts.medications:
                # Sanitize all fact values to prevent prompt injection
                name = PromptSanitizer.sanitize(
                    str(med.get("name", "Unknown")), max_length=200
                )
                date = PromptSanitizer.sanitize(
                    str(med.get("start_date", "date unknown")), max_length=50
                )
                parts.append(f"  - {name} ({date})")

        if facts.lab_results:
            parts.append("\nLab Results:")
            for lab in facts.lab_results:
                test = PromptSanitizer.sanitize(
                    str(lab.get("test", "Unknown")), max_length=200
                )
                value = PromptSanitizer.sanitize(
                    str(lab.get("value", "N/A")), max_length=100
                )
                date = PromptSanitizer.sanitize(
                    str(lab.get("date", "date unknown")), max_length=50
                )
                parts.append(f"  - {test}: {value} ({date})")

        if facts.procedures:
            parts.append("\nProcedures:")
            for proc in facts.procedures:
                name = PromptSanitizer.sanitize(
                    str(proc.get("name", "Unknown")), max_length=200
                )
                date = PromptSanitizer.sanitize(
                    str(proc.get("date", "date unknown")), max_length=50
                )
                parts.append(f"  - {name} ({date})")

        if facts.diagnoses:
            parts.append("\nDiagnoses:")
            for dx in facts.diagnoses:
                condition = PromptSanitizer.sanitize(
                    str(dx.get("condition", "Unknown")), max_length=200
                )
                date = PromptSanitizer.sanitize(
                    str(dx.get("date", "date unknown")), max_length=50
                )
                parts.append(f"  - {condition} ({date})")

        if facts.temporal_facts:
            parts.append("\nTimeline:")
            for tf in facts.temporal_facts:
                description = PromptSanitizer.sanitize(
                    str(tf.get("description", "")), max_length=300
                )
                parts.append(f"  - {description}")

        return "\n".join(parts) if parts else "No facts extracted"

    def _format_violations_for_prompt(self, evaluation: EvaluationScore) -> str:
        """Format violations for correction prompt with sanitization to prevent injection."""
        if not evaluation.violations or not evaluation.violations.violations:
            return "No violations detected"

        parts = []
        for i, v in enumerate(evaluation.violations.violations, 1):
            # Sanitize all violation fields to prevent prompt injection
            violated_text = PromptSanitizer.sanitize(
                str(v.violated_text), max_length=500
            )
            violation_type = PromptSanitizer.sanitize(
                str(v.violation_type), max_length=100
            )
            severity = PromptSanitizer.sanitize(str(v.severity), max_length=50)
            explanation = PromptSanitizer.sanitize(str(v.explanation), max_length=500)

            parts.append(f'{i}. "{violated_text}" - {violation_type} ({severity})')
            parts.append(f"   Explanation: {explanation}")
            if v.suggested_replacement:
                suggested_replacement = PromptSanitizer.sanitize(
                    str(v.suggested_replacement), max_length=500
                )
                parts.append(f'   Fix: "{suggested_replacement}"')
            parts.append("")

        return "\n".join(parts)
