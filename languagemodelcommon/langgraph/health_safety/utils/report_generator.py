# baileyai/langgraph/health_safety/utils/report_generator.py
"""
Generates downloadable health safety analysis reports.
Includes original response with footnotes, tool calls, and iteration history.
Sanitizes sensitive tokens.
"""

import time
import re
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING
from loguru import logger

from languagemodelcommon.langgraph.health_safety.models import EvaluationScore
from languagemodelcommon.langgraph.health_safety.utils.fact_extractor import (
    ExtractedFacts,
)

if TYPE_CHECKING:
    from languagemodelcommon.langgraph.health_safety.config import IterationResult


class ReportGenerator:
    """Generates structured text reports for health safety analysis."""

    def __init__(self, output_directory: str, max_report_age_hours: int = 24):
        """
        Initialize report generator.

        Args:
            output_directory: Directory to save report files
            max_report_age_hours: Delete reports older than this (default: 24)
        """
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_report_age_hours = max_report_age_hours
        logger.debug(f"📁 [REPORT GENERATOR] Output: {self.output_dir}")

    def generate_report(
        self,
        facts: ExtractedFacts,
        evaluation: EvaluationScore,
        original_response: str,
        original_raw: str = "",
        footnote_legend: str = "",
        corrected_response: Optional[str] = None,
        content_length: int = 0,
        violation_count: int = 0,
        iteration_history: Optional[List["IterationResult"]] = None,
    ) -> tuple[str, str]:
        """
        Generate a health safety analysis report.

        Args:
            facts: Extracted medical facts
            evaluation: Evaluation scores
            original_response: Original response with footnote markers
            original_raw: Raw response with tool calls included
            footnote_legend: Footnote explanations
            corrected_response: Corrected response (if generated)
            content_length: Character count of response
            violation_count: Number of violations detected
            iteration_history: List of iteration results (if iterative refinement used)

        Returns:
            Tuple of (file_path, file_path)
        """
        try:
            # Generate filename with timestamp
            timestamp = int(time.time())
            filename = f"health_safety_analysis_{timestamp}.txt"
            file_path = self.output_dir / filename

            # Sanitize raw content before adding to report
            sanitized_raw = self._sanitize_tokens(original_raw)

            # Build report content
            report_content = self._build_report_content(
                facts=facts,
                evaluation=evaluation,
                original_response=original_response,
                original_raw=sanitized_raw,
                footnote_legend=footnote_legend,
                corrected_response=corrected_response,
                content_length=content_length,
                violation_count=violation_count,
                iteration_history=iteration_history,
            )

            # Write to file
            file_path.write_text(report_content, encoding="utf-8")

            logger.debug(
                f"✅ [REPORT GENERATOR] Created: {filename} | "
                f"Size: {len(report_content):,} chars | "
                f"Path: {file_path}"
            )

            # Cleanup old reports
            self._cleanup_old_reports()

            # Return absolute path as string
            return str(file_path.absolute()), str(file_path.absolute())

        except Exception as e:
            logger.exception(f"❌ [REPORT GENERATOR] Failed: {e}")
            raise

    def _sanitize_tokens(self, text: str) -> str:
        """Remove bearer tokens and sensitive auth data from text."""
        if not text:
            return text

        # Pattern 1: Remove JWT tokens (eyJ... pattern)
        text = re.sub(
            r"eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*",
            "[REDACTED_TOKEN]",
            text,
        )

        # Pattern 2: Remove Bearer tokens from auth_token fields
        text = re.sub(r"'auth_token':\s*'[^']*'", "'auth_token': '[REDACTED]'", text)
        text = re.sub(r"auth_token:\s*'?[^'\s]*'?", "auth_token: '[REDACTED]'", text)

        # Pattern 3: Remove other sensitive fields
        sensitive_fields = ["password", "api_key", "access_token", "refresh_token"]
        for field in sensitive_fields:
            text = re.sub(rf"'{field}':\s*'[^']*'", f"'{field}': '[REDACTED]'", text)
            text = re.sub(rf"{field}:\s*'?[^'\s]*'?", f"{field}: '[REDACTED]'", text)

        return text

    def _build_report_content(
        self,
        facts: ExtractedFacts,
        evaluation: EvaluationScore,
        original_response: str,
        original_raw: str,
        footnote_legend: str,
        corrected_response: Optional[str],
        content_length: int,
        violation_count: int,
        iteration_history: Optional[List["IterationResult"]],
    ) -> str:
        """Build the structured report content."""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("HEALTH SAFETY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
        )
        lines.append(f"Content Length: {content_length:,} characters")
        lines.append(f"Violations Detected: {violation_count}")

        if iteration_history and len(iteration_history) > 1:
            lines.append(f"Refinement Iterations: {len(iteration_history)}")

        lines.append("")

        # Executive Summary
        lines.append("-" * 80)
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Overall Score: {evaluation.total_score:.1f}/100")
        lines.append(f"Compliance Level: {evaluation.compliance_level}")
        lines.append("")

        if violation_count > 0:
            lines.append(
                f"⚠️  {violation_count} safety issue(s) identified and corrected"
            )
        else:
            lines.append("✅ No safety violations detected")
        lines.append("")

        # Iterative Refinement History (if applicable)
        if iteration_history and len(iteration_history) > 1:
            lines.append("-" * 80)
            lines.append("ITERATIVE REFINEMENT HISTORY")
            lines.append("-" * 80)
            lines.append(
                f"Target Score: 80/100 | Max Iterations: {len(iteration_history)}"
            )
            lines.append("")

            for result in iteration_history:
                lines.append(f"Attempt {result.iteration}:")
                lines.append(f"  • Score: {result.evaluation.total_score:.1f}/100")
                lines.append(f"  • Violations: {len(result.violations)}")
                lines.append(f"  • Duration: {result.duration:.2f}s")
                lines.append(f"  • Compliance: {result.evaluation.compliance_level}")

                # Show what changed
                if result.iteration > 1:
                    prev_result = iteration_history[result.iteration - 2]
                    score_change = (
                        result.evaluation.total_score
                        - prev_result.evaluation.total_score
                    )
                    change_symbol = (
                        "↗" if score_change > 0 else "↘" if score_change < 0 else "→"
                    )
                    lines.append(
                        f"  • Change: {change_symbol} {score_change:+.1f} points"
                    )

                    # Highlight improvements
                    if len(result.violations) < len(prev_result.violations):
                        removed_violations = len(prev_result.violations) - len(
                            result.violations
                        )
                        lines.append(f"  • Fixed {removed_violations} violation(s)")

                lines.append("")

            # Why did refinement stop?
            final_result = iteration_history[-1]
            if final_result.evaluation.total_score >= 80:
                lines.append(
                    f"✅ Refinement completed: Target score reached ({final_result.evaluation.total_score:.1f} ≥ 80)"
                )
            else:
                lines.append(
                    f"🛑 Refinement stopped: Max iterations reached "
                    f"(best score: {max(r.evaluation.total_score for r in iteration_history):.1f}/100)"
                )

            lines.append("")

        # Dimension Scores
        lines.append("-" * 80)
        lines.append("QUALITY SCORES BY DIMENSION")
        lines.append("-" * 80)
        lines.append(
            f"• Communication:        {evaluation.patient_communication_score:.0f}/20"
        )
        lines.append(
            f"• Information Accuracy: {evaluation.information_accuracy_score:.0f}/20"
        )
        lines.append(
            f"• Scope & Boundaries:   {evaluation.scope_boundaries_score:.0f}/20"
        )
        lines.append(f"• Privacy:              {evaluation.privacy_score:.0f}/20")
        lines.append(f"• Uncertainty Handling: {evaluation.uncertainty_score:.0f}/20")
        lines.append("")

        # Evaluator Notes
        if any(
            [
                evaluation.communication_notes,
                evaluation.accuracy_notes,
                evaluation.scope_notes,
                evaluation.privacy_notes,
                evaluation.uncertainty_notes,
            ]
        ):
            lines.append("-" * 80)
            lines.append("EVALUATOR NOTES")
            lines.append("-" * 80)

            if evaluation.communication_notes:
                lines.append("Communication:")
                lines.append(f"  {evaluation.communication_notes}")
                lines.append("")

            if evaluation.accuracy_notes:
                lines.append("Accuracy:")
                lines.append(f"  {evaluation.accuracy_notes}")
                lines.append("")

            if evaluation.scope_notes:
                lines.append("Scope:")
                lines.append(f"  {evaluation.scope_notes}")
                lines.append("")

            if evaluation.privacy_notes:
                lines.append("Privacy:")
                lines.append(f"  {evaluation.privacy_notes}")
                lines.append("")

            if evaluation.uncertainty_notes:
                lines.append("Uncertainty:")
                lines.append(f"  {evaluation.uncertainty_notes}")
                lines.append("")

        # Violations Detail
        if evaluation.violations and evaluation.violations.violations:
            lines.append("-" * 80)
            lines.append("VIOLATIONS DETECTED")
            lines.append("-" * 80)

            severity_emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
            }

            for i, v in enumerate(evaluation.violations.violations, 1):
                emoji = severity_emoji.get(v.severity, "⚠️")
                lines.append(f"{emoji} Violation {i}: {v.violation_type.upper()}")
                lines.append(f"   Severity: {v.severity.upper()}")
                lines.append(f'   Text: "{v.violated_text}"')
                lines.append(f"   Explanation: {v.explanation}")
                if v.suggested_replacement:
                    lines.append(f'   Suggested Fix: "{v.suggested_replacement}"')
                lines.append("")

        # Extracted Facts
        lines.append("-" * 80)
        lines.append("EXTRACTED MEDICAL FACTS")
        lines.append("-" * 80)

        facts_text = self._format_facts(facts)
        lines.append(facts_text if facts_text else "No medical facts extracted")
        lines.append("")

        # Original Response with Violation Markers
        lines.append("-" * 80)
        lines.append("ORIGINAL RESPONSE (WITH VIOLATION MARKERS)")
        lines.append("-" * 80)
        lines.append(original_response)
        lines.append("")

        # Footnote Legend
        if footnote_legend:
            lines.append(footnote_legend)
            lines.append("")

        # Final Corrected Response
        if corrected_response:
            lines.append("-" * 80)
            if iteration_history and len(iteration_history) > 1:
                lines.append(
                    f"FINAL REFINED RESPONSE (Iteration {len(iteration_history)})"
                )
            else:
                lines.append("CORRECTED RESPONSE")
            lines.append("-" * 80)
            lines.append(corrected_response)
            lines.append("")

        # Appendix: Tool Execution Details
        if original_raw:
            lines.append("")
            lines.append("=" * 80)
            lines.append("APPENDIX: TOOL EXECUTION DETAILS")
            lines.append("=" * 80)
            lines.append("")
            lines.append(
                "This section contains the raw tool calls and responses from the AI agent."
            )
            lines.append("Included for debugging and transparency purposes.")
            lines.append("(Sensitive tokens have been redacted)")
            lines.append("")
            lines.append("-" * 80)
            lines.append(original_raw)
            lines.append("-" * 80)
            lines.append("")

        # Footer
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _format_facts(self, facts: ExtractedFacts) -> str:
        """Format facts as bullet list."""
        bullets = []

        if facts.medications:
            bullets.append("Medications/Vaccinations:")
            for med in facts.medications:
                bullet = f"  • {med.get('name', 'Unknown')}"
                if med.get("dose"):
                    bullet += f" {med['dose']}"
                if med.get("start_date"):
                    bullet += f" (started {med['start_date']})"
                bullets.append(bullet)

        if facts.lab_results:
            bullets.append("\nLab Results:")
            for lab in facts.lab_results:
                bullets.append(
                    f"  • {lab.get('test')}: {lab.get('value')} "
                    f"({lab.get('date', 'date unknown')})"
                )

        if facts.procedures:
            bullets.append("\nProcedures:")
            for proc in facts.procedures:
                bullets.append(
                    f"  • {proc.get('name')} ({proc.get('date', 'date unknown')})"
                )

        if facts.diagnoses:
            bullets.append("\nDiagnoses:")
            for dx in facts.diagnoses:
                bullets.append(
                    f"  • {dx.get('condition')} ({dx.get('date', 'date unknown')})"
                )

        if facts.temporal_facts:
            bullets.append("\nTimeline:")
            for tf in facts.temporal_facts:
                bullets.append(f"  • {tf.get('description', str(tf))}")

        return "\n".join(bullets) if bullets else ""

    def _cleanup_old_reports(self) -> None:
        """Delete reports older than max_report_age_hours."""
        try:
            current_time = time.time()
            max_age_seconds = self.max_report_age_hours * 3600
            removed_count = 0

            for file_path in self.output_dir.glob("health_safety_analysis_*.txt"):
                file_age = current_time - file_path.stat().st_mtime

                if file_age > max_age_seconds:
                    file_path.unlink()
                    removed_count += 1

            if removed_count > 0:
                logger.debug(
                    f"🗑️  [REPORT GENERATOR] Cleaned up {removed_count} old reports"
                )

        except Exception as e:
            logger.warning(f"⚠️  [REPORT GENERATOR] Cleanup failed: {e}")
