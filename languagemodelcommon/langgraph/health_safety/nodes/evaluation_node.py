"""
Health Safety Evaluation Node for LangGraph.

Implements the HealthSafetyNode that can be added to LangGraph workflows
to evaluate and refine AI responses before they are returned to users.
"""

import time
from typing import Optional, List, Dict, Any, Sequence, Tuple

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from loguru import logger

from baileyai.langgraph.state import MyMessagesState
from baileyai.langgraph.health_safety.config import (
    HealthSafetyConfig,
    IterationResult,
)
from baileyai.langgraph.health_safety.models import EvaluationScore, ViolationDetail
from baileyai.langgraph.health_safety.utils.fact_extractor import ExtractedFacts
from baileyai.langgraph.health_safety.utils.corrector import ResponseCorrector
from baileyai.langgraph.health_safety.evaluator import Evaluator


class HealthSafetyNode:
    """
    LangGraph node that performs health/safety evaluation on the last AI message.

    This node can be added to any LangGraph workflow to evaluate and optionally
    refine AI responses before they are returned to the user.
    """

    def __init__(self, config: Optional[HealthSafetyConfig] = None):
        self.config = config or HealthSafetyConfig()
        self.evaluator: Optional[Evaluator] = None
        self.corrector: Optional[ResponseCorrector] = None

        if self.config.enabled:
            self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize evaluator and corrector components."""

        try:
            logger.debug("🔧 [HEALTH_SAFETY_NODE] Initializing evaluator...")
            self.evaluator = Evaluator(
                model=self.config.evaluation_model,
                enable_violation_detection=True,
            )
            logger.debug("✅ [HEALTH_SAFETY_NODE] Evaluator initialized")
        except Exception as e:
            logger.exception(f"❌ [HEALTH_SAFETY_NODE] Evaluator init failed: {e}")
            self.evaluator = None

        try:
            logger.debug("🔧 [HEALTH_SAFETY_NODE] Initializing corrector...")
            if self.config.correction_model is None:
                logger.warning("❌ [HEALTH_SAFETY_NODE] Corrector model not configured")
                self.corrector = None
                return
            self.corrector = ResponseCorrector(model=self.config.correction_model)
            logger.debug("✅ [HEALTH_SAFETY_NODE] Corrector initialized")
        except Exception as e:
            logger.exception(f"❌ [HEALTH_SAFETY_NODE] Corrector init failed: {e}")
            self.corrector = None

    async def __call__(self, state: MyMessagesState) -> Dict[str, Any]:
        """
        Process the state and evaluate/refine the last AI message.

        Args:
            state: The current graph state containing messages

        Returns:
            Updated state with evaluation results and potentially modified message
        """
        if not self.config.enabled:
            logger.debug("[HEALTH_SAFETY_NODE] Disabled - passing through")
            return {
                "passed_evaluation": True,
                "evaluation_notes": "Health safety evaluation disabled",
            }

        # Get the last AI message
        messages = state["messages"]
        if not messages:
            logger.warning("[HEALTH_SAFETY_NODE] No messages in state")
            return {
                "passed_evaluation": True,
                "evaluation_notes": "No messages to evaluate",
            }

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            logger.debug(
                "[HEALTH_SAFETY_NODE] Last message is not AI message - skipping"
            )
            return {
                "passed_evaluation": True,
                "evaluation_notes": "Last message is not an AI response",
            }

        # Extract the response content
        response_content = self._extract_content(last_message)
        if not response_content:
            logger.warning("[HEALTH_SAFETY_NODE] Empty AI message content")
            return {
                "passed_evaluation": True,
                "evaluation_notes": "Empty response - no evaluation needed",
            }

        # Get the user query from previous message if available
        user_query = self._get_user_query(messages)

        # Evaluate and potentially refine the response
        (
            final_response,
            evaluation,
            iteration_history,
            passed,
        ) = await self._evaluate_and_refine(
            response_content=response_content,
            user_query=user_query,
        )

        # Build evaluation notes
        notes = self._build_evaluation_notes(evaluation, iteration_history, passed)

        # If the response was modified, update the messages
        if final_response != response_content:
            logger.debug("[HEALTH_SAFETY_NODE] Response was refined - updating message")
            # Create a new AI message with the refined content
            new_message = AIMessage(
                content=final_response,
                additional_kwargs=last_message.additional_kwargs,
                response_metadata={
                    **last_message.response_metadata,
                    "health_safety_evaluation": {
                        "passed": passed,
                        "score": evaluation.total_score,
                        "iterations": len(iteration_history),
                    },
                },
            )
            # Replace the last message
            updated_messages = list(messages[:-1]) + [new_message]
            return {
                "messages": updated_messages,
                "passed_evaluation": passed,
                "evaluation_notes": notes,
            }
        else:
            return {
                "passed_evaluation": passed,
                "evaluation_notes": notes,
            }

    async def _evaluate_and_refine(
        self,
        response_content: str,
        user_query: Optional[str] = None,
        facts: Optional[ExtractedFacts] = None,
    ) -> Tuple[str, EvaluationScore, List[IterationResult], bool]:
        """
        Evaluate and iteratively refine a response until it meets the target score.

        Args:
            response_content: The AI response to evaluate
            user_query: The original user query (for context during correction)
            facts: Extracted medical facts (optional, for correction context)

        Returns:
            Tuple of (final_response, final_evaluation, iteration_history, passed)
        """
        if not self.evaluator:
            logger.warning("[HEALTH_SAFETY_NODE] Evaluator not initialized")
            default_eval = EvaluationScore(
                patient_communication_score=0.0,
                information_accuracy_score=0.0,
                scope_boundaries_score=0.0,
                privacy_score=0.0,
                uncertainty_score=0.0,
            ).with_max_scores()
            return response_content, default_eval, [], False

        iteration_history: List[IterationResult] = []
        best_result: Optional[IterationResult] = None
        current_response = response_content

        logger.debug(
            f"🔄 [HEALTH_SAFETY_NODE] Starting evaluation | "
            f"Target: {self.config.target_score}/100 | "
            f"Max iterations: {self.config.max_iterations}"
        )

        for iteration in range(1, self.config.max_iterations + 1):
            iteration_start = time.time()
            logger.debug(f"📊 [ITERATION {iteration}] Evaluating...")

            # Evaluate current response
            try:
                evaluation = await self.evaluator.acheck_message(current_response)
            except Exception as e:
                logger.exception(f"❌ [ITERATION {iteration}] Evaluation failed: {e}")
                break

            violations: List[ViolationDetail] = (
                evaluation.violations.violations
                if evaluation.violations and evaluation.violations.violations
                else []
            )

            iteration_duration = time.time() - iteration_start

            # Store iteration result
            result = IterationResult(
                iteration=iteration,
                response=current_response,
                evaluation=evaluation,
                violations=violations,
                duration=iteration_duration,
            )
            iteration_history.append(result)

            logger.debug(
                f"✅ [ITERATION {iteration}] Score: {evaluation.total_score:.1f}/100 | "
                f"Violations: {len(violations)} | Duration: {iteration_duration:.2f}s"
            )

            # Track best result
            if (
                best_result is None
                or evaluation.total_score > best_result.evaluation.total_score
            ):
                best_result = result
                logger.debug(
                    f"🌟 [ITERATION {iteration}] New best score: {evaluation.total_score:.1f}/100"
                )

            # Check if target reached
            if evaluation.total_score >= self.config.target_score:
                logger.debug(
                    f"🎯 [HEALTH_SAFETY_NODE] Target reached! "
                    f"({evaluation.total_score:.1f} ≥ {self.config.target_score})"
                )
                break

            # Check if max iterations reached
            if iteration >= self.config.max_iterations:
                logger.debug(
                    f"🛑 [HEALTH_SAFETY_NODE] Max iterations reached ({self.config.max_iterations}). "
                    f"Using best score: {best_result.evaluation.total_score:.1f}/100"
                )
                break

            # Generate improved response for next iteration
            if self.corrector:
                logger.debug(f"🔧 [ITERATION {iteration}] Generating improvement...")
                try:
                    current_response = await self.corrector.generate_corrected_response(
                        original_response=current_response,
                        facts=facts or ExtractedFacts(),
                        evaluation=evaluation,
                        user_query=user_query,
                        iteration=iteration,
                    )
                    logger.debug(
                        f"✅ [ITERATION {iteration}] Improvement generated | "
                        f"Length: {len(current_response)} chars"
                    )
                except Exception as e:
                    logger.exception(
                        f"❌ [ITERATION {iteration}] Correction failed: {e}"
                    )
                    break
            else:
                logger.warning(
                    "[HEALTH_SAFETY_NODE] Corrector not available, stopping refinement"
                )
                break

        # Return best result
        if best_result:
            passed = best_result.evaluation.total_score >= self.config.target_score
            logger.debug(
                f"✅ [HEALTH_SAFETY_NODE] Complete | "
                f"{len(iteration_history)} iteration(s) | "
                f"Best score: {best_result.evaluation.total_score:.1f}/100 | "
                f"Passed: {passed}"
            )
            return (
                best_result.response,
                best_result.evaluation,
                iteration_history,
                passed,
            )
        else:
            logger.warning("[HEALTH_SAFETY_NODE] No valid iterations completed")
            default_eval = EvaluationScore(
                patient_communication_score=0.0,
                information_accuracy_score=0.0,
                scope_boundaries_score=0.0,
                privacy_score=0.0,
                uncertainty_score=0.0,
            ).with_max_scores()
            return response_content, default_eval, [], False

    def _extract_content(self, message: AIMessage) -> str:
        """Extract string content from an AI message."""
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list):
            # Extract text from list of content blocks
            text_parts = []
            for item in message.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return " ".join(text_parts).strip()
        else:
            return str(message.content) if message.content else ""

    def _get_user_query(self, messages: Sequence[BaseMessage]) -> Optional[str]:
        """Get the user's query from the message history."""
        for msg in reversed(messages[:-1]):  # Exclude the last message (AI response)
            if isinstance(msg, HumanMessage):
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    # Extract text from list
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            return str(item.get("text", ""))
                        elif isinstance(item, str):
                            return item
        return None

    def _build_evaluation_notes(
        self,
        evaluation: EvaluationScore,
        iteration_history: List[IterationResult],
        passed: bool,
    ) -> str:
        """Build a summary of the evaluation."""
        notes_parts = [
            f"Score: {evaluation.total_score:.1f}/100",
            f"Passed: {passed}",
            f"Iterations: {len(iteration_history)}",
        ]

        # Add dimension scores
        notes_parts.append(
            f"Communication: {evaluation.patient_communication_score:.1f}/{evaluation.patient_communication_score_max}"
        )
        notes_parts.append(
            f"Accuracy: {evaluation.information_accuracy_score:.1f}/{evaluation.information_accuracy_score_max}"
        )
        notes_parts.append(
            f"Scope: {evaluation.scope_boundaries_score:.1f}/{evaluation.scope_boundaries_score_max}"
        )
        notes_parts.append(
            f"Privacy: {evaluation.privacy_score:.1f}/{evaluation.privacy_score_max}"
        )
        notes_parts.append(
            f"Uncertainty: {evaluation.uncertainty_score:.1f}/{evaluation.uncertainty_score_max}"
        )

        return " | ".join(notes_parts)


def create_health_safety_node(
    config: Optional[HealthSafetyConfig] = None,
) -> HealthSafetyNode:
    """Factory function to create a health safety node."""
    return HealthSafetyNode(config)
