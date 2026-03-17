"""
Evaluator for health AI responses.

Comprehensive health AI response evaluator with dimension scoring and violation detection.
Compatible with LangChain + AWS Bedrock.
"""

import asyncio
import json
import os
from typing import Optional, List, Tuple, Dict, Any

from boto3 import Session
from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from loguru import logger

from languagemodelcommon.langgraph.health_safety.models import (
    MAX_SCORES,
    ViolationReport,
    CommunicationEval,
    AccuracyEval,
    ScopeEval,
    PrivacyEval,
    UncertaintyEval,
    EvaluationScore,
)
from languagemodelcommon.langgraph.health_safety.utils.violation_detector import (
    HybridViolationDetector,
)
from languagemodelcommon.langgraph.health_safety.utils.llm_utils import (
    extract_text_content,
)
from languagemodelcommon.langgraph.health_safety.prompts import (
    get_evaluation_dimension_prompts,
)
from languagemodelcommon.utilities.security.prompt_sanitizer import PromptSanitizer


class Evaluator:
    """Evaluator with dimension scoring + violation detection."""

    def __init__(
        self, model: Optional[str] = None, enable_violation_detection: bool = True
    ):
        self.model = model or "us.anthropic.claude-sonnet-4-20250514-v1:0"
        self.llm: Optional[ChatBedrockConverse] = None
        self.enable_violation_detection = enable_violation_detection

        # Initialize parsers
        self.parsers: Dict[str, Any] = {
            "communication": PydanticOutputParser(pydantic_object=CommunicationEval),
            "accuracy": PydanticOutputParser(pydantic_object=AccuracyEval),
            "scope": PydanticOutputParser(pydantic_object=ScopeEval),
            "privacy": PydanticOutputParser(pydantic_object=PrivacyEval),
            "uncertainty": PydanticOutputParser(pydantic_object=UncertaintyEval),
        }

        # Load prompts
        raw_prompts = get_evaluation_dimension_prompts()
        format_values = {
            "communication_max": MAX_SCORES["patient_communication_score_max"],
            "accuracy_max": MAX_SCORES["information_accuracy_score_max"],
            "scope_max": MAX_SCORES["scope_boundaries_score_max"],
            "privacy_max": MAX_SCORES["privacy_score_max"],
            "uncertainty_max": MAX_SCORES["uncertainty_score_max"],
            "total_max": sum(MAX_SCORES.values()),
        }

        self.dimension_prompts = {
            k: v.format(**format_values) for k, v in raw_prompts.items()
        }

        # Initialize LLM
        self._init_llm()

        # Initialize violation detector
        if self.enable_violation_detection and self.llm:
            self.violation_detector: Optional[HybridViolationDetector] = (
                HybridViolationDetector(self.llm)
            )
        else:
            self.violation_detector = None

        # Store prompts metadata
        self.prompts_payload = {
            "prompt_type": "multi_dimension_split_v1",
            "dimensions": list(self.dimension_prompts.keys()),
            "dimension_prompts": self.dimension_prompts,
            "dimension_max_scores": {
                k.replace("_max", ""): v for k, v in MAX_SCORES.items()
            },
            "total_max_score": sum(MAX_SCORES.values()),
        }
        self.prompts_json = json.dumps(
            self.prompts_payload, ensure_ascii=False, separators=(",", ":")
        )

    def _init_llm(self) -> None:
        """Initialize Bedrock LLM client."""
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
            max_tokens=1000,
        )

        logger.debug(f"✅ [EVALUATOR] Initialized with {self.model}")

    async def _evaluate_dimension(
        self, message: str, dimension: str, retry: int = 0
    ) -> Tuple[float, str]:
        """Evaluate a single dimension."""
        dim_config = {
            "communication": {
                "parser": self.parsers["communication"],
                "score_field": "patient_communication_score",
                "notes_field": "communication_notes",
                "max": MAX_SCORES["patient_communication_score_max"],
            },
            "accuracy": {
                "parser": self.parsers["accuracy"],
                "score_field": "information_accuracy_score",
                "notes_field": "accuracy_notes",
                "max": MAX_SCORES["information_accuracy_score_max"],
            },
            "scope": {
                "parser": self.parsers["scope"],
                "score_field": "scope_boundaries_score",
                "notes_field": "scope_notes",
                "max": MAX_SCORES["scope_boundaries_score_max"],
            },
            "privacy": {
                "parser": self.parsers["privacy"],
                "score_field": "privacy_score",
                "notes_field": "privacy_notes",
                "max": MAX_SCORES["privacy_score_max"],
            },
            "uncertainty": {
                "parser": self.parsers["uncertainty"],
                "score_field": "uncertainty_score",
                "notes_field": "uncertainty_notes",
                "max": MAX_SCORES["uncertainty_score_max"],
            },
        }

        cfg = dim_config[dimension]
        parser = cfg["parser"]
        format_instructions = parser.get_format_instructions()

        # Sanitize the message to prevent prompt injection
        sanitized_message = PromptSanitizer.sanitize_for_evaluation(
            message, max_length=2000
        )

        prompt = f"""{self.dimension_prompts[dimension]}

{format_instructions}

Response to evaluate:
{sanitized_message}

Evaluate and return JSON:"""

        try:
            if not self.llm:
                raise ValueError("LLM not initialized")

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = extract_text_content(response.content)
            result = parser.parse(content)

            score_value = getattr(result, cfg["score_field"], cfg["max"] * 0.8)
            notes_value = getattr(
                result, cfg["notes_field"], f"No notes for {dimension}"
            )

            return float(score_value), str(notes_value)

        except Exception as e:
            logger.warning(f"⚠️ [EVALUATOR] {dimension} failed: {e}")

            if retry == 0:
                await asyncio.sleep(2)
                return await self._evaluate_dimension(message, dimension, retry=1)
            else:
                default_score = cfg["max"] * 0.8
                return default_score, f"Evaluation failed: {str(e)[:150]}"

    async def acheck_message(self, message: str) -> EvaluationScore:
        """Evaluate response with scores + violations."""
        try:
            logger.debug("🔬 [EVALUATOR] Starting evaluation + violation detection...")

            # Run dimension evaluation and violation detection in parallel
            tasks: List[Any] = [
                self._evaluate_dimension(message, "communication"),
                self._evaluate_dimension(message, "accuracy"),
                self._evaluate_dimension(message, "scope"),
                self._evaluate_dimension(message, "privacy"),
                self._evaluate_dimension(message, "uncertainty"),
            ]

            if self.violation_detector:
                tasks.append(self.violation_detector.detect_violations(message))

            results = await asyncio.gather(*tasks)

            # Extract scores
            (comm_score, comm_notes) = results[0]
            (acc_score, acc_notes) = results[1]
            (scope_score, scope_notes) = results[2]
            (priv_score, priv_notes) = results[3]
            (unc_score, unc_notes) = results[4]

            # Extract violations
            violations: Optional[ViolationReport] = None
            if len(results) > 5:
                violations = results[5]

            if violations:
                logger.debug(
                    f"🚨 [VIOLATIONS] {violations.total_violations} found | "
                    f"Critical: {violations.severity_breakdown.get('critical', 0)}"
                )

            total = comm_score + acc_score + scope_score + priv_score + unc_score
            logger.debug(f"✅ [EVALUATOR] Complete | Score: {total:.1f}/100")

            evaluation = EvaluationScore(
                patient_communication_score=comm_score,
                communication_notes=comm_notes,
                information_accuracy_score=acc_score,
                accuracy_notes=acc_notes,
                scope_boundaries_score=scope_score,
                scope_notes=scope_notes,
                privacy_score=priv_score,
                privacy_notes=priv_notes,
                uncertainty_score=unc_score,
                uncertainty_notes=unc_notes,
                violations=violations,
                system_prompt_used=self.prompts_json,
                user_input=message[:500],
            )

            return evaluation.with_max_scores()

        except Exception as e:
            logger.exception(f"❌ [EVALUATOR] Failed: {e}")

            default_eval = EvaluationScore(
                patient_communication_score=0.0,
                information_accuracy_score=0.0,
                scope_boundaries_score=0.0,
                privacy_score=0.0,
                uncertainty_score=0.0,
                communication_notes=f"Evaluation failed: {e}",
                accuracy_notes="Evaluation failed",
                scope_notes="Evaluation failed",
                privacy_notes="Evaluation failed",
                uncertainty_notes="Evaluation failed",
                system_prompt_used=self.prompts_json,
                user_input=message[:500],
            )

            return default_eval.with_max_scores()

    async def acheck(self, messages: List[str]) -> List[EvaluationScore]:
        """Evaluate multiple responses."""
        return await asyncio.gather(*[self.acheck_message(msg) for msg in messages])

    def get_max_scores(self) -> Dict[str, int]:
        """Get max scores."""
        return MAX_SCORES.copy()
