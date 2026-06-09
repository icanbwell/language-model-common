import logging

from langchain_core.messages import AIMessage

from languagemodelcommon.pipeline.context import PipelineContext
from languagemodelcommon.utilities.chat_message_helpers import (
    convert_message_content_to_string,
)
from languagemodelcommon.utilities.security.off_topic_detector import (
    detect_off_topic_manipulation,
)
from languagemodelcommon.utilities.security.prompt_extraction_detector import (
    detect_prompt_extraction,
)

logger = logging.getLogger(__name__)


class PromptExtractionAttempt(Exception):
    pass


class OffTopicAttempt(Exception):
    def __init__(self, *, category: str) -> None:
        self.category = category
        super().__init__(category)


class SecurityScanStep:
    """Scan the current turn for prompt extraction and off-topic manipulation.

    Only scans messages after the last assistant response to avoid
    re-triggering on previously-blocked messages the client re-sends.
    """

    def __init__(self, *, endpoint: str = "pipeline") -> None:
        self._endpoint = endpoint

    async def run(self, *, context: PipelineContext) -> None:
        user_id = (
            context.request_information.user_id if context.request_information else None
        )
        messages = context.messages

        last_ai_idx = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage):
                last_ai_idx = i
        current_turn = messages[last_ai_idx + 1 :]

        for msg in current_turn:
            content = convert_message_content_to_string(msg.content)
            matched = detect_prompt_extraction(content)
            if matched:
                logger.warning(
                    "Security detection: prompt_extraction | user=%s | endpoint=%s | pattern=%s",
                    user_id,
                    self._endpoint,
                    matched,
                )
                raise PromptExtractionAttempt()
            category = detect_off_topic_manipulation(content)
            if category:
                logger.warning(
                    "Security detection: %s | user=%s | endpoint=%s",
                    category,
                    user_id,
                    self._endpoint,
                )
                raise OffTopicAttempt(category=category)
