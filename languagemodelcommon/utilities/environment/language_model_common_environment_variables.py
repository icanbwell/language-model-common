import os
import logging
from typing import Optional

from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.skill_loader import (
    SkillLoaderEnvironmentVariables,
)

from languagemodelcommon.configs.prompt_library.prompt_library_environment_variables import (
    PromptLibraryEnvironmentVariables,
)

logger = logging.getLogger(__name__)

DEFAULT_STREAMING_BUFFER_FLUSH_INTERVAL_SECONDS = 0.05
DEFAULT_LANGGRAPH_MAX_CONCURRENCY = 4

# Default generic error message when not exposing technical details
DEFAULT_GENERIC_ERROR_MESSAGE = (
    "I ran into an issue processing your request. "
    "Could you try asking again? If it persists, rephrasing might help."
)


class LanguageModelCommonEnvironmentVariables(
    LangchainAISkillsFrameworkEnvironmentVariables,
    PromptLibraryEnvironmentVariables,
    SkillLoaderEnvironmentVariables,
):
    @property
    def streaming_buffer_flush_interval_seconds(self) -> float:
        """Interval in seconds for flushing the streaming buffer when processing LLM responses."""
        return float(
            os.environ.get("STREAMING_BUFFER_FLUSH_INTERVAL_SECONDS")
            or os.environ.get("BUFFER_FLUSH_INTERVAL_SECONDS")
            or DEFAULT_STREAMING_BUFFER_FLUSH_INTERVAL_SECONDS
        )

    @property
    def client_ids_for_debug_output(self) -> set[str] | None:
        # read the CLIENT_IDS_FOR_DEBUG_OUTPUT environment variable and split it by commas
        client_ids = os.environ.get("CLIENT_IDS_FOR_DEBUG_OUTPUT", "aiden")
        if client_ids and client_ids.strip():
            return set(client_id.strip() for client_id in client_ids.split(","))
        else:
            return None

    @property
    def generic_error_message(self) -> str:
        return os.environ.get(
            "GENERIC_ERROR_MESSAGE",
            DEFAULT_GENERIC_ERROR_MESSAGE,
        )

    @property
    def prompt_library_path(self) -> Optional[str]:
        configured = os.environ.get("PROMPT_LIBRARY_PATH")
        if configured and configured.strip():
            return configured
        return None
