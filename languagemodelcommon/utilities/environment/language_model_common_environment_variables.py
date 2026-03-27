import os
import logging
from typing import Optional

from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)
from oidcauthlib.utilities.environment.oidc_environment_variables import (
    OidcEnvironmentVariables,
)

from languagemodelcommon.configs.prompt_library.prompt_library_environment_variables import (
    PromptLibraryEnvironmentVariables,
)

logger = logging.getLogger(__name__)

DEFAULT_STREAMING_BUFFER_FLUSH_INTERVAL_SECONDS = 0.05
DEFAULT_LANGGRAPH_MAX_CONCURRENCY = 4
DEFAULT_LANGGRAPH_RECURSION_LIMIT = 100

# Default generic error message when not exposing technical details
DEFAULT_GENERIC_ERROR_MESSAGE = (
    "I ran into an issue processing your request. "
    "Could you try asking again? If it persists, rephrasing might help."
)


class LanguageModelCommonEnvironmentVariables(
    LangchainAISkillsFrameworkEnvironmentVariables,
    PromptLibraryEnvironmentVariables,
    OidcEnvironmentVariables,
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
    def enable_streaming_buffering(self) -> bool:
        """Enable token buffering for streamed chunks."""
        return self.str2bool(os.environ.get("ENABLE_STREAMING_BUFFERING", "true"))

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

    @property
    def maximum_inline_tool_output_size(self) -> int:
        """Maximum size in characters for tool output to be inlined in responses."""
        return int(os.environ.get("MAXIMUM_INLINE_TOOL_OUTPUT_SIZE", "100"))

    @property
    def enable_llm_memory(self) -> bool:
        return self.str2bool(os.environ.get("ENABLE_LLM_MEMORY", "false"))

    @property
    def llm_storage_type(self) -> str:
        return os.environ.get("LLM_STORAGE_TYPE", "memory")

    @property
    def mongo_llm_storage_uri(self) -> Optional[str]:
        return os.environ.get("MONGO_LLM_STORAGE_URI") or self.mongo_uri

    @property
    def mongo_llm_storage_db_name(self) -> Optional[str]:
        return os.environ.get("MONGO_LLM_STORAGE_DB_NAME", "llm_storage")

    @property
    def mongo_llm_storage_db_username(self) -> Optional[str]:
        return os.environ.get("MONGO_LLM_STORAGE_DB_USERNAME") or self.mongo_db_username

    @property
    def mongo_llm_storage_db_password(self) -> Optional[str]:
        return os.environ.get("MONGO_LLM_STORAGE_DB_PASSWORD") or self.mongo_db_password

    @property
    def mongo_llm_storage_store_collection_name(self) -> str:
        return os.environ.get("MONGO_LLM_STORAGE_STORE_COLLECTION_NAME", "stores")

    @property
    def mongo_llm_storage_checkpointer_collection_name(self) -> str:
        return os.environ.get(
            "MONGO_LLM_STORAGE_CHECKPOINTER_COLLECTION_NAME", "checkpoints"
        )

    @property
    def enable_llm_store(self) -> bool:
        return self.str2bool(os.environ.get("ENABLE_LLM_STORE", "false"))

    @property
    def enable_llm_checkpointer(self) -> bool:
        return self.str2bool(os.environ.get("ENABLE_LLM_CHECKPOINTER", "false"))

    @property
    def write_tool_output_to_file(self) -> bool:
        return self.str2bool(os.environ.get("WRITE_TOOL_OUTPUT_TO_FILE", "false"))

    @property
    def langgraph_recursion_limit(self) -> int:
        value = os.environ.get("LANGGRAPH_RECURSION_LIMIT")
        if value is None:
            return DEFAULT_LANGGRAPH_RECURSION_LIMIT
        try:
            parsed = int(value)
            return max(1, parsed)
        except ValueError:
            logger.warning(
                "Invalid LANGGRAPH_RECURSION_LIMIT value '%s'; using default=%s",
                value,
                DEFAULT_LANGGRAPH_RECURSION_LIMIT,
            )
            return DEFAULT_LANGGRAPH_RECURSION_LIMIT
