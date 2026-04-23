import os
import logging
import tempfile
from pathlib import Path
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
    def github_cache_folder(self) -> Optional[str]:
        return self._resolve_path(os.environ.get("GITHUB_CACHE_FOLDER"))

    @property
    def models_official_path(self) -> str:
        value = self._resolve_path(os.environ.get("MODELS_OFFICIAL_PATH", ""))
        if not value:
            raise ValueError("MODELS_OFFICIAL_PATH environment variable is not set")
        return value

    @property
    def models_testing_path(self) -> Optional[str]:
        return self._resolve_path(os.environ.get("MODELS_TESTING_PATH"))

    @property
    def mcp_json_path(self) -> Optional[str]:
        return self._resolve_path(os.environ.get("MCP_JSON_PATH"))

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
        configured = self._resolve_path(os.environ.get("PROMPT_LIBRARY_PATH"))
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
    def snapshot_cache_type(self) -> str:
        """Cache backend type: 'mongo', 'memory', or '' (disabled).

        Replaces the old ENABLE_SNAPSHOT_CACHE boolean.
        Falls back to ENABLE_SNAPSHOT_CACHE for backward compatibility.
        """
        explicit = os.environ.get("SNAPSHOT_CACHE_TYPE", "").strip().lower()
        if explicit:
            return explicit
        # Backward compat: ENABLE_SNAPSHOT_CACHE=true → "mongo"
        if self.str2bool(os.environ.get("ENABLE_SNAPSHOT_CACHE", "false")):
            return "mongo"
        return "memory"

    @property
    def snapshot_cache_collection_name(self) -> str:
        return os.environ.get("SNAPSHOT_CACHE_COLLECTION_NAME", "snapshot_cache")

    @property
    def snapshot_cache_ttl_seconds(self) -> int:
        """TTL for snapshot cache entries in seconds.

        Defaults to 3600 (1 hour).  This is independent of
        ``config_cache_timeout_seconds`` which controls the in-memory
        cache.  The snapshot cache should persist long enough to
        survive restarts and new worker processes.
        """
        return int(os.environ.get("SNAPSHOT_CACHE_TTL_SECONDS", "3600"))

    @property
    def snapshot_cache_model_configs_collection(self) -> str | None:
        """Optional separate collection for model config snapshots.

        When set, ConfigReader stores its snapshot in this collection
        instead of the store's default collection.
        """
        return os.environ.get("SNAPSHOT_CACHE_MODEL_CONFIGS_COLLECTION") or None

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

    @property
    def mongo_db_token_collection_name(self) -> Optional[str]:
        return os.environ.get("MONGO_DB_TOKEN_COLLECTION_NAME")

    @property
    def mongo_db_dcr_collection_name(self) -> str:
        return os.environ.get("MONGO_DB_DCR_COLLECTION_NAME", "dcr_registrations")

    @property
    def mcp_tools_metadata_cache_ttl_seconds(self) -> int:
        """TTL for MCP tool list cache entries in seconds.

        Falls back to MCP_TOOLS_METADATA_CACHE_TIMEOUT_SECONDS for
        backward compatibility.  Defaults to 3600 (1 hour).
        """
        return int(
            os.environ.get("MCP_TOOLS_METADATA_CACHE_TTL_SECONDS")
            or os.environ.get("MCP_TOOLS_METADATA_CACHE_TIMEOUT_SECONDS")
            or 3600
        )

    @property
    def tool_output_token_limit(self) -> Optional[int]:
        limit = os.environ.get("TOOL_OUTPUT_TOKEN_LIMIT")
        return int(limit) if limit and limit.isdigit() else None

    @property
    def tool_call_timeout_seconds(self) -> int:
        """Timeout in seconds for tool calls."""
        return int(os.environ.get("TOOL_CALL_TIMEOUT_SECONDS", "600"))

    @property
    def app_login_uri(self) -> str:
        value = os.environ.get("APP_LOGIN_URI")
        return value if value else "/app/login"

    @property
    def app_token_save_uri(self) -> str:
        value = os.environ.get("APP_TOKEN_SAVE_URI")
        return value if value else "/app/token"

    @property
    def log_input_and_output(self) -> bool:
        return os.environ.get("LOG_INPUT_AND_OUTPUT", "0") == "1"

    @property
    def image_generation_path(self) -> Optional[str]:
        return os.environ.get("IMAGE_GENERATION_PATH")

    @property
    def image_generation_url(self) -> Optional[str]:
        return os.environ.get("IMAGE_GENERATION_URL")

    @property
    def aws_bedrock_retry_mode(self) -> str:
        return os.environ.get("AWS_BEDROCK_RETRY_MODE", "adaptive")

    @property
    def aws_credentials_profile(self) -> Optional[str]:
        return os.environ.get("AWS_CREDENTIALS_PROFILE")

    @property
    def aws_region(self) -> str:
        return os.environ.get("AWS_REGION", "us-east-1")

    @property
    def default_model_provider(self) -> str:
        return os.environ.get("DEFAULT_MODEL_PROVIDER", "bedrock")

    @property
    def default_model_name(self) -> str:
        return os.environ.get(
            "DEFAULT_MODEL_NAME",
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        )

    @property
    def default_llm_model(self) -> str:
        return os.environ.get("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")

    @property
    def google_credentials_json(self) -> Optional[str]:
        return os.environ.get("GOOGLE_CREDENTIALS_JSON")

    @property
    def openai_api_key(self) -> Optional[str]:
        return os.environ.get("OPENAI_API_KEY")

    @property
    def config_cache_timeout_seconds(self) -> int:
        try:
            return int(os.environ.get("CONFIG_CACHE_TIMEOUT_SECONDS", "3600"))
        except (ValueError, TypeError):
            return 3600

    @property
    def github_config_cache_dir(self) -> str:
        return os.environ.get(
            "GITHUB_CONFIG_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "github_config_cache"),
        )

    @property
    def github_config_repo_url(self) -> Optional[str]:
        return os.environ.get("GITHUB_CONFIG_REPO_URL")

    @property
    def github_timeout(self) -> int:
        try:
            return int(os.environ.get("GITHUB_TIMEOUT", "300"))
        except (ValueError, TypeError):
            return 300

    @property
    def github_token(self) -> Optional[str]:
        return os.environ.get("GITHUB_TOKEN")
