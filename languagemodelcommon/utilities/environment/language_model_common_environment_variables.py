import os
import logging
import tempfile
from pathlib import Path
from typing import Optional

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
    PromptLibraryEnvironmentVariables,
    OidcEnvironmentVariables,
):
    @staticmethod
    def _resolve_path(value: str | None) -> str | None:
        """Replace ``{pid}`` with the current process ID.

        When multiple gunicorn workers share the same environment, this
        gives each worker its own directory tree so they don't collide
        on reads/writes.
        """
        if value and "{pid}" in value:
            return value.replace("{pid}", str(os.getpid()))
        return value

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
    def debug_prefixes(self) -> tuple[str, ...]:
        raw = os.environ.get("DEBUG_PREFIXES", "DEBUG:,/debug ")
        return tuple(p for p in raw.split(",") if p)

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
    def model_config_cache_collection_name(self) -> str:
        return os.environ.get("MODEL_CONFIG_CACHE_COLLECTION_NAME", "models")

    @property
    def model_config_cache_ttl_seconds(self) -> int:
        """TTL for model config cache entries in seconds.

        Defaults to 3600 (1 hour).  The model config cache should persist
        long enough to survive restarts and new worker processes.
        """
        return int(os.environ.get("MODEL_CONFIG_CACHE_TTL_SECONDS", "3600"))

    @property
    def mcp_tool_cache_db_name(self) -> str:
        """MongoDB database name for MCP tool list cache.

        Falls back to MONGO_LLM_STORAGE_DB_NAME for backward compatibility.
        """
        return (
            os.environ.get("MCP_TOOL_CACHE_DB_NAME")
            or self.mongo_llm_storage_db_name
            or "language_model_gateway"
        )

    @property
    def mcp_tool_cache_db_collection(self) -> str:
        """MongoDB collection name for MCP tool list cache."""
        return os.environ.get("MCP_TOOL_CACHE_DB_COLLECTION", "mcp-tool-cache")

    @property
    def prompt_store_type(self) -> str:
        """Backend for the prompt store: 'mongo' or '' (disabled)."""
        explicit = os.environ.get("PROMPT_STORE_TYPE", "").strip().lower()
        if explicit:
            return explicit
        return "mongo"

    @property
    def prompt_store_db_name(self) -> str:
        """MongoDB database name for prompt store.

        Falls back to MONGO_LLM_STORAGE_DB_NAME for backward compatibility.
        """
        return (
            os.environ.get("PROMPT_STORE_DB_NAME")
            or self.mongo_llm_storage_db_name
            or "language_model_gateway"
        )

    @property
    def prompt_store_collection(self) -> str:
        """MongoDB collection name for prompt store."""
        return os.environ.get("PROMPT_STORE_COLLECTION", "prompts")

    @property
    def token_cache_schema_version(self) -> str:
        """Schema version for token cache entries.

        Changing this value automatically obsoletes all existing token
        cache entries without migration — queries filter by version,
        so old-version entries are never matched.
        """
        return os.environ.get("TOKEN_CACHE_SCHEMA_VERSION", "1")

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
    def context_compaction_enabled(self) -> bool:
        """When True, automatically compact conversation context on input-too-long errors."""
        return self.str2bool(os.environ.get("CONTEXT_COMPACTION_ENABLED", "true"))

    @property
    def rate_limit_retry_enabled(self) -> bool:
        """When True, retry upstream model rate limits (HTTP 429) before surfacing them."""
        return self.str2bool(os.environ.get("RATE_LIMIT_RETRY_ENABLED", "true"))

    @property
    def rate_limit_max_retries(self) -> int:
        """Maximum number of retries for an upstream rate limit before giving up."""
        return int(os.environ.get("RATE_LIMIT_MAX_RETRIES", "3"))

    @property
    def rate_limit_retry_base_delay_ms(self) -> int:
        """Base delay in milliseconds for rate-limit retry exponential backoff."""
        return int(os.environ.get("RATE_LIMIT_RETRY_BASE_DELAY_MS", "500"))

    @property
    def mongo_db_token_collection_name(self) -> Optional[str]:
        return os.environ.get("MONGO_DB_TOKEN_COLLECTION_NAME")

    @property
    def mongo_db_dcr_collection_name(self) -> str:
        return os.environ.get("MONGO_DB_DCR_COLLECTION_NAME", "dcr_registrations")

    @property
    def emit_task_progress_in_chat_completions(self) -> bool:
        """When True, MCP task progress updates are emitted as content deltas
        in the Chat Completions streaming format."""
        return self.str2bool(
            os.environ.get("EMIT_TASK_PROGRESS_IN_CHAT_COMPLETIONS", "false")
        )

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
    def mcp_tool_heartbeat_interval_seconds(self) -> float:
        """Interval in seconds between synthetic heartbeat events emitted
        while an MCP tool call is in flight without reporting progress."""
        return float(os.environ.get("MCP_TOOL_HEARTBEAT_INTERVAL_SECONDS", "15"))

    @property
    def emit_tool_heartbeat_in_chat_completions(self) -> bool:
        """When True, synthetic MCP tool heartbeats are emitted as content
        deltas in the Chat Completions streaming format. Separate from
        emit_task_progress_in_chat_completions so enabling one does not
        change the volume/behavior of the other."""
        return self.str2bool(
            os.environ.get("EMIT_TOOL_HEARTBEAT_IN_CHAT_COMPLETIONS", "false")
        )

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
    def bedrock_use_anthropic_client(self) -> bool:
        return os.environ.get("BEDROCK_USE_ANTHROPIC_CLIENT", "").lower() in (
            "1",
            "true",
            "yes",
        )

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
        return self._resolve_path(
            os.environ.get(
                "GITHUB_CONFIG_CACHE_DIR",
                str(Path(tempfile.gettempdir()) / "github_config_cache"),
            )
        ) or str(Path(tempfile.gettempdir()) / "github_config_cache")

    @property
    def github_token(self) -> Optional[str]:
        return os.environ.get("GITHUB_TOKEN")

    @property
    def github_app_id(self) -> Optional[str]:
        return os.environ.get("GITHUB_APP_ID")

    @property
    def github_app_private_key(self) -> Optional[str]:
        return os.environ.get("GITHUB_APP_PRIVATE_KEY")

    @property
    def github_app_installation_id(self) -> Optional[str]:
        return os.environ.get("GITHUB_APP_INSTALLATION_ID")

    @property
    def plugins_mcp_server(self) -> Optional[str]:
        """URL of the MCP server that returns plugin MCP configs.

        When set, ``ConfigReader`` fetches MCP server definitions by
        calling the ``get_mcp_servers_config`` tool on this server
        instead of reading a local ``.mcp.json`` file.
        """
        return os.environ.get("PLUGINS_MCP_SERVER")

    @property
    def mcp_app_proxy_base_url(self) -> Optional[str]:
        """Base URL for the MCP Apps proxy endpoint.

        When set, the injected bridge JavaScript in MCP App iframes can
        proxy ``tools/call`` and ``resources/read`` requests through this
        URL back to the MCP server.  Typically points to the gateway's
        own base URL (e.g., ``http://localhost:5000``).
        """
        return os.environ.get("MCP_APP_PROXY_BASE_URL")
