import os
import logging
from pathlib import Path
from typing import Optional

from oidcauthlib.utilities.environment.oidc_environment_variables import (
    OidcEnvironmentVariables,
)
from languagemodelcommon.utilities.logger.exception_logger import (
    DEFAULT_GENERIC_ERROR_MESSAGE,
)

logger = logging.getLogger(__name__)

DEFAULT_STREAMING_BUFFER_FLUSH_INTERVAL_SECONDS = 0.05
DEFAULT_LANGGRAPH_MAX_CONCURRENCY = 4


class BaileyAIEnvironmentVariables(OidcEnvironmentVariables):
    @property
    def default_model(self) -> str:
        return os.environ.get(
            "DEFAULT_MODEL_NAME", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )

    @property
    def model_override(self) -> Optional[str]:
        return os.environ.get("MODEL_OVERRIDE")

    @property
    def pss_server_url(self) -> Optional[str]:
        return os.environ.get("PSS_BASE_URL")

    @property
    def api_gateway_url(self) -> Optional[str]:
        return os.environ.get("API_GATEWAY_BASE_URL")

    @property
    def pass_through_headers(self) -> set[str] | None:
        # read the PASS_THROUGH_HEADERS environment variable and split it by commas
        headers = os.environ.get(
            "PASS_THROUGH_HEADERS",
            "bwell-client-fhir-patient-id,bwell-client-fhir-person-id,bwell-client-key,bwell-fhir-patient-id,bwell-fhir-person-id,bwell-managing-organization,traceparent",
        )
        if headers and headers.strip():
            return set(header.strip().lower() for header in headers.split(","))
        else:
            return None

    @property
    def client_ids_for_debug_output(self) -> set[str] | None:
        # read the CLIENT_IDS_FOR_DEBUG_OUTPUT environment variable and split it by commas
        client_ids = os.environ.get("CLIENT_IDS_FOR_DEBUG_OUTPUT", "aiden")
        if client_ids and client_ids.strip():
            return set(client_id.strip() for client_id in client_ids.split(","))
        else:
            return None

    @property
    def mcp_tool_allowed_domains(self) -> set[str]:
        # read the MCP_TOOL_ALLOWED_DOMAINS environment variable and split it by commas
        domains = os.environ.get(
            "MCP_TOOL_ALLOWED_DOMAINS",
            "bwell.zone,bwell.com,icanbwell.com,mcp-fhir-agent,test-mcp-server",
        )
        if domains and domains.strip():
            return set(domain.strip().lower() for domain in domains.split(","))
        else:
            return set()

    @property
    def generic_error_message(self) -> str:
        return os.environ.get(
            "GENERIC_ERROR_MESSAGE",
            DEFAULT_GENERIC_ERROR_MESSAGE,
        )

    @property
    def streaming_buffer_flush_interval_seconds(self) -> float:
        """Interval in seconds for flushing the streaming buffer when processing LLM responses."""
        return float(
            os.environ.get("STREAMING_BUFFER_FLUSH_INTERVAL_SECONDS")
            or os.environ.get("BUFFER_FLUSH_INTERVAL_SECONDS")
            or DEFAULT_STREAMING_BUFFER_FLUSH_INTERVAL_SECONDS
        )

    @property
    def enable_parallel_tool_calls(self) -> bool:
        """Whether to enable parallel execution of tool calls in the agent graph."""
        return OidcEnvironmentVariables.str2bool(
            os.environ.get("ENABLE_PARALLEL_TOOL_CALLS", "true")
        )

    @property
    def langgraph_max_concurrency(self) -> int:
        """Maximum concurrency level for LangGraph execution when parallel tool calls are enabled."""
        return int(
            os.environ.get("LANGGRAPH_MAX_CONCURRENCY")
            or DEFAULT_LANGGRAPH_MAX_CONCURRENCY
        )

    @property
    def warm_start_model_names(self) -> list[str]:
        """List of model names to warm start on application startup."""
        raw_value = os.environ.get("WARM_START_MODEL_NAMES")
        if not raw_value:
            return [self.default_model]
        models = [item.strip() for item in raw_value.split(",") if item.strip()]
        return models or [self.default_model]

    @property
    def mcp_fhir_url(self) -> Optional[str]:
        """
        URL of the MCP FHIR agent for demographics and other FHIR-related calls.

        Returns:
            The MCP FHIR agent URL, or None if not configured.
        """
        return os.environ.get("MCP_FHIR_URL")

    @property
    def mcp_fhir_demographics_enabled(self) -> bool:
        """
        Whether to fetch demographics from the MCP FHIR agent before processing requests.

        When enabled, the system will make a preliminary call to get_demographics
        to inject user context as a system message.

        By default this feature is disabled (opt-in). To enable it, set the
        MCP_FHIR_DEMOGRAPHICS_ENABLED environment variable to a truthy value
        (e.g., "true", "1", or "yes").

        Returns:
            True if demographics fetching is enabled, False otherwise.
        """
        return self.str2bool(os.environ.get("MCP_FHIR_DEMOGRAPHICS_ENABLED", "false"))

    @property
    def mcp_fhir_demographics_timeout_seconds(self) -> float:
        """
        Timeout in seconds for the MCP FHIR demographics fetch.

        This is a time-box to ensure the demographics call doesn't block
        the main request if the MCP service is slow or unavailable.

        Returns:
            The timeout in seconds (default: 3.0).
        """
        try:
            return float(os.environ.get("MCP_FHIR_DEMOGRAPHICS_TIMEOUT_SECONDS", "5.0"))
        except ValueError:
            return 5.0

    @property
    def excluded_skills(self) -> set[str]:
        """List of skill names to skip when loading Agent Skills."""
        raw_value = os.environ.get("SKILLS_EXCLUDED")
        if not raw_value or not raw_value.strip():
            return set()
        return {item.strip() for item in raw_value.split(",") if item.strip()}

    @property
    def excluded_skill_groups(self) -> set[str]:
        """List of skill group names to skip when loading Agent Skills."""
        raw_value = os.environ.get("SKILL_GROUPS_EXCLUDED")
        if not raw_value or not raw_value.strip():
            return set()
        return {item.strip() for item in raw_value.split(",") if item.strip()}

    @property
    def skills_directory(self) -> str:
        """Return the absolute path to the Agent Skills directory."""

        configured = os.environ.get("SKILLS_DIRECTORY")
        if configured and configured.strip():
            return configured

        # Compute repository root (three levels up from this file):
        # repo_root / baileyai / utilities / environment / baileyai_environment_variables.py
        repo_root = Path(__file__).resolve().parents[3]
        default_skills_dir = repo_root / "baileyai" / "skills" / "skills"
        if default_skills_dir.is_dir():
            return str(default_skills_dir)

        # Fallback to legacy Docker path for backward compatibility
        return "/usr/src/baileyai/baileyai/skills/skills"

    @property
    def tool_friendly_name_config_path(self) -> str:
        configured = os.environ.get("TOOL_FRIENDLY_NAME_CONFIG_PATH")
        if configured and configured.strip():
            return configured

        repo_root = Path(__file__).resolve().parents[3]
        return str(
            repo_root
            / "baileyai"
            / "language-model-gateway-configs"
            / "tool_friendly_names.json"
        )

    @property
    def prompt_library_path(self) -> Optional[str]:
        configured = os.environ.get("PROMPT_LIBRARY_PATH")
        if configured and configured.strip():
            return configured
        return None
