import json
import logging
import os
from pathlib import Path
from typing import List

from languagemodelcommon.configs.config_reader.github_directory_helper import (
    GitHubDirectoryHelper,
)
from languagemodelcommon.utilities.config_substitution import substitute_env_vars
from languagemodelcommon.configs.schemas.config_schema import (
    AgentConfig,
    ChatModelConfig,
    McpOAuthConfig,
)
from languagemodelcommon.configs.schemas.mcp_json_schema import (
    McpJsonConfig,
)
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)

MCP_JSON_PATH_ENV = (
    "MCP_JSON_PATH"  # Environment variable to override the path to .mcp.json
)

MCP_JSON_FILENAME = ".mcp.json"


class McpJsonReader:
    """Reads and parses ``.mcp.json``, resolving the path via environment variables."""

    def __init__(
        self,
        *,
        environment_variables: LanguageModelCommonEnvironmentVariables | None = None,
        github_directory_helper: GitHubDirectoryHelper | None = None,
    ) -> None:
        self._environment_variables = environment_variables
        self._github_directory_helper = github_directory_helper or (
            GitHubDirectoryHelper(environment_variables=environment_variables)
            if environment_variables
            else None
        )

    def read_mcp_json(
        self,
        config_dir: str | None = None,
    ) -> McpJsonConfig | None:
        """Read and parse ``.mcp.json``.

        Resolution order for the file path:
        1. ``mcp_json_path`` from *environment_variables* (supports ``{pid}``
           substitution for per-worker isolation).
        2. ``MCP_JSON_PATH`` environment variable (fallback when no
           environment_variables is provided).
        3. ``.mcp.json`` in *config_dir* (the model-configs directory).

        Returns ``None`` when no ``.mcp.json`` is found.
        """
        if self._environment_variables is not None:
            env_path: str | None = self._environment_variables.mcp_json_path
        else:
            env_path = os.environ.get(MCP_JSON_PATH_ENV)

        if env_path:
            if GitHubDirectoryHelper.is_github_path(env_path):
                if self._github_directory_helper is None:
                    raise RuntimeError(
                        "GitHubDirectoryHelper is required to resolve GitHub paths"
                    )
                resolved_dir = self._github_directory_helper.resolve_github_path(
                    env_path
                )
            else:
                resolved_dir = Path(env_path).resolve()
            mcp_json_path = (resolved_dir / MCP_JSON_FILENAME).resolve()
        elif config_dir:
            resolved_dir = Path(config_dir).resolve()
            mcp_json_path = (resolved_dir / MCP_JSON_FILENAME).resolve()
            if not mcp_json_path.parent == resolved_dir:
                logger.warning(
                    ".mcp.json path resolved outside config directory: %s",
                    mcp_json_path,
                )
                return None
        else:
            return None

        if not mcp_json_path.is_file():
            logger.debug(".mcp.json not found at %s", mcp_json_path)
            return None

        logger.info("Loading MCP server registry from %s", mcp_json_path)
        with open(mcp_json_path, "r", encoding="utf-8") as f:
            data = substitute_env_vars(json.load(f))
        return McpJsonConfig(**data)


# Keep module-level function for backward compatibility with existing callers
def read_mcp_json(
    config_dir: str | None = None,
) -> McpJsonConfig | None:
    """Backward-compatible wrapper around :class:`McpJsonReader`.

    Prefer injecting ``McpJsonReader`` via the container instead of
    calling this function directly.
    """
    return McpJsonReader().read_mcp_json(config_dir=config_dir)


def _compute_oauth_provider_key(server_key: str, oauth: McpOAuthConfig) -> str:
    """Compute normalized auth_provider key for token scoping.

    Tokens are scoped per authorization-server + client_id pair.
    For Dynamic Client Registration (no pre-configured client_id),
    fall back to the server key so each MCP server gets its own
    token scope.
    """
    if oauth.client_id:
        return f"mcp_oauth_{oauth.client_id}"
    return server_key


def resolve_mcp_servers(
    configs: List[ChatModelConfig],
    mcp_config: McpJsonConfig,
) -> None:
    """Resolve ``mcp_server`` references on every ``AgentConfig``.

    For each agent/tool that has ``mcp_server`` set, look up the key in
    *mcp_config* and populate ``url``, ``headers``, ``auth``, and related
    fields from the server entry.  The ``mcp_server`` value is left intact
    for traceability.

    All connection details for tools with ``mcp_server`` come exclusively
    from ``.mcp.json``; inline values on the tool config are not supported.
    """
    servers = mcp_config.mcpServers

    for model in configs:
        agents: List[AgentConfig] = model.get_agents()

        # Expand wildcard mcp_server="*" into one agent per .mcp.json entry
        expanded: List[AgentConfig] = []
        for agent in agents:
            if agent.mcp_server == "*":
                for server_key in servers:
                    expanded.append(AgentConfig(name=server_key, mcp_server=server_key))
            else:
                expanded.append(agent)

        # Replace agents list on model with expanded version
        if model.agents:
            model.agents = expanded
        else:
            model.tools = expanded
        agents = expanded

        for agent in agents:
            if not agent.mcp_server:
                continue
            entry = servers.get(agent.mcp_server)
            if entry is None:
                logger.warning(
                    "MCP server '%s' referenced by tool '%s' in model '%s' "
                    "not found in .mcp.json (available: %s).",
                    agent.mcp_server,
                    agent.name,
                    model.name,
                    list(servers.keys()),
                )
                continue
            if entry.url:
                agent.url = entry.url
            if entry.display_name:
                agent.display_name = entry.display_name
            if entry.description:
                agent.description = entry.description
            if entry.headers:
                agent.headers = entry.headers
            if entry.auth:
                agent.auth = entry.auth
            if entry.auth_optional is not None:
                agent.auth_optional = entry.auth_optional
            if entry.auth_providers:
                agent.auth_providers = entry.auth_providers
            if entry.issuers:
                agent.issuers = entry.issuers
            if entry.oauth:
                agent.oauth = entry.oauth
                # Compute normalized auth_provider key for token scoping
                provider_key = _compute_oauth_provider_key(
                    agent.mcp_server, entry.oauth
                )
                agent.auth = "jwt_token"
                agent.auth_providers = [provider_key]
            # Auto-set auth to "headers" when headers contain Authorization
            if (
                not agent.auth
                and agent.headers
                and any(k.lower() == "authorization" for k in agent.headers)
            ):
                agent.auth = "headers"
            logger.info(
                "Resolved mcp_server '%s' -> url '%s' for tool '%s' in model '%s'",
                agent.mcp_server,
                agent.url,
                agent.name,
                model.name,
            )

    # Second pass: union scopes for agents sharing the same OAuth provider key
    oauth_groups: dict[str, list[AgentConfig]] = {}
    for model in configs:
        for agent in model.get_agents():
            if agent.oauth and agent.auth_providers:
                key = agent.auth_providers[0]
                oauth_groups.setdefault(key, []).append(agent)

    for key, grouped_agents in oauth_groups.items():
        if len(grouped_agents) <= 1:
            continue
        all_scopes: set[str] = set()
        for a in grouped_agents:
            if a.oauth and a.oauth.scopes:
                all_scopes.update(a.oauth.scopes)
        if all_scopes:
            scope_list = sorted(all_scopes)
            for a in grouped_agents:
                if a.oauth:
                    a.oauth.scopes = scope_list
