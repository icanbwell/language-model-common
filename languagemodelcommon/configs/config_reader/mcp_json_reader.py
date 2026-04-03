import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

from oidcauthlib.auth.config.auth_config import AuthConfig

from languagemodelcommon.utilities.config_substitution import substitute_env_vars
from languagemodelcommon.configs.schemas.config_schema import (
    AgentConfig,
    ChatModelConfig,
    McpOAuthConfig,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)

MCP_JSON_PATH_ENV = (
    "MCP_JSON_PATH"  # Environment variable to override the path to .mcp.json
)

MCP_JSON_FILENAME = ".mcp.json"


class McpServerEntry(BaseModel):
    """A single server entry inside .mcp.json ``mcpServers``."""

    type: str | None = None
    url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: Dict[str, str] | None = None

    headers: Dict[str, str] | None = None
    auth: Literal["None", "jwt_token", "oauth", "headers"] | None = None
    auth_optional: bool | None = None
    auth_providers: List[str] | None = None
    issuers: List[str] | None = None

    oauth: McpOAuthConfig | None = None
    """OAuth configuration for this MCP server (clientId, authServerMetadataUrl)."""

    model_config = ConfigDict(extra="allow")


class AuthProviderEntry(BaseModel):
    """A single auth provider entry inside .mcp.json ``authProviders``."""

    model_config = ConfigDict(populate_by_name=True)

    issuer: str | None = None
    audience: str
    client_id: str | None = Field(None, alias="clientId")
    client_secret: str | None = Field(None, alias="clientSecret")
    well_known_uri: str | None = Field(None, alias="wellKnownUri")
    display_name: str | None = Field(None, alias="displayName")
    scope: str | None = None
    extra_info: Dict[str, Any] | None = Field(None, alias="extraInfo")
    authorization_endpoint: str | None = Field(None, alias="authorizationEndpoint")
    token_endpoint: str | None = Field(None, alias="tokenEndpoint")
    registration_url: str | None = Field(None, alias="registrationUrl")


class McpJsonConfig(BaseModel):
    """Root model for the ``.mcp.json`` file."""

    mcpServers: Dict[str, McpServerEntry] = Field(default_factory=dict)
    authProviders: Dict[str, AuthProviderEntry] = Field(default_factory=dict)


def read_mcp_json(config_dir: str | None = None) -> McpJsonConfig | None:
    """Read and parse ``.mcp.json``.

    Resolution order for the file path:
    1. ``MCP_JSON_PATH`` environment variable (absolute path to the file).
    2. ``.mcp.json`` in *config_dir* (the model-configs directory).

    Returns ``None`` when no ``.mcp.json`` is found.
    """
    env_path = os.environ.get(MCP_JSON_PATH_ENV)

    if env_path:
        mcp_json_path = Path(env_path).resolve()
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


def build_auth_configs_from_mcp_json(
    mcp_config: McpJsonConfig,
) -> list[AuthConfig]:
    """Build ``AuthConfig`` objects from the ``authProviders`` section of .mcp.json.

    Each key in ``authProviders`` becomes the ``auth_provider`` name.
    Returns an empty list if no ``authProviders`` section is present.
    """
    configs: list[AuthConfig] = []
    for provider_name, entry in mcp_config.authProviders.items():
        config = AuthConfig(
            auth_provider=provider_name,
            friendly_name=entry.display_name or provider_name,
            audience=entry.audience,
            issuer=entry.issuer,
            client_id=entry.client_id,  # type: ignore[arg-type]  # Optional in oidcauthlib>=3.1
            client_secret=entry.client_secret,
            well_known_uri=entry.well_known_uri,
            scope=entry.scope or "openid profile email",
            extra_info=entry.extra_info,
            authorization_endpoint=entry.authorization_endpoint,
            token_endpoint=entry.token_endpoint,
            registration_url=entry.registration_url,
        )
        configs.append(config)
        logger.info("Parsed auth provider '%s' from .mcp.json", provider_name)
    return configs


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
    *mcp_config* and populate ``url`` from the server entry.  The
    ``mcp_server`` value is left intact for traceability.

    If a referenced server key is not found, a warning is logged and the
    agent's existing ``url`` (if any) is left unchanged as a fallback.
    """
    servers = mcp_config.mcpServers

    for model in configs:
        agents: List[AgentConfig] = model.get_agents()
        for agent in agents:
            if not agent.mcp_server:
                continue
            entry = servers.get(agent.mcp_server)
            if entry is None:
                logger.warning(
                    "MCP server '%s' referenced by tool '%s' in model '%s' "
                    "not found in .mcp.json (available: %s). "
                    "Falling back to inline url '%s'.",
                    agent.mcp_server,
                    agent.name,
                    model.name,
                    list(servers.keys()),
                    agent.url,
                )
                continue
            if entry.url:
                agent.url = entry.url
            if entry.headers and not agent.headers:
                agent.headers = entry.headers
            if entry.auth and not agent.auth:
                agent.auth = entry.auth
            if entry.auth_optional is not None and agent.auth_optional is None:
                agent.auth_optional = entry.auth_optional
            if entry.auth_providers and not agent.auth_providers:
                agent.auth_providers = entry.auth_providers
            if entry.issuers and not agent.issuers:
                agent.issuers = entry.issuers
            if entry.oauth and not agent.oauth:
                agent.oauth = entry.oauth
                # Compute normalized auth_provider key for token scoping
                provider_key = _compute_oauth_provider_key(
                    agent.mcp_server, entry.oauth
                )
                if not agent.auth:
                    agent.auth = "jwt_token"
                if not agent.auth_providers:
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
