from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

from languagemodelcommon.configs.schemas.config_schema import McpOAuthConfig


class McpServerEntry(BaseModel):
    """A single server entry inside .mcp.json ``mcpServers``."""

    type: str | None = None
    """Transport type (e.g., ``"http"`` for HTTP-based MCP servers)."""

    url: str | None = None
    """The MCP server endpoint URL.  Supports ``${ENV_VAR}`` substitution."""

    command: str | None = None
    """Command to launch a stdio-based MCP server."""

    args: list[str] | None = None
    """Arguments passed to ``command`` when launching a stdio server."""

    env: Dict[str, str] | None = None
    """Environment variables set when launching a server via ``command``."""

    headers: Dict[str, str] | None = None
    """HTTP headers sent with every request.  Supports ``${ENV_VAR}``
    substitution.  When an ``Authorization`` header is present and ``auth``
    is not set, ``auth`` is automatically set to ``"headers"``."""

    auth: Literal["None", "jwt_token", "oauth", "headers"] | None = None
    """Authentication mode.  Auto-set to ``"jwt_token"`` when ``oauth`` is
    present, or ``"headers"`` when ``headers`` contains an Authorization key."""

    auth_optional: bool | None = None
    """When ``True``, requests are allowed even without a valid auth token."""

    auth_providers: List[str] | None = None
    """Auth provider keys used for token acquisition."""

    issuers: List[str] | None = None
    """Allowed JWT issuer values for token validation."""

    oauth: McpOAuthConfig | None = None
    """OAuth configuration for this MCP server (clientId, authServerMetadataUrl)."""

    model_config = ConfigDict(extra="allow")


class McpJsonConfig(BaseModel):
    """Root model for the ``.mcp.json`` file."""

    mcpServers: Dict[str, McpServerEntry] = Field(
        default_factory=dict,
        description="Map of server name to MCP server configuration.",
    )
