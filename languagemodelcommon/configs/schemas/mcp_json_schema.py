from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

from languagemodelcommon.configs.schemas.config_schema import McpOAuthConfig


class McpServerEntry(BaseModel):
    """A single server entry inside .mcp.json ``mcpServers``."""

    type: str | None = Field(
        None,
        description="Transport type (e.g., 'http' for HTTP-based MCP servers).",
    )

    url: str | None = Field(
        None,
        description="The MCP server endpoint URL. Supports ${ENV_VAR} and ${ENV_VAR:-default} substitution.",
    )

    display_name: str | None = Field(
        None,
        alias="displayName",
        description="Human-readable display name for this MCP server. "
        "Used in login prompts and UI instead of the server key.",
    )

    description: str | None = Field(
        None,
        description="Description of the category of tools available from this MCP server. "
        "Used in the system prompt to guide the LLM's tool discovery and in BM25 search indexing.",
    )

    command: str | None = Field(
        None,
        description="Command to launch a stdio-based MCP server.",
    )

    args: list[str] | None = Field(
        None,
        description="Arguments passed to command when launching a stdio server.",
    )

    env: Dict[str, str] | None = Field(
        None,
        description="Environment variables set when launching a server via command.",
    )

    headers: Dict[str, str] | None = Field(
        None,
        description="HTTP headers sent with every request. Supports ${ENV_VAR} substitution. "
        "When an Authorization header is present and auth is not set, auth is automatically set to 'headers'.",
    )

    auth: Literal["None", "jwt_token", "oauth", "headers"] | None = Field(
        None,
        description="Authentication mode. Auto-set to 'jwt_token' when oauth is present, "
        "or 'headers' when headers contains an Authorization key.",
    )

    auth_optional: bool | None = Field(
        None,
        description="When true, requests are allowed even without a valid auth token.",
    )

    auth_providers: List[str] | None = Field(
        None,
        description="Auth provider keys used for token acquisition. "
        "Auto-populated from oauth config when mcp_server is resolved.",
    )

    issuers: List[str] | None = Field(
        None,
        description="Allowed JWT issuer values for token validation. "
        "If multiple are provided, the tool accepts ANY of those issuers.",
    )

    oauth: McpOAuthConfig | None = Field(
        None,
        description="OAuth configuration for this MCP server (clientId, authServerMetadataUrl, etc.).",
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class McpJsonConfig(BaseModel):
    """Root model for the ``.mcp.json`` file."""

    mcpServers: Dict[str, McpServerEntry] = Field(
        default_factory=dict,
        description="Map of server name to MCP server configuration.",
    )
