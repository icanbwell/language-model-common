from typing import List, Optional, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field


class PromptConfig(BaseModel):
    """Prompt configuration"""

    role: str = Field(
        "system", description='The role of the prompt (e.g., "system", "user").'
    )

    name: str | None = Field(
        None,
        description="Prompt library name to load content from the prompts/ directory. Mutually exclusive with content.",
    )

    content: str | None = Field(
        None,
        description="Inline prompt content. Mutually exclusive with name.",
    )

    hub_id: str | None = Field(None, description="Hub identifier for the prompt.")

    cache: bool | None = Field(None, description="Whether to cache the prompt.")


class ModelParameterConfig(BaseModel):
    """Model parameter configuration"""

    key: str = Field(
        ...,
        description='Parameter name (e.g., "temperature", "max_tokens", "top_p").',
    )

    value: float | str | int | bool = Field(..., description="Parameter value.")


class FewShotExampleConfig(BaseModel):
    """Few shot example configuration"""

    input: str = Field(..., description="The example input.")

    output: str = Field(..., description="The expected output.")


class HeaderConfig(BaseModel):
    """Header configuration"""

    key: str = Field(..., description="Header name.")

    value: str = Field(
        ...,
        description='Header value. Supports special tokens like "Bearer OPENAI_API_KEY".',
    )


class AgentParameterConfig(BaseModel):
    """Tool parameter configuration"""

    key: str = Field(..., description="Parameter name.")

    value: str = Field(..., description="Parameter value.")


class McpOAuthClientMetadata(BaseModel):
    """Client metadata for OAuth 2.1 Dynamic Client Registration (RFC 7591).

    These fields are sent in the registration request body when dynamically
    registering with an authorization server.
    """

    model_config = ConfigDict(populate_by_name=True)

    client_name: str | None = Field(
        None, alias="clientName", description="Human-readable client name."
    )
    redirect_uris: list[str] | None = Field(
        None, alias="redirectUris", description="Redirect URIs for the client."
    )
    grant_types: list[str] | None = Field(
        None,
        alias="grantTypes",
        description='Grant types the client will use (e.g., ["authorization_code"]).',
    )
    response_types: list[str] | None = Field(
        None,
        alias="responseTypes",
        description='Response types the client will use (e.g., ["code"]).',
    )
    token_endpoint_auth_method: str | None = Field(
        None,
        alias="tokenEndpointAuthMethod",
        description='Authentication method at the token endpoint (e.g., "none", "client_secret_post").',
    )
    client_uri: str | None = Field(
        None, alias="clientUri", description="URL of the client's home page."
    )
    logo_uri: str | None = Field(
        None, alias="logoUri", description="URL of the client's logo."
    )
    contacts: list[str] | None = Field(
        None, description="Contact emails for the client."
    )


class McpOAuthConfig(BaseModel):
    """OAuth configuration from .mcp.json server entries.

    Supports two styles:
    1. **Discovery-based** – provide ``authServerMetadataUrl`` (OIDC well-known)
       and endpoints are discovered automatically.
    2. **Explicit endpoints** – provide ``authorizationUrl`` and ``tokenUrl``
       directly (no discovery needed).
    """

    model_config = ConfigDict(populate_by_name=True)

    client_id: str | None = Field(
        None, alias="clientId", description="The OIDC / OAuth2 client ID."
    )

    display_name: str | None = Field(
        None,
        alias="displayName",
        description="Human-readable name for this OAuth provider (used in UI and logging).",
    )

    audience: str | None = Field(
        None,
        description="Expected audience claim for JWT validation. Defaults to client_id when not set.",
    )

    issuer: str | None = Field(
        None,
        description="Expected issuer claim for JWT validation. Discovered from authServerMetadataUrl when omitted.",
    )

    auth_server_metadata_url: str | None = Field(
        None,
        alias="authServerMetadataUrl",
        description="The OIDC well-known / server metadata URL (discovery-based flow).",
    )

    authorization_url: str | None = Field(
        None,
        alias="authorizationUrl",
        description="The authorization endpoint URL (explicit-endpoints flow).",
    )

    token_url: str | None = Field(
        None,
        alias="tokenUrl",
        description="The token endpoint URL (explicit-endpoints flow).",
    )

    client_secret: str | None = Field(
        None,
        alias="clientSecret",
        description="Client secret for confidential clients.",
    )

    scopes: List[str] | None = Field(
        None,
        description='OAuth scopes to request. Defaults to ["openid", "profile", "email"].',
    )

    redirect_uri: str | None = Field(
        None,
        alias="redirectUri",
        description="Redirect URI override for the OAuth callback.",
    )

    callback_port: int | None = Field(
        None,
        alias="callbackPort",
        description="Local callback port for PKCE flows (client-side only).",
    )

    registration_url: str | None = Field(
        None,
        alias="registrationUrl",
        description="RFC 7591 Dynamic Client Registration endpoint URL.",
    )

    use_pkce: bool = Field(
        True,
        alias="usePKCE",
        description="Whether to use PKCE. Defaults to True (OAuth 2.1 standard).",
    )

    pkce_method: Literal["S256", "plain"] | None = Field(
        "S256", alias="pkceMethod", description="PKCE challenge method."
    )

    client_metadata: McpOAuthClientMetadata | None = Field(
        None,
        alias="clientMetadata",
        description="Client metadata for Dynamic Client Registration.",
    )

    @property
    def scope_string(self) -> str:
        """Return scopes as a space-separated string suitable for OAuth requests."""
        if self.scopes:
            return " ".join(self.scopes)
        return "openid profile email"

    @property
    def is_dcr(self) -> bool:
        """Whether this config uses Dynamic Client Registration (no pre-configured client_id)."""
        return self.client_id is None and self.registration_url is not None


class AuthenticationConfig(BaseModel):
    """Authentication configuration"""

    name: str = Field(..., description="Name of the authentication configuration.")

    url: str | None = Field(
        None, description="URL to access the authenticated resource."
    )

    headers: Dict[str, str] | None = Field(
        None, description="Headers to pass with requests (map of header name to value)."
    )

    auth: Literal["None", "jwt_token", "oauth", "headers"] | None = Field(
        None,
        description="Authentication mode. Auto-set to 'jwt_token' when oauth is present, or 'headers' when headers contain an Authorization key.",
    )

    auth_optional: bool | None = Field(
        None,
        description="When true, requests are allowed even without a valid auth token.",
    )

    auth_providers: List[str] | None = Field(
        None,
        description="Auth provider keys for token acquisition. If multiple are provided, the tool accepts ANY of those providers. The first provider is used when auth is needed.",
    )

    issuers: List[str] | None = Field(
        None,
        description="Allowed JWT issuer values for token validation. If multiple are provided, the tool accepts ANY of those issuers. Falls back to the default issuer from the OIDC provider when not set.",
    )

    oauth: McpOAuthConfig | None = Field(
        None,
        description="OAuth configuration resolved from .mcp.json. Provides the OIDC client_id and well-known URL for this tool's MCP server.",
    )

    oauth_providers: List[McpOAuthConfig] | None = Field(
        None,
        description="Inline OAuth provider definitions for model-level authentication. Each entry defines an OAuth/OIDC provider that callers must authenticate with. At runtime these are registered as auth providers and auth_providers is auto-populated.",
    )


class ToolDefinitionConfig(BaseModel):
    """Static tool definition for lazy-loaded MCP tools"""

    name: str = Field(..., description="Tool name.")

    description: str = Field(..., description="Description of what the tool does.")


class AgentConfig(AuthenticationConfig):
    """Tool configuration"""

    description: str | None = Field(
        None,
        description="Human-readable description of the category of tools available from this MCP server. Used in the system prompt to guide the LLM's tool discovery.",
    )

    display_name: str | None = Field(
        None, description="Human-readable display name for the tool in the UI."
    )

    emoji: str | None = Field(
        None, description="Emoji to display alongside the tool in the UI."
    )

    mcp_server: str | None = Field(
        None,
        description="Key into the .mcp.json mcpServers registry. When set, url, headers, auth, and oauth are resolved from the matching server entry at config-load time.",
    )

    parameters: List[AgentParameterConfig] | None = Field(
        None, description="Parameters for the tool."
    )

    tools: str | None = Field(
        None,
        description="Comma-separated names of specific tools to use from the MCP server. If omitted, all tools at the URL are used.",
    )

    public_url: str | None = Field(
        None,
        description="Unauthenticated URL for MCP tool discovery (tools/list). Used when tool invocation requires auth but discovery does not.",
    )

    lazy_load: bool | None = Field(
        None,
        description="If true, skip tool discovery from the MCP server and use tool_definitions instead.",
    )

    tool_definitions: List[ToolDefinitionConfig] | None = Field(
        None,
        description="Static tool definitions for lazy-loaded tools. Each entry provides a tool name and description.",
    )


class ModelConfig(BaseModel):
    """Model configuration"""

    provider: str = Field(
        ...,
        description='The provider of the model (e.g., "bedrock", "openai").',
    )

    model: str = Field(
        ...,
        description='The model identifier (e.g., "us.anthropic.claude-sonnet-4-6").',
    )


class ChatModelConfig(BaseModel):
    """Model configuration for chat models"""

    id: str = Field(
        ...,
        description="Unique identifier for the model. Used in the API path.",
    )

    name: str = Field(..., description="Human-readable name for the model.")

    description: str | None = Field(
        None,
        description="Description of the model's purpose and capabilities.",
    )

    type: str = Field(
        "langchain",
        description='Model type (e.g., "langchain", "passthru").',
    )

    owner: Optional[str] = Field(None, description="Owner of the model configuration.")

    url: str | None = Field(
        None,
        description="URL to access the model (used with passthru type).",
    )

    disabled: bool | None = Field(
        None,
        description="When true, the model is not loaded at startup.",
    )

    model: ModelConfig | None = Field(
        None, description="Model provider and identifier configuration."
    )

    system_prompts: List[PromptConfig] | None = Field(
        None, description="System prompts for the model."
    )

    model_parameters: List[ModelParameterConfig] | None = Field(
        None,
        description="Model parameters (e.g., temperature, max_tokens).",
    )

    headers: List[HeaderConfig] | None = Field(
        None,
        description="Headers to pass when calling the model URL.",
    )

    tools: List[AgentConfig] | None = Field(
        None, description="Tools available to the model."
    )

    agents: List[AgentConfig] | None = Field(
        None,
        description="Alias for tools. If both are provided, agents takes precedence.",
    )

    skills: List[str] | None = Field(
        None,
        description='Skills to enable for the model. Use ["*"] for all skills.',
    )

    example_prompts: List[PromptConfig] | None = Field(
        None, description="Example prompts shown in the UI."
    )

    auth_config: AuthenticationConfig | None = Field(
        None,
        description="Model-level authentication configuration.",
    )

    request_timeout_seconds: float | None = Field(
        None,
        description="Outbound request timeout override in seconds. Default is 60.",
    )

    streaming_enabled: bool | None = Field(
        None,
        description="Whether the upstream model supports streaming responses. Defaults to true.",
    )

    use_tool_discovery: bool | None = Field(
        None,
        description="When true, uses meta-tool discovery (search_tools + call_tool) instead of loading all MCP tools directly into the LLM context.",
    )

    def get_agents(self) -> List[AgentConfig]:
        """Get the agents for the model"""
        return self.agents or self.tools or []
