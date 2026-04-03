from typing import List, Optional, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field


class PromptConfig(BaseModel):
    """Prompt configuration"""

    role: str = "system"
    """The role of the prompt"""

    name: str | None = None
    """Optional prompt library name to load content"""

    content: str | None = None
    """The content of the prompt"""

    hub_id: str | None = None
    """The hub id of the prompt"""

    cache: bool | None = None
    """Whether to cache the prompt"""


class ModelParameterConfig(BaseModel):
    """Model parameter configuration"""

    key: str
    """The key of the parameter"""

    value: float | str | int | bool
    """The value of the parameter"""


class FewShotExampleConfig(BaseModel):
    """Few shot example configuration"""

    input: str
    """The input"""

    output: str
    """The output"""


class HeaderConfig(BaseModel):
    """Header configuration"""

    key: str
    """The key of the header"""

    value: str
    """The value of the header"""


class AgentParameterConfig(BaseModel):
    """Tool parameter configuration"""

    key: str
    """The key of the parameter"""

    value: str
    """The value of the parameter"""


class McpOAuthClientMetadata(BaseModel):
    """Client metadata for OAuth 2.1 Dynamic Client Registration (RFC 7591).

    These fields are sent in the registration request body when dynamically
    registering with an authorization server.
    """

    model_config = ConfigDict(populate_by_name=True)

    client_name: str | None = Field(None, alias="clientName")
    redirect_uris: list[str] | None = Field(None, alias="redirectUris")
    grant_types: list[str] | None = Field(None, alias="grantTypes")
    response_types: list[str] | None = Field(None, alias="responseTypes")
    token_endpoint_auth_method: str | None = Field(
        None, alias="tokenEndpointAuthMethod"
    )
    client_uri: str | None = Field(None, alias="clientUri")
    logo_uri: str | None = Field(None, alias="logoUri")
    contacts: list[str] | None = None


class McpOAuthConfig(BaseModel):
    """OAuth configuration from .mcp.json server entries.

    Supports two styles:
    1. **Discovery-based** – provide ``authServerMetadataUrl`` (OIDC well-known)
       and endpoints are discovered automatically.
    2. **Explicit endpoints** – provide ``authorizationUrl`` and ``tokenUrl``
       directly (no discovery needed).
    """

    model_config = ConfigDict(populate_by_name=True)

    client_id: str | None = Field(None, alias="clientId")
    """The OIDC / OAuth2 client ID."""

    display_name: str | None = Field(None, alias="displayName")
    """Human-readable name for this OAuth provider (used in UI and logging)."""

    audience: str | None = None
    """Expected audience claim for JWT validation.  Defaults to ``client_id``
    when not set.  Useful for IdPs like Okta where the audience differs from
    the client ID."""

    issuer: str | None = None
    """Expected issuer claim for JWT validation.  When omitted, the issuer is
    discovered from ``authServerMetadataUrl``."""

    auth_server_metadata_url: str | None = Field(None, alias="authServerMetadataUrl")
    """The OIDC well-known / server metadata URL (discovery-based flow)."""

    authorization_url: str | None = Field(None, alias="authorizationUrl")
    """The authorization endpoint URL (explicit-endpoints flow)."""

    token_url: str | None = Field(None, alias="tokenUrl")
    """The token endpoint URL (explicit-endpoints flow)."""

    client_secret: str | None = Field(None, alias="clientSecret")
    """Optional client secret for confidential clients."""

    scopes: List[str] | None = None
    """OAuth scopes to request. Defaults to ``["openid", "profile", "email"]``."""

    redirect_uri: str | None = Field(None, alias="redirectUri")
    """Optional redirect URI override for the OAuth callback."""

    callback_port: int | None = Field(None, alias="callbackPort")
    """Optional local callback port for PKCE flows (client-side only)."""

    registration_url: str | None = Field(None, alias="registrationUrl")
    """RFC 7591 Dynamic Client Registration endpoint URL."""

    use_pkce: bool = Field(True, alias="usePKCE")
    """Whether to use PKCE. Defaults to True (OAuth 2.1 standard)."""

    pkce_method: Literal["S256", "plain"] | None = Field("S256", alias="pkceMethod")
    """PKCE challenge method."""

    client_metadata: McpOAuthClientMetadata | None = Field(None, alias="clientMetadata")
    """Client metadata for Dynamic Client Registration."""

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

    name: str
    """The name of the authentication configuration"""

    url: str | None = None
    """The URL to access the authenticated resource"""

    headers: Dict[str, str] | None = None
    """The headers to pass to the MCP tool"""

    auth: Literal["None", "jwt_token", "oauth", "headers"] | None = None
    """The authentication method to use when calling the tool"""

    auth_optional: bool | None = None
    """Whether authentication is optional when calling the tool.  Default is None."""

    auth_providers: List[str] | None = None
    """The auth providers for the authentication. If multiple are provided then the tool accepts ANY of those auth providers.  If auth is needed, we will use the first auth provider."""

    issuers: List[str] | None = None
    """
    The issuers for the authentication.
    If multiple are provided then the tool accepts ANY of those issuers.
    If auth is needed, we will use the first issuer.
    If none is provided then we use the default issuer from the OIDC provider.
    """

    oauth: McpOAuthConfig | None = None
    """OAuth configuration resolved from .mcp.json.  When present, provides the
    OIDC client_id and well-known URL for this tool's MCP server."""

    oauth_providers: List[McpOAuthConfig] | None = None
    """Inline OAuth provider definitions for model-level authentication.
    Each entry defines an OAuth/OIDC provider that callers must authenticate
    with.  At runtime these are registered as auth providers and
    ``auth_providers`` is auto-populated with the generated provider keys."""


class ToolDefinitionConfig(BaseModel):
    """Static tool definition for lazy-loaded MCP tools"""

    name: str
    """The name of the tool"""

    description: str
    """A description of what the tool does"""


class AgentConfig(AuthenticationConfig):
    """Tool configuration"""

    display_name: str | None = None
    """An optional human-readable display name for the tool in the UI"""

    emoji: str | None = None
    """An optional emoji to display alongside the tool in the UI"""

    mcp_server: str | None = None
    """Key into the .mcp.json mcpServers registry.  When set, the url field
    is resolved automatically from the matching server entry at config-load
    time.  This decouples the tool name from the server name and centralises
    MCP server definitions in a single .mcp.json file."""

    parameters: List[AgentParameterConfig] | None = None
    """The parameters for the tool"""

    tools: str | None = None
    """The names of the tool to use in the MCP call.  If none is provided then all tools at the URL will be used. Separate multiple tool names with commas."""

    public_url: str | None = None
    """An unauthenticated URL for MCP tool discovery (tools/list). When specified, this URL is used for listing available tools without authentication, while the main url is used for authenticated tool invocation. This is needed because the MCP spec does not allow unauthenticated access to tools/list when tool invocation requires auth."""

    lazy_load: bool | None = None
    """If true, skip tool discovery from the MCP server and use tool_definitions instead."""

    tool_definitions: List[ToolDefinitionConfig] | None = None
    """Static tool definitions to use when lazy_load is enabled. Each entry provides a tool name and description."""


class ModelConfig(BaseModel):
    """Model configuration"""

    provider: str
    """The provider of the model"""

    model: str
    """The model to use"""


class ChatModelConfig(BaseModel):
    """Model configuration for chat models"""

    id: str
    """The unique identifier for the model"""

    name: str
    """The name of the model"""

    description: str | None = None
    """A description of the model"""

    type: str = "langchain"
    """The type of model"""

    owner: Optional[str] = None
    """The owner of the model"""

    url: str | None = None
    """The URL to access the model"""

    disabled: bool | None = None

    model: ModelConfig | None = None
    """The model configuration"""

    system_prompts: List[PromptConfig] | None = None
    """The system prompts for the model"""

    model_parameters: List[ModelParameterConfig] | None = None
    """The model parameters"""

    headers: List[HeaderConfig] | None = None
    """The headers to pass to url when calling the model"""

    tools: List[AgentConfig] | None = None
    """The tools to use with the model"""

    agents: List[AgentConfig] | None = None
    """The tools to use with the model"""

    skills: List[str] | None = None
    """The skills to enable for the model"""

    example_prompts: List[PromptConfig] | None = None
    """Example prompts for the model"""

    auth_config: AuthenticationConfig | None = None
    """The authentication configuration for the model"""

    request_timeout_seconds: float | None = None
    """Optional override for outbound request timeout when invoking this model.  Default is to use the global default timeout which is 60 seconds."""

    streaming_enabled: bool | None = None
    """Whether the upstream model supports streaming responses; defaults to True"""

    def get_agents(self) -> List[AgentConfig]:
        """Get the agents for the model"""
        return self.agents or self.tools or []
