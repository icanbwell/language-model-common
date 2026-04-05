from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any, Dict, List

import httpx
from httpx import HTTPStatusError
from langchain_core.tools import BaseTool
from mcp.types import (
    CallToolResult,
    LoggingMessageNotificationParams,
    Tool as MCPTool,
)
from oidcauthlib.auth.exceptions.authorization_needed_exception import (
    AuthorizationNeededException,
)

from languagemodelcommon.auth.exceptions.authorization_mcp_tool_token_invalid_exception import (
    AuthorizationMcpToolTokenInvalidException,
)
from languagemodelcommon.auth.pass_through_token_manager import (
    PassThroughTokenManager,
)
from languagemodelcommon.auth.tools.tool_auth_manager import ToolAuthManager
from languagemodelcommon.configs.schemas.config_schema import AgentConfig
from languagemodelcommon.mcp.auth_server_metadata_discovery import (
    McpAuthServerDiscoveryProtocol,
)
from languagemodelcommon.mcp.callbacks import Callbacks, CallbackContext
from languagemodelcommon.mcp.exceptions.mcp_tool_unauthorized_exception import (
    McpToolUnauthorizedException,
)
from languagemodelcommon.mcp.interceptors.auth import (
    AuthMcpCallInterceptor,
)
from languagemodelcommon.mcp.interceptors.tracing import (
    TracingMcpCallInterceptor,
)
from languagemodelcommon.mcp.interceptors.truncation import (
    TruncationMcpCallInterceptor,
)
from languagemodelcommon.mcp.mcp_client import (
    MCPConnectionConfig,
    create_mcp_session,
    list_all_tools,
    mcp_tool_to_langchain_tool,
    call_mcp_tool_raw,
)
from languagemodelcommon.mcp.tool_catalog import ToolCatalog, ToolResolverProtocol
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS
from languagemodelcommon.utilities.logger.logging_transport import LoggingTransport
from languagemodelcommon.utilities.token_reducer.token_reducer import TokenReducer

# OpenTelemetry propagation for trace context

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


class MCPToolProvider:
    """
    A class to provide tools for the MCP (Model Control Protocol) gateway.
    This class is responsible for managing and providing access to various tools
    that can be used in conjunction with the MCP.
    """

    def __init__(
        self,
        *,
        tool_auth_manager: ToolAuthManager,
        environment_variables: LanguageModelCommonEnvironmentVariables,
        token_reducer: TokenReducer,
        truncation_interceptor: TruncationMcpCallInterceptor,
        tracing_interceptor: TracingMcpCallInterceptor,
        pass_through_token_manager: PassThroughTokenManager,
        auth_server_metadata_discovery: McpAuthServerDiscoveryProtocol,
    ) -> None:
        """
        Initialize the MCPToolProvider with authentication and token management.
        Accepts:
            tool_auth_manager (ToolAuthManager): Manages tool authentication.
            environment_variables (LanguageModelCommonEnvironmentVariables): Provides environment configuration.
            token_reducer (TokenReducer): Handles token management and reduction.
        """
        self.tool_auth_manager = tool_auth_manager
        if self.tool_auth_manager is None:
            raise ValueError("ToolAuthManager must be provided")
        if not isinstance(self.tool_auth_manager, ToolAuthManager):
            raise TypeError("auth_manager must be an instance of ToolAuthManager")

        self.environment_variables = environment_variables
        if self.environment_variables is None:
            raise ValueError("EnvironmentVariables must be provided")
        if not isinstance(
            self.environment_variables, LanguageModelCommonEnvironmentVariables
        ):
            raise TypeError(
                "environment_variables must be an instance of EnvironmentVariables"
            )

        self.token_reducer = token_reducer
        if self.token_reducer is None:
            raise ValueError("TokenReducer must be provided")
        if not isinstance(self.token_reducer, TokenReducer):
            raise TypeError("token_reducer must be an instance of TokenReducer")

        self.truncation_interceptor = truncation_interceptor
        if self.truncation_interceptor is None:
            raise ValueError("TruncationMcpCallInterceptor must be provided")
        if not isinstance(self.truncation_interceptor, TruncationMcpCallInterceptor):
            raise TypeError(
                "truncation_interceptor must be an instance of TruncationMcpCallInterceptor"
            )

        self.tracing_interceptor = tracing_interceptor
        if self.tracing_interceptor is None:
            raise ValueError("TracingMcpCallInterceptor must be provided")
        if not isinstance(self.tracing_interceptor, TracingMcpCallInterceptor):
            raise TypeError(
                "tracing_interceptor must be an instance of TracingMcpCallInterceptor"
            )

        self.pass_through_token_manager = pass_through_token_manager
        if self.pass_through_token_manager is None:
            raise ValueError("PassThroughTokenManager must be provided")
        if not isinstance(self.pass_through_token_manager, PassThroughTokenManager):
            raise TypeError(
                "pass_through_token_manager must be an instance of PassThroughTokenManager"
            )

        self.auth_server_metadata_discovery = auth_server_metadata_discovery

    @staticmethod
    def get_httpx_async_client(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        """
        Get an async HTTP client for making requests to MCP tools.

        Returns:
            An instance of httpx.AsyncClient configured for MCP tool requests.
        """
        return httpx.AsyncClient(
            auth=auth,
            headers=headers,
            timeout=timeout,
            transport=LoggingTransport(httpx.AsyncHTTPTransport()),
        )

    @staticmethod
    async def on_mcp_tool_logging(
        params: LoggingMessageNotificationParams,
        context: CallbackContext,
    ) -> None:
        """Execute callback on logging message notification."""
        logger.info(
            f"MCP Tool Logging - Server: {context.server_name}, Level: {params.level}, Message: {params.data}"
        )

    @staticmethod
    async def on_mcp_tool_progress(
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ) -> None:
        logger.info(
            f"MCP Tool Progress - Server: {context.server_name}, Progress: {progress}, Total: {total}, Message: {message}"
        )

    def _build_connection_config(self, tool_config: AgentConfig) -> MCPConnectionConfig:
        """Build an MCPConnectionConfig from an AgentConfig."""
        url = tool_config.url
        if url is None:
            raise ValueError(f"Tool URL must be provided for: {tool_config.name}")

        tool_call_timeout_seconds: int = (
            self.environment_variables.tool_call_timeout_seconds
        )
        config: MCPConnectionConfig = {
            "url": url,
            "transport": "streamable_http",
            "httpx_client_factory": self.get_httpx_async_client,
            "timeout": timedelta(seconds=tool_call_timeout_seconds),
            "sse_read_timeout": timedelta(seconds=tool_call_timeout_seconds),
        }
        if tool_config.headers:
            config["headers"] = {
                key: os.path.expandvars(value)
                for key, value in tool_config.headers.items()
            }
        return config

    def get_lazy_tools(
        self,
        *,
        tool_config: AgentConfig,
        headers: Dict[str, str],
        auth_interceptor: AuthMcpCallInterceptor,
    ) -> List[BaseTool]:
        """
        Create tools from static definitions without contacting the MCP server.
        When lazy_load is enabled, tool metadata comes from the config's
        tool_definitions instead of being discovered from the server.  The
        actual MCP connection is deferred to tool invocation time.
        """
        if not tool_config.tool_definitions:
            logger.warning(
                f"lazy_load is enabled for {tool_config.name} but no tool_definitions provided"
            )
            return []

        mcp_tool_config = self._build_connection_config(tool_config)

        tools: List[BaseTool] = []
        for tool_def in tool_config.tool_definitions:
            mcp_tool = MCPTool(
                name=tool_def.name,
                description=tool_def.description,
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            )
            langchain_tool = mcp_tool_to_langchain_tool(
                mcp_tool,
                connection=mcp_tool_config,
                callbacks=Callbacks(
                    on_progress=self.on_mcp_tool_progress,
                    on_logging_message=self.on_mcp_tool_logging,
                ),
                tool_interceptors=[
                    auth_interceptor.get_tool_interceptor_auth(),
                    self.tracing_interceptor.get_tool_interceptor_tracing(),
                    self.truncation_interceptor.get_tool_interceptor_truncation(),
                ],
                server_name=tool_config.name,
            )
            tools.append(langchain_tool)

        logger.info(
            f"Created {len(tools)} lazy-loaded tools for {tool_config.name}: "
            f"{[t.name for t in tools]}"
        )
        return tools

    async def _discover_tools_via_public_url(
        self,
        *,
        tool_config: AgentConfig,
        invocation_config: MCPConnectionConfig,
        tool_call_timeout_seconds: int,
        auth_interceptor: AuthMcpCallInterceptor,
    ) -> List[BaseTool]:
        """
        Discover tools via an unauthenticated public_url, then create
        LangChain tools that invoke via the authenticated main url.
        """
        public_url: str = tool_config.public_url  # type: ignore[assignment]
        logger.info(
            f"Discovering tools for {tool_config.name} via public_url: {public_url}"
        )

        # Build an unauthenticated discovery connection
        discovery_config: MCPConnectionConfig = {
            "url": public_url,
            "transport": "streamable_http",
            "httpx_client_factory": self.get_httpx_async_client,
            "timeout": timedelta(seconds=tool_call_timeout_seconds),
            "sse_read_timeout": timedelta(seconds=tool_call_timeout_seconds),
        }
        if tool_config.headers:
            discovery_config["headers"] = {
                key: os.path.expandvars(value)
                for key, value in tool_config.headers.items()
            }

        callbacks = Callbacks(
            on_progress=self.on_mcp_tool_progress,
            on_logging_message=self.on_mcp_tool_logging,
        )
        mcp_callbacks = callbacks.to_mcp_format(
            context=CallbackContext(server_name=tool_config.name)
        )

        # Discover tool metadata from the public (unauthenticated) endpoint
        async with create_mcp_session(
            discovery_config, mcp_callbacks=mcp_callbacks
        ) as session:
            await session.initialize()
            mcp_tools: List[MCPTool] = await list_all_tools(session)

        # Create LangChain tools that use the main url for invocation
        tool_interceptors = [
            auth_interceptor.get_tool_interceptor_auth(),
            self.tracing_interceptor.get_tool_interceptor_tracing(),
            self.truncation_interceptor.get_tool_interceptor_truncation(),
        ]
        tools: List[BaseTool] = [
            mcp_tool_to_langchain_tool(
                mcp_tool,
                connection=invocation_config,
                callbacks=callbacks,
                tool_interceptors=tool_interceptors,
                server_name=tool_config.name,
            )
            for mcp_tool in mcp_tools
        ]

        logger.info(
            f"Discovered {len(tools)} tools for {tool_config.name} via public_url: "
            f"{[t.name for t in tools]}"
        )
        return tools

    async def get_tools_by_url_async(
        self,
        *,
        tool_config: AgentConfig,
        headers: Dict[str, str],
        auth_interceptor: AuthMcpCallInterceptor,
    ) -> List[BaseTool]:
        """
        Get tools by their MCP URL asynchronously.
        This method retrieves tools from the MCP based on the provided URL and headers.
        Args:
            tool_config: An AgentConfig instance containing the tool's configuration.
            headers: A dictionary of headers to include in the request, such as Authorization.
            auth_interceptor: An AuthMcpCallInterceptor instance.
        Returns:
            A list of BaseTool instances retrieved from the MCP.
        """
        if tool_config.lazy_load:
            return self.get_lazy_tools(
                tool_config=tool_config,
                headers=headers,
                auth_interceptor=auth_interceptor,
            )

        safe_header_keys = list(headers.keys()) if headers else []
        logger.info(
            "get_tools_by_url_async called for tool: %s, url: %s, "
            "public_url: %s, header_keys: %s",
            tool_config.name,
            tool_config.url,
            tool_config.public_url,
            safe_header_keys,
        )

        try:
            invocation_config = self._build_connection_config(tool_config)
            tool_call_timeout_seconds: int = (
                self.environment_variables.tool_call_timeout_seconds
            )

            tool_names: List[str] | None = (
                tool_config.tools.split(",") if tool_config.tools else None
            )

            if tool_config.public_url:
                # When public_url is set, use it for unauthenticated tool discovery
                # and the main url for authenticated tool invocation.
                tools = await self._discover_tools_via_public_url(
                    tool_config=tool_config,
                    invocation_config=invocation_config,
                    tool_call_timeout_seconds=tool_call_timeout_seconds,
                    auth_interceptor=auth_interceptor,
                )
            else:
                # Standard path: single URL for both discovery and invocation.
                # Attach auth headers for discovery if needed.
                discovery_config: MCPConnectionConfig = dict(invocation_config)  # type: ignore[assignment]
                if headers and tool_config.auth:
                    if tool_config.auth_providers:
                        resolved_header: (
                            str | None
                        ) = await auth_interceptor.resolve_auth_header_for_discovery(
                            tool_config
                        )
                        if resolved_header:
                            existing_headers = discovery_config.get("headers") or {}
                            discovery_config["headers"] = {
                                **existing_headers,
                                "Authorization": resolved_header,
                            }
                    else:
                        auth_header: str | None = (
                            AuthMcpCallInterceptor._extract_auth_header(headers)
                        )
                        if auth_header:
                            existing_headers = discovery_config.get("headers") or {}
                            discovery_config["headers"] = {
                                **existing_headers,
                                "Authorization": auth_header,
                            }

                callbacks = Callbacks(
                    on_progress=self.on_mcp_tool_progress,
                    on_logging_message=self.on_mcp_tool_logging,
                )
                mcp_callbacks = callbacks.to_mcp_format(
                    context=CallbackContext(server_name=tool_config.name)
                )
                tool_interceptors = [
                    auth_interceptor.get_tool_interceptor_auth(),
                    self.tracing_interceptor.get_tool_interceptor_tracing(),
                    self.truncation_interceptor.get_tool_interceptor_truncation(),
                ]

                try:
                    async with create_mcp_session(
                        discovery_config, mcp_callbacks=mcp_callbacks
                    ) as session:
                        await session.initialize()
                        mcp_tools = await list_all_tools(session)

                    tools = [
                        mcp_tool_to_langchain_tool(
                            mcp_tool,
                            connection=invocation_config,
                            callbacks=callbacks,
                            tool_interceptors=tool_interceptors,
                            server_name=tool_config.name,
                        )
                        for mcp_tool in mcp_tools
                    ]
                except BaseException as e:
                    tool_url = tool_config.url or "unknown"

                    if (
                        self._contains_http_401(e)
                        and tool_config.oauth is None
                        and not tool_config.oauth_providers
                        and tool_url != "unknown"
                    ):
                        discovered = await self._attempt_auth_server_discovery(
                            tool_config=tool_config,
                        )
                        if discovered:
                            raise

                    logger.error(
                        "get_tools_by_url_async Failed to discover tools from "
                        "%s at %s: %s: %s",
                        tool_config.name,
                        tool_url,
                        type(e).__name__,
                        e,
                    )
                    return []

            if tool_names and tools:
                tools = [t for t in tools if t.name in tool_names]
            return tools
        except* HTTPStatusError as e:
            tool_url = tool_config.url or "unknown"
            first_exception = e.exceptions[0]

            tool_name = tool_config.name
            # Attempt auth server discovery on 401 when no OAuth is configured
            if (
                isinstance(first_exception, HTTPStatusError)
                and first_exception.response.status_code == 401
                and tool_config.oauth is None
                and not tool_config.oauth_providers
                and tool_url != "unknown"
            ):
                discovered = await self._attempt_auth_server_discovery(
                    tool_config=tool_config,
                )
                if discovered:
                    raise AuthorizationMcpToolTokenInvalidException(
                        message=AuthorizationMcpToolTokenInvalidException.build_login_required_message(
                            tool_name
                        ),
                        tool_url=tool_url,
                        token=None,
                    ) from e

            logger.error(
                "get_tools_by_url_async HTTP error loading MCP tools from %s: %s %s",
                tool_url,
                type(first_exception).__name__,
                first_exception,
            )
            raise AuthorizationMcpToolTokenInvalidException(
                message=AuthorizationMcpToolTokenInvalidException.build_login_required_message(
                    tool_name
                ),
                tool_url=tool_url,
                token=None,
            ) from e
        except* McpToolUnauthorizedException as e:
            tool_url = tool_config.url or "unknown"
            tool_name = tool_config.name
            unauth_exception = e.exceptions[0]
            logger.error(
                "get_tools_by_url_async MCP Tool Unauthorized error loading MCP tools from %s: %s %s",
                tool_url,
                type(unauth_exception).__name__,
                unauth_exception,
            )
            raise AuthorizationMcpToolTokenInvalidException(
                message=AuthorizationMcpToolTokenInvalidException.build_login_required_message(
                    tool_name
                ),
                tool_url=tool_url,
                token=None,
            ) from e
        except* Exception as e:
            tool_url = tool_config.url or "unknown"
            logger.error(
                "get_tools_by_url_async Failed to load MCP tools from %s: %s %s",
                tool_url,
                type(e.exceptions[0]).__name__,
                e,
            )
            raise e

    @staticmethod
    def _contains_http_401(exc: BaseException) -> bool:
        """Check if an exception tree contains an HTTPStatusError with status 401."""
        if isinstance(exc, HTTPStatusError) and exc.response.status_code == 401:
            return True
        if isinstance(exc, BaseExceptionGroup):
            return any(MCPToolProvider._contains_http_401(e) for e in exc.exceptions)
        return False

    async def _attempt_auth_server_discovery(
        self,
        *,
        tool_config: AgentConfig,
    ) -> bool:
        """Attempt RFC 8414 / OIDC Discovery for an MCP server.

        Returns True if discovery succeeded and the tool_config was updated
        with the discovered OAuth configuration.
        """
        tool_url = tool_config.url
        if not tool_url:
            return False

        logger.info(
            "Attempting auth server discovery for %s at %s",
            tool_config.name,
            tool_url,
        )
        discovered = await self.auth_server_metadata_discovery.discover(
            mcp_server_url=tool_url,
        )
        if discovered is None:
            logger.info(
                "Auth server discovery returned no results for %s", tool_config.name
            )
            return False

        tool_config.oauth = discovered

        # Use the tool's display name so login prompts show a human-readable
        # name instead of the generated provider key (e.g. "oauth_discovered_...").
        if not discovered.display_name:
            discovered.display_name = tool_config.display_name or tool_config.name

        provider_key = (
            f"oauth_{discovered.client_id}"
            if discovered.client_id
            else f"oauth_discovered_{hash(tool_url)}"
        )
        tool_config.auth = "jwt_token"
        tool_config.auth_providers = [provider_key]

        await self.pass_through_token_manager._ensure_oauth_provider_registered(
            auth_provider=provider_key,
            oauth=discovered,
        )

        logger.info(
            "Auth server discovery succeeded for %s — registered provider '%s'",
            tool_config.name,
            provider_key,
        )
        return True

    async def get_tools_async(
        self,
        *,
        tools: list[AgentConfig],
        headers: Dict[str, str],
        auth_interceptor: AuthMcpCallInterceptor,
    ) -> list[BaseTool]:
        # get list of tools from the tools from each agent and then concatenate them
        all_tools: List[BaseTool] = []
        for tool in tools:
            if tool.url is not None:
                logger.info(
                    "get_tools_async Fetching tools from url: %s for tool: %s",
                    tool.url,
                    tool.name,
                )
                try:
                    tools_by_url: List[BaseTool] = await self.get_tools_by_url_async(
                        tool_config=tool,
                        headers=headers,
                        auth_interceptor=auth_interceptor,
                    )
                    all_tools.extend(tools_by_url)
                except* AuthorizationMcpToolTokenInvalidException as auth_eg:
                    auth_exception = auth_eg.exceptions[0]
                    logger.warning(
                        f"get_tools_async No valid auth token for {tool.name} from {tool.url}, "
                        f"skipping tool discovery: {type(auth_exception).__name__}: {auth_exception}"
                    )
                except* AuthorizationNeededException as auth_needed_eg:
                    auth_needed_exception = auth_needed_eg.exceptions[0]
                    logger.warning(
                        f"get_tools_async Authorization needed for {tool.name} from {tool.url}, "
                        f"prompting user to login: {type(auth_needed_exception).__name__}: {auth_needed_exception}"
                    )
                    raise
                except* Exception as conn_eg:
                    conn_exception = conn_eg.exceptions[0]
                    logger.warning(
                        f"get_tools_async Failed to connect to MCP server for {tool.name} at {tool.url}, "
                        f"skipping tool: {type(conn_exception).__name__}: {conn_exception}"
                    )
        return all_tools

    def discover_tool_catalog(
        self,
        *,
        tools: list[AgentConfig],
    ) -> "ToolCatalog":
        """Build a ToolCatalog with lazily-resolved MCP servers.

        Registers each server's metadata (name, description) in the catalog
        without contacting the MCP servers. Actual tool discovery is deferred
        until the LLM calls ``search_tools`` with a matching category.
        """
        catalog = ToolCatalog()
        for tool_config in tools:
            if tool_config.url is None:
                continue
            logger.info(
                "discover_tool_catalog Registering server: %s (url: %s, category: %s)",
                tool_config.name,
                tool_config.url,
                tool_config.description,
            )
            catalog.register_server(
                server_name=tool_config.name,
                category=tool_config.description,
                agent_config=tool_config,
            )
        return catalog

    async def _list_mcp_tools_for_config(
        self,
        *,
        tool_config: AgentConfig,
        headers: Dict[str, str],
        auth_interceptor: AuthMcpCallInterceptor,
    ) -> List[MCPTool]:
        """List raw MCP tools from a configured server (no LangChain conversion)."""
        config = self._build_connection_config(tool_config)

        # Attach auth headers for discovery if needed
        if headers and tool_config.auth:
            if tool_config.auth_providers:
                resolved_header = (
                    await auth_interceptor.resolve_auth_header_for_discovery(
                        tool_config
                    )
                )
                if resolved_header:
                    existing = config.get("headers") or {}
                    config["headers"] = {**existing, "Authorization": resolved_header}
            else:
                auth_header = AuthMcpCallInterceptor._extract_auth_header(headers)
                if auth_header:
                    existing = config.get("headers") or {}
                    config["headers"] = {**existing, "Authorization": auth_header}

        callbacks = Callbacks(
            on_progress=self.on_mcp_tool_progress,
            on_logging_message=self.on_mcp_tool_logging,
        )
        mcp_callbacks = callbacks.to_mcp_format(
            context=CallbackContext(server_name=tool_config.name)
        )

        tool_url = tool_config.url or "unknown"

        try:
            if tool_config.public_url:
                # Use public URL for unauthenticated discovery
                discovery_config: MCPConnectionConfig = {
                    "url": tool_config.public_url,
                    "transport": "streamable_http",
                    "httpx_client_factory": self.get_httpx_async_client,
                    "timeout": config.get("timeout", timedelta(seconds=30)),
                    "sse_read_timeout": config.get(
                        "sse_read_timeout", timedelta(seconds=300)
                    ),
                }
                if tool_config.headers:
                    discovery_config["headers"] = {
                        key: os.path.expandvars(value)
                        for key, value in tool_config.headers.items()
                    }
                async with create_mcp_session(
                    discovery_config, mcp_callbacks=mcp_callbacks
                ) as session:
                    await session.initialize()
                    mcp_tools = await list_all_tools(session)
            else:
                async with create_mcp_session(
                    config, mcp_callbacks=mcp_callbacks
                ) as session:
                    await session.initialize()
                    mcp_tools = await list_all_tools(session)
        except BaseException as e:
            if not self._contains_http_401(e):
                raise

            tool_name = tool_config.name
            if (
                tool_config.oauth is None
                and not tool_config.oauth_providers
                and tool_url != "unknown"
            ):
                discovered = await self._attempt_auth_server_discovery(
                    tool_config=tool_config,
                )
                if discovered:
                    raise AuthorizationMcpToolTokenInvalidException(
                        message=AuthorizationMcpToolTokenInvalidException.build_login_required_message(
                            tool_name
                        ),
                        tool_url=tool_url,
                        token=None,
                    ) from e

            raise AuthorizationMcpToolTokenInvalidException(
                message=AuthorizationMcpToolTokenInvalidException.build_login_required_message(
                    tool_name
                ),
                tool_url=tool_url,
                token=None,
            ) from e

        # Filter by tool names if specified
        if tool_config.tools:
            tool_names = tool_config.tools.split(",")
            mcp_tools = [t for t in mcp_tools if t.name in tool_names]

        return mcp_tools

    def create_tool_resolver(
        self,
        *,
        headers: Dict[str, str],
        auth_interceptor: AuthMcpCallInterceptor,
    ) -> ToolResolverProtocol:
        """Create a resolver that lazily fetches tools from MCP servers.

        The returned object satisfies ``ToolResolverProtocol`` and can be
        passed to ``SearchToolsTool`` for on-demand tool resolution.
        """
        return _BoundToolResolver(
            provider=self,
            headers=headers,
            auth_interceptor=auth_interceptor,
        )

    async def execute_mcp_tool(
        self,
        *,
        tool_name: str,
        arguments: Dict[str, Any],
        agent_config: AgentConfig,
        auth_interceptor: AuthMcpCallInterceptor,
    ) -> CallToolResult:
        """Execute an MCP tool by name, applying the full interceptor chain."""
        config = self._build_connection_config(agent_config)
        callbacks = Callbacks(
            on_progress=self.on_mcp_tool_progress,
            on_logging_message=self.on_mcp_tool_logging,
        )
        return await call_mcp_tool_raw(
            config=config,
            tool_name=tool_name,
            arguments=arguments,
            server_name=agent_config.name,
            callbacks=callbacks,
            tool_interceptors=[
                auth_interceptor.get_tool_interceptor_auth(),
                self.tracing_interceptor.get_tool_interceptor_tracing(),
                self.truncation_interceptor.get_tool_interceptor_truncation(),
            ],
        )


class _BoundToolResolver:
    """Adapts MCPToolProvider into a ToolResolverProtocol for a specific request context."""

    def __init__(
        self,
        *,
        provider: MCPToolProvider,
        headers: Dict[str, str],
        auth_interceptor: AuthMcpCallInterceptor,
    ) -> None:
        self._provider = provider
        self._headers = headers
        self._auth_interceptor = auth_interceptor

    async def resolve_tools(
        self,
        agent_config: AgentConfig,
    ) -> List[MCPTool]:
        try:
            return await self._provider._list_mcp_tools_for_config(
                tool_config=agent_config,
                headers=self._headers,
                auth_interceptor=self._auth_interceptor,
            )
        except AuthorizationMcpToolTokenInvalidException:
            # The MCP server returned 401. Trigger the full token
            # resolution flow which either raises
            # AuthorizationNeededException with login links (user must
            # log in) or succeeds when the token was refreshed.
            await self._auth_interceptor.resolve_auth_for_tool_with_login_links(
                tool_config=agent_config,
            )
            # Token was refreshed — retry the MCP call.
            return await self._provider._list_mcp_tools_for_config(
                tool_config=agent_config,
                headers=self._headers,
                auth_interceptor=self._auth_interceptor,
            )
