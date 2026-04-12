import logging
from typing import Any, Dict, List, cast
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from oidcauthlib.auth.auth_helper import AuthHelper
from oidcauthlib.auth.auth_manager import AuthManager
from oidcauthlib.auth.config.auth_config import AuthConfig
from oidcauthlib.auth.config.auth_config_reader import AuthConfigReader
from oidcauthlib.auth.models.auth import AuthInformation

from languagemodelcommon.configs.schemas.config_schema import (
    AgentConfig,
    ChatModelConfig,
    AuthenticationConfig,
    McpOAuthConfig,
)
from languagemodelcommon.auth.exceptions.authorization_mcp_tool_token_invalid_exception import (
    AuthorizationMcpToolTokenInvalidException,
)
from languagemodelcommon.auth.models.token_cache_item import TokenCacheItem
from languagemodelcommon.auth.oauth_provider_registrar import OAuthProviderRegistrar
from languagemodelcommon.auth.tools.tool_auth_manager import ToolAuthManager
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.LLM)


class PassThroughTokenManager:
    def __init__(
        self,
        *,
        auth_manager: AuthManager,
        auth_config_reader: AuthConfigReader,
        tool_auth_manager: ToolAuthManager,
        environment_variables: LanguageModelCommonEnvironmentVariables,
        oauth_provider_registrar: OAuthProviderRegistrar,
    ) -> None:
        self.auth_manager: AuthManager = auth_manager
        if self.auth_manager is None:
            raise ValueError("auth_manager must not be None")
        if not isinstance(self.auth_manager, AuthManager):
            raise TypeError("auth_manager must be an instance of AuthManager")

        self.auth_config_reader: AuthConfigReader = auth_config_reader
        if self.auth_config_reader is None:
            raise ValueError("auth_config_reader must not be None")
        if not isinstance(self.auth_config_reader, AuthConfigReader):
            raise TypeError(
                "auth_config_reader must be an instance of AuthConfigReader"
            )

        self.tool_auth_manager: ToolAuthManager = tool_auth_manager
        if self.tool_auth_manager is None:
            raise ValueError("tool_auth_manager must not be None")
        if not isinstance(self.tool_auth_manager, ToolAuthManager):
            raise TypeError("tool_auth_manager must be an instance of ToolAuthManager")

        self.environment_variables: LanguageModelCommonEnvironmentVariables = (
            environment_variables
        )
        if self.environment_variables is None:
            raise ValueError("environment_variables must not be None")
        if not isinstance(
            self.environment_variables, LanguageModelCommonEnvironmentVariables
        ):
            raise TypeError(
                "environment_variables must be an instance of LanguageModelCommonEnvironmentVariables"
            )

        self._oauth_provider_registrar = oauth_provider_registrar
        if self._oauth_provider_registrar is None:
            raise ValueError("oauth_provider_registrar must not be None")
        if not isinstance(self._oauth_provider_registrar, OAuthProviderRegistrar):
            raise TypeError(
                "oauth_provider_registrar must be an instance of OAuthProviderRegistrar"
            )

    async def check_tokens_are_valid_for_tools(
        self,
        *,
        auth_information: AuthInformation,
        headers: Dict[str, Any],
        model_config: ChatModelConfig,
    ) -> None:
        tools_using_authentication: List[AgentConfig] = (
            [a for a in model_config.get_agents() if a.auth == "jwt_token"]
            if model_config.get_agents() is not None
            else []
        )
        if not tools_using_authentication:
            logger.debug("No tools require authentication.")
            return

        auth_headers = [
            headers.get(key) for key in headers if key.lower() == "authorization"
        ]
        auth_header: str | None = auth_headers[0] if auth_headers else None
        for tool_using_authentication in tools_using_authentication:
            await self.check_tokens_are_valid_for_tool(
                auth_header=auth_header,
                auth_information=auth_information,
                authentication_config=tool_using_authentication,
            )

    async def _ensure_oauth_provider_registered(
        self,
        *,
        auth_provider: str,
        oauth: McpOAuthConfig,
        server_url: str | None = None,
    ) -> AuthConfig:
        """Dynamically register an auth provider from .mcp.json oauth config.

        Delegates to ``OAuthProviderRegistrar`` which handles discovery,
        DCR, AuthConfig construction, per-provider locking, and registration.
        """
        return await self._oauth_provider_registrar.register_provider(
            auth_provider=auth_provider,
            oauth=oauth,
            server_url=server_url,
            auth_manager=self.auth_manager,
        )

    async def _resolve_oauth_providers(
        self,
        authentication_config: AuthenticationConfig,
    ) -> None:
        """Register inline ``oauth_providers`` and populate ``auth_providers``.

        When a model's ``auth_config`` specifies ``oauth_providers`` instead of
        named ``auth_providers``, this method dynamically registers each OAuth
        config and fills in the ``auth_providers`` list so the rest of the auth
        flow can proceed normally.
        """
        if not authentication_config.oauth_providers:
            return
        if authentication_config.auth_providers:
            logger.debug(
                "OAuth providers already resolved for '%s' — auth_providers=%s",
                authentication_config.name,
                authentication_config.auth_providers,
            )
            return

        logger.info(
            "Resolving %d inline oauth_providers for '%s'",
            len(authentication_config.oauth_providers),
            authentication_config.name,
        )

        provider_names: list[str] = []
        for oauth in authentication_config.oauth_providers:
            provider_key = (
                f"oauth_{oauth.client_id}"
                if oauth.client_id
                else f"oauth_{hash(oauth.auth_server_metadata_url)}"
            )
            # Ensure the display_name is set so login prompts show a
            # human-readable name instead of the generated provider key.
            if not oauth.display_name:
                oauth.display_name = (
                    getattr(authentication_config, "display_name", None)
                    or authentication_config.name
                )
            logger.info(
                "Resolving oauth_provider '%s' for tool '%s' (metadata_url=%s)",
                provider_key,
                authentication_config.name,
                oauth.auth_server_metadata_url,
            )
            await self._ensure_oauth_provider_registered(
                auth_provider=provider_key,
                oauth=oauth,
                server_url=authentication_config.url,
            )
            provider_names.append(provider_key)

        authentication_config.auth_providers = provider_names
        if not authentication_config.auth:
            authentication_config.auth = "jwt_token"
        logger.info(
            "Resolved oauth_providers for '%s' — auth_providers=%s",
            authentication_config.name,
            provider_names,
        )

    async def check_tokens_are_valid_for_tool(
        self,
        *,
        auth_header: str | None,
        auth_information: AuthInformation,
        authentication_config: AuthenticationConfig,
    ) -> TokenCacheItem | None:
        # Resolve inline oauth_providers into named auth_providers
        await self._resolve_oauth_providers(authentication_config)

        tool_auth_providers: list[str] | None = authentication_config.auth_providers
        if (
            authentication_config.auth_providers is None
            or len(authentication_config.auth_providers) == 0
        ):
            logger.debug(
                f"Tool {authentication_config.name} doesn't have auth providers."
            )
            return None
        if not auth_information.redirect_uri:
            logger.debug("AuthInformation doesn't have redirect_uri.")
            return None

        if not auth_information.subject:
            logger.debug("AuthInformation doesn't have subject.")
            return None

        tool_first_auth_provider: str | None = (
            tool_auth_providers[0] if tool_auth_providers is not None else None
        )
        auth_config: AuthConfig | None = (
            self.auth_config_reader.get_config_for_auth_provider(
                auth_provider=tool_first_auth_provider
            )
            if tool_first_auth_provider is not None
            else None
        )
        # If not found in static config, try to register from .mcp.json oauth
        if (
            auth_config is None
            and tool_first_auth_provider
            and authentication_config.oauth
        ):
            auth_config = await self._ensure_oauth_provider_registered(
                auth_provider=tool_first_auth_provider,
                oauth=authentication_config.oauth,
                server_url=authentication_config.url,
            )
        if auth_config is None:
            raise ValueError(
                f"AuthConfig not found for auth provider {tool_first_auth_provider}"
                f" used by tool {authentication_config.name}."
            )
        if not tool_first_auth_provider:
            raise ValueError("Tool using authentication must have an auth provider.")
        tool_auth_provider: str = tool_first_auth_provider
        tool_client_id: str | None = (
            auth_config.client_id if auth_config is not None else None
        )
        if not tool_client_id:
            raise ValueError("Tool using authentication must have a client ID.")

        error_message = await self.build_login_message_for_tool(
            auth_information=auth_information,
            authentication_config=authentication_config,
            tool_auth_provider=tool_auth_provider,
        )
        return await self.tool_auth_manager.get_token_for_tool_async(
            auth_header=auth_header,
            error_message=error_message,
            tool_config=authentication_config,
        )

    async def build_login_message_for_tool(
        self,
        *,
        auth_information: AuthInformation,
        authentication_config: AuthenticationConfig,
        tool_auth_provider: str | None = None,
    ) -> str:
        """Build a user-facing error message with login links for a tool.

        Can be called independently of the full token-check flow to produce
        an actionable message when an MCP server rejects a request.
        """
        if tool_auth_provider is None:
            await self._resolve_oauth_providers(authentication_config)
            providers = authentication_config.auth_providers
            tool_auth_provider = providers[0] if providers else None

        authorization_url: str | None = None
        if (
            tool_auth_provider
            and auth_information.redirect_uri
            and auth_information.subject
        ):
            auth_config: AuthConfig | None = (
                self.auth_config_reader.get_config_for_auth_provider(
                    auth_provider=tool_auth_provider
                )
            )
            if auth_config is None and authentication_config.oauth:
                auth_config = await self._ensure_oauth_provider_registered(
                    auth_provider=tool_auth_provider,
                    oauth=authentication_config.oauth,
                    server_url=authentication_config.url,
                )
            if auth_config is not None:
                try:
                    authorization_url = (
                        await self.auth_manager.create_authorization_url(
                            auth_provider=tool_auth_provider,
                            redirect_uri=auth_information.redirect_uri,
                            url=authentication_config.url,
                            referring_email=auth_information.email,
                            referring_subject=auth_information.subject,
                        )
                    )
                except Exception:
                    logger.warning(
                        "Could not create authorization URL for %s",
                        authentication_config.name,
                        exc_info=True,
                    )

        app_login_url_with_parameters: str | None = None
        app_login_uri = self.environment_variables.app_login_uri
        if (
            app_login_uri
            and tool_auth_provider
            and auth_information.email
            and auth_information.subject
        ):
            parsed_login_uri = urlparse(app_login_uri)
            existing_query_params = dict(
                parse_qsl(parsed_login_uri.query, keep_blank_values=True)
            )
            sanitized_login_query_params = {
                "state": AuthHelper.encode_state(
                    content={
                        "auth_provider": tool_auth_provider,
                        "referring_email": auth_information.email,
                        "referring_subject": auth_information.subject,
                    }
                ),
            }
            app_login_url_with_parameters = cast(  # type: ignore[redundant-cast]
                str,
                urlunparse(
                    parsed_login_uri._replace(
                        query=urlencode(
                            {**existing_query_params, **sanitized_login_query_params}
                        )
                    )
                ),
            )

        app_token_save_uri_with_parameters: str | None = None
        app_token_save_uri = self.environment_variables.app_token_save_uri
        if (
            app_token_save_uri
            and tool_auth_provider
            and auth_information.email
            and auth_information.subject
        ):
            parsed_token_save_uri = urlparse(app_token_save_uri)
            existing_query_params = dict(
                parse_qsl(parsed_token_save_uri.query, keep_blank_values=True)
            )
            sanitized_token_save_query_params = {
                "state": AuthHelper.encode_state(
                    content={
                        "auth_provider": tool_auth_provider,
                        "referring_email": auth_information.email,
                        "referring_subject": auth_information.subject,
                    }
                ),
            }
            app_token_save_uri_with_parameters = cast(  # type: ignore[redundant-cast]
                str,
                urlunparse(
                    parsed_token_save_uri._replace(
                        query=urlencode(
                            {
                                **existing_query_params,
                                **sanitized_token_save_query_params,
                            }
                        )
                    )
                ),
            )

        tool_display_name: str = (
            getattr(authentication_config, "display_name", None)
            or authentication_config.name
        )
        error_message: str = (
            "\n"
            + AuthorizationMcpToolTokenInvalidException.build_login_required_message(
                tool_display_name
            )
        )

        login_options: list[str] = []
        if authorization_url:
            oauth_display_name: str | None = (
                authentication_config.oauth.display_name
                if authentication_config.oauth
                and authentication_config.oauth.display_name
                else None
            )
            if oauth_display_name and oauth_display_name != tool_display_name:
                login_display_name = f"{tool_display_name} ({oauth_display_name})"
            else:
                login_display_name = tool_display_name
            login_options.append(
                f"[Login to {login_display_name}]({authorization_url})"
            )
        app_login_allowed: bool = (
            authentication_config.oauth.app_login_allowed
            if authentication_config.oauth
            else False
        )
        if app_login_allowed and app_login_url_with_parameters:
            login_options.append(
                f"[Login to b.well App]({app_login_url_with_parameters})"
            )
        if app_token_save_uri_with_parameters:
            login_options.append(f"[Paste Token]({app_token_save_uri_with_parameters})")

        if len(login_options) == 1:
            error_message += f"\nClick here to {login_options[0]}."
        elif len(login_options) > 1:
            error_message += "\n\nChoose one of the following options to authenticate:"
            for i, option in enumerate(login_options, 1):
                error_message += f"\n{i}. {option}"

        return error_message
