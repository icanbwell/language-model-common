import logging
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from languagemodelcommon.configs.schemas.config_schema import McpOAuthConfig
from oidcauthlib.auth.well_known_configuration.auth_server_metadata import (
    AuthServerMetadata,
)
from oidcauthlib.auth.well_known_configuration.auth_server_metadata_discovery import (
    AuthServerMetadataDiscovery as OidcAuthServerMetadataDiscovery,
    AuthServerMetadataDiscoveryProtocol as OidcDiscoveryProtocol,
)

from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)

_DISCOVERY_TIMEOUT = httpx.Timeout(10.0)


class McpAuthServerDiscoveryProtocol(Protocol):
    """Discovers OAuth metadata for an MCP server and returns an McpOAuthConfig."""

    async def discover(self, *, mcp_server_url: str) -> McpOAuthConfig | None: ...


class McpAuthServerDiscovery:
    """Discovers OAuth metadata for MCP servers using RFC 9728 Protected Resource
    Metadata, falling back to RFC 8414 path-aware authorization server metadata.

    Discovery order:
    1. Fetch the RFC 9728 Protected Resource Metadata document at
       {origin}/.well-known/oauth-protected-resource{path} — if found, use its
       authorization_servers to discover auth endpoints.
    2. Fall back to the oidc-auth-lib generic discovery (RFC 8414 at the
       server root).
    """

    def __init__(
        self,
        *,
        discovery: OidcDiscoveryProtocol | None = None,
    ) -> None:
        self._discovery = discovery or OidcAuthServerMetadataDiscovery()

    @staticmethod
    def _build_protected_resource_metadata_url(mcp_server_url: str) -> str:
        """Build RFC 9728 protected resource metadata URL.

        Per RFC 9728 §3.1 the URL is:
            {origin}/.well-known/oauth-protected-resource{path}
        """
        parsed = urlparse(mcp_server_url)
        resource_path = parsed.path.rstrip("/") if parsed.path != "/" else ""
        return f"{parsed.scheme}://{parsed.netloc}/.well-known/oauth-protected-resource{resource_path}"

    @staticmethod
    def _to_mcp_oauth_config(metadata: AuthServerMetadata) -> McpOAuthConfig:
        return McpOAuthConfig.model_validate(
            {
                "authorization_url": metadata.authorization_endpoint,
                "token_url": metadata.token_endpoint,
                "registration_url": metadata.registration_endpoint,
                "issuer": metadata.issuer,
                "scopes": metadata.scopes_supported,
            }
        )

    @staticmethod
    def _to_mcp_oauth_config_from_dict(
        metadata: dict[str, Any],
    ) -> McpOAuthConfig | None:
        authorization_endpoint = metadata.get("authorization_endpoint")
        token_endpoint = metadata.get("token_endpoint")
        if not authorization_endpoint or not token_endpoint:
            return None
        scopes: list[str] | None = None
        scopes_supported = metadata.get("scopes_supported")
        if isinstance(scopes_supported, list):
            scopes = [s for s in scopes_supported if isinstance(s, str)]
        return McpOAuthConfig.model_validate(
            {
                "authorization_url": authorization_endpoint,
                "token_url": token_endpoint,
                "registration_url": metadata.get("registration_endpoint"),
                "issuer": metadata.get("issuer"),
                "scopes": scopes,
            }
        )

    async def _discover_via_protected_resource_metadata(
        self, *, mcp_server_url: str
    ) -> McpOAuthConfig | None:
        """RFC 9728: Fetch protected resource metadata, then discover the auth server."""
        prm_url = self._build_protected_resource_metadata_url(mcp_server_url)
        logger.debug("Fetching RFC 9728 protected resource metadata from %s", prm_url)

        async with httpx.AsyncClient(timeout=_DISCOVERY_TIMEOUT) as client:
            try:
                response = await client.get(prm_url)
                if response.status_code != 200:
                    logger.debug(
                        "Protected resource metadata at %s returned %s",
                        prm_url,
                        response.status_code,
                    )
                    return None
                prm = response.json()
            except (httpx.HTTPError, ValueError) as e:
                logger.debug(
                    "Failed to fetch protected resource metadata from %s: %s",
                    prm_url,
                    e,
                )
                return None

        auth_servers = prm.get("authorization_servers")
        if not auth_servers or not isinstance(auth_servers, list):
            logger.debug(
                "Protected resource metadata at %s missing authorization_servers",
                prm_url,
            )
            return None

        scopes_from_resource = prm.get("scopes_supported")
        registration_endpoint = prm.get("registration_endpoint")

        for auth_server_url in auth_servers:
            if not isinstance(auth_server_url, str):
                continue
            config = await self._discover_auth_server(
                auth_server_url=auth_server_url,
                scopes_from_resource=scopes_from_resource,
                registration_endpoint_override=registration_endpoint,
            )
            if config is not None:
                logger.info(
                    "Discovered auth server metadata via RFC 9728 for %s "
                    "(auth server: %s)",
                    mcp_server_url,
                    auth_server_url,
                )
                return config

        return None

    async def _discover_auth_server(
        self,
        *,
        auth_server_url: str,
        scopes_from_resource: list[str] | None = None,
        registration_endpoint_override: str | None = None,
    ) -> McpOAuthConfig | None:
        """Discover auth server metadata from an authorization server URL.

        Tries RFC 8414 path-aware discovery, then falls back to root-level.
        """
        parsed = urlparse(auth_server_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path.rstrip("/") if parsed.path and parsed.path != "/" else ""

        well_known_urls: list[str] = []
        if path:
            well_known_urls.append(
                f"{origin}/.well-known/oauth-authorization-server{path}"
            )
            well_known_urls.append(f"{origin}/.well-known/openid-configuration{path}")
        well_known_urls.append(f"{origin}/.well-known/oauth-authorization-server")
        well_known_urls.append(f"{origin}/.well-known/openid-configuration")

        async with httpx.AsyncClient(timeout=_DISCOVERY_TIMEOUT) as client:
            for url in well_known_urls:
                try:
                    response = await client.get(url)
                    if response.status_code != 200:
                        continue
                    metadata = response.json()
                    config = self._to_mcp_oauth_config_from_dict(metadata)
                    if config is not None:
                        if scopes_from_resource and not config.scopes:
                            config.scopes = scopes_from_resource
                        if (
                            registration_endpoint_override
                            and not config.registration_url
                        ):
                            config.registration_url = registration_endpoint_override
                        return config
                except (httpx.HTTPError, ValueError):
                    continue

        return None

    async def discover(self, *, mcp_server_url: str) -> McpOAuthConfig | None:
        # 1. Try RFC 9728 Protected Resource Metadata (path-aware)
        config = await self._discover_via_protected_resource_metadata(
            mcp_server_url=mcp_server_url
        )
        if config is not None:
            return config

        # 2. Fall back to generic oidc-auth-lib discovery (root-level only)
        result = await self._discovery.discover(resource_url=mcp_server_url)
        if result is None:
            return None

        config = self._to_mcp_oauth_config(result)
        logger.info(
            "Mapped discovered auth server metadata to McpOAuthConfig for %s",
            mcp_server_url,
        )
        return config
