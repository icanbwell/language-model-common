"""Shared OAuth provider registration: discovery, DCR, and AuthConfig building.

Consolidates the logic previously duplicated between
``PassThroughTokenManager._ensure_oauth_provider_registered`` and
``GatewayTokenStorageAuthManager._try_register_from_mcp_json``.
"""

import asyncio
import logging
from typing import Dict

import httpx
from oidcauthlib.auth.auth_manager import AuthManager
from oidcauthlib.auth.config.auth_config import AuthConfig
from oidcauthlib.auth.config.auth_config_reader import AuthConfigReader
from oidcauthlib.auth.dcr.dcr_manager import DcrManager

from languagemodelcommon.configs.schemas.config_schema import McpOAuthConfig
from languagemodelcommon.mcp.auth.auth_server_metadata_discovery import (
    McpAuthServerDiscoveryProtocol,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.LLM)


class OAuthProviderRegistrar:
    """Resolves OAuth credentials and registers providers with per-provider locking.

    Handles:
    - In-memory fast-path check (already registered?)
    - Per-provider asyncio locking (prevents duplicate DCR for concurrent requests)
    - RFC 8414 / OIDC Discovery (when endpoints are missing)
    - Dynamic Client Registration via ``DcrManager`` (with MongoDB caching)
    - ``AuthConfig`` construction and registration

    ``auth_manager`` is passed to ``register_provider()`` rather than the
    constructor to avoid circular dependencies when the caller IS the
    auth manager (e.g. ``GatewayTokenStorageAuthManager``).
    """

    def __init__(
        self,
        *,
        dcr_manager: DcrManager,
        auth_config_reader: AuthConfigReader,
        auth_server_metadata_discovery: McpAuthServerDiscoveryProtocol | None = None,
    ) -> None:
        self._dcr_manager = dcr_manager
        self._auth_config_reader = auth_config_reader
        self._auth_server_metadata_discovery = auth_server_metadata_discovery
        self._dcr_locks: Dict[str, asyncio.Lock] = {}

    async def register_provider(
        self,
        *,
        auth_provider: str,
        oauth: McpOAuthConfig,
        server_url: str | None = None,
        auth_manager: AuthManager,
    ) -> AuthConfig:
        """Resolve credentials and register an OAuth provider.

        Returns the ``AuthConfig`` (existing or newly created).  Raises
        ``ValueError`` if a ``client_id`` cannot be resolved.
        """
        # Fast path: already registered in-memory (no lock needed)
        existing = self._auth_config_reader.get_config_for_auth_provider(
            auth_provider=auth_provider
        )
        if existing is not None:
            logger.debug(
                "OAuth provider '%s' already registered in-memory "
                "(client_id=%s) — skipping registration",
                auth_provider,
                existing.client_id,
            )
            return existing

        # Per-provider lock: only one coroutine performs DCR per provider
        if auth_provider not in self._dcr_locks:
            self._dcr_locks[auth_provider] = asyncio.Lock()
        lock = self._dcr_locks[auth_provider]

        async with lock:
            # Re-check after acquiring — another coroutine may have
            # completed registration while we were waiting.
            existing = self._auth_config_reader.get_config_for_auth_provider(
                auth_provider=auth_provider
            )
            if existing is not None:
                logger.info(
                    "OAuth provider '%s' was registered by another request "
                    "while waiting for lock (client_id=%s)",
                    auth_provider,
                    existing.client_id,
                )
                return existing

            # --- Discovery ---
            await self._discover_endpoints_if_needed(
                auth_provider=auth_provider,
                oauth=oauth,
                server_url=server_url,
            )

            # --- DCR ---
            client_id, client_secret = await self._resolve_credentials(
                auth_provider=auth_provider,
                oauth=oauth,
            )

            if not client_id:
                raise ValueError(
                    f"Could not resolve client_id for auth_provider '{auth_provider}'"
                )

            # --- Eagerly resolve endpoints from well-known metadata ---
            # When authorization_url or token_url are missing but
            # auth_server_metadata_url is set, fetch the OIDC discovery
            # document now so the AuthConfig has explicit endpoints.
            # This avoids a fragile lazy fetch inside authlib's
            # create_authorization_url which, if it fails, silently
            # drops the login link.
            if oauth.auth_server_metadata_url and (
                not oauth.authorization_url or not oauth.token_url
            ):
                await self._resolve_well_known_endpoints(
                    auth_provider=auth_provider, oauth=oauth
                )

            # --- Build and register ---
            auth_config = AuthConfig(
                auth_provider=auth_provider,
                friendly_name=oauth.display_name or auth_provider,
                audience=oauth.audience or client_id,
                issuer=oauth.issuer,
                client_id=client_id,
                client_secret=client_secret,
                well_known_uri=oauth.auth_server_metadata_url,
                scope=oauth.scope_string,
                authorization_endpoint=oauth.authorization_url,
                token_endpoint=oauth.token_url,
                use_pkce=oauth.use_pkce,
                pkce_method=oauth.pkce_method,
                registration_url=oauth.registration_url,
            )

            self._auth_config_reader.register_auth_configs(configs=[auth_config])
            await auth_manager.register_dynamic_provider(auth_config=auth_config)

            logger.info(
                "Registered OAuth provider '%s' (client_id=%s, pkce=%s/%s, "
                "audience=%s, token_url=%s)",
                auth_provider,
                client_id,
                oauth.use_pkce,
                oauth.pkce_method,
                auth_config.audience,
                oauth.token_url,
            )

            return auth_config

    async def _discover_endpoints_if_needed(
        self,
        *,
        auth_provider: str,
        oauth: McpOAuthConfig,
        server_url: str | None,
    ) -> None:
        """Perform RFC 8414 / OIDC Discovery when key endpoints are missing.

        Covers two scenarios:
        1. No ``client_id`` and no ``registration_url`` — discover both.
        2. Has ``client_id`` but no well-known URL and no explicit
           ``authorization_url`` — discover endpoints for the login link.
        """
        if not self._auth_server_metadata_discovery or not server_url:
            return

        needs_discovery = (not oauth.client_id and not oauth.registration_url) or (
            oauth.client_id
            and not oauth.auth_server_metadata_url
            and not oauth.authorization_url
        )

        if not needs_discovery:
            return

        logger.info(
            "Discovering auth server metadata for '%s' from %s",
            auth_provider,
            server_url,
        )
        discovered = await self._auth_server_metadata_discovery.discover(
            mcp_server_url=server_url,
        )
        if discovered is None:
            return

        if discovered.registration_url:
            oauth.registration_url = discovered.registration_url
        if discovered.authorization_url and not oauth.authorization_url:
            oauth.authorization_url = discovered.authorization_url
        if discovered.token_url and not oauth.token_url:
            oauth.token_url = discovered.token_url
        if discovered.issuer and not oauth.issuer:
            oauth.issuer = discovered.issuer
        if discovered.scopes and not oauth.scopes:
            oauth.scopes = discovered.scopes

        logger.info(
            "Discovery populated OAuth config for '%s' "
            "(registration_url=%s, authorization_url=%s, token_url=%s)",
            auth_provider,
            oauth.registration_url,
            oauth.authorization_url,
            oauth.token_url,
        )

    async def _resolve_well_known_endpoints(
        self,
        *,
        auth_provider: str,
        oauth: McpOAuthConfig,
    ) -> None:
        """Fetch OIDC discovery metadata and populate missing endpoints.

        Populates ``authorization_url``, ``token_url``, and ``issuer`` on
        *oauth* in-place so the ``AuthConfig`` built afterwards has explicit
        endpoints.  This prevents authlib from needing to do a lazy metadata
        fetch inside ``create_authorization_url``, which can silently fail
        and cause the login link to be omitted.
        """
        url = oauth.auth_server_metadata_url
        if not url:
            return
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                metadata = resp.json()
        except Exception:
            logger.warning(
                "Failed to fetch well-known metadata for '%s' from %s "
                "— login link may be unavailable",
                auth_provider,
                url,
                exc_info=True,
            )
            return

        if not oauth.authorization_url and metadata.get("authorization_endpoint"):
            oauth.authorization_url = metadata["authorization_endpoint"]
        if not oauth.token_url and metadata.get("token_endpoint"):
            oauth.token_url = metadata["token_endpoint"]
        if not oauth.issuer and metadata.get("issuer"):
            oauth.issuer = metadata["issuer"]

        logger.info(
            "Well-known metadata resolved for '%s': "
            "authorization_url=%s, token_url=%s, issuer=%s",
            auth_provider,
            oauth.authorization_url,
            oauth.token_url,
            oauth.issuer,
        )

    async def _resolve_credentials(
        self,
        *,
        auth_provider: str,
        oauth: McpOAuthConfig,
    ) -> tuple[str | None, str | None]:
        """Return ``(client_id, client_secret)`` — from config or via DCR."""
        client_id = oauth.client_id
        client_secret = oauth.client_secret

        dcr_client_name: str | None = (
            (oauth.client_metadata.client_name if oauth.client_metadata else None)
            or oauth.display_name
            or auth_provider
        )

        logger.info(
            "Resolving credentials for '%s' (registration_url=%s, has_client_id=%s)",
            auth_provider,
            oauth.registration_url,
            client_id is not None,
        )

        dcr_result = await self._dcr_manager.resolve_dcr_credentials(
            auth_provider=auth_provider,
            registration_url=oauth.registration_url,
            client_id=oauth.client_id,
            client_name=dcr_client_name,
            client_uri=(
                oauth.client_metadata.client_uri if oauth.client_metadata else None
            ),
            logo_uri=(
                oauth.client_metadata.logo_uri if oauth.client_metadata else None
            ),
            contacts=(
                oauth.client_metadata.contacts if oauth.client_metadata else None
            ),
        )

        if dcr_result is not None:
            logger.info(
                "DCR resolved credentials for '%s' — client_id=%s",
                auth_provider,
                dcr_result.client_id,
            )
            client_id = dcr_result.client_id
            client_secret = dcr_result.client_secret

        return client_id, client_secret
