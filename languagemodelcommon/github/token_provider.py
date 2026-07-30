from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Protocol, runtime_checkable

import httpx
import jwt

logger = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"
_TOKEN_EXPIRY_BUFFER_SECONDS = 300


@runtime_checkable
class GitHubTokenProvider(Protocol):
    """Protocol for providing GitHub authentication tokens."""

    async def get_token(self) -> str: ...

    def get_token_sync(self) -> str: ...


class StaticTokenProvider:
    """Returns a pre-configured token (PAT or pre-minted installation token)."""

    def __init__(self, *, token: str) -> None:
        self._token = token

    async def get_token(self) -> str:
        return self._token

    def get_token_sync(self) -> str:
        return self._token


class GitHubAppTokenProvider:
    """Mints short-lived installation tokens via GitHub App credentials.

    Tokens are cached and automatically refreshed 5 minutes before expiry.
    The async path is serialized via asyncio.Lock to prevent duplicate mints.
    """

    def __init__(
        self,
        *,
        app_id: str,
        private_key: str,
        installation_id: str,
    ) -> None:
        self._app_id = app_id
        self._private_key = private_key
        self._installation_id = installation_id
        self._cached_token: str | None = None
        self._cached_expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        async with self._lock:
            if self._is_token_valid():
                return self._cached_token  # type: ignore[return-value]
            return await self._mint_token()

    def get_token_sync(self) -> str:
        if self._is_token_valid():
            return self._cached_token  # type: ignore[return-value]
        return self._mint_token_sync()

    def _is_token_valid(self) -> bool:
        return (
            self._cached_token is not None
            and time.time() < self._cached_expires_at - _TOKEN_EXPIRY_BUFFER_SECONDS
        )

    def _build_jwt(self) -> str:
        now = int(time.time())
        payload = {
            "iat": now - 60,
            "exp": now + 600,
            "iss": self._app_id,
        }
        return jwt.encode(payload, self._private_key, algorithm="RS256")

    def _request_url(self) -> str:
        return f"{_GITHUB_API_BASE}/app/installations/{self._installation_id}/access_tokens"

    def _request_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._build_jwt()}",
            "Accept": "application/vnd.github+json",
        }

    def _cache_response(self, *, data: dict[str, str]) -> str:
        self._cached_token = data["token"]
        self._cached_expires_at = datetime.fromisoformat(
            data["expires_at"].replace("Z", "+00:00")
        ).timestamp()
        logger.debug(
            "Minted new GitHub App installation token (expires %s)",
            data["expires_at"],
        )
        return self._cached_token

    async def _mint_token(self) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._request_url(), headers=self._request_headers()
            )
        response.raise_for_status()
        return self._cache_response(data=response.json())

    def _mint_token_sync(self) -> str:
        with httpx.Client() as client:
            response = client.post(self._request_url(), headers=self._request_headers())
        response.raise_for_status()
        return self._cache_response(data=response.json())
