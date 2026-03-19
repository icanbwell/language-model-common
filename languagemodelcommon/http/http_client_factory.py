from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Optional

import httpx


class HttpClientFactory:
    @asynccontextmanager
    async def create_http_client(
        self,
        *,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = 5.0,
    ) -> AsyncGenerator[httpx.AsyncClient, None]:
        async with httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout
        ) as client:
            yield client
