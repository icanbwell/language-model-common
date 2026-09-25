import logging
from collections.abc import AsyncGenerator
from typing import override

import httpx2

from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.HTTP)


class LoggingResponse(httpx2.Response):
    """
    A custom HTTP response class that logs the request and response details.
    This class extends httpx2.Response to log the request method, URL, status code,
    and response content in bytes as they are streamed.
    """

    @override
    async def aiter_bytes(
        self, chunk_size: int | None = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Asynchronously iterate over the response content in bytes, logging each chunk.
        This method overrides the default aiter_bytes method to include logging.
        Args:
            chunk_size: Passed through to the parent method.
        Yields:
            bytes: The next chunk of response content in bytes.
        """
        logger.debug(
            f"====== Response: {self.request.method} {self.url} {self.status_code} ====="
        )
        chunk_count = 0
        total_bytes = 0
        async for chunk in super().aiter_bytes(chunk_size):
            chunk_count += 1
            total_bytes += len(chunk)
            logger.debug(f"Chunk #{chunk_count}: {len(chunk)} bytes")
            yield chunk
        logger.debug(
            f"====== End Response: {self.request.method} {self.url} {self.status_code} "
            f"| chunks: {chunk_count} | total bytes: {total_bytes} ====="
        )
