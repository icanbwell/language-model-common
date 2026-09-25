import logging
from typing import override

import httpx2

from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS
from languagemodelcommon.utilities.logger.logging_response import (
    LoggingResponse,
)
from languagemodelcommon.utilities.logger.summarization import summarize_for_logging

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.HTTP)


class LoggingTransport(httpx2.AsyncBaseTransport):
    """
    A custom HTTP transport that logs request and response details.
    This class extends httpx2.AsyncBaseTransport to log the request method, URL,
    headers, and content before sending the request, and logs the response status code,
    headers, and content as it is streamed back.
    It is designed to be used with httpx2 for asynchronous HTTP requests.
    It logs the request method, URL, headers, and content before sending the request,
    and logs the response status code, headers, and content as it is streamed back.
    This transport can be used to monitor and debug HTTP requests and responses in an application.
    """

    def __init__(self, transport: httpx2.AsyncBaseTransport) -> None:
        """
        Initialize the LoggingTransport with a given transport.
        Args:
            transport (httpx2.AsyncBaseTransport): The underlying transport to wrap.
            This transport will handle the actual HTTP requests and responses.
        """
        self.transport: httpx2.AsyncBaseTransport = transport

    @override
    async def handle_async_request(self, request: httpx2.Request) -> LoggingResponse:
        """
        Handle an asynchronous HTTP request, logging the request details and returning a LoggingResponse.
        Args:
            request (httpx.Request): The HTTP request to handle.
        Returns:
            LoggingResponse: A custom response object that logs the response details.
        """
        # log the request
        logger.debug(f" ====== Request: {request.method} {request.url} =====")
        # Log header names only - never values. httpx2's SENSITIVE_HEADERS
        # masking only covers authorization/proxy-authorization, so any
        # other sensitive header (Cookie, X-Api-Key, etc.) would otherwise
        # render raw here.
        logger.debug(f"Header names: {sorted(request.headers.keys())}")
        # Log presence/length of the Authorization header only - never its raw value.
        if "authorization" in request.headers:
            logger.debug(
                f"Authorization header present: {summarize_for_logging(value=request.headers['authorization'])}"
            )
        if request.content:
            logger.debug(
                f"Content (redacted): {summarize_for_logging(value=request.content)}"
            )

        try:
            response = await self.transport.handle_async_request(request)

            return LoggingResponse(
                status_code=response.status_code,
                headers=response.headers,
                stream=response.stream,
                extensions=response.extensions,
            )
        except httpx2.HTTPError as e:
            logger.exception(f"HTTP error occurred: {e}")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}")
            raise
