"""Tests for LoggingTransport's DEBUG-log hardening.

BAI-955: `handle_async_request` used to log the raw Authorization header
value and the full decoded request body at DEBUG. Neither is a live leak
today (LOG_LEVEL is INFO everywhere in Helm values), but this closes the
rollout-hardening gap so enabling DEBUG never leaks either.
"""

import logging

import httpx2
import pytest

from languagemodelcommon.utilities.logger.logging_transport import LoggingTransport


class TestLoggingTransportAuthorizationHeader:
    @pytest.mark.asyncio
    async def test_does_not_log_raw_authorization_header_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        raw_token = "Bearer super-secret-token-value"

        async def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, request=request)

        transport = LoggingTransport(transport=httpx2.MockTransport(handler))
        request = httpx2.Request(
            "GET", "https://example.test/resource", headers={"authorization": raw_token}
        )

        with caplog.at_level(
            logging.DEBUG,
            logger="languagemodelcommon.utilities.logger.logging_transport",
        ):
            await transport.handle_async_request(request)

        assert raw_token not in caplog.text


class TestLoggingTransportRequestBody:
    @pytest.mark.asyncio
    async def test_does_not_log_raw_request_body_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        marker = "MARKER-DO-NOT-LEAK-REQUEST-BODY"

        async def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, request=request)

        transport = LoggingTransport(transport=httpx2.MockTransport(handler))
        request = httpx2.Request(
            "POST",
            "https://example.test/resource",
            json={"note": marker},
        )

        with caplog.at_level(
            logging.DEBUG,
            logger="languagemodelcommon.utilities.logger.logging_transport",
        ):
            await transport.handle_async_request(request)

        assert marker not in caplog.text
