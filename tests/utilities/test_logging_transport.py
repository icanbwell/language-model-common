"""Tests for LoggingTransport's DEBUG-log hardening.

BAI-955: `handle_async_request` used to log the raw Authorization header
value and the full decoded request body at DEBUG. Neither is a live leak
today (LOG_LEVEL is INFO everywhere in Helm values), but this closes the
rollout-hardening gap so enabling DEBUG never leaks either. It also used to
log every header value via `f"Headers: {request.headers}"` - httpx2's
`SENSITIVE_HEADERS` masking only covers authorization/proxy-authorization,
so any other sensitive header (Cookie, X-Api-Key, etc.) rendered raw. Only
header names are logged now.
"""

import io
import logging

import httpx2
import pytest

from languagemodelcommon.utilities.logger.logging_transport import LoggingTransport

_TRANSPORT_LOGGER_NAME = "languagemodelcommon.utilities.logger.logging_transport"


def _capture_rendered_log_output() -> tuple[logging.Logger, io.StringIO]:
    """Attach a real logging.Handler + Formatter to capture rendered DEBUG text.

    caplog record attributes alone don't prove what a real handler would
    render - this exercises the actual formatting path.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))

    target_logger = logging.getLogger(_TRANSPORT_LOGGER_NAME)
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.DEBUG)
    return target_logger, stream


class TestLoggingTransportHeaderNames:
    @pytest.mark.asyncio
    async def test_logs_header_names_never_values(self) -> None:
        cookie_value = "MARKER-DO-NOT-LEAK-COOKIE-VALUE"
        api_key_value = "MARKER-DO-NOT-LEAK-API-KEY-VALUE"  # pragma: allowlist secret

        async def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, request=request)

        transport = LoggingTransport(transport=httpx2.MockTransport(handler))
        request = httpx2.Request(
            "GET",
            "https://example.test/resource",
            headers={"Cookie": cookie_value, "X-Api-Key": api_key_value},
        )

        target_logger, stream = _capture_rendered_log_output()
        try:
            await transport.handle_async_request(request)
        finally:
            target_logger.handlers.clear()

        rendered = stream.getvalue()
        assert "cookie" in rendered
        assert "x-api-key" in rendered
        assert cookie_value not in rendered
        assert api_key_value not in rendered


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
